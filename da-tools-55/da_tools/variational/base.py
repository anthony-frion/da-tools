import math
import warnings
from typing import Callable, List, Tuple, Type, Union

import torch
from da_tools.observation.data import ObservationSet
from da_tools.observation.operators import ObservationOperator
from da_tools.probability.distributions import DiagonalGaussian, Distribution
from da_tools.util.data import impose_batch_size
from da_tools.util.initialization import naive_initialization
from da_tools.util.state_space import rollout, same_shape
from da_tools.util.typing import ListOfStates, State
from tensordict import TensorDict
from torch import full_like, searchsorted, Tensor
from torch.optim.optimizer import Optimizer


def sliding_windows(
    obs_times: Tensor,
    window_duration: Union[float, int],
    window_shift: Union[float, int] = None,
    discard_partial: bool = False,
) -> Tuple:
    """Get time windows for sliding window data assimilation.

    Args:
        obs_times (Tensor): 1D tensor containing times for each observation
        window_duration (Union[float, int]): length of assimilation window in time units (float) or time steps (int)
        window_shift (Union[float, int], optional): shift between consecutive windows,
            as a fraction of window size (float) or as a number of time steps (int). Defaults to 0.25.
        discard_partial (bool, optional): Discard windows extending beyond final observation, defaults False.
    """
    assert isinstance(obs_times, Tensor), f"Invalid type for obs_times: {type(obs_times)}"
    assert len(obs_times.shape) == 1, f"Invalid shape for obs_times: {obs_times.shape}"
    assert type(window_duration) in [float, int], f"Invalid type for window_duration: {type(window_duration)}"
    assert type(window_shift) in [float, int], f"Invalid type for window_shift: {type(window_shift)}"
    t0 = obs_times[0]
    trange = obs_times[-1] - t0
    dt = obs_times.diff()
    mean_dt = dt.mean()

    fixed_dt = torch.allclose(dt, mean_dt, atol=1e-6)
    if obs_times.dtype is torch.float32:
        warnings.warn(
            "using float32 for observation times can lead to precision issues when checking for fixed dt in long "
            "sequences. HINT: try generating and providing time tensor using double precision if shape mismatch "
            "assertion presents."
        )
    # try to do everything with integers if the shift and duration are multiples of a fixed dt
    if fixed_dt:
        r = (window_duration / mean_dt).item() if isinstance(window_duration, float) else window_duration
        s = r * window_shift if isinstance(window_shift, float) else window_shift
        if abs(r - round(r)) < 1e-6 and abs(s - round(s)) < 1e-6:
            r, s = round(r), round(s)
            assert r > 0 and s > 0, "window duration and shift must remain positive after rounding"
            if discard_partial:
                n_windows = (len(obs_times) - r) // s
            else:
                n_windows = math.ceil(len(obs_times) / s)
            idx_ranges = [(n * s, min(n * s + r, len(obs_times))) for n in range(n_windows)]
            windows = [(obs_times[s] - mean_dt / 2.0, obs_times[e - 1] + mean_dt / 2.0) for (s, e) in idx_ranges]
            return torch.tensor(windows), idx_ranges

    # time shift between consecutive windows
    assert window_shift > 0 and window_duration > 0, "shift and window duration must be strictly positive"

    if isinstance(window_shift, float):
        assert window_shift <= 1, "shifts of more than one full window are not supported"
        if isinstance(window_duration, float):
            shift = window_duration * window_shift if window_shift < 1.0 else window_duration
            if isinstance(window_duration, int) and shift % 1 == 0 and obs_times.dtype in (torch.int32, torch.int64):
                shift = int(shift)
            if discard_partial:
                n_windows = int((trange - window_duration) // shift)  # round down
            else:
                n_windows = math.ceil(trange / shift)  # this doesn't do a window starting on the final obs. correct?

            windows = [(t0 + n * shift, t0 + n * shift + window_duration) for n in range(n_windows)]

        elif isinstance(window_duration, int):
            shift = int(window_duration * window_shift) if window_shift < 1.0 else window_duration  # integer shift
            if discard_partial:
                n_windows = len(obs_times) // shift
            else:
                n_windows = math.ceil(len(obs_times) / shift)

            windows = [(obs_times[n * shift], obs_times[n * shift + window_duration]) for n in range(n_windows)]

    elif isinstance(window_shift, int):  # window_shift and window_duration are both int
        shift = window_shift  # integer shift
        if discard_partial:
            n_windows = len(obs_times) // shift
        else:
            n_windows = math.ceil(len(obs_times) / shift)
        if isinstance(window_duration, int):
            assert window_shift <= window_duration, "shifts of more than one full window are not supported"
            windows = [(obs_times[n * shift], obs_times[n * shift + window_duration]) for n in range(n_windows)]
        elif isinstance(window_duration, float):
            windows = [(obs_times[n * shift], obs_times[n * shift] + window_duration) for n in range(n_windows)]
    assert shift > 0, "shift must be strictly positive"

    idx_ranges = []  # index ranges into observation_times for each window
    s = 0
    for t_start, t_end in windows:
        s += searchsorted(obs_times[s:], t_start, right=False).item()
        e = (
            s + searchsorted(obs_times[s:], t_end, right=True).item()
        )  # index of first observation after the current window
        idx_ranges.append((s, e))

    return torch.tensor(windows), idx_ranges


def process_extra_inputs(dynamic_inputs: State, static_inputs: State, time_axis: Tensor):
    """Check and process dynamic/static inputs for 4dvar.

    Args:
        dynamic_inputs (State): dynamic inputs to check
        static_inputs (State): static inputs to check
        time_axis (Tensor): timeaxis

    Returns:
        dynamic_inputs, static_inputs: checked inputs as State objects
    """
    T = time_axis.nelement()
    if dynamic_inputs is None:
        dynamic_inputs = State(TensorDict(batch_size=(1, T)), time_axis=time_axis)  # State with no fields
    else:
        assert isinstance(dynamic_inputs, State), "extra_inputs must be of type State"
        assert dynamic_inputs.fields.batch_size == (1, T), "shape mismatch"
    if static_inputs is None:
        static_inputs = State(TensorDict(batch_size=(1, 1)))  # State with no fields
    return dynamic_inputs, static_inputs


def process_4dvar_inputs(
    obs_op: ObservationOperator,
    m_dyn: Callable,
    dynamic_inputs: State,
    static_inputs: State,
    x_init: State,
    background_prior: Distribution,
    observations: ObservationSet,
    model_error_distribs: Distribution = None,
    dt: float = None,
    window_duration: Union[float, int] = None,
    window_shift: Union[float, int] = None,
    discard_partial: bool = False,
    rollout_init: bool = True,
) -> Tuple[Tensor, Union[List, Tensor], State, Distribution]:
    """Check/update inputs for 4dvar routines.

    Args:
        obs_op (ObservationOperator): mapping from system state sequence to observations
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        x_init (State): first guess for initial state (hard constraint) or all states (weak constraint).
          Uses prior mean if None. should have a leading batch dimension, followed by time dimension for weak
            constraint.
        background_prior (Distribution): prior distribution in initial state
        observations (ObservatationSet): observations data to be assimilated
        model_error_distribs (StateDistribution): distributions of model errors at
          each time step. Must have length 1 on batch axis.
        dt (float, optional): the prediction time step.
            Only used for weak-constraint when obs_op has an irregular time axis.
        window_duration (Union[float, int]): length of assimilation window in time units (float) or time steps (int)
        window_shift (Union[float, int], optional): shift between consecutive windows,
            as a fraction of the window size (float) or as a number of time steps (int). Defaults to 0.25.
        discard_partial (bool, optional): Discard windows extending beyond final observation, defaults False.
        rollout_init (bool): indicates whether to copy or rollout the input state if it is only one time step,
            for weak-constraint 4DVar
    """
    is_weak = model_error_distribs is not None
    T = obs_op.time_axis.nelement()

    assert isinstance(observations, ObservationSet), "Observations must be of type ObservationSet"
    if hasattr(observations, "time_axis"):
        assert (obs_op.time_axis == observations.time_axis).all(), "time axis mismatch"
    if hasattr(observations, "valid_interval"):
        assert (
            observations.valid_time_interval[0] >= obs_op.time_axis[0]
            and observations.valid_time_interval[1] <= obs_op.time_axis[-1]
        ), "observations outside operator time range"
    if isinstance(window_duration, Tensor):
        assert window_duration.nelement() == 1, " window_duration should have a single value or be a float or int"
        window_duration = window_duration.item()

    if window_duration is not None:
        windows, idx_ranges = sliding_windows(
            obs_op.time_axis, window_duration, window_shift=window_shift, discard_partial=discard_partial
        )
        ntimepoints_init = idx_ranges[0][1] - idx_ranges[0][0]  # window length to define/check wc4dvar initialization
    else:
        ntimepoints_init, windows, idx_ranges = T, None, None

    dynamic_inputs, static_inputs = process_extra_inputs(dynamic_inputs, static_inputs, obs_op.time_axis)

    if background_prior is not None:
        assert isinstance(background_prior, Distribution), "background_prior must be of type"
        " da_tools.probability.distrubution"
        assert isinstance(background_prior, DiagonalGaussian), "only diagonal Gaussian background priors"
        " are currently suported"
        background_prior.mu = impose_batch_size(background_prior.mu, (1, 1))
        background_prior.sigma = impose_batch_size(background_prior.sigma, (1, 1))

    if x_init is not None:
        assert x_init.fields.shape[0] == 1, "batching is not supported for classical 4dvar"
        if background_prior is not None:
            assert same_shape(x_init.fields[:1, :1], background_prior.mean), "size mismsatch"
        if is_weak:
            obs_op_dt = obs_op.time_axis.diff()
            if dt is None:
                assert torch.allclose(
                    obs_op_dt, obs_op_dt[0], atol=1e-6, rtol=1e-3
                ), "When the observation operator has an irregular time axis, dt is expected to be specified."
                assert x_init.fields.shape[1] == ntimepoints_init, "sequence length mismatch"
            else:
                assert not torch.allclose(
                    obs_op_dt, obs_op_dt[0], atol=1e-6, rtol=1e-3
                ), "When the observation operator has a regular time axis, dt is expected to be unspecified."
        else:
            assert x_init.fields.shape[1] == 1, "sequence length mismatch"
    elif not is_weak:
        mu = background_prior.mean
        time_axis = obs_op.time_axis[:1]
        x_init = State(mu, time_axis=time_axis)

    if is_weak:
        assert isinstance(model_error_distribs, Distribution), "model_error_distribs must be of type"
        " da_tools.probability.distrubution"
        assert isinstance(model_error_distribs, DiagonalGaussian), "only diagonal Gaussian model error distributions"
        " are currently suported"
        x_init = process_wc4dvar_init(x_init, m_dyn, background_prior, observations, obs_op, rollout_init=rollout_init)
        if dt is None:
            model_error_distribs.mu = impose_batch_size(model_error_distribs.mu, (1, T - 1))
            model_error_distribs.sigma = impose_batch_size(model_error_distribs.sigma, (1, T - 1))
            assert same_shape(
                x_init.fields[:, 1:], model_error_distribs.mu[:, : ntimepoints_init - 1]
            ), "dimension mismatch between state sequence and model error prior"
        else:
            T_ = x_init.time_axis.nelement()
            model_error_distribs.mu = impose_batch_size(model_error_distribs.mu, (1, T_ - 1))
            model_error_distribs.sigma = impose_batch_size(model_error_distribs.sigma, (1, T_ - 1))
    x_init = x_init.detach().clone()  # might need to change this if we're differentiating through the 4dvar operations
    x_init.fields.requires_grad_(True)
    return dynamic_inputs, static_inputs, x_init, windows, idx_ranges, model_error_distribs


def process_wc4dvar_init(
    x_init: State,
    m_dyn: Callable,
    background_prior: Distribution,
    observations: ObservationSet,
    obs_op: ObservationOperator,
    rollout_init: bool = True,
    dynamic_inputs: State = None,
    static_inputs: State = None,
):
    """Returns the time series corresponding to the initial state of the weak-constraint 4DVar optimization.

    Args:
        x_init (State): a State passed as input to wc4dvar, containing either one time step or the whole time range
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        background_prior (Distribution): prior distribution of the initial state
        observations (ObservationSet): list of observations at each time point
        obs_op (ObservationOperator): the observation operator for wc4dvar,
            notably containing information on the time range of the problem
        rollout_init (bool): if x_init is just the state at one time steps, decides whether to copy it or roll it out

    Returns:
        x_init (State): the processed initial state with the right format
    """
    if x_init is None and background_prior is not None:
        x_init = State(background_prior.mean)
    if isinstance(x_init, State):
        if x_init.fields.batch_size[1] == obs_op.time_axis.nelement():
            x_init.time_axis = obs_op.time_axis
        elif x_init.fields.batch_size[1] == 1:
            if rollout_init:
                x_init = rollout(m_dyn, obs_op.time_axis, x_init, dynamic_inputs, static_inputs)
            else:
                x_init = x_init.expand_time(obs_op.time_axis)
    else:
        assert x_init is None, "at this point of the processing x_init has to be a State or None"
        x_init = naive_initialization(observations)
        assert not torch.isnan(x_init).any(), (
            "naive initialization failed: some fields/locations have no data. To "
            "solve this problem, provide an x_init or background_prior"
        )
    return x_init


def sliding_window_4dvar(
    da_function: Callable,
    m_dyn: Callable,
    observations: ObservationSet,  # array or less
    obs_op: ObservationOperator,
    window_duration: Union[float, int],
    window_shift: Union[float, int] = 0.25,
    model_error_distribs: Distribution = None,
    discard_partial: bool = False,
    x_init: State = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    background_prior: Distribution = None,
    dt: float = None,
    optimizer_class: Type[Optimizer] = torch.optim.LBFGS,
    optimizer_pars: dict = None,
    n_steps: int = 10,
    **kwargs,
) -> ListOfStates:
    """Weak or hard constraint 4DVAR for overlapping windows of observations.

    Args:
        da_function (Callable): data assimiliation function to be used on each window
        m_dyn (Callable): time stepping operator, taking inputs x, t, dt, extra_inputs
        observations (ObservatationSet): set of observations to be assimilated.
        obs_op (ObservationOperator): observation operator used to map between system states and observations
        window_duration (Union[float, int]): window length in time units
        window_shift (Union[float, int], optional): shift between consecutive windows,
            as a fraction of window size (float) or as a number of time steps (int). Defaults to 0.25.
        model_error_distribs (Distribution): distributions of model errors at each time step (None for hard constraint)
        dt (float, optional): the prediction time step. Only used when obs_op has an irregular time axis.
        discard_partial (bool, optional): Drop windows extending beyond last obs, default False.
        x_init (State): first guess of initial state (hard constriant) or state trajectory (weak constraint).
          Uses prior mean if None. Should have leading batch, then time dimension (of length 1 for hard constraint).
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        background_prior (StateDistribution, optional): background prior for first window. Defaults to None.
        optimizer_class: class of a pytorch optimizer, default is LBFGS
        optimizer_pars (dict): parameters for optimizer (learning rate etc.)
        n_steps (int): the number of optimization steps
        verbose (bool): if True, print loss at each iteration. default is False

    Returns:
        ListOfStates: initial state for each assimilation window
    """
    is_weak = model_error_distribs is not None
    dynamic_inputs, static_inputs, x_init, windows, idx_ranges, model_error_distribs = process_4dvar_inputs(
        obs_op,
        m_dyn,
        dynamic_inputs,
        static_inputs,
        x_init,
        background_prior,
        observations,
        model_error_distribs=model_error_distribs,
        dt=dt,
        window_duration=window_duration,
        window_shift=window_shift,
        discard_partial=discard_partial,
    )
    if background_prior is not None:
        assert isinstance(background_prior, DiagonalGaussian), "background prior must be diagonal Gaussian."

    x_eachwin = []  # IC (hc) or state trajectory (wc) for each window

    for i, ((s, e), (t_s, t_e)) in enumerate(zip(idx_ranges, windows)):
        # positional arguments for this window
        da_args = [m_dyn, observations.restrict_time_domain(t_s, t_e), obs_op.restrict_time_domain(t_s, t_e)]
        if is_weak:
            da_args.append(model_error_distribs[:, s : e - 1])  # Distribution class supports indexing

        # keyword arguments for this window
        da_kwargs = dict(
            x_init=x_init,
            dynamic_inputs=dynamic_inputs[:, s : e - 1],
            static_inputs=static_inputs,
            background_prior=background_prior,
            optimizer_class=optimizer_class,
            optimizer_pars=optimizer_pars,
            n_steps=n_steps,
            check_inputs=False,  # already checked once, don't need to check again for each window
        )

        x = da_function(*da_args, **da_kwargs, **kwargs)
        x_eachwin.append(x)

        if i + 1 < len(idx_ranges):  # prepare for next window
            next_s, next_e = idx_ranges[i + 1]
            with torch.no_grad():
                if is_weak:
                    # to initialize the next window's trajectory, we use some of this window's trajectory and extend
                    # with a rollout as needed
                    x_init_fromprev = x[:, next_s - s :]
                    if (
                        e == next_e
                    ):  # this window has reached the end of the observations, no need for a forecast rollout
                        x_init = x_init_fromprev
                    else:  # generate a forecast to initialize unknown system states for next window
                        t_rollout = obs_op.time_axis[e - 1 : next_e]
                        x_init_fromroll = rollout(
                            m_dyn,
                            t_rollout,
                            x[:, -1:],
                            dynamic_inputs=dynamic_inputs[:, e - 1 : next_e - 1],  # include dynamic_inputs for IC
                            static_inputs=static_inputs,
                        )[
                            :, 1:
                        ]  # discard IC

                        x_init = State(
                            torch.cat([x_init_fromprev.fields, x_init_fromroll.fields], dim=1),
                            time_axis=torch.cat([x_init_fromprev.time_axis, x_init_fromroll.time_axis], dim=0),
                        )
                else:  # for hard constraint, we roll from start of this window to start of the next
                    t_rollout = obs_op.time_axis[s : next_s + 1]
                    x_init = rollout(m_dyn, t_rollout, x, dynamic_inputs[:, s:next_s], static_inputs)[
                        :, -1:
                    ]  # only final state

            x_init = x_init.detach().clone()
            x_init.fields.requires_grad_(True)
            if background_prior is not None:  # shift mean, keep covariance of background prior
                mu = x_init.fields[:, :1] if is_weak else x_init.fields
                mu = mu.detach().clone().reshape(background_prior.mu.shape)  # remove batch dimension
                background_prior = DiagonalGaussian(mu, background_prior.sigma)

    return x_eachwin, windows, idx_ranges


def assemble_analysis(
    x_eachwin: ListOfStates,
    idx_ranges,
    time_axis: Tensor,
    is_weak: bool,
    m_dyn: Callable = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
) -> State:
    """Combines the results of a windowed DA method into a single analysis for all time steps.

    Args:
        x_eachwin (ListOfStates): assimilated state(s) for each analysis window
        idx_ranges: range of observations for each window
        time_axis (Tensor): global list of observation times for all windows
        is_weak (bool): was this a weak_constraint method
        m_dyn (Callable): time stepping operator, taking inputs x, t, dt, extra_inputs
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
    """
    dynamic_inputs, static_inputs = process_extra_inputs(dynamic_inputs, static_inputs, time_axis)
    T = time_axis.nelement()
    analysis_fields = (
        full_like(x_eachwin[0].fields[:1, :1].clone().detach(), torch.nan).expand(1, T).clone()
    )  # clone() to actually copy the data
    # analysis is now a TensorDict with a singleton batch and time dimensions

    for i, (s, e) in enumerate(idx_ranges):
        # don't use the part of this analyis window that overlaps later windows
        e = idx_ranges[i + 1][0] if i < len(idx_ranges) - 1 else e
        n_thiswindow = e - s

        if is_weak:
            assert (time_axis[s:e] == x_eachwin[i].time_axis[:n_thiswindow]).all(), "time axis mismatch"
            analysis_fields[:, s:e] = x_eachwin[i].fields[:, :n_thiswindow]
        else:
            x0 = x_eachwin[i]
            assert x0.time_axis[0] == time_axis[s], "time axis mismatch"
            # rollout to get state trajectory and remove batch dimension
            analysis_fields[:, s:e] = rollout(
                m_dyn, time_axis[s:e], x0, dynamic_inputs[:, s : e - 1], static_inputs
            ).fields

    return State(analysis_fields, time_axis=time_axis)
