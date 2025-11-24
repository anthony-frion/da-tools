from typing import Callable, Type

import torch
from da_tools.observation.data import ObservationSet
from da_tools.observation.operators import ObservationOperator
from da_tools.probability.distributions import Distribution
from da_tools.system.state import State
from da_tools.util.optimization import optimize
from da_tools.util.typing import ListOfStates
from da_tools.variational.base import process_4dvar_inputs, sliding_window_4dvar
from tensordict import TensorDict
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class WC4DVarLoss(_Loss):
    def __init__(
        self,
        m_dyn: Callable,
        observations: ObservationSet,
        obs_op: ObservationOperator,
        dynamic_inputs: State,
        static_inputs: State,
        model_error_distribs: Distribution,
        alpha: float = 1.0,
        background_prior: Distribution = None,
        normalize: bool = False,
    ):
        """Loss for weak constraint 4dvar.

        Args:
            one_step_func (Callable): callable that computes advancement of the state by one time step
            observations (ObservationSet): observations for each time step
            obs_op (ObservationOperator): observation operators for each time step
            dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
            static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
            model_error_distribs (Distribution): probability distributions of the model errors at each time step
            alpha (float, optional): weighting factor for model error loss term. Defaults to 1.0.
            background_prior (Distribution, optional): prior on initial state. Defaults to None.
            normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
        """
        super().__init__()
        (
            self.m_dyn,
            self.observations,
            self.obs_op,
            self.dynamic_inputs,
            self.static_inputs,
            self.model_error_distribs,
            self.alpha,
            self.background_prior,
        ) = (m_dyn, observations, obs_op, dynamic_inputs, static_inputs, model_error_distribs, alpha, background_prior)
        T = obs_op.time_axis.nelement()
        self.x = State(TensorDict(batch_size=(1, T)), time_axis=obs_op.time_axis)
        dt = obs_op.time_axis.diff()
        assert torch.allclose(dt, dt[0], atol=1e-6, rtol=1e-3), "requirement for fixed time step is not satisfied"
        self.dt = dt[0]
        self.normalize = normalize

    def forward(self, fields: TensorDict) -> Tensor:
        """Compute loss function.

        Args:
            fields (TensorDict): variable fields for system state trajectory

        Returns:
            Tensor: loss value, summed over fields and all dimensions, and averaged over the batch
        """
        self.x.fields = fields

        x0, x1 = self.x[:, :-1], self.x[:, 1:]
        x1_pred = self.m_dyn(x0, self.dt, dynamic_inputs=self.dynamic_inputs, static_inputs=self.static_inputs)

        logp = 0.0 if self.background_prior is None else self.background_prior.log_prob(self.x.fields[:, :1]).squeeze(1)

        # logp now has only a single singleton batch dimension

        logp = logp + self.obs_op.log_prob(self.x, self.observations).reshape(
            1,
        )
        logp += self.alpha * self.model_error_distribs.log_prob(x1_pred.fields - x1.fields).sum(
            axis=1
        )  # sum over time axis
        if self.normalize:
            logp /= self.obs_op.time_axis.nelement()
        return -logp.mean(dim=0)  # mean over batch axis


def wc4dvar_single_window(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationOperator,
    model_error_distribs: Distribution,
    x_init: State = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    alpha: float = 1.0,
    background_prior: Distribution = None,
    normalize: bool = False,
    optimizer_class: Type[Optimizer] = torch.optim.LBFGS,
    optimizer_pars: dict = None,
    scheduler_class: Type[LRScheduler] = None,
    scheduler_pars: dict = None,
    n_steps: int = 10,
    verbose: bool = False,
    check_inputs: bool = True,
    rollout_init: bool = True,
):
    """Weak constraint 4DVAR for a single window of observations.

    Args:
        m_dyn: time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations: list of observations at each time point
        obs_operators: global obs_op, or list of observation operators at each time point
        model_error_distribs: distributions of model errors at each time step
        x_init: first guess of state trajectory. Uses background prior mean if None. Should have a leading batch
          dimension, followed by a time dimension.
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        alpha (float, optional): weighting factor for model error loss term. Defaults to 1.0.
        background_prior: prior distribution of the initial state, defaults to None.
        normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
        optimizer: class of a pytorch optimizer
        optimizer_pars: dictionary containing parameters for optimizer (learning rate etc.)
        n_steps: int indicating the number of optimizer steps
        verbose (bool): if True, print loss at each iteration. default is False
        check_inputs: if False, input checking is skipped
        rollout_init (bool): indicates whether to copy or rollout the input state if it is only one time step
    Returns:
        x0: optimized state trajectory. first dimension is batch, second is time.
    """
    if check_inputs:
        dynamic_inputs, static_inputs, x_init, _, _, model_error_distribs = process_4dvar_inputs(
            obs_op,
            m_dyn,
            dynamic_inputs,
            static_inputs,
            x_init=x_init,
            background_prior=background_prior,
            observations=observations,
            model_error_distribs=model_error_distribs,
            rollout_init=rollout_init,
        )
    loss = WC4DVarLoss(
        m_dyn,
        observations,
        obs_op,
        dynamic_inputs,
        static_inputs,
        model_error_distribs,
        alpha=alpha,
        background_prior=background_prior,
        normalize=normalize,
    )

    fields = optimize(
        loss,
        x_init.fields,
        optimizer_class,
        optimizer_pars,
        scheduler_class,
        scheduler_pars,
        n_steps=n_steps,
        verbose=verbose,
    )
    return State(fields, time_axis=x_init.time_axis)


def wc4dvar_sliding_window(*args, **kwargs) -> ListOfStates:
    """Weak constraint 4DVAR for overlapping windows of observations.

    Args:
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations (ObservatationSet): tensor or list of observations at each time point
        obs_op (ObservationOperator): list of observation operators at each time point
        window_duration (Union[float, int]): window length in time units
        shift_fraction (float, optional): shift as fraction of window. Defaults to 0.25.
        model_error_distribs (Distribution): distributions of model errors at each time step (None for hard constraint)
        discard_partial (bool, optional): Drop windows extending beyond last obs, default False.
        x_init (State): first guess of state trajectory. Uses prior mean if None.
          Should have leading batch, then time dimension
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        background_prior (StateDistribution, optional): background prior for first window. Defaults to None.
        optimizer_class: class of a pytorch optimizer, default is LBFGS
        optimizer_pars (dict): parameters for optimizer (learning rate etc.)
        n_steps (int): the number of optimization steps
        verbose (bool): if True, print loss at each iteration. default is False

    Returns:
        ListOfStates: state trajectory for each assimilation window
    """
    return sliding_window_4dvar(wc4dvar_single_window, *args, **kwargs)
