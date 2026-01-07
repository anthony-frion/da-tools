from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import torch
from da_tools.observation.data import (
    FullStateObservationSet,
    GriddedPointObservationSet,
    MaskedStateObservationSet,
    ObservationSet,
)
from da_tools.system.state import State
from da_tools.util.mask import mask_like, mask_like_from_tensor, mask_state
from da_tools.util.state_space import same_shape, weighted_sse
from tensordict import TensorDict
from torch import full_like, Tensor


class ObservationOperator(ABC):
    """Base class for all observation operators."""

    @abstractmethod
    def log_prob(self, x: State, y: ObservationSet) -> float:
        """Compute log likelihood probability.

        Args:
            x (State): current state
            y (ObservationSet): current observed state
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, x: State) -> ObservationSet:
        """Sample observations given a state.

        Args:
            x (State): current state
        """
        raise NotImplementedError

    @abstractmethod
    def restrict_time_domain(self, tmin: float, tmax: float):
        """Returns a new operator with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        raise NotImplementedError


class IidGaussianObsOp(ObservationOperator):
    """Unmasked Identity Gaussian observation operator."""

    def __init__(self, sigma: State):
        """Args:
        sigma (State): variance per each variable
        """
        self.sigma, self.time_axis = sigma, sigma.time_axis

    def log_prob(self, x: State, y: ObservationSet, normalized: bool = False) -> Tensor:
        """Compute log likelihood probability.

        Args:
            x (State): current system state variables
            y (ObservationSet): current observed system state
            normalized (bool): whether to normalize log probability
        """
        assert isinstance(y, FullStateObservationSet), "invalid ObservationSet class"
        logp = -weighted_sse(x.fields, y.values, self.sigma)
        if normalized:
            raise NotImplementedError
        return logp.sum(dim=1)  # sum over time axis

    def sample(self, x: State) -> State:
        """Sample observations given a state.

        Args:
            x (State): current system state variables
        """
        fields = x.fields + torch.randn_like(x.fields) * self.sigma.fields
        return FullStateObservationSet(State(fields, time_axis=x.time_axis))

    def restrict_time_domain(self, tmin: float, tmax: float):
        """Returns a new operator with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        return IidGaussianObsOp(self.sigma.restrict_time_domain(tmin, tmax))

    def to(self, *args):
        """Returns a copy of the observation operator with its sigma set as specified by the arguments"""
        return IidGaussianObs(self.sigma.to(*args), self.sample_points, self.time_axis)


class MaskedIidGaussianObsOp(ObservationOperator):
    """Masked Identity Gaussian observation operator.

    Args:
        mask (State): mask fields per each variable
        sigma (State): varience per each variable
    """

    def __init__(self, mask: State, sigma: State, sample_points: bool = False):
        self.mask, self.sigma, self.sample_points, self.time_axis = mask, sigma, sample_points, mask.time_axis

    def log_prob(self, x: State, y: State, normalized: bool = False) -> float:
        """Compute log likelihood probability.

        Args:
            x (State): curren system state variables
            y (State): current observed system state
            normalized (bool): whether to normalize log probability
        """
        if isinstance(y, MaskedStateObservationSet):
            yhat = mask_state(x, self.mask)
            sigma = mask_state(self.sigma, self.mask)
            logp = -weighted_sse(yhat, y.values, sigma)
        elif isinstance(y, GriddedPointObservationSet):
            raise NotImplementedError
        else:
            raise TypeError("invalid ObservationSet class")

        if normalized:
            raise NotImplementedError
        return logp

    def sample(self, x: State, sample_points: bool = None, mult_precision=torch.float16) -> State:
        """Sample observations given a state.

        Args:
            x (State): current system state variables
            mult_precision: a torch dtype for the precision at which to compute the noise
        """
        sample_points = self.sample_points if sample_points is None else sample_points
        if sample_points:  # return a GriddedPointObservationSet, more compact for sparse observations
            raise NotImplementedError
        else:
            fields = x.fields + torch.randn_like(x.fields, dtype=mult_precision) * self.sigma.fields.to(mult_precision)
            return MaskedStateObservationSet(State(fields, time_axis=x.time_axis), self.mask)

    def restrict_time_domain(self, tmin: float, tmax: float):
        """Returns a new operator with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        return MaskedIidGaussianObsOp(
            self.mask.restrict_time_domain(tmin, tmax),
            self.sigma.restrict_time_domain(tmin, tmax),
            sample_points=self.sample_points,
        )

    def linearize(self, idx_time: int) -> State:
        """Linearize the observation operator at a specific time index.

        Args:
            idx_time (int): Time index to linearize at

        Returns:
            State: State object containing H matrix for the specified time step
        """
        n_variables = self.mask.fields["x"].shape[2]

        # Ensure index is within bounds
        max_time_idx = self.mask.fields["x"].shape[1] - 1
        if idx_time > max_time_idx:
            print(f"Warning: Time index {idx_time} out of bounds, using last available index {max_time_idx}")
            idx_time = max_time_idx
        elif idx_time < 0:
            print(f"Warning: Time index {idx_time} is negative, using index 0")
            idx_time = 0

        # Extract mask for the specified time step
        try:
            mask_t = self.mask.fields["x"][0, idx_time, :]  # Shape: (n_variables,)
            mask_t = mask_t.squeeze()
        except IndexError as e:
            print(f"Error accessing mask at index {idx_time}: {e}")
            print(f"Mask shape: {self.mask.fields['x'].shape}")
            print(f"Available time indices: 0 to {self.mask.fields['x'].shape[1]-1}")
            # Fallback to first time step
            mask_t = self.mask.fields["x"][0, 0, :].squeeze()
            idx_time = 0

        # Count number of observations at this time step
        n_obs_t = mask_t.sum().item()

        # Create H matrix
        if n_obs_t > 0:
            # Get indices of observed variables
            observed_indices = torch.where(mask_t)[0]  # Get indices of True values
            H_t = torch.zeros(n_obs_t, n_variables, dtype=torch.float32)
            H_t[torch.arange(n_obs_t), observed_indices] = 1.0
        else:
            # No observations - create empty matrix
            H_t = torch.zeros(0, n_variables, dtype=torch.float32)

        # # Create H matrix
        # if n_obs_t > 0:
        #     H_t = torch.zeros(n_obs_t, n_variables, dtype=torch.float32)
        #     obs_idx = 0
        #     for var_idx in range(n_variables):
        #         if mask_t[var_idx].item():  # Convert tensor to Python bool
        #             H_t[obs_idx, var_idx] = 1.0
        #             obs_idx += 1
        # else:
        #     # No observations - create empty matrix
        #     H_t = torch.zeros(0, n_variables, dtype=torch.float32)

        # Create TensorDict with proper batch structure
        # Shape: (1, 1, n_obs, n_variables) to match expected indexing H_t = H_state.fields['x'][0, 0, :, :]
        H_tensor = H_t.unsqueeze(0).unsqueeze(0)

        H_fields = TensorDict(x=H_tensor, batch_size=(1, 1))

        # Create time axis for the result using the specified time index
        result_time_axis = None
        if self.time_axis is not None and idx_time < len(self.time_axis):
            result_time_axis = self.time_axis[idx_time : idx_time + 1]  # Single time point

        return State(H_fields, time_axis=result_time_axis)

    def to(self, *args):
        """Returns a copy of the observation operator with its mask and sigma set as specified by the arguments"""
        return MaskedIidGaussianObsOp(self.mask.to(*args), self.sigma.to(*args), self.sample_points)


def expand_sigma(sigma: Union[float, Tensor, dict, TensorDict], state_shape: Union[torch.Size, List, Tuple, Dict]):
    """Expand inputs to match a TensorDict.

    Args:
        sigma (Union[float, torch.Tensor, TensorDict]): _description_
        state_shape (Union[torch.Size, List, Tuple, Dict]): _description_

    Returns:
        TensorDict: Expanded values.
    """
    if isinstance(sigma, float):
        if isinstance(state_shape, dict):
            sigma_td = TensorDict()
            for k, s in state_shape.items():
                sigma_td[k] = torch.full(state_shape[k], sigma)
            sigma = sigma_td
        else:
            sigma = torch.full(state_shape, sigma)
    elif isinstance(sigma, dict):
        sigma_td = TensorDict()
        for k, s in state_shape.items():
            sigma_td[k] = torch.full(state_shape[k], sigma[k])
        sigma = sigma_td
    return sigma


def noisy_obs_from_mask(
    x: Union[State, Tensor],
    obs_noise_sd: Union[State, Tensor, float, int, Dict],
    mask_t: Union[Tensor, TensorDict],
    time_axis=None,
) -> Tuple:
    """Generates noisy observations from system state sequence, using the provided mask and noise standard deviation.

    Args:
        x (Union[State, Tensor]): a complete state sequence
        obs_noise_sd (Union[State, Tensor, float, int, Dict]): standard deviation of the observation noise
        mask_t (Union[Tensor, TensorDict]): a Tensor containing either boolean values
            or floating values representing the probability of observation for every variable and time step

    Returns:
        x: input state sequence, converted to State class
        obs_op: observation operator
        obs: observations
    """
    if type(mask_t) not in [Tensor, TensorDict]:
        raise TypeError(f"Invalid type for mask_t: {type(mask_t)}")
    if isinstance(obs_noise_sd, int):
        obs_noise_sd = float(obs_noise_sd)
    if isinstance(x, Tensor):
        if isinstance(obs_noise_sd, float):
            obs_noise_sd = full_like(x, obs_noise_sd)
        x, obs_noise_sd = State(x, time_axis=time_axis), State(obs_noise_sd, time_axis=time_axis)
    elif isinstance(x, State):
        assert time_axis is None, "time_axis cannot be specified when x is a State"
        if isinstance(obs_noise_sd, float) or isinstance(obs_noise_sd, dict):
            obs_noise_sd = x.detach().clone().fill_(obs_noise_sd)
        else:
            assert isinstance(obs_noise_sd, State), "sigma must be float, dict or State when x is a State"
            assert same_shape(
                x, obs_noise_sd
            ), f"shape mismatch: got x.shape={x.shape} and obs_noise_sd.shape={obs_noise_sd.shape}"
            assert torch.all(
                x.time_axis == obs_noise_sd.time_axis
            ), f"time axis mismatch: got x.time_axis={x.time_axis} and obs_noise_sd.time_axis={obs_noise_sd.time_axis}"
    else:
        raise TypeError(f"Invalid type for x: {type(x)}")
    # x should now be a State
    if isinstance(mask_t, Tensor):
        assert (
            len(x.fields.keys()) == 1
        ), "mask_t cannot be a Tensor when x has several fields, use a TensorDict instead"
        mask_t = TensorDict({field: mask_t for field in x.fields.keys()})
    mask = mask_like_from_tensor(x, mask_t)
    print(mask)
    obs_op = MaskedIidGaussianObsOp(mask, obs_noise_sd)

    obs = obs_op.sample(x)

    return x, obs_op, obs


def random_sparse_noisy_obs(
    x: Union[State, Tensor],
    obs_noise_sd: Union[State, Tensor, float, int, Dict],
    p_obs: Union[Tensor, float, Dict],
    time_axis=None,
    constant_obs_count_per_step: bool = True,
    constant_obs_count: bool = True,
) -> Tuple:
    """Generate sparse and noisy observations from system state sequence.

    Args:
        x (Union[State, Tensor]): a complete state sequence
        obs_noise_sd (Union[State, Tensor, float, int, Dict]): standard deviation of the observation noise
        p_obs (Union[float, Dict]): proportion of observed variables
        : if provided, it is used instead of p_obs to define the mask.
        time_axis (_type_, optional): time axis associated to the state sequence. Defaults to None.
        contant_obs_count_per_step: if this is true, enforces that the number of observed variables remains the same
          for each time step. Assumes first axis is time. Defaults to True.
        constant_obs_count: if this is true, the total number of observed variables is calculated deterministically.
          It can still vary for each time step. Applied only if constant_obs_per_step is False. Defaults to True.

    Returns:
        x: input state sequence, converted to State class
        obs_op: observation operator
        obs: observations
    """
    if isinstance(obs_noise_sd, int):
        obs_noise_sd = float(obs_noise_sd)
    if isinstance(p_obs, int):
        p_obs = float(p_obs)
    if isinstance(x, Tensor):
        if isinstance(obs_noise_sd, float):
            obs_noise_sd = full_like(x, obs_noise_sd)
        assert isinstance(obs_noise_sd, Tensor), "sigma must be tensor or float when x is tensor"
        assert isinstance(p_obs, float), "p_obs must be float when x is tensor"
        x, obs_noise_sd = State(x, time_axis=time_axis), State(obs_noise_sd, time_axis=time_axis)
        p_obs = dict(x=p_obs)
    elif isinstance(x, State):
        assert time_axis is None, "time_axis cannot be specified when x is a State"
        if isinstance(obs_noise_sd, float) or isinstance(obs_noise_sd, dict):
            obs_noise_sd = x.detach().clone().fill_(obs_noise_sd)
        else:
            assert isinstance(obs_noise_sd, State), "sigma must be float, dict or State when x is a State"
            assert same_shape(
                x, obs_noise_sd
            ), f"shape mismatch: got x.shape={x.shape} and obs_noise_sd.shape={obs_noise_sd.shape}"
            assert torch.all(
                x.time_axis == obs_noise_sd.time_axis
            ), f"time axis mismatch: got x.time_axis={x.time_axis} and obs_noise_sd.time_axis={obs_noise_sd.time_axis}"
        if isinstance(p_obs, float) or isinstance(p_obs, Tensor):
            p_obs_dict = dict()
            for key in x.fields.keys():
                p_obs_dict[key] = p_obs
            p_obs = p_obs_dict
    else:
        raise TypeError(f"Invalid type for x: {type(x)}")
    # x and sigma are now both of type State, p_obs is a dict

    mask = mask_like(x, p_obs, constant_obs_count_per_step, constant_obs_count)

    obs_op = MaskedIidGaussianObsOp(mask, obs_noise_sd)

    obs = obs_op.sample(x)

    return x, obs_op, obs
