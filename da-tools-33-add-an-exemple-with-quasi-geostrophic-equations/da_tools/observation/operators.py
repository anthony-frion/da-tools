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
from da_tools.util.mask import mask_like, mask_state
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

    def sample(self, x: State, sample_points: bool = None) -> State:
        """Sample observations given a state.

        Args:
            x (State): current system state variables
        """
        sample_points = self.sample_points if sample_points is None else sample_points
        if sample_points:  # return a GriddedPointObservationSet, more compact for sparse observations
            raise NotImplementedError
        else:
            fields = x.fields + torch.randn_like(x.fields) * self.sigma.fields
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


def random_sparse_noisy_obs(
    x: Union[State, Tensor],
    obs_noise_sd: Union[State, Tensor, float, int, Dict],
    p_obs: Union[Tensor, float, Dict],
    time_axis=None,
    **kwargs,
) -> Tuple:
    """Generate sparse and noisy observations from system state sequence.

    Args:
        x (Union[State, Tensor]): a complete state sequence
        obs_noise_sd (Union[State, Tensor, float, int, Dict]): standard deviation of the observation noise
        p_obs (Union[Tensor, float, Dict]): proportion of observed variables
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

    mask = mask_like(x, p_obs, **kwargs)

    obs_op = MaskedIidGaussianObsOp(mask, obs_noise_sd)

    obs = obs_op.sample(x)

    return x, obs_op, obs
