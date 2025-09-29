from typing import Callable, Type

import torch
from da_tools.observation.data import ObservationSet
from da_tools.observation.operators import ObservationOperator
from da_tools.probability.distributions import Distribution
from da_tools.system.state import State
from da_tools.util.optimization import optimize
from da_tools.util.state_space import rollout
from da_tools.util.typing import ListOfStates
from da_tools.variational.base import process_4dvar_inputs, sliding_window_4dvar
from tensordict import TensorDict
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class HC4DVarLoss(_Loss):
    """Loss function class for hard-constraint 4dvar."""

    def __init__(
        self,
        m_dyn: Callable,
        observations: ObservationSet,
        obs_op: ObservationOperator,
        dynamic_inputs: State = None,
        static_inputs: State = None,
        background_prior: Distribution = None,
    ):
        """Initialize loss.

        Args:
            m_dyn (Callable): function that takes 4 inputs: x, dt, dynamic_inputs, static_inputs. All inputs are State
            objects except dt, which can be a float or Tensor. In general, this will be a user-supplied function that
            describes the known dynamics of a state space model.
            observations (ObservationSet): set of observations to be assimilated.
            obs_op (ObservationOperator): observation operator used to map between system states and observations
            dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
            static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
            background_prior (StateDistribution): prior distribution on initial state. Default None.
        """
        super().__init__()
        self.m_dyn, self.observations, self.obs_op, self.dynamic_inputs, self.static_inputs, self.background_prior = (
            m_dyn,
            observations,
            obs_op,
            dynamic_inputs,
            static_inputs,
            background_prior,
        )
        self.x0 = State(TensorDict(batch_size=(1, 1)), time_axis=obs_op.time_axis[0:1])

    def forward(self, fields: TensorDict) -> Tensor:
        """Compute loss function.

        Args:
            fields (TensorDict): variable fields for initial system state

        Returns:
            Tensor: loss value, summed over fields and all dimensions, and averaged over the batch
        """
        self.x0.fields = fields
        x_all = rollout(self.m_dyn, self.obs_op.time_axis, self.x0, self.dynamic_inputs, self.static_inputs)
        logp = (
            0
            if self.background_prior is None
            else self.background_prior.log_prob(self.x0.fields).squeeze(1)
        )
        # logp now has only a single singleton batch dimension

        logp = logp + self.obs_op.log_prob(x_all, self.observations)

        return -logp.mean(dim=0)  # mean over batch axis


def hc4dvar_single_window(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationOperator,
    x_init: State = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    background_prior: Distribution = None,
    optimizer_class: Type[Optimizer] = torch.optim.LBFGS,
    optimizer_pars: dict = None,
    n_steps: int = 10,
    verbose: bool = False,
    check_inputs: bool = True,
) -> State:
    """Hard constraint 4DVAR for a single window of observations.

    Args:
        m_dyn: time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        oobservations (ObservatationSet): set of observations to be assimilated.
        obs_op (ObservationOperator): observation operator used to map between system states and observations
        x_init (State): first guess of initial state for the first window. uses background prior mean if None.
          Should have a leading time dimension of length 1.
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        background_prior (StateDistribution): prior distribution on initial state
        optimizer_class: class of a pytorch optimizer, default is LBFGS
        optimizer_pars (dict): parameters for optimizer (learning rate etc.)
        n_steps (int): the number of optimization steps
        verbose (bool): if True, print loss at each iteration. default is False
        check_inputs: if False, input checking is skipped

    Returns:
        x0 (State): optimized initial state
    """
    if check_inputs:
        dynamic_inputs, static_inputs, x_init, _, _, _ = process_4dvar_inputs(
            obs_op,
            dynamic_inputs,
            static_inputs,
            x_init=x_init,
            background_prior=background_prior,
            observations=observations,
        )

    loss = HC4DVarLoss(m_dyn, observations, obs_op, background_prior=background_prior)  # instantiate loss

    fields = optimize(loss, x_init.fields, optimizer_class, optimizer_pars, n_steps, verbose=verbose)
    return State(fields, time_axis=x_init.time_axis)


def hc4dvar_sliding_window(*args, **kwargs) -> ListOfStates:
    """Hard constraint 4DVAR for overlapping windows of observations.

    Args:
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations (ObservatationSet): set of observations to be assimilated.
        obs_op (ObservationOperator): observation operator used to map between system states and observations
        window_duration (Union[float, int]): window length in time units
        shift_fraction (float, optional): shift as fraction of window. Defaults to 0.25.
        discard_partial (bool, optional): Drop windows extending beyond last obs, default False.
        x_init (State, optional): first guess of initial state. Uses prior mean if None.
          Should have leading batch, then time dimension of length 1
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
    return sliding_window_4dvar(hc4dvar_single_window, *args, **kwargs)
