from typing import Dict, List, Tuple, Union

import torch
from da_tools.system.state import State
from tensordict import TensorDict
from torch import Tensor


def mask_state(x: State, mask: State) -> TensorDict:
    """Apply mask to the system state variable(s)

    Args:
        x (State): system state variable(s)
        mask (State): mask with boolean
    """
    x_masked = TensorDict(batch_size=())  # no batch dimensions after masking
    for key, v in x.fields.items():
        x_masked[key] = v[mask.fields[key]]
    return x_masked


def mask_like(
    x: State,
    p_obs: Dict,
    constant_obs_count_per_step: bool = True,
    constant_obs_count: bool = True,
) -> State:
    """Generate mask like State object.

    Args:
        x (State): _description_
        p_obs (Dict): _description_
        constant_obs_count_per_step (bool, optional): _description_. Defaults to True.
        constant_obs_count (bool, optional): _description_. Defaults to True.

    Returns:
        State: _description_
    """
    m = TensorDict(batch_size=x.fields.batch_size)
    for key, val in x.fields.items():
        m[key] = generate_mask(
            val.shape,
            p_obs[key],
            constant_obs_count_per_step=constant_obs_count_per_step,
            constant_obs_count=constant_obs_count,
        )
    return State(m, x.time_axis)


def generate_mask(
    shape: Union[torch.Size, List, Tuple],
    p_obs: Union[float, Tensor],
    constant_obs_count_per_step: bool = True,
    constant_obs_count: bool = True,
) -> Union[TensorDict, Tensor]:
    """Generates a mask for a State object with batch and time axes shared across all fields.

    Args:
        shape: the shape of the mask
        p_obs: the proportion of observed variables across the trajectory.
        If p_obs is a boolean tensor than it is directly used as the mask.
        contant_obs_count_per_step: if this is true, enforces that the number of observed variables remains the same
          for each time step. Assumes first axis is time.
        constant_obs_count: if this is true, the total number of observed variables is calculated detemrinistically.
          It can still vary for each time step. Applied only if constant_obs_per_step is False.
    """
    shape = torch.Size(shape)
    if isinstance(p_obs, Tensor):
        # FIXME: we might want to add support for float tensors
        # to handle cases where different times/variables have different obs proportions
        assert p_obs.dtype == torch.bool, f"non-boolean Tensors are not supported for input p_obs, got {p_obs.dtype}"
        mask = p_obs.reshape(shape)
    elif constant_obs_count_per_step:
        field_dim = shape[1:].numel()
        n_obs_per_step = int(field_dim * p_obs)
        shape_flat = (shape[0], field_dim)
        sample_indexes = torch.rand(shape_flat).topk(n_obs_per_step, dim=-1).indices
        mask = torch.zeros(shape_flat, dtype=bool).scatter_(dim=-1, index=sample_indexes, value=True)
        mask = mask.reshape(shape)
    elif constant_obs_count:
        field_dim = shape.numel()
        n_obs = int(p_obs * field_dim)
        mask = torch.zeros(field_dim, dtype=bool)
        mask[:n_obs] = True
        mask = mask[torch.randperm(field_dim)].reshape(shape)
    else:
        mask = torch.rand(shape) < p_obs
    return mask
