import torch
from da_tools.observation.data import FullStateObservationSet, MaskedStateObservationSet, ObservationSet
from da_tools.system.state import State
from torch import full_like


def naive_initialization(observations: ObservationSet, default_value=torch.nan) -> State:
    """A basic initialization where each variable takes the value of the closest observation in the future. If no future
    observations are available, the most recent past observation will be used.

    Args:
        observations: a set of possibly noisy observations.
        default_value: the value to attribute to variables that are not observed.
    """
    if isinstance(ObservationSet, FullStateObservationSet):
        return observations.state
    assert isinstance(
        observations, MaskedStateObservationSet
    ), f"naive initialization not supported for observations of type {type(observations)}"
    init = full_like(observations.state.fields, torch.nan)
    B, T = observations.state.fields.batch_size
    assert B == 1, "batching is not support by naive_initialization"
    with torch.no_grad():
        for key, mask in observations.mask.fields.items():
            init_vals, mask_vals, obs_vals = (  # remove batch axis from each tensor
                init[key].squeeze(0),
                mask.squeeze(0),
                observations.state.fields[key].squeeze(0),
            )

            last_element = torch.full((1, *init_vals.shape[1:]), default_value, dtype=obs_vals.dtype, device=obs_vals.device)
            for t in range(T - 1, -1, -1):  # reverse order
                mask_t = mask_vals[t, ...]
                y = obs_vals[t : t + 1, mask_t]
                init_vals[: t + 1, mask_t] = y
                last_element[:, mask_t] = y

            missing_data = torch.isnan(init_vals)
            init_vals = torch.where(missing_data, last_element.expand(init_vals.shape), init_vals)

            init[key] = init_vals.unsqueeze(0)  # re-add batch dimension
    
    init = init.detach().clone()  # to prevent sharing memory with observations, which should not change. necessary?
    return State(init, observations.state.time_axis)
