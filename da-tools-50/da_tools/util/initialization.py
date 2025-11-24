import torch
from da_tools.observation.data import FullStateObservationSet, MaskedStateObservationSet, ObservationSet
from da_tools.system.state import State
from scipy import interpolate
from torch import full_like


def naive_initialization(observations: ObservationSet, default_value=torch.nan) -> State:
    """A shortcut to interp_initialization with selection of next available observation."""
    return interp_initialization(observations, kind="next", default_value=default_value)


def interp_initialization(observations: ObservationSet, kind="linear", default_value=torch.nan) -> State:
    """A basic initialization where each variable takes the value of the closest available observations. This method
    acts sequentially on each variable, so it might be slow for states with many variables.

    Args:
        observations: a set of possibly noisy observations.
        kind: the kind of interpolation to perform. Refer to scipy.interpolate.interp1d for details.
        default_value: the value to attribute to variables that are not observed.
    """
    if isinstance(ObservationSet, FullStateObservationSet):
        return observations.state
    assert isinstance(
        observations, MaskedStateObservationSet
    ), f"nearest initialization not supported for observations of type {type(observations)}"
    init = full_like(observations.state.fields, torch.nan)
    B, T = observations.state.fields.batch_size
    assert B == 1, "batching is not support by naive_initialization"
    time_axis = observations.state.time_axis
    with torch.no_grad():
        for key, mask in observations.mask.fields.items():
            init_vals, mask_vals, obs_vals = (  # remove batch axis from each tensor
                init[key].squeeze(0),
                mask.squeeze(0),
                observations.state.fields[key].squeeze(0),
            )
            orig_shape = init_vals.shape[1:]
            init_vals_flat = init_vals.flatten(1, -1)
            mask_vals_flat = mask_vals.flatten(1, -1)
            obs_vals_flat = obs_vals.flatten(1, -1)
            for var in range(init_vals_flat.shape[1]):
                indexes = torch.where(mask_vals_flat[:, var] == 1)[0]
                x = time_axis[indexes]
                if len(x) == 0:
                    init_vals_flat[:, var] = default_value
                elif len(x) == 1:
                    init_vals_flat[:, var] = obs_vals_flat[indexes[0], var]
                else:
                    y = obs_vals_flat[indexes, var]
                    interp = interpolate.interp1d(x.detach().cpu(), y.detach().cpu(), kind=kind)
                    min_ind, max_ind = int(indexes[0]), int(indexes[-1])
                    init_vals_flat[min_ind:max_ind, var] = torch.Tensor(interp(time_axis[min_ind:max_ind]))
                    init_vals_flat[:min_ind, var] = y[0]
                    init_vals_flat[max_ind:, var] = y[-1]
            init_vals = init_vals_flat.reshape(1, T, *orig_shape)
            init[key] = init_vals

    init = init.detach().clone()  # to prevent sharing memory with observations, which should not change. necessary?
    return State(init, time_axis)
