from abc import ABC
from textwrap import indent
from typing import Dict, Union

import torch
from da_tools.util.data import isnumber
from da_tools.util.index import sorted_index_range
from tensordict import TensorDict
from torch import full_like, Tensor


class State(ABC):
    """Object representing a system state (trajectory).

    Its fields are a Tensordict with two shared (batch) dimensions: the batch dimension, followed by the time dimension.
    It also contains a time axis.
    """

    def __init__(self, x: Union[Tensor, TensorDict], time_axis: Tensor = None):
        """Initialize system state from Tensor or TensorDict input.

        Args:
            x (Union[Tensor, TensorDict]): Input data describing fields at each time point. If x is a tensor, it will be
             converted to a TensorDict with a single field "x"
            time_axis (Tensor, optional): Time values for each system state. Should match the second shared dimension of
             x. If ommited, will be populated with 64 bit signed integers starting at 0.
        """
        if isinstance(x, Tensor):
            assert x.ndim >= 2, "State tensor must have batch and time dimensions"
            x = TensorDict(x=x, batch_size=x.shape[:2])
        else:
            assert isinstance(x, TensorDict), "x must be Tensor or TensorDict"
            for v in x.values():
                assert isinstance(v, Tensor), "nested TensorDicts are not supported"
                assert v.ndim >= 2, "each field must have batch and time dimensions"
            if x.ndim < 2:
                x.batch_size = v.shape[:2]
            else:
                assert x.ndim == 2, "only batch and time dimensions can appear in the tensordict batch_size"

        assert isinstance(x, TensorDict), "x must be Tensor or TensorDict"
        self.time_axis = check_time_axis(time_axis, x.shape[1])
        self.fields = x

        self.shape = dict()
        for k, v in self.fields.items():
            self.shape[k] = v.shape

    def restrict_time_domain(self, tmin, tmax):
        """Returns a new State with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        s, e = sorted_index_range(self.time_axis, tmin, tmax)
        new_fields = TensorDict()
        for key in self.fields.keys():
            new_fields[key] = self.fields[key][:, s:e, ...]
        return State(new_fields, self.time_axis[s:e])

    def expand_time(self, time_axis):
        """Returns a new state with an expanded batch size and time axis."""
        assert self.fields.batch_size[1] == 1, "only a State with a single time step can be expanded in time"
        B, T = self.fields.batch_size[0], time_axis.nelement()
        new_state = self.clone()
        new_state.fields = new_state.fields.expand(B, T)
        new_state.time_axis = check_time_axis(time_axis, T)
        return new_state

    def detach(self):
        """Detach from computation graph. See torch.Tensor.detach().

        Returns:
            State: detached State.
        """
        return State(self.fields.detach(), self.time_axis.detach())

    def clone(self):
        """Clone State. See torch.Tensor.clone().

        Returns:
            State: cloned State.
        """
        return State(self.fields.clone(), self.time_axis.clone())

    def fill_(self, val: Union[float, Dict]):
        """Fill values of state trajectory. See fill_ methods of Tensor and TensorDict.

        Args:
            val (Union[float, Dict]): Value to fill in. Can be a dict with an entry for each field, or a single float.

        Returns:
            State: State with filled in values.
        """
        if isinstance(val, float):
            fields = full_like(self.fields, val)
        elif isinstance(val, dict):
            fields = self.fields.clone()
            for k in val:
                fields[k][...] = val[k]
        return State(fields, self.time_axis)

    def __getitem__(self, key):
        """Indexing operator for State.

        Args:
            key: indexing key

        Returns:
            State: indexed key
        """
        if isinstance(key, tuple):
            if len(key) == 1:  # batch dimension
                return State(self.fields[key], self.time_axis)
            elif len(key) == 2:  # batch and time dimensions
                return State(self.fields[key], self.time_axis[key[1]])
            else:
                raise KeyError("invalid key")
        elif isinstance(key, int):  # batch dimension
            raise TypeError(
                "a State requires a batch dimension and cannot be indexed with a single int, use a range instead"
            )
        elif isinstance(key, slice) or key is Ellipsis:
            return State(self.fields[key], self.time_axis)
        elif isinstance(key, Tensor) and key.dtype is torch.bool:  # binary mask on batch dimension
            assert key.ndim == 1, "binary masking along batch dimension only"  # allow time also?
            assert key.nelement() == self.fields.batch_size[0], "size mismatch"
            return State(self.fields[key], self.time_axis)
        else:
            raise KeyError("invalid key")

    def __setitem__(self, key, value):
        """Indexed assignement operator for State.

        Args:
            key: indexing key
            value: values to be assigned
        """
        if isinstance(key, tuple):
            if len(key) == 0 or len(key) > 2:
                raise KeyError("tuple key must have 1 or 2 entries")
        elif isinstance(key, Tensor) and key.dtype is torch.bool:
            if key.ndim == 0 or key.ndim > 2:
                raise KeyError("only 1 or 2 dims for logical index")
        elif not (isinstance(key, int) or isinstance(key, slice) or (key is Ellipsis)):
            raise KeyError("invalid key")
        self.fields[key] = value

    def __mul__(self, other):
        """Multiplication operator."""
        if isinstance(other, State):
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            return State(self.fields * other.fields, self.time_axis)
        elif isnumber(other):
            return State(self.fields * other, self.time_axis)
        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """Right multiplication."""
        if isnumber(other):
            return State(self.fields * other, self.time_axis)
        else:
            raise NotImplementedError()

    def __div__(self, other):
        """Division operator."""
        if isinstance(other, State):
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            return State(self.fields / other.fields, self.time_axis)
        elif isnumber(other):
            return State(self.fields / other, self.time_axis)
        else:
            raise NotImplementedError()

    def __rdiv__(self, other):
        """Right-to-left division."""
        if isnumber(other):
            return State(other / self.fields, self.time_axis)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        """Addition operator."""
        if isinstance(other, State):
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            return State(self.fields + other.fields, self.time_axis)
        elif isnumber(other):
            return State(self.fields + other, self.time_axis)
        else:
            raise NotImplementedError()

    def __radd__(self, other):
        "Right addition"
        if isnumber(other):
            return State(self.fields + other, self.time_axis)
        else:
            raise NotImplementedError()

    def __sub__(self, other):
        """Subtraction operator."""
        if isinstance(other, State):
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            return State(self.fields - other.fields, self.time_axis)
        if isnumber(other):
            return State(self.fields - other, self.time_axis)

    def __rsub__(self, other):
        """Right subtraction."""
        if isnumber(other):
            return State(other - self.fields, self.time_axis)

    def __pow__(self, other):
        """Exponentiation operator."""
        if isnumber(other):
            return State(self.fields**other, self.time_axis)
        else:
            raise NotImplementedError()

    def __repr__(self) -> str:
        string_fields = indent(f"Fields: {self.fields}", 4 * " ")
        string_time_axis = indent(f"Time axis: {self.time_axis}", 4 * " ")
        return f"{type(self).__name__}(\n{string_fields}\n{string_time_axis})"


def check_time_axis(time_axis, T):
    """Initialize time_axis if none, otherwise check validity."""
    if time_axis is None:
        return torch.arange(T)
    else:
        assert (
            isinstance(time_axis, Tensor) and time_axis.ndim == 1 and time_axis.nelement() == T and T > 0
        ), "invalid time_axis"
        return time_axis
