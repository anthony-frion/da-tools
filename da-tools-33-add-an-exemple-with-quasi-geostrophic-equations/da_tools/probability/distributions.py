from abc import ABC, abstractmethod
from textwrap import indent
from da_tools.system.state import State
from da_tools.util.data import as_TensorDict
from da_tools.util.state_space import same_shape, weighted_sse
from tensordict import TensorDict
from torch import randn_like, Tensor


class Distribution(ABC):
    """Base class for distributions over a TensorDict."""

    @abstractmethod
    def log_prob(self, x: TensorDict) -> float:
        """Compute log probability of state. This sums over the batch and time axes.

        Args:
            x (State): input state
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> TensorDict:
        """Sample from distribution on states."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mean(self):
        """Mean of distribution on states."""
        raise NotImplementedError

    def __getitem__(self, key):
        """Indexing to obtain marginal distribution over a subset of variables."""
        raise NotImplementedError


class DiagonalGaussian(Distribution):
    """Diagonal normal distribution.

    Mu/Sigma are tensordicts, with optional batch dimensions.

    Args:
        mu (TensorDict): mean of distribution per each system variable
        sigma (TensorDict): standard deviation of distribution per each system variable
    """

    def __init__(self, mu: TensorDict, sigma: TensorDict):
        self.mu, self.sigma = as_TensorDict(mu), as_TensorDict(sigma)
        assert same_shape(self.mu, self.sigma), "mu/sigma mismatch"

    def log_prob(self, x: TensorDict, normalized: bool = False) -> Tensor:
        """Compute log probability, summing over all non-batch axes.

        Args:
            x (TensorDict): variables for which to evaluate the density
            normalized (bool): whether to normalize log probability
        """
        if normalized:
            raise NotImplementedError
        return -weighted_sse(x, self.mu, self.sigma)

    def sample(self, n_samples: int = None) -> State:
        """Sample from the distribution on states.

        Args:
            n_samples (int): number of samples to draw from the distribution
        """
        mu, sigma = self.mu, self.sigma
        if n_samples is not None:
            assert mu.shape[0] == 1, "batch dimension of mu/sigma must have length 1 when n_samples is specified"
            new_shape = (n_samples, *mu.shape[1:])
            mu, sigma = mu.expand(new_shape), sigma.expand(new_shape)
        return mu + randn_like(mu) * sigma

    @property
    def mean(self):
        """Mean of distribution."""
        return self.mu

    def __getitem__(self, key):
        """Indexing operator.

        Args:
            key: indexing key, which be applied to index the mean and s.d. of this distribution

        Returns:
            DiagonalGaussian: indexed distribution
        """
        return DiagonalGaussian(self.mu[key], self.sigma[key])
    
    def __repr__(self) -> str:
        string_mu = indent(f"mu: {self.mu}", 4 * " ")
        string_sigma = indent(f"sigma: {self.sigma}", 4 * " ")
        return f"{type(self).__name__}(\n{string_mu}\n{string_sigma})"
