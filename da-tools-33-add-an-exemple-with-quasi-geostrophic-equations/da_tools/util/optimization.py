from typing import Callable, Type, Union

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def optimize(
    f: Callable,
    x: Union[Tensor, TensorDict],
    optimizer_class: Type[Optimizer],
    optimizer_pars: dict,
    scheduler_class: Type[LRScheduler] = None,
    scheduler_pars: dict = None,
    n_steps: int = 10,
    verbose: bool = False,
) -> Union[Tensor, TensorDict]:
    """

    Args:
        f (Callable): function to be minimized
        x (Union[Tensor, TensorDict]): initial estimate of inputs
        optimizer_class: class of pytorch optimizer
        optimizer_pars (dict): dict of optimizer parameters
        scheduler_class: class of pytorch learning rate scheduler
        scheduler_pars (dict): dict of scheduler parameters
        n_steps (int): number of optimizer steps
        verbose (bool): if True, print loss at each iteration. default is False

    Returns:
        x (Union[Tensor, TensorDict]): optimized inputs to f
    """
    if isinstance(x, TensorDict):
        inputs = [v for v in x.values() if isinstance(v, Tensor)]
    else:
        assert isinstance(x, torch.Tensor)
        inputs = [x]

    optimizer = optimizer_class(
        params=inputs,
        **optimizer_pars,
    )

    if scheduler_class is not None:
        scheduler_pars = scheduler_pars or {}
        scheduler = scheduler_class(optimizer=optimizer, **scheduler_pars)

    for i_opt in range(n_steps):

        def closure():
            """Closure function that updates and zeros gradients as needed. used by LBFGS which sometimes, but not
            always needs gradients.

            Returns: None
            """
            loss = f(x)
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
            if verbose:
                print(f"iteration {i_opt}: loss = {loss}")
            return loss

        optimizer.step(closure)
        if scheduler_class is not None:
            scheduler.step()
            for param_group in optimizer.param_groups:
                if verbose:
                    print(f"iteration {i_opt}: lr = {param_group['lr']}")

        for param in inputs:
            if torch.isnan(param).any():
                raise ValueError("NaN encountered during optimization")

    return x
