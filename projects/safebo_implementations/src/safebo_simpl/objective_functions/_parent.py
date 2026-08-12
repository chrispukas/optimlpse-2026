from typing import Callable, List, Tuple, Any

import torch
from torch import Tensor

from safebo_simpl.util.typing import AllowUndefined

class ObjectiveFunction():
    def __init__(
            self,
            dim: int,
            negate: bool = False,
            bounds: AllowUndefined[Tensor] = None,

            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,

            _default_bounds: Tuple[float, float] = (-1, 1),
            ) -> None:
        self.dim: int = dim
        self.negate: bool = negate
        
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        if isinstance(bounds, Tensor):
            self.bounds: Tensor = bounds.to(device=self.device, dtype=self.dtype)
        else:
            self.bounds: Tensor = self._generate_default_bounds(
                lb=_default_bounds[0], 
                ub=_default_bounds[1], 
                dim=self.dim)

    def __call__(
            self,
            X: Tensor,
    ) -> Tensor:
        x: Tensor = X.to(device=self.device, dtype=self.dtype)
        if not self._check_in_bounds(X=x, bounds=self.bounds):
            raise ValueError(f"Input {x} provided is outside of the specified bounds: {self.bounds}!")
        if not isinstance(X, Tensor):
            raise ValueError("X must be of type Tensor!")
        if X.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension to be of size ({self.dim}), got ({X.shape[-1]})")

        result: Tensor = self.forward(X=x)
        return -result if self.negate else result


    def _generate_default_bounds(
        self,
        lb: float,
        ub: float,
        dim: int,
    ) -> Tensor:
        return torch.tensor(
            [[lb] * dim, [ub] * dim],
            device=self.device,
            dtype=self.dtype,
        )
    def _check_in_bounds(
            self,
            X: Tensor,
            bounds: AllowUndefined[Tensor]
    ) -> bool:
        if not isinstance(bounds, Tensor):
            raise ValueError("Bounds are not of type Tensor!")
        
        lb_m: Tensor = ((bounds[0, :] - X) <= 0) # [batch_size, dim] : True if in bounds (difference is positive)
        ub_m: Tensor = ((X - bounds[1, :]) <= 0) # [batch_size, dim] : True if in bounds (also positive diff)

        mask: Tensor = lb_m & ub_m
        return bool(mask.all().item())

    def forward(
            self,
            X: Tensor,
            *args: Any,
            **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError(f"Inherited subclass {self.__class__.__name__} must implement a forward pass.")