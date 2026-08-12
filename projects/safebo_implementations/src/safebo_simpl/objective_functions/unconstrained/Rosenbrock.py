
import torch
from torch import Tensor

from safebo_simpl.util.typing import AllowUndefined, InitializerCallable
from safebo_simpl.util.params import BOParams
from safebo_simpl.objective_functions._parent import ObjectiveFunction

from botorch import test_functions as b_tfuncs

class Rosenbrock(ObjectiveFunction):
    def __init__(
        self,

        dim: int,
        negate: bool = False,
        bounds: AllowUndefined[Tensor] = None,

        device: torch.device = torch.device("mps"),
        dtype: torch.dtype = torch.float32
    ) -> None:

        super().__init__(
            dim=dim,
            negate=negate,
            bounds=bounds,

            device=device,
            dtype=dtype,

            _default_bounds=(-float('inf'), float('inf'))
        )

    def forward(
        self, 
        X: Tensor,
        dim: int,
    ) -> Tensor:
        return self.__func(
            X=X,
            dim=dim,
        )
    
    @staticmethod
    def __func(
        X: Tensor,
        dim: int = 3,
    ) -> Tensor:
        
        X_t: Tensor = X.narrow(dim=-1, start=0, length=dim - 2) # t -> n-1
        X_tp1: Tensor = X.narrow(dim=-1, start=1, length=dim - 1) # t -> n-1
        return torch.sum(
            100 * torch.square(X_tp1 - torch.square(X_t)) + torch.square(1 - X_t),
            dim=0
            )