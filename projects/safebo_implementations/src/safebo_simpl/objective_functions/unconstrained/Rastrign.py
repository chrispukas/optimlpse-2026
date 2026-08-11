
import torch
from torch import Tensor

from safebo_simpl.util.typing import AllowUndefined, InitializerCallable
from safebo_simpl.util.params import BOParams
from safebo_simpl.objective_functions.parent import ObjectiveFunction

from botorch import test_functions as b_tfuncs

class Rastrigin(ObjectiveFunction):
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

            _default_bounds=(-5.11, 5.11)
        )

    def forward(
        self, 
        X: Tensor
    ) -> Tensor:
        return self.__func(
            X=X,
            dim=self.dim,
        )
    
    @staticmethod
    def __func(
        X: Tensor,
        dim: int = 3,
        A: float = 10,
    ) -> Tensor:
        return A * dim + torch.sum(
            torch.square(X) - A * torch.cos(2 * torch.pi * X),
            dim=-1)