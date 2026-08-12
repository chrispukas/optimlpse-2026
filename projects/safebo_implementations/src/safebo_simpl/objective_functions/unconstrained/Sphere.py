
import torch
from torch import Tensor

from safebo_simpl.util.typing import AllowUndefined, InitializerCallable
from safebo_simpl.util.params import BOParams
from safebo_simpl.objective_functions._parent import ObjectiveFunction

from botorch import test_functions as b_tfuncs

class Sphere(ObjectiveFunction):
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

            _default_bounds=(-float("inf"), float("inf"))
        )

    def forward(
        self, 
        X: Tensor
    ) -> Tensor:
        return self.__func(
            X=X,
        )
    
    @staticmethod
    def __func(
        X: Tensor,
    ) -> Tensor:
        return torch.sum(
            torch.square(X),
            dim=-1,
        )