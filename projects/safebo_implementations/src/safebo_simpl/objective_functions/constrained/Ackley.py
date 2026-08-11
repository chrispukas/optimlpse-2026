
import torch
from torch import Tensor

from safebo_simpl.util.typing import AllowUndefined, InitializerCallable
from safebo_simpl.util.params import BOParams
from safebo_simpl.objective_functions.parent import ObjectiveFunction

from botorch import test_functions as b_tfuncs

class Ackley(ObjectiveFunction):
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
    ) -> Tensor:
        num: Tensor = torch.sum(torch.pow(torch.cos(X), exponent=4), dim=-1,) - 2 * torch.prod(torch.pow(torch.cos(X), exponent=2), dim=-1,)
        idx: Tensor = torch.arange(start=0, end=dim+1, device=X.device, dtype=X.dtype)
        denom: Tensor = torch.sqrt(torch.sum(torch.mul(idx, torch.square(X)), dim=-1))

        return - torch.abs(num / denom)