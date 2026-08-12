
import torch
from torch import Tensor

from safebo_simpl.util.typing import AllowUndefined, InitializerCallable
from safebo_simpl.util.params import BOParams
from safebo_simpl.objective_functions._parent import ObjectiveFunction

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

            _default_bounds=(-5, 10)
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
        d_rec: float = 1/float(dim)
        sqr_: Tensor = -20 * torch.exp(-0.2 * torch.sqrt(d_rec * torch.sum(torch.square(X), dim=-1)))
        trig_: Tensor = torch.exp(d_rec * torch.sum(torch.cos(2 * torch.pi * X), dim=-1))

        return sqr_ - trig_ + 20 + torch.e