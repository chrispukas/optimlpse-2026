
from typing import Callable

import torch
from torch import Tensor

from botorch import test_functions as b_tfuncs
from safebo_simpl.util import params as su_prms

type InitializerCallable[A, T] = Callable[[A], T] | None

class ObjectiveFunction():
    def __init__(self) -> None:
        self.acq: InitializerCallable[Tensor, Tensor] = None
    def forward(
            self,
            X: Tensor
            ) -> Tensor:
        if self.acq is None:
            raise NotImplementedError("Acquisition function is currently not implemented!")
        return self.acq(X)
        
class Ackley[
    T_BOParams: su_prms.BOParams
    ](ObjectiveFunction):
    def __init__(
            self,
            state: T_BOParams,
            device: torch.device = torch.device("mps"),
            dtype: torch.dtype = torch.float32
            ) -> None:
        super().__init__()

        ackley: b_tfuncs.Ackley = b_tfuncs.Ackley(
                    dim=state.data.dimensions,
                    negate=state.data.negate,
                ).to(
                    device=device,
                    dtype=dtype
                )
        self.acq: InitializerCallable[Tensor, Tensor] = ackley

    def forward(
            self, 
            X: Tensor
            ) -> Tensor:
        return super().forward(X=X)