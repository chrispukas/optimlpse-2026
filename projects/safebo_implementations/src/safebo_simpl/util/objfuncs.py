
from typing import Callable, List, Tuple

import torch
from torch import Tensor

from botorch import test_functions as b_tfuncs
from botorch.utils.transforms import unnormalize
from safebo_simpl.util import params as su_prms
from safebo_simpl.util.typing import AllowUndefined, InitializerCallable

class ObjectiveFunction():
    def __init__(self) -> None:
        self.acq: InitializerCallable[Tensor, Tensor] = None
        self.bounds: AllowUndefined[Tensor] = None
    def forward(
            self,
            X: Tensor
            ) -> Tensor:
        if self.acq is None:
            raise NotImplementedError("Acquisition function is currently not implemented!")
        if self.bounds is None:
            raise NotImplementedError("Bounds are not strictly defined!")

        return self.acq(
            unnormalize(
                X=X, 
                bounds=self.bounds
                ))

class Ackley[
    T_BOParams: su_prms.BOParams
    ](ObjectiveFunction):
    def __init__(
            self,
            state: T_BOParams,
            standardize: bool = True,
            bounds: AllowUndefined[Tensor] = None,
            device: torch.device = torch.device("mps"),
            dtype: torch.dtype = torch.float32
            ) -> None:
        super().__init__()

        self.state: T_BOParams = state
        self.bounds: AllowUndefined[Tensor] = bounds if isinstance(bounds, list) else torch.tensor([
                [-5., 10.] for _ in range(state.data.dimensions)
            ]).to(device=device, dtype=dtype).T
            
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        ackley: b_tfuncs.Ackley = b_tfuncs.Ackley(
                            dim=self.state.data.dimensions,
                            negate=self.state.data.negate,
                        ).to(
                            device=self.device,
                            dtype=self.dtype
                        )
        self.acq: InitializerCallable[Tensor, Tensor] = ackley

    def forward(
            self, 
            X: Tensor
            ) -> Tensor:
        return super().forward(X=X)