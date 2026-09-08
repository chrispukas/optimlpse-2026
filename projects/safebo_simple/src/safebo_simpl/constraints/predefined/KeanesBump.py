from typing import Any

import torch
from torch import Tensor

from safebo_simpl.util import objfuncs, params
from safebo_simpl.constraints._parent import Constraint

class KeanesBump(Constraint):
    def __init__(
            self,
            dim: int,
            dtype: torch.dtype,
            device: torch.device,

            *args: Any,
            **kwargs: Any,
            ) -> None:
        super().__init__(
            dtype=dtype,
            device=device,

            *args,
            **kwargs,
        )

        self.dim: int = dim

    def forward(
            self,
            X: Tensor,
            boolmask: bool = True,
            ) -> Tensor:
        if not self._is_valid_tensor(X=X, dim=self.dim):
            raise ValueError(f"Dim[-1] is of size: {X.shape[-1]}, expected {self.dim}")
        
        mask: Tensor = self.__func(
            X=X,
            dim=self.dim
        )
        return mask if boolmask else X[mask]

    @staticmethod
    def __func(
        X: Tensor, # [-1, batch_size, dim]
        dim: int,
    ) -> Tensor:
        b_1: Tensor = ((0.75 - torch.prod(X, dim=-1)) < 0)
        b_2: Tensor = ((torch.sum(X, dim=-1) - 7.5 * dim) < 0)
        return b_1 & b_2 # [batch_size, 1]