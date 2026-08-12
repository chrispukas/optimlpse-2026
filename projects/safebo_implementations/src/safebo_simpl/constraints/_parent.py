from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import generics, params
from safebo_simpl.util import typing as su_typing
from botorch import posteriors

class Constraint():
    def __init__(
            self,
            dtype: torch.dtype,
            device: torch.device,

            *args: Any,
            **kwargs: Any,
            ) -> None:
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

    def __call__(
            self, 
            X: Tensor,
            *args: Any, 
            **kwargs: Any
        ) -> Tensor:
        return self.forward(
            X=X,
        )

    @staticmethod
    def _is_valid_tensor(
        X: Tensor,
        dim: int
    ) -> bool:
        return X.shape[-1] == dim

    def fit(
            self,
            X: Tensor,
            **kwargs: Any
    ) -> None:
        raise NotImplementedError(f"The function fit is not implemented for class: {self.__class__.__name__}!")
    def forward(
            self,
            X: Tensor
    ) -> Tensor:
        raise NotImplementedError(f"The forward pass is not implemented for class: {self.__class__.__name__}!")
