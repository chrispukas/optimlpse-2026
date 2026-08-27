from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import generics, params
from safebo_simpl.util.s_typing import AllowUndefined
from safebo_simpl.util.continuity import NormTensor
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

        self.x_normtensor: AllowUndefined[NormTensor] = None
        self.y_normtensor: AllowUndefined[NormTensor] = None

    def _set_norms(
                self,
                X_normtensor: NormTensor,
                Y_normtensor: NormTensor
        ) -> None:
            self.x_normtensor: AllowUndefined[NormTensor] = X_normtensor
            self.y_normtensor: AllowUndefined[NormTensor] = Y_normtensor

    def __call__(
            self, 
            X: Tensor,
            *args: Any, 
            **kwargs: Any
        ) -> Tensor:
        """
            Accepts a normalized X tensor, returns a boolmask
        """
        if not isinstance(self.x_normtensor, NormTensor):
             raise ValueError("Normalization Tensor for X is not set!")

        X_denorm: Tensor = self.x_normtensor.denormalize(X)
        result: Tensor = self.forward(
            X=X_denorm,
        )
        return result

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
        pass
        #print(f"The function fit is not implemented for class: {self.__class__.__name__}!")
    def forward(
            self,
            X: Tensor
    ) -> Tensor:
        pass
        #print(f"The forward pass is not implemented for class: {self.__class__.__name__}!")
