from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import generics, params
from safebo_simpl.util.s_typing import AllowUndefined
from safebo_simpl.util.continuity import NormTensor
from botorch import posteriors

class Constraint():
    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
    def evaluate_real(self, X: Tensor) -> Tensor:
        raise NotImplementedError(f"The callable is not implemented for class: {self.__class__.__name__}!")
    def fit(self, X: Tensor, **kwargs: Any) -> None:
        raise NotImplementedError(f"The function fit is not implemented for class: {self.__class__.__name__}!")
    def forward(self, X: Tensor) -> Tensor:
        """
        Should return a constraint bitmask
        """
        raise NotImplementedError(f"The forward pass is not implemented for class: {self.__class__.__name__}!")
