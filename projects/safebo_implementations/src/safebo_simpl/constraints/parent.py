from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import objfuncs, generics, params
from safebo_simpl.util import typing as su_typing
from botorch import posteriors

class Constraint[
    T_BOParams: params.BOParams,
    T_ObjectiveFunction: objfuncs.ObjectiveFunction,
]:
    def __init__(
            self,
            dtype: torch.dtype,
            device: torch.device,

            state: T_BOParams,
            constraint_function: T_ObjectiveFunction | type[T_ObjectiveFunction],

            **kwargs: Any,
            ) -> None:
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

        self.constraint_function: T_ObjectiveFunction = su_typing._factory(constraint_function)
        self.state: T_BOParams = state
    def __call__(
            self, 
            X: Tensor,
            *args: Any, 
            **kwargs: Any
            ) -> Tensor:
        return self.forward(
            X=X,
        )


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
