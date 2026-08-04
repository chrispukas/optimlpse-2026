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

class NonSurrogateConstraint[
    T_BOParams: params.BOParams,
    T_ObjectiveFunction: objfuncs.ObjectiveFunction
](Constraint[T_BOParams, T_ObjectiveFunction]):
    def __init__(
            self,
            dtype: torch.dtype,
            device: torch.device,

            state: T_BOParams,
            constraint_function: T_ObjectiveFunction,
            bounds: Tensor,

            **kwargs: Any,
            ) -> None:
        
        # Pre-define types to expose them to the IDE        
        self.dtype: torch.dtype
        self.device: torch.device

        self.state: T_BOParams
        self.constraint_function: T_ObjectiveFunction
        self.bounds: Tensor

        super().__init__(
            dtype=dtype,
            device=device,
            state=state,
            constraint_function=constraint_function,
            bounds=bounds,
            **kwargs
        )


class SurrogateConstraint[
    T_BOParams: params.BOParams,
    T_ObjectiveFunction: objfuncs.ObjectiveFunction
](Constraint[T_BOParams, T_ObjectiveFunction]):
    def __init__(
            self,
            dtype: torch.dtype,
            device: torch.device,

            X: Tensor,
            Y: Tensor,

            state: T_BOParams,
            constraint_function: T_ObjectiveFunction | type[T_ObjectiveFunction],

            bounds: Tensor,

            **kwargs: Any,
            ) -> None:
        
        # Pre-define types to expose them to the IDE
        self.dtype: torch.dtype
        self.device: torch.device

        self.state: T_BOParams
        self.constraint_function: T_ObjectiveFunction

        self.bounds: Tensor = bounds.to(device=device, dtype=dtype)
        self.X: Tensor = X
        self.Y: Tensor = Y
        
        super().__init__(
            dtype=dtype,
            device=device,
            state=state,
            constraint_function=constraint_function,
            **kwargs
        )
        self.surrogate: generics.Surrogate = generics.Surrogate(
            dtype=self.dtype,
            device=self.device,

            X=X,
            Y=Y,

            state=self.state,
        )

    def forward(
            self,
            X: Tensor,
            boolmask: bool = True,
            ) -> Tensor:
        acq: Tensor = self.surrogate.get_lcb(X=X, beta=self.state.convergence.confidence_level)
        lb_mask: Tensor = (acq >= self.bounds[0, :]).any(dim=1)
        ub_mask: Tensor = (acq <= self.bounds[1, :]).any(dim=1)

        mask: Tensor = lb_mask & ub_mask

        return mask if boolmask else X[mask]

    def fit(
            self,
            X: Tensor,
            **kwargs: Any,
            ) -> None:
        Y_new: Tensor = self.constraint_function.forward(X=X)

        self.X = torch.cat([self.X, X])
        self.Y = torch.cat([self.Y, Y_new])

        self.surrogate.refresh_surrogate(
            X=self.X,
            Y=self.Y,
        )