from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import objfuncs, generics, params
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
            constraint_function: T_ObjectiveFunction,
            bounds: Tensor,

            **kwargs: Any,
            ) -> None:
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

        self.constraint_function: T_ObjectiveFunction = constraint_function
        self.state: T_BOParams = state
        self.bounds: Tensor = bounds

class NonSurrogateConstraint[
    T_BOParams: params.BOParams,
    T_ObjectiveFunction: objfuncs.ObjectiveFunction
](Constraint):
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
](Constraint):
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
        self.surrogate: generics.Surrogate = generics.Surrogate(
            dtype=self.dtype,
            device=self.device
        )

    def apply(
            self,
            X: Tensor,
            truncate: bool = False
            ) -> Tensor:

        posterior: posteriors.GPyTorchPosterior = self.surrogate.posterior(X=X)
        acq: Tensor = self.surrogate.get_lcb(X=X, beta=self.state.convergence.confidence_level)
        
        ...

    def train() -> Tensor:
        ...