from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import objfuncs, generics, params
from safebo_simpl.util import typing as su_typing
from botorch import posteriors

from safebo_simpl.constraints.parent import Constraint

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
        acq: Tensor = self.surrogate.get_ucb(X=X, beta=self.state.convergence.confidence_level)

        mask: Tensor = torch.ones_like(acq, dtype=torch.bool)
        mask &= (acq >= self.bounds[0, :]).all(dim=1)
        mask &= (acq <= self.bounds[1, :]).all(dim=1)

        return mask if boolmask else X[mask]

    def fit(
            self,
            X: Tensor,
            **kwargs: Any,
            ) -> None:
        Y_new: Tensor = self.constraint_function.forward(X=X).unsqueeze(-1)

        self.X = torch.cat([self.X, X])
        self.Y = torch.cat([self.Y, Y_new])

        self.surrogate.refresh_surrogate(
            X=self.X,
            Y=self.Y,
        )