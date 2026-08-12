from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import generics, params
from safebo_simpl.util import typing as su_typing
from botorch import posteriors

from safebo_simpl.constraints._parent import Constraint

class SurrogateConstraint[
    T_BOParams: params.BOParams
](Constraint):
    def __init__(
            self,
            dtype: torch.dtype,
            device: torch.device,

            X: Tensor,
            Y: Tensor,

            state: T_BOParams,
            bounds: Tensor,

            *args: Any,
            **kwargs: Any,
            ) -> None:    
        super().__init__(
            dtype=dtype,
            device=device,
            **kwargs
        )
        self.surrogate: generics.Surrogate = generics.Surrogate(
            dtype=self.dtype,
            device=self.device,

            X=X,
            Y=Y,

            state=state,
        )

        self.state: T_BOParams = state

        self.X: Tensor = X
        self.Y: Tensor = Y
        self.bounds: Tensor = bounds.to(device=device, dtype=dtype)

    def forward(
            self,
            X: Tensor,
            boolmask: bool = True,
            ) -> Tensor:
        mask: Tensor = self.__func(
            X=X,
            beta=self.state.convergence.confidence_level,

            bounds=self.bounds,
            acqf=self.surrogate.get_ucb
        )
        return mask if boolmask else X[mask]

    @staticmethod
    def __func(
        X: Tensor,
        beta: float,

        bounds: Tensor,
        acqf: Callable[[Tensor, float], Tensor]

    ) -> Tensor:
        acq: Tensor = acqf(X, beta)
        
        ub_m: Tensor = (acq >= bounds[0, :])
        lb_m: Tensor = (acq <= bounds[1, :])

        return (ub_m & lb_m).all(dim=1)

    def fit(
            self,
            X: Tensor,

            *args: Any,
            **kwargs: Any,
            ) -> None:
        Y_new: Tensor = self.state.constraints(X=X).unsqueeze(-1)

        self.X = torch.cat([self.X, X])
        self.Y = torch.cat([self.Y, Y_new])

        self.surrogate.refresh_surrogate(
            X=self.X,
            Y=self.Y,
        )