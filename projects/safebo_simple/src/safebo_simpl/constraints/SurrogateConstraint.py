from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import generics, params
from safebo_simpl.util.s_typing import AllowUndefined
from botorch import posteriors

from safebo_simpl.constraints._parent import Constraint
from safebo_simpl.util.continuity import NormTensor, StandardizationType
from safebo_simpl.util.s_typing import AllowUndefined

class SurrogateConstraint[T_BOParams: params.BOParams](Constraint):
    def __init__(self, dtype: torch.dtype, device: torch.device, state: T_BOParams, **kwargs: Any,) -> None:    
        super().__init__(dtype=dtype, device=device, **kwargs)
        self.surrogate: AllowUndefined[generics.Surrogate] = None
        self.state: T_BOParams = state

        self.X: Tensor = torch.tensor([], device=self.device, dtype=self.dtype)
        self.Y: Tensor = self.X.clone()

        self.x_normalization_object: AllowUndefined[NormTensor] = None
        self.y_normalization_object: AllowUndefined[NormTensor] = None

    def get_ucb(self, X: Tensor, beta: float, already_normalized: bool = True) -> Tensor:
        self._is_surrogate_active(self.get_ucb)
        val: Tensor = self.surrogate.get_ucb(X=X if already_normalized else self.x_normalization_object.normalize(X), beta=beta) # type: ignore
        return self.y_normalization_object.denormalize(val) # type: ignore
    
    def get_lcb(self, X: Tensor, beta: float, already_normalized: bool = True) -> Tensor:
        self._is_surrogate_active(self.get_lcb)
        val: Tensor = self.surrogate.get_lcb(X=X if already_normalized else self.x_normalization_object.normalize(X), beta=beta) # type: ignore
        return self.y_normalization_object.denormalize(val) # type: ignore

    def get_mean(self, X: Tensor, already_normalized: bool = True) -> Tensor:
        self._is_surrogate_active(self.get_mean)
        val: Tensor = self.surrogate.posterior(X=X if already_normalized else self.x_normalization_object.normalize(X)).mean # type: ignore
        return self.y_normalization_object.denormalize(val) # type: ignore

    def _is_surrogate_active(self, method: Callable[[Tensor, Any], Tensor]) -> None:
        if not isinstance(self.surrogate, generics.Surrogate):
            raise ValueError("Surrogate called before being initialized!")
        if not isinstance(self.x_normalization_object, NormTensor):
            raise ValueError(f"Normalization object for X must be initialized before calling {method.__name__}!")

    def evaluate_real(self, X: Tensor) -> Tensor:
        raise NotImplementedError(f"Evaluation of the real constraint function is not implemented for class: {self.__class__.__name__}")
    def initial_safe_candidates(self, n_points: int) -> Tensor:
        raise NotImplementedError(f"Initial safe candidates are not implemented for class: {self.__class__.__name__}")

    def fit_all(self, X: Tensor, bounds: Tensor, **kwargs: Any) -> None:
        Y: Tensor = self.evaluate_real(X=X)
        self.fit(X=X, Y=Y, bounds=bounds, **kwargs)

    def fit_append(self, X: Tensor, bounds: Tensor, **kwargs: Any) -> None:
        Y: Tensor = self.evaluate_real(X=X)
        
        new_X = torch.cat([self.X, X], dim=0)
        new_Y = torch.cat([self.Y, Y], dim=0)
        
        self.fit(X=new_X, Y=new_Y, bounds=bounds, **kwargs)
    
    def fit(self, X: Tensor, Y: Tensor, bounds: Tensor, *args: Any, **kwargs: Any,) -> None:
        """
            Assumes X, and Y are denormalized on input
        """

        self.x_normalization_object: AllowUndefined[NormTensor] = NormTensor(bounds, method=StandardizationType.MinMax)
        self.y_normalization_object: AllowUndefined[NormTensor] = NormTensor(Y, method=StandardizationType.ZScore)

        self.X: Tensor = X
        self.Y: Tensor = Y

        X_normalized: Tensor = self.x_normalization_object.normalize(self.X)
        Y_normalized: Tensor = self.y_normalization_object.normalize(self.Y)

        if isinstance(self.surrogate, generics.Surrogate):
            self.surrogate.refresh_surrogate(X=X_normalized, Y=Y_normalized)
            return
        self.surrogate: AllowUndefined[generics.Surrogate] = generics.Surrogate(self.dtype, self.device, X_normalized, Y_normalized, self.state)