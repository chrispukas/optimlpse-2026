import copy

from typing import Any, List
from dataclasses import dataclass, field

from safebo_simpl.util.typing import AllowUndefined, _factory
from safebo_simpl.constraints import Constraint

import torch
from torch import Tensor

@dataclass
class BOParams_Sampling():
    initial_candidates: int = 64
    batch_size: int = 256
    max_iterations: int = 30

@dataclass
class BOParams_Convergence():
    confidence_level: float = 2.5
    max_cholesky_size: float = float("inf")

@dataclass
class BOParams_Constraints[
    T_Constraint: Constraint
    ]:
    constraints: AllowUndefined[List[T_Constraint]] = None

    def __call__(
            self, 
            X: Tensor,

            *args: Any, 
            **kwds: Any
        ) -> Any:
        return self._get_constraints(
            X=X
        )
        
    def is_available(
            self,
        ) -> bool:
        return bool(self.constraints)

    def _get_constraints(
            self,
            X: Tensor
        ) -> Tensor:
        DEFAULT: Tensor = torch.ones(X.shape[:-1], dtype=torch.bool, device=X.device)
        if not self.constraints:
            return copy.copy(DEFAULT)
        mask: Tensor = copy.copy(DEFAULT)

        for constraint in self.constraints:
            mask: Tensor = mask & constraint.forward(X=X)
        return mask
    
    def refresh_constraints(
            self,
            X: Tensor
        ) -> None:
        if not self.constraints:
            return

        for constraint in self.constraints:
            constraint.fit(X=X)


@dataclass
class BOParams_Data():
    dimensions: int = 3
    negate: bool = True

@dataclass
class BOParams_Dynamics():
    max_iterations: int = 10
    current_iteration: int = 1

@dataclass 
class BOParams[
    T_BOParams_Sampling:    BOParams_Sampling,
    T_BOParams_Convergence: BOParams_Convergence,
    T_BOParams_Constraints: BOParams_Constraints,
    T_BOParams_Data:        BOParams_Data,
    T_BOParams_Dynamics:    BOParams_Dynamics,
    ]():

    sampling:    T_BOParams_Sampling
    convergence: T_BOParams_Convergence
    constraints: T_BOParams_Constraints
    data:        T_BOParams_Data
    dynamics:    T_BOParams_Dynamics

    type Instantiable[T] = T | type[T]

    def __init__(
            self, 
            sampling:    Instantiable[T_BOParams_Sampling]  =   BOParams_Sampling,
            convergence: Instantiable[T_BOParams_Convergence] = BOParams_Convergence,
            constraints: Instantiable[T_BOParams_Constraints] = BOParams_Constraints,
            data:        Instantiable[T_BOParams_Data] =        BOParams_Data,
            dynamics:    Instantiable[T_BOParams_Dynamics] =    BOParams_Dynamics,
            ) -> None:
        
        self.sampling:    T_BOParams_Sampling =    _factory(sampling)
        self.convergence: T_BOParams_Convergence = _factory(convergence)
        self.constraints: T_BOParams_Constraints = _factory(constraints)
        self.data:        T_BOParams_Data =        _factory(data)
        self.dynamics:    T_BOParams_Dynamics =    _factory(dynamics)
