from typing import Any, Tuple, List, TypeVar, Generic, Callable, overload
from dataclasses import dataclass, field

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
class BOParams_Constraints():
    ...

@dataclass
class BOParams_Data():
    dimensions: int = 3
    negate: bool = True

@dataclass
class BOParams_Dynamics():
    max_iterations: int = 10
    current_iteration: int = 1

    def is_exceeded(self):
            return self.current_iteration > self.max_iterations
    def increment(self):
        print(f"Current iteration: {self.current_iteration}")
        self.current_iteration = self.current_iteration + 1

@dataclass 
class BOParams[
    T_BOParams_Sampling:    BOParams_Sampling,
    T_BOParams_Convergence: BOParams_Convergence,
    T_BOParams_Constraints: BOParams_Constraints,
    T_BOParams_Data:        BOParams_Data,
    T_BOParams_Dynamics:    BOParams_Dynamics,
    ]():

    sampling: T_BOParams_Sampling
    convergence: T_BOParams_Convergence
    constraints: T_BOParams_Constraints
    data: T_BOParams_Data
    dynamics: T_BOParams_Dynamics

    type Instantiable[T] = T | type[T]

    def __init__(
            self, 
            sampling: Instantiable[T_BOParams_Sampling]  = BOParams_Sampling,
            convergence: Instantiable[T_BOParams_Convergence] = BOParams_Convergence,
            constraints: Instantiable[T_BOParams_Constraints] = BOParams_Constraints,
            data: Instantiable[T_BOParams_Data] = BOParams_Data,
            dynamics: Instantiable[T_BOParams_Dynamics] = BOParams_Dynamics,
            ) -> None:
        
        self.sampling: T_BOParams_Sampling = self._factory(sampling)
        self.convergence: T_BOParams_Convergence = self._factory(convergence)
        self.constraints: T_BOParams_Constraints = self._factory(constraints)
        self.data: T_BOParams_Data = self._factory(data)
        self.dynamics: T_BOParams_Dynamics = self._factory(dynamics)

    def _factory[T](
            self, 
            val: T | type[T],
            **kwargs: Any
            ) -> T:
        return val(**kwargs) if isinstance(val, type) else val # type: ignore
