from __future__ import annotations
import traceback

import math
from typing import Any, Generic, Callable, Generator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from ddo_suite.problems import BenchmarkProblem
from ddo_suite.algorithms.base import (
    BudgetExhausted,
    OptimizationResult,
    budget_remaining,
    build_result,
    make_budgeted_callable,
)

from safebo_simpl.util.generics import SafeBOAlgorithm
from safebo_simpl.util.params import BOParams
from safebo_simpl.util.typing import AllowUndefined
from safebo_simpl.objective_functions import ObjectiveFunction

from botorch.exceptions import ModelFittingError


class DDOSuite_ObjectiveWrapper(ObjectiveFunction):
    """
    Adapter bridging SafeBOAlgorithm's expected tensor operations with
    ddo-suite's numpy-based budgeted callable and real-world bounds.
    """
    def __init__(
        self,
        f_budgeted: Callable,
        dim: int,
        lb: NDArray[np.float64],
        span: NDArray[np.float64],
        dtype: torch.dtype,
        device: torch.device,
        maximize: bool = True,
    ) -> None:
        unit_bounds = torch.tensor(
            [[0.0] * dim, [1.0] * dim], 
            dtype=dtype, 
            device=device
        )
        super().__init__(
            dim=dim,
            negate=maximize,
            bounds=unit_bounds,
            device=device,
            dtype=dtype,
        )
        
        self.f_budgeted: Callable = f_budgeted
        self.lb: NDArray[np.float64] = lb
        self.span: NDArray[np.float64] = span

    def forward(self, X: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Maps unit-cube tensor X to real bounds and evaluates via the suite.
        The parent __call__ handles the negation.
        """
        X_np: NDArray[np.float64] = X.detach().cpu().numpy()
        y_list: list[float] = []
        
        for xi in X_np:
            x_real: NDArray[np.float64] = self.lb + xi * self.span
            val: float = float(self.f_budgeted(x_real))
            y_list.append(val)

        Y_tensor: Tensor = torch.tensor(y_list, dtype=self.dtype, device=self.device)
        if Y_tensor.ndim == 1:
            Y_tensor: Tensor = Y_tensor.unsqueeze(0)
            
        return torch.tensor(y_list, dtype=self.dtype, device=self.device)

class DDOSuite_AlgorithmWrapper[T_Algorithm: SafeBOAlgorithm, T_Params: BOParams]():
    def __init__(
        self,
        algorithm: type[T_Algorithm], # Uninitialized algorithm class
        params: T_Params,
        maximize_objective: bool = True,

        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.uninitialized_algorithm: type[T_Algorithm] = algorithm
        self.params: T_Params = params
        self.id: str = algorithm.__name__.lower().replace(" ", "_")
        self.maximize_objective: bool = maximize_objective

        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

    def __call__(
        self, 
        problem: BenchmarkProblem,
        max_evals: int,
        bounds: AllowUndefined[NDArray[np.float64]] = None,
        rng_seed: AllowUndefined[int] = None,
        *args: Any, 
        **kwargs: Any
    ) -> OptimizationResult:
        if max_evals < 0:
            raise ValueError(f"Max_evals must be positive, got {max_evals}")
        
        dim: int = problem.n_x
        bounds_arr = problem.bounds if bounds is None else np.asarray(bounds, dtype=np.float64)
        if bounds_arr.shape != (dim, 2):
            raise ValueError(f"Bounds must be of shape ({dim}, 2), got {bounds_arr.shape}")
            
        if budget_remaining(problem, max_evals) == 0:
            return self._error_object(
                dim=dim,
                reason="budget exhausted before start",
                metadata={"algorithm": self.id}
            )
        rng: np.random.Generator = np.random.default_rng(rng_seed if rng_seed is not None else problem.seed)
        seed: int = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        lb = bounds_arr[:, 0]
        span = bounds_arr[:, 1] - bounds_arr[:, 0]

        f_budgeted: Callable[[NDArray[np.float64]], float] = make_budgeted_callable(problem, max_evals)
        adapted_objective: DDOSuite_ObjectiveWrapper = DDOSuite_ObjectiveWrapper(
            f_budgeted=f_budgeted,
            dim=dim,                     
            lb=lb,
            span=span,
            dtype=self.dtype,
            device=self.device,
            maximize=self.maximize_objective
        )
        state: ModelState = ModelState()
        res: OptimizationResult = self._error_object(
            dim=problem.n_x,
            metadata={"termination": "failed_all_run_attempts"}
        )

        break_conditions: set[str] = {"failed_all_run_attempts", "fitting_error"}

        for _ in range(state.max_fit_restarts):
            problem.reset()
            state.increment()
            res: OptimizationResult = self._attempt_run(
                problem=problem,
                state=state,
                adapted_objective=adapted_objective,
                max_evals=max_evals,

                rng=rng,
            )

            if res.metadata.get("termination") not in break_conditions:
                break
        return res

    def _attempt_run(
            self,
            problem: BenchmarkProblem,
            state: ModelState,
            adapted_objective: DDOSuite_ObjectiveWrapper,
            max_evals: int,

            rng: np.random.Generator,
            ) -> OptimizationResult:
        try: 
            dim: int = problem.n_x

            while (rem := budget_remaining(problem=problem, max_evals=max_evals)) > 0:

                init_seed: int = int(rng.integers(0, 2**31 - 1))
                batch_size: int = self._get_rem_samples(maximum=self.params.sampling.initial_candidates, remaining=rem)
                sobol: SobolEngine = SobolEngine(dimension=dim, scramble=True, seed=init_seed)

                self.params.data.dimensions = dim
                self.params.data.bounds = problem.bounds
                    
                X: Tensor = sobol.draw(n=batch_size).to(dtype=self.dtype, device=self.device)
                Y: Tensor = adapted_objective.forward(X).unsqueeze(-1)

                self.algorithm: T_Algorithm = self.uninitialized_algorithm(
                    X=X,
                    Y=Y,
                    dtype=self.dtype,
                    device=self.device,
                    state=self.params,
                    objective_function=adapted_objective,
                )

                self._refresh()
                self.algorithm.train()
        except ModelFittingError:
            return self._error_object(
                dim=problem.n_x,
                reason="model fitting error",
                metadata={
                    "termination": "fitting_error",
                    "n_restarts": state.n_restarts,
                }
            )
        except BudgetExhausted:
            state.termination = "budget_exhausted"
        except Exception as exc:
            traceback.print_exc()
            exc_traceback: str = repr(exc)
            print(f"CRASH in {self.id}: {exc_traceback}")
            return self._build_check(
                problem=problem,
                success=False,
                metadata={
                    "termination": "exception",
                    "n_restarts": state.n_restarts,
                    "traceback": exc_traceback,
                }
            )
        return self._build_check(
            problem=problem,
            success=True,
            metadata={
                "termination": state.termination,
                "n_restarts": state.n_restarts,
            }
        )


    def _get_rem_samples(
            self,
            maximum: int,
            remaining: int,
        ) -> int:
        n_pts: float = min(maximum, remaining)
        if n_pts <= 0: 
            raise BudgetExhausted
        return n_pts

    def _refresh(
            self,
        ) -> None:
        if not hasattr(self.algorithm.state, "dynamics"):
            return
        if not hasattr(self.algorithm.state.dynamics, "reset"):
            return
        self.algorithm.state.dynamics.reset()

    
    def _build_check(
        self,
        problem: BenchmarkProblem,
        success: bool,
        metadata: dict[str, str | int] = {},
    ) -> OptimizationResult:
        meta: dict[str, str | int] = {
            **metadata,
            "algorithm": self.id,
        }
        if not problem.f_list:
            return self._error_object(
                dim=problem.n_x,
                elapsed=problem.elapsed,
                metadata=meta
            )
        return build_result(problem, success=success, metadata=meta)

    @staticmethod
    def _error_object(
        dim: int,
        elapsed: float = 0.,
        reason: str = "no evaluations completed",
        metadata: dict[str, str | int] = {},
    ) -> OptimizationResult:
        return OptimizationResult(
            best_x=np.full(dim, np.nan),
            best_f=float("inf"),
            n_evals=0,
            success=False,
            elapsed=elapsed,
            metadata={**metadata, "reason": reason}
        )

@dataclass
class ModelState():
    n_restarts: int = 0
    termination: str = "normal"
    max_fit_restarts: int = 50

    def increment(
            self
            ) -> None:
        self.n_restarts += 1
