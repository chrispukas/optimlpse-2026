from __future__ import annotations
import traceback

import math
from typing import Any, Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from ddo_suite.problems import BenchmarkProblem, ConstrainedProblem
from ddo_suite.algorithms.base import (
    BudgetExhausted,
    OptimizationResult,
    budget_remaining,
    build_result,
    make_budgeted_callable,
    make_constrained_callable
)

from safebo_simpl.util.generics import SafeBOAlgorithm
from safebo_simpl.util.params import BOParams
from safebo_simpl.util.s_typing import AllowUndefined
from safebo_simpl.objective_functions import ObjectiveFunction
from safebo_simpl.constraints import Constraint, SurrogateConstraint
from safebo_simpl.util.continuity import NormTensor, StandardizationType

from botorch.exceptions import ModelFittingError


class DDOSuite_ConstraintWrapper[T_Params: BOParams](SurrogateConstraint):
    def __init__(
        self,
        problem: ConstrainedProblem,
        f_budgeted: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
        params: T_Params,
        constraint_idx: int = 0,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(dtype=dtype, device=device, state=params)
        self.f_budgeted: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]] = f_budgeted
        self.problem: ConstrainedProblem = problem
        
        self.bounds = torch.tensor(self.problem.bounds, dtype=self.dtype, device=self.device)
        self.x_normtensor = NormTensor(X=self.bounds, method=StandardizationType.MinMax)

        self.constraint_idx = constraint_idx

    def evaluate_real(self, X: Tensor) -> Tensor:
        evaluated_constraints: list[Tensor] = []
        
        for entry in X.detach().cpu().numpy():
            _, constraints = self.f_budgeted(entry)
            evaluated_constraints.append(torch.tensor(constraints[self.constraint_idx], device=X.device, dtype=X.dtype))

        return -torch.stack(evaluated_constraints, dim=0).unsqueeze(-1)
    
    def initial_safe_candidates(self, n_points: int) -> Tensor:
        return torch.tensor(self.problem.safe_initial_design(n=n_points), dtype=self.dtype, device=self.device)


class DDOSuite_ObjectiveWrapper(ObjectiveFunction):
    def __init__(
        self,
        maximize: bool,
        f_budgeted: Callable[[NDArray[np.float64]], float],
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(dim=dim, negate=maximize, device=device, dtype=dtype,)
        self.f_budgeted: Callable[[NDArray[np.float64]], float] = f_budgeted

    def forward(self, X: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        evaluated_objectives: list[float] = []
        for xi in X.detach().cpu().numpy():
            f_evaluated: float = float(self.f_budgeted(xi))
            evaluated_objectives.append(f_evaluated)
            
        return torch.tensor(np.stack(evaluated_objectives), dtype=self.dtype, device=self.device).nan_to_num(0.)


class DDOSuite_AlgorithmWrapper[T_Algorithm: SafeBOAlgorithm, T_Params: BOParams]():
    safe_break_conditions: set[str] = {"failed_all_run_attempts", "fitting_error"}

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
        rng_seed: AllowUndefined[int] = None,
        *args: Any, 
        **kwargs: Any
    ) -> OptimizationResult:
        self._check_validity(max_evals=max_evals, problem=problem)
        if budget_remaining(problem, max_evals) == 0:
            return self._error_object(
                dim=problem.n_x,
                reason="budget exhausted before start",
                metadata={"algorithm": self.id}
            )
        # Initialize deterministic random number generator
        (_rgen, seed) = self._set_seed(seed=(problem.seed if rng_seed is None else rng_seed))

        # Initialize constraints, and objective functions from the wrappers
        (adapted_objective_function, adapted_constraint_functions) = self._initialize_funcs(params=self.params, problem=problem, max_evals=max_evals)
        self._set_params(params=self.params, dim=problem.n_x, constraint=adapted_constraint_functions)

        # Initialize states, and a default result object
        state: ModelState = ModelState()
        result: OptimizationResult = self._error_object(
            dim=problem.n_x,
            metadata={"termination": "failed_all_run_attempts"}
        )

        for _r in range(state.max_fit_restarts):
            state.n_restarts += 1
            problem.reset()
            self._reset_constraints(problem)

            result: OptimizationResult = self._attempt_run(
                problem=problem,
                state=state,
                adapted_objective=adapted_objective_function,
                max_evals=max_evals,
                seed=seed,
            )

            if result.metadata.get("termination") not in self.safe_break_conditions:
                break
        return result

    @staticmethod
    def _check_validity(max_evals: int, problem: BenchmarkProblem) -> None:
        if max_evals < 0:
            raise ValueError(f"Max_evals must be positive, got {max_evals}")
        if problem.bounds.shape != (problem.n_x, 2):
            raise ValueError(f"Bounds must be of shape ({problem.n_x}, 2), got {problem.bounds.shape}")

    @staticmethod
    def _set_seed(seed: AllowUndefined[int]) -> tuple[np.random.Generator, int]:
        if not isinstance(seed, int):
            raise ValueError("Seed not defined!")
        rng: np.random.Generator = np.random.default_rng(seed=seed)
        deterministic_seed: int = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(deterministic_seed)
        return (rng, deterministic_seed)

    @staticmethod
    def _reset_constraints(problem: BenchmarkProblem) -> None:
        if not isinstance(problem, ConstrainedProblem):
            return
        problem.c_list.clear()

    @staticmethod
    def _set_params(params: T_Params, dim: int, constraint: AllowUndefined[list[DDOSuite_ConstraintWrapper]]) -> None:
        params.data.dimensions = dim
        constraint_list: list[DDOSuite_ConstraintWrapper] = constraint if constraint else []
        params.constraints.constraints = (params.constraints.constraints or []) + constraint_list # type: ignore

    def _initialize_funcs(self, params: T_Params, problem: BenchmarkProblem, max_evals: int) -> tuple[DDOSuite_ObjectiveWrapper, list[DDOSuite_ConstraintWrapper] | None]:
        problem.reset()
        params.reset()

        objective: DDOSuite_ObjectiveWrapper = self._initialize_objective(problem=problem, max_evals=max_evals)
        constraints: list[DDOSuite_ConstraintWrapper] | None = self._initialize_constraint(problem=problem, max_evals=max_evals)
        return (objective, constraints)

    def _initialize_objective(self, problem: BenchmarkProblem, max_evals: int) -> DDOSuite_ObjectiveWrapper:
        f_budgeted: Callable[[NDArray[np.float64]], float] = make_budgeted_callable(problem, max_evals)
        adapted_objective: DDOSuite_ObjectiveWrapper = DDOSuite_ObjectiveWrapper(
            f_budgeted=f_budgeted,
            dim=problem.n_x,                     
            dtype=self.dtype,
            device=self.device,
            maximize=self.maximize_objective
        )
        adapted_objective.bounds = torch.tensor(problem.bounds, dtype=self.dtype, device=self.device)
        adapted_objective.x_normtensor = NormTensor(X=adapted_objective.bounds, method=StandardizationType.MinMax)
        return adapted_objective

    def _initialize_constraint(self, problem: BenchmarkProblem, max_evals: int) -> AllowUndefined | list[DDOSuite_ConstraintWrapper]:
        if not isinstance(problem, ConstrainedProblem):
            return None

        final: list[DDOSuite_ConstraintWrapper] = []

        f_budgeted: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]] = make_constrained_callable(problem=problem, max_evals=max_evals)
        num_constraints: int = problem._constraints(problem.safe_initial_design(n=1)).shape[0]

        for idx in range(num_constraints):
            adapted_constraint: DDOSuite_ConstraintWrapper = DDOSuite_ConstraintWrapper(
                problem=problem,
                f_budgeted=f_budgeted,
                params=self.params,
                constraint_idx=idx,
                dtype=self.dtype,
                device=self.device,
            )
            final.append(adapted_constraint)
        return final

    def _get_initial_candidates[T_BenchmarkProblem: BenchmarkProblem](self, batch_size: int, seed: int, bounds: Tensor, problem: BenchmarkProblem) -> Tensor:
        if isinstance(problem, ConstrainedProblem):
            constraint: DDOSuite_ConstraintWrapper = self.params.constraints.constraints[0]
            X_denormalized: Tensor = constraint.initial_safe_candidates(batch_size) # Returns X_denormalized
            x_normalization_object: NormTensor = NormTensor(X=bounds, method=StandardizationType.MinMax) 
            return x_normalization_object.normalize(X_denormalized) # X_denormalized -> X_normalized
        else:
            sobol: SobolEngine = SobolEngine(dimension=problem.n_x, scramble=True, seed=seed)
            X_normalized: Tensor = sobol.draw(n=batch_size).to(dtype=self.dtype, device=self.device) # Returns a tensor between 0, and 1
            return X_normalized

    def _attempt_run(
            self,
            problem: BenchmarkProblem,
            state: ModelState,
            adapted_objective: DDOSuite_ObjectiveWrapper,
            max_evals: int,
            seed: int,
            ) -> OptimizationResult:
        try: 
            while (rem := budget_remaining(problem=problem, max_evals=max_evals)) > 0:
                assert adapted_objective.x_normtensor

                batch_size: int = self._get_remaining_samples(maximum=self.params.sampling.initial_candidates, remaining=rem)

                # X is normalized
                X_normalized: Tensor = self._get_initial_candidates(batch_size=batch_size, seed=seed, bounds=adapted_objective.bounds, problem=problem)
                Y_denormalized: Tensor = adapted_objective(X_normalized).unsqueeze(-1) # X_normalized -> Y_denormalized

                # X_normalized -> X_denormalized, Y_denormalized -> f(x) -> GP_trained
                self.algorithm: T_Algorithm = self.uninitialized_algorithm(
                    X=adapted_objective.x_normtensor.denormalize(X_normalized),
                    Y=Y_denormalized,
                    dtype=self.dtype,
                    device=self.device,
                    state=self.params,
                    objective_function=adapted_objective,
                )

                self._refresh()
                self.algorithm.train()

        except ModelFittingError:
            return self._error_object(dim=problem.n_x, reason="model fitting error",
                metadata={"termination": "fitting_error", "n_restarts": state.n_restarts,})
        except BudgetExhausted:
            state.termination = "budget_exhausted"
        except Exception as exc:
            traceback.print_exc()
            return self._safe_build_result(problem=problem, success=False,
                metadata={"termination": "exception", "n_restarts": state.n_restarts, "traceback": repr(exc),})
        
        return self._safe_build_result(problem=problem, success=True, 
            metadata={"termination": state.termination, "n_restarts": state.n_restarts,})

    def _get_remaining_samples(
            self,
            maximum: int,
            remaining: int,
        ) -> int:
        remaining_points: float = min(maximum, remaining)
        if remaining_points <= 0: 
            raise BudgetExhausted
        return remaining_points

    def _refresh(
            self,
        ) -> None:
        if not hasattr(self.algorithm.state, "dynamics"):
            return
        if not hasattr(self.algorithm.state.dynamics, "reset"):
            return
        self.algorithm.state.dynamics.reset()

    def _safe_build_result(
        self,
        problem: BenchmarkProblem,
        success: bool,
        metadata: dict[str, str | int] = {},
    ) -> OptimizationResult:
        """
            Safely builds the result fromt the problem, throwing error objects if the objective function has not been called at all.
        """
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
            metadata={**metadata, "reason": reason}
        )

@dataclass
class ModelState():
    n_restarts: int = 0
    termination: str = "normal"
    max_fit_restarts: int = 8
