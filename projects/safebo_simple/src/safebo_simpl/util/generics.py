from __future__ import annotations
from typing import Tuple, Any, Callable
from dataclasses import dataclass

from safebo_simpl.util import params as su_prms
from safebo_simpl.util.s_typing import AllowUndefined
from safebo_simpl.constraints import SurrogateConstraint, Constraint

from safebo_simpl.objective_functions import ObjectiveFunction
from safebo_simpl.util.continuity import NormTensor, StandardizationType

import torch
from torch import Tensor
from torch import quasirandom as t_qrand

import gpytorch
from gpytorch import constraints as g_constraints
from gpytorch import likelihoods as g_lh
from gpytorch import mlls as g_mlls
from gpytorch import kernels as g_kern

import botorch
from botorch import models as b_models
from botorch.posteriors import gpytorch as bp_gpytorch
from botorch import fit as b_fit

import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution, Bounds

class Surrogate[
    T_BOParams: su_prms.BOParams
    ]:
    def __init__(self, dtype: torch.dtype, device: torch.device, X: Tensor, Y: Tensor, state: T_BOParams) -> None:
        super().__init__()
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
        self.state: T_BOParams = state
        
        self.refresh_surrogate(X=X, Y=Y,)

    def posterior(self, X: Tensor,) -> bp_gpytorch.GPyTorchPosterior:
        self.posterior_state.forward(
            x=X,
            model=self.surrogate_model
        )
        if self.posterior_state.posterior is None:
            raise ValueError("Poster is not defined!")
        return self.posterior_state.posterior

    def refresh_surrogate(self, X: Tensor, Y: Tensor,) -> Surrogate:
        (self.surrogate_model, self.surrogate_likelihood) = self._create_surrogate(
            X=X.detach().nan_to_num(0.),
            Y=Y.detach().nan_to_num(0.),
        )
        with gpytorch.settings.max_cholesky_size(self.state.convergence.max_cholesky_size):
            b_fit.fit_gpytorch_mll(mll=self.surrogate_likelihood)
        self.posterior_state = PosteriorState()
        return self

    def _create_surrogate(
                self,
                X: Tensor,
                Y: Tensor,
                noise_interval: g_constraints.Interval = g_constraints.Interval(1e-6, 1e-4),
                matern_smoothness: float = 2.5,
            ) -> Tuple[b_models.SingleTaskGP, g_mlls.ExactMarginalLogLikelihood]:
        self._check_tensor_safety(X=X, Y=Y,)
        likelihood: g_lh.GaussianLikelihood = g_lh.GaussianLikelihood(noise_constraint=noise_interval,)
        kernel: g_kern.ScaleKernel = g_kern.ScaleKernel(base_kernel=g_kern.MaternKernel(nu=matern_smoothness,))
        model: b_models.SingleTaskGP = b_models.SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood, covar_module=kernel,)
        mll: g_mlls.ExactMarginalLogLikelihood = g_mlls.ExactMarginalLogLikelihood(likelihood=likelihood, model=model)
        return (model, mll)

    def _check_tensor_safety(self, X: Tensor, Y: Tensor) -> None:
        if X is None:
            raise ValueError("X tensor is None!")
        if Y is None:
            raise ValueError("Y tensor is None!")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Dim mismatch: X tensor is of shape {X.shape[0]}. and Y tensor is of shape {Y.shape[0]}")
        if X.dtype != self.dtype or Y.dtype != self.dtype:
            raise ValueError(f"Datatype of either X, or Y are incorrectly configured, currently X: {X.dtype}, and Y: {Y.dtype} must be: {self.dtype}")
        if X.device.type != self.device.type or Y.device.type != self.device.type:
            raise ValueError(f"Device type of either X, or Y are incorrectly configured, currently X: {X.device}, and Y: {Y.device}, must be: {self.device}")

    def get_ucb(self, X: Tensor, beta: float,) -> Tensor:
            (mean, std) = self.get_properties(X=X)
            return mean + beta * std
            
    def get_lcb(self, X: Tensor, beta: float,) -> Tensor:
        (mean, std) = self.get_properties(X=X)
        return mean - beta * std

    def get_properties(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """
            Returns: (mean, std): tuple[Tensor, Tensor]
        """
        self.posterior_state.forward(x=X, model=self.surrogate_model)
        properties: Tuple[Tensor, Tensor] | None = self.posterior_state.properties
        if properties is None:
            raise ValueError("Unable to extract properties from the posterior to calculate the ucb!")
        return properties


class SafeBOAlgorithm[
    T_BOParams: su_prms.BOParams
    ]():
    def __init__(self, X: Tensor, Y: Tensor, dtype: torch.dtype, device: torch.device, 
                 state: T_BOParams, objective_function: ObjectiveFunction, *args: Any, **kwargs: Any,) -> None:
        super().__init__()
        self.X: Tensor = X # Denormalized X
        self.Y: Tensor = Y # Denormalized Y

        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
        self.state: T_BOParams = state

        self.objective_function: ObjectiveFunction = objective_function
        self.surrogate: Surrogate = Surrogate(dtype=dtype, device=device, X=X, Y=Y, state=state,)

        self.unit_bounds: Tensor = torch.tensor([[0.0, 1.0] for _ in range(self.state.data.dimensions)], dtype=self.dtype, device=self.device)

    def _train[T_Constraint: Constraint](self, single_pass: Callable[[Tensor, ObjectiveFunction], AllowUndefined[Tensor]], metrics: bool = False,) -> None:

        constraints: list[T_Constraint] = self.state.constraints.constraints or []
        X_normalization_object: NormTensor = NormTensor(self.objective_function.bounds, method=StandardizationType.MinMax)

        for _ in range(self.state.dynamics.max_iterations):

            X_denormalized: Tensor = self.X.detach() # Denormalized X
            Y_denormalized: Tensor = self.Y.detach() # Denormalized Y

            with gpytorch.settings.max_cholesky_size(self.state.convergence.max_cholesky_size):

                Y_normalization_object: NormTensor = NormTensor(Y_denormalized, method=StandardizationType.ZScore)

                # Normalize X, and Y here, provide this information to the objective function.
                X_normalized: Tensor = X_normalization_object.normalize(X_denormalized)
                Y_normalized: Tensor = Y_normalization_object.normalize(Y_denormalized)

                self.objective_function._set_norms(X_normtensor=X_normalization_object, Y_normtensor=Y_normalization_object)
                self.surrogate.refresh_surrogate(X=X_normalized, Y=Y_normalized) # Train the surrogate model on the normalized inputs, and outputs

                for constraint in constraints:
                    if not isinstance(constraint, SurrogateConstraint): continue
                    constraint.fit_append(X=X_denormalized, bounds=self.objective_function.bounds)

                # X_normalized_train -> | GP | -> Y_normalized_train, the GP is trained on the normalized/standardized data            
                unsafe_X_normalized_candidates: Tensor | None = single_pass(X_normalized, self.objective_function)
                if not isinstance(unsafe_X_normalized_candidates, Tensor): continue
                X_normalized_candidates: Tensor = self.sanitize_tensors(unsafe_X_normalized_candidates)

                # X_norm -> | GP | -> Y_norm -> Y_denorm (Y_candidates) (implicit transformation within the objective function wrapper)
                Y_denormalized_candidates: Tensor = self.sanitize_tensors(self.objective_function(X=X_normalized_candidates,))

                # (X_norm_test, Y_norm_test) -> | denormalization| -> (X_test, Y_test)
                X_denormalized_candidates: Tensor = X_normalization_object.denormalize(X_normalized_candidates)

            self.X: Tensor = torch.cat((self.X, X_denormalized_candidates), dim=0,)
            self.Y: Tensor = torch.cat((self.Y, Y_denormalized_candidates.T), dim=0,)

            if metrics:
                print(f"Minimum y-value: {torch.amin(self.X)}, Maximum y-value: {torch.amax(self.X)}, Latest: {Y_denormalized_candidates}")

    def train(
            self,
    ) -> None:
        raise NotImplementedError("Training logic not implemented!")

    # Discrete 'Monte Carlo' sampling
    def sobol_sampler(self, n: int, dim: int, scramble: bool = False, requires_grad: bool = False, center: bool = False,) -> Tensor:
        sobol: t_qrand.SobolEngine = t_qrand.SobolEngine(dimension=dim, scramble=scramble)
        draw: Tensor = sobol.draw(n=n).to(device=self.device, dtype=self.dtype).requires_grad_(requires_grad)
        return draw * 2 - 1 if center else draw

    def de_sampler(
            self,
            acq_func: Callable[[Tensor], Tensor],
            bounds: Tensor,

            de_samples_per_loop: int = 8,
            batch_size: int = 32,
            maxiter: int = 10000,
            strategy: str = "rand2bin",
            vectorized: bool = True,
            **kwargs,
            ):

        kw_args = {
            "popsize": batch_size,
            "bounds": self.sanitize_bounds(bounds.detach().cpu().numpy()),
            "maxiter": maxiter,
            "strategy": strategy,
            "vectorized": vectorized,
            "updating": "deferred",
            **kwargs
        }

        return self._default_continuous_wrapper(acq_func, differential_evolution, top_k=de_samples_per_loop, **kw_args)

    def cmaes_sampler(self,
                      acq_func: Callable[[Tensor], Tensor],
                      bounds: Tensor,

                      top_k: int = 8,
                      batch_size: int = 32,
                      maxiter: int = 10000,
                      **kwargs,
                      ) -> Tensor:
        # Lazy import when required
        from fcmaes import cmaes

        sanitized = self.sanitize_bounds(bounds.detach().cpu().numpy())

        lb = sanitized[:, 0].flatten().astype(float)  # shape: (dim,)
        ub = sanitized[:, 1].flatten().astype(float)
        
        kw_args = {
                    "bounds": Bounds(lb, ub),
                    "max_iterations": maxiter,
                    "popsize": batch_size, 
                }

        return self._default_continuous_wrapper(acq_func, custom_cmaes_minimize, top_k=top_k, **kw_args)

    def _default_continuous_wrapper(self, acq_func: Callable[[Tensor], Tensor], optimizer: Callable[[Any], Any], top_k: int, *args, **kwargs) -> Tensor:
        vectorized: bool = kwargs.get("vectorized", True)
        wrapper: ContinuousOptimizerWrapper = ContinuousOptimizerWrapper(acq_func=acq_func, device=self.device, dtype=self.dtype)

        _: npt.NDArray[np.float32] = optimizer(wrapper, **kwargs).x
        curr_loss: Tensor = wrapper.curr_loss
        curr_proposed_candidates: Tensor = wrapper.curr_proposed_candidates
        
        k_elements = min(top_k, curr_loss.numel())
        _, idx = torch.topk(curr_loss.flatten(), k=k_elements, largest=False) # type: ignore
        return curr_proposed_candidates[idx, :] if vectorized else curr_proposed_candidates[idx]
    
    @staticmethod
    def sanitize_bounds(
        bounds: npt.NDArray[np.float32],
        limit: float = 1e5
        ) -> npt.NDArray[np.float32]:
        return np.nan_to_num(
            bounds, 
            nan=0.0, 
            posinf=limit, 
            neginf=-limit
        )

    @staticmethod
    def sanitize_tensors(X: Tensor) -> Tensor:
        return X.unsqueeze(0) if X.ndim == 1 else X
@dataclass
class PosteriorState():
    posterior: bp_gpytorch.GPyTorchPosterior | None = None
    X: Tensor | None = None

    def is_cached(self, x: Tensor) -> bool:
        if self.X is None:
            return False
        if x.shape != self.X.shape:
            return False
        if x.requires_grad or (self.X.requires_grad if self.X is not None else False):
            return False
            
        return torch.equal(x, self.X)
    
    def forward(self, x: Tensor, model: b_models.SingleTaskGP,) -> None:
        if x is None:
            return
        if  not isinstance(self.posterior, bp_gpytorch.GPyTorchPosterior) \
            or not isinstance(self.X, Tensor) \
            or not self.is_cached(x=x):

            posterior: bp_gpytorch.GPyTorchPosterior | Any = model.posterior(X=x)
            if not isinstance(posterior, bp_gpytorch.GPyTorchPosterior):
                raise ValueError(f"Posterior is the incorrect class ({posterior.__class__.__name__})!")
            
            self.posterior: bp_gpytorch.GPyTorchPosterior | None = posterior
            self.X: Tensor | None = x
            return

    @property
    def properties(self,) -> Tuple[Tensor, Tensor] | None:
            if self.posterior is None:
                return None
            return (self.posterior.mean, torch.sqrt(self.posterior.variance))


class ContinuousOptimizerWrapper:
    def __init__(self, acq_func: Callable[[Tensor], Tensor], device: torch.device, dtype: torch.dtype) -> None:
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        self.curr_proposed_candidates: torch.Tensor = torch.tensor([], device=device, dtype=dtype)
        self.curr_loss: torch.Tensor = self.curr_proposed_candidates.clone()
        self.acq_func: Callable[[Tensor], Tensor] = acq_func
    def __call__(self, x: npt.NDArray[np.float32]) -> Any:
        return self.wrapper(x)

    def wrapper(
            self,
            x: npt.NDArray[np.float32]
            ) -> npt.NDArray[np.float32]:        
        X: Tensor = torch.permute(torch.tensor(x, dtype=self.dtype, device=self.device,), (1, 0))
        with torch.no_grad():
            loss: torch.Tensor = self.acq_func(X)

        self.curr_proposed_candidates = X
        self.curr_loss = loss

        return loss.detach().cpu().numpy()


from scipy.optimize import OptimizeResult
from fcmaes.cmaes import Cmaes

def custom_cmaes_minimize(fun: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]], **kwargs: Any) -> OptimizeResult:
    bounds = kwargs.get("bounds", None)
    x0 = kwargs.get("x0", None)
    popsize = kwargs.get("popsize", 32)
    max_iterations = kwargs.get("max_iterations", 10000)
    input_sigma = kwargs.get("input_sigma", 0.3)

    es: Cmaes = Cmaes(
        bounds=bounds,
        x0=x0,
        input_sigma=input_sigma,
        popsize=popsize,
        max_evaluations=max_iterations * popsize
    )

    best_x = None
    best_fun = float("inf")

    for _ in range(max_iterations):
        if es.stop:
            break
        xs: npt.NDArray[np.float32] = es.ask().T  # shape: (dim, pop_size)
        loss_vals: npt.NDArray[np.float32] = fun(xs)
        stop = es.tell(loss_vals)

        min_idx = np.argmin(loss_vals)
        if loss_vals[min_idx] < best_fun:
            best_fun = loss_vals[min_idx]
            best_x = xs[:, min_idx]

        if stop:
            break

    res = OptimizeResult()
    res.x = best_x if best_x is not None else np.zeros(1)
    res.fun = best_fun
    return res