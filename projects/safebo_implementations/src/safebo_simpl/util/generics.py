from __future__ import annotations
from typing import Tuple, Any, Callable, Generic
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
from scipy.optimize import differential_evolution

from contextlib import contextmanager

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
        if self.posterior_state.posterior == None:
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
            self.posterior_state.forward(x=X, model=self.surrogate_model)
            properties: Tuple[Tensor, Tensor] | None = self.posterior_state.properties
            if properties is None:
                raise ValueError("Unable to extract properties from the posterior to calculate the ucb!")
            (mean, std) = properties
            return mean + beta * std
            
    def get_lcb(self, X: Tensor, beta: float,) -> Tensor:
        self.posterior_state.forward(x=X, model=self.surrogate_model)
        properties: Tuple[Tensor, Tensor] | None = self.posterior_state.properties
        if properties is None:
            raise ValueError("Unable to extract properties from the posterior to calculate the lcb!")
        (mean, std) = properties
        print(f"X shape: {X.shape} | Mean min/max: {mean.min().item():.4f}, {mean.max().item():.4f} | Std min/max: {std.min().item():.4f}, {std.max().item():.4f}")
        return mean - beta * std

class SafeBOAlgorithm[
    T_BOParams: su_prms.BOParams
    ]():
    def __init__(self, X: Tensor, Y: Tensor, dtype: torch.dtype, device: torch.device, 
                 state: T_BOParams, objective_function: ObjectiveFunction, *args: Any, **kwargs: Any,) -> None:
        super().__init__()
        self.X: Tensor = X
        self.Y: Tensor = Y

        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
        self.state: T_BOParams = state

        self.objective_function: ObjectiveFunction = objective_function
        self.surrogate: Surrogate = Surrogate(dtype=dtype, device=device, X=X, Y=Y, state=state,)

        self.unit_bounds: npt.NDArray[np.float32] = np.array([[0.0, 1.0] for _ in range(self.state.data.dimensions)], dtype=np.float32)

    def _train[T_Constraint: Constraint](self, single_pass: Callable[[Tensor, ObjectiveFunction], AllowUndefined[Tensor]], metrics: bool = False,) -> None:

        constraints: list[T_Constraint] = self.state.constraints.constraints or []
        X_normalization_object: NormTensor = NormTensor(self.objective_function.bounds, method=StandardizationType.MinMax)

        for constraint in constraints:
            if not isinstance(constraint, SurrogateConstraint):
                continue
            constraint.fit_all(X=X_normalization_object.denormalize(self.X), bounds=self.objective_function.bounds,)
        for _ in range(self.state.dynamics.max_iterations):

            X: Tensor = self.X.detach()
            Y: Tensor = self.Y.detach()

            with gpytorch.settings.max_cholesky_size(self.state.convergence.max_cholesky_size):

                Y_normalization_object: NormTensor = NormTensor(Y, method=StandardizationType.ZScore)

                # Normalize X, and Y here, provide this information to the objective function.
                X_normalized_train: Tensor = X_normalization_object.normalize(X)
                Y_normalized_train: Tensor = Y_normalization_object.normalize(Y)

                self.objective_function._set_norms(X_normtensor=X_normalization_object, Y_normtensor=Y_normalization_object)
                self.surrogate.refresh_surrogate(X=X_normalized_train, Y=Y_normalized_train)

                # X_normalized_train -> | GP | -> Y_normalized_train, the GP is trained on the normalized/standardized data            
                X_normalized_candidates: AllowUndefined[Tensor] = single_pass(X_normalized_train, self.objective_function)

                if not isinstance(X_normalized_candidates, Tensor): continue

                # X_norm -> | GP | -> Y_norm -> Y_denorm (Y_candidates) (implicit transformation within the objective function wrapper)
                X_candidates: Tensor = X_normalized_candidates.unsqueeze(0) if X_normalized_candidates.ndim == 1 else X_normalized_candidates
                Y_candidates: Tensor = self.objective_function(X=X_candidates,)
                Y_candidates: Tensor = Y_candidates.unsqueeze(-1) if Y_candidates.ndim == 1 else Y_candidates

                # (X_norm_test, Y_norm_test) -> | denormalization| -> (X_test, Y_test)
                X_cand_denorm: Tensor = X_normalization_object.denormalize(X_candidates)

            for constraint in constraints:
                if not isinstance(constraint, SurrogateConstraint):
                    continue
                constraint.fit_append(X=X_cand_denorm, bounds=self.objective_function.bounds)

            self.X: Tensor = torch.cat((self.X, X_candidates), dim=0,)
            self.Y: Tensor = torch.cat((self.Y, Y_candidates), dim=0,)

            if metrics:
                print(f"Minimum y-value: {torch.amin(self.X)}, Maximum y-value: {torch.amax(self.X)}, Latest: {Y_candidates}")

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
            de_samples_per_loop: int,
            batch_size: int,
            acq_func: Callable[[Tensor], Tensor],
            bounds: npt.NDArray[np.float32],
            maxiter: int = 50,
            strategy: str = "rand2bin",
            vectorized: bool = True,
            candidates: AllowUndefined[torch.Tensor] = None
            ):

        curr_proposed_candidates: torch.Tensor = torch.tensor([], device=self.device, dtype=self.dtype)
        curr_loss: torch.Tensor = curr_proposed_candidates.clone()
        
        def wrapper(
                x: npt.NDArray[np.float32]
                ) -> npt.NDArray[np.float32]:
            nonlocal curr_proposed_candidates, curr_loss
            
            X: Tensor = torch.permute(torch.tensor(x, dtype=self.dtype, device=self.device,), (1, 0))
            with torch.no_grad():
                loss: torch.Tensor = acq_func(X)

            curr_proposed_candidates = X
            curr_loss = loss

            return loss.detach().cpu().numpy()

        init: str | npt.NDArray[np.float64] = "latinhypercube" if not isinstance(candidates, Tensor) else candidates.detach().cpu().numpy()

        _: npt.NDArray[np.float32] = differential_evolution(func=wrapper, bounds=self.sanitize_bounds(bounds), popsize=de_samples_per_loop, maxiter=maxiter, 
                                                                 strategy=strategy, vectorized=vectorized, updating="deferred", polish=False, mutation=(0.5, 1.0), 
                                                                 init=init).x
    
        k_elements = min(batch_size, curr_loss.numel())
        _, idx = torch.topk(curr_loss.flatten(), k=k_elements, largest=False)
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