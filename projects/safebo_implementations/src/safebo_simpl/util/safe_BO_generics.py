from __future__ import annotations
from typing import Tuple, Any, Callable
from dataclasses import dataclass

from safebo_simpl.util import bayesian_parameters as su_prms
from safebo_simpl.util import objective_functions as su_objfuncs

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

class SafeBOAlgorithm[
    T_BOParams: su_prms.BOParams
    ]():
    def __init__(
            self,
            X_safe: Tensor,
            Y_safe: Tensor,

            dtype: torch.dtype,
            device: torch.device,

            state: T_BOParams,

            objective_function: su_objfuncs.ObjectiveFunction,
            ) -> None:
        super().__init__()
        self.X_safe: Tensor = X_safe
        self.Y_safe: Tensor = Y_safe

        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

        self.state: T_BOParams = state

        self.objective_function: su_objfuncs.ObjectiveFunction = objective_function

    def _train(
            self,
            single_pass: Callable[[Tensor, su_objfuncs.ObjectiveFunction], Tensor],
            metrics: bool = False,
            ) -> None:

        while not self.state.dynamics.is_exceeded():
            self.state.dynamics.increment()
            self.refresh_surrogate(X=self.X_safe.detach(), Y=self.Y_safe.detach())

            with gpytorch.settings.max_cholesky_size(self.state.convergence.max_cholesky_size):
                b_fit.fit_gpytorch_mll(mll=self.surrogate_likelihood)

                X_candidates: Tensor = single_pass(
                    self.X_safe,
                    self.objective_function
                ).unsqueeze(0)
                Y_candidates: Tensor = self.objective_function.forward(
                    X=X_candidates,
                ).unsqueeze(-1)


            print(self.X_safe.shape, X_candidates.shape)

            self.X_safe: Tensor = torch.cat(
                (self.X_safe, X_candidates),
                dim=0,
            )
            self.Y_safe: Tensor = torch.cat(
                (self.Y_safe, Y_candidates),
                dim=0,
            )

            if metrics:
                print(f"Minimum y-value: {torch.amin(self.Y_safe)}")
                
    
    def refresh_surrogate(
            self,
            X: Tensor,
            Y: Tensor,
            ) -> SafeBOAlgorithm:
        (self.surrogate_model, self.surrogate_likelihood) = self._create_surrogate(
            X=X,
            Y=Y,
        )
        self.posterior_state = PosteriorState()
        return self

    def _check_tensor_safety(
            self,
            X: Tensor,
            Y: Tensor
        ) -> None:
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
    
    def _create_surrogate(
            self,
            X: Tensor,
            Y: Tensor,
            noise_interval: g_constraints.Interval = g_constraints.Interval(1e-6, 1e-4),
            matern_smoothness: float = 2.5,
        ) -> Tuple[b_models.SingleTaskGP, g_mlls.ExactMarginalLogLikelihood]:
        self._check_tensor_safety(
            X=X,
            Y=Y,
        )
        likelihood: g_lh.GaussianLikelihood = g_lh.GaussianLikelihood(
            noise_constraint=noise_interval,
        )
        kernel: g_kern.ScaleKernel = g_kern.ScaleKernel(
            base_kernel=g_kern.MaternKernel(
                nu=matern_smoothness,
            )
        )
        model: b_models.SingleTaskGP = b_models.SingleTaskGP(
            train_X=X,
            train_Y=Y,
            likelihood=likelihood,
            covar_module=kernel,
        )
        mll: g_mlls.ExactMarginalLogLikelihood = g_mlls.ExactMarginalLogLikelihood(
            likelihood=likelihood,
            model=model
        )
        return (model, mll)

    def sample_discrete(
            self,
            n: int,
            dim: int,
            scramble: bool = False,
            requires_grad: bool = False,
            center: bool = False,
            ) -> Tensor:
        sobol: t_qrand.SobolEngine = t_qrand.SobolEngine(
            dimension=dim,
            scramble=scramble
        )
        draw: Tensor = sobol.draw(n=n).to(device=self.device, dtype=self.dtype).requires_grad_(requires_grad)
        return draw * 2 - 1 if center else draw

    def get_ucb(
            self,
            X: Tensor,
            beta: float,
        ) -> Tensor:
        self.posterior_state.forward(
            x=X,
            model=self.surrogate_model
            )
        properties: Tuple[Tensor, Tensor] | None = self.posterior_state.get_properties()
        if properties is None:
            raise ValueError("Unable to extract properties from the posterior to calculate the ucb!")
        (mean, std) = properties
        return mean + beta * std
        
    def get_lcb(
            self,
            X: Tensor,
            beta: float,
        ) -> Tensor:
        self.posterior_state.forward(
            x=X,
            model=self.surrogate_model
            )
        properties: Tuple[Tensor, Tensor] | None = self.posterior_state.get_properties()
        if properties is None:
            raise ValueError("Unable to extract properties from the posterior to calculate the lcb!")
        (mean, std) = properties
        return mean - beta * std

    def posterior(
            self,
            X: Tensor,
        ) -> bp_gpytorch.GPyTorchPosterior:
        self.posterior_state.forward(
            x=X,
            model=self.surrogate_model
        )
        if self.posterior_state.posterior == None:
            raise ValueError("Poster is not defined!")
        return self.posterior_state.posterior

@dataclass
class PosteriorState():
    posterior: bp_gpytorch.GPyTorchPosterior | None = None
    X: Tensor | None = None

    def is_cached(
                self,
                x: Tensor
            ) -> bool:
        if self.X is None:
            return False
        if x.shape != self.X.shape:
            return False
        return torch.equal(x, self.X)
    
    def forward(
            self,
            x: Tensor,
            model: b_models.SingleTaskGP,
        ) -> None:
        if x is None:
            return
        if  not isinstance(self.posterior, bp_gpytorch.GPyTorchPosterior) \
            or not isinstance(self.X, Tensor) \
            or not self.is_cached(x=x):

            posterior: bp_gpytorch.GPyTorchPosterior | Any = model.posterior(X=x)
            if not isinstance(posterior, bp_gpytorch.GPyTorchPosterior):
                raise ValueError(f"Posterior is the incorrect class ({posterior.__class__.__name__})!")
            
            self.posterior = posterior
            self.X = x
            return
        
    def get_properties(
                self,
            ) -> Tuple[Tensor, Tensor] | None:
            if self.posterior is None:
                return None
            return (self.posterior.mean, torch.sqrt(self.posterior.variance))