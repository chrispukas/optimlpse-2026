from typing import Tuple, List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from safebo_simpl.util import generics as su_safe
from safebo_simpl.util import params as su_prms

from safebo_simpl.objective_functions import ObjectiveFunction

import torch
from torch import Tensor

import botorch
from botorch import models as b_models
from botorch.posteriors import gpytorch as bp_gpytorch

@dataclass
class BOParams_SafeOpt(
    su_prms.BOParams[
        su_prms.BOParams_Sampling,
        su_prms.BOParams_Convergence,
        su_prms.BOParams_Constraints,
        su_prms.BOParams_Data,
        su_prms.BOParams_Dynamics
        ]
    ):
    def __init__(
            self,
            ) -> None:
        super().__init__()

    def __post_init__(
                self,
            ) -> None:

        # Setting initial config for dynamic variables across a run
        self.dynamics.max_iterations = 30    
        self.data.negate = False

class SafeOpt(su_safe.SafeBOAlgorithm):
    def __init__(
            self, 
            X: Tensor, 
            Y: Tensor,

            dtype: torch.dtype,
            device: torch.device,

            state: BOParams_SafeOpt,
            objective_function: ObjectiveFunction,
            ) -> None:
        super().__init__(
            X, 
            Y, 
            dtype=dtype, 
            device=device,
            state=state,
            objective_function=objective_function
            )

    def train(
            self
            ) -> None:
        super()._train(
            single_pass=self.forward, 
            metrics=True
            )

    def forward(
                self,
                X: Tensor,
                objective_function: ObjectiveFunction,
                **kwargs: Any
            ) -> Tensor:
        X_candidates: Tensor = self.sobol_sampler(
            n=self.state.sampling.batch_size,
            dim=self.state.data.dimensions,
            scramble=True
        )
        Z_candidates: Tensor = self.sobol_sampler(
            n=self.state.sampling.batch_size,
            dim=self.state.data.dimensions,
            scramble=True
        )

        uncertainty_candidates: Tensor = self.get_uncertainty(
            X=X_candidates
        )
        minimizer: Tensor = self.get_minimizer_set_guess(
            X=X_candidates,
            Z=Z_candidates,
            W=uncertainty_candidates,
        )
        expander: Tensor = self.get_expander_set_guess(
            X=X_candidates,
            Z=Z_candidates,
            W=uncertainty_candidates,
        )

        next_candidates: Tensor = self.select_next_candidate(
            minimizer=minimizer,
            expander=expander,
        )
        return next_candidates

    
    def get_minimizer_set_guess(
            self,
            X: Tensor,
            Z: Tensor,
            W: Tensor,
        ) -> Tensor:
        beta: float = self.state.convergence.confidence_level
        lcb_X: Tensor = self.surrogate.get_lcb(X=X, beta=beta)
        ucb_Z: Tensor = self.surrogate.get_ucb(X=Z, beta=beta)

        safety: Tensor = torch.min(ucb_Z) - lcb_X
        masked: Tensor = safety >= 0

        minimized: Tensor = torch.where(masked, W, float("-inf"))
        return X[torch.argmax(minimized, dim=0)]

    def get_expander_set_guess(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
    ) -> Tensor:
        beta: float = self.state.convergence.confidence_level

        eucl_distances: Tensor = torch.cdist(
            x1=X, 
            x2=Z
        )
        posterior_X: bp_gpytorch.GPyTorchPosterior = self.surrogate.posterior(X=X)
        mean_flat: Tensor = posterior_X.mean.flatten()

        gradients: Tensor = torch.abs(torch.autograd.grad(
            outputs=mean_flat, 
            inputs=X, 
            grad_outputs=torch.ones_like(mean_flat), 
            retain_graph=True
        )[0])

        L_i: Tensor = torch.max(torch.abs(gradients)) # i-constraint Lipschitz constant
        ucb_i: Tensor = self.surrogate.get_ucb(X=X, beta=beta)

        safety: Tensor = ucb_i - L_i * eucl_distances
        masked: Tensor = safety.any(dim=1, keepdim=True)

        maximized: Tensor = torch.where(masked, W, float("-inf"))
        return X[torch.argmax(maximized, dim=0)]

    def select_next_candidate(
            self,
            minimizer: Tensor,
            expander: Tensor,
        ) -> Tensor:
        cat: Tensor = torch.cat(
            (minimizer, expander), 
            dim=0
            )
        uncertainty: Tensor = self.get_uncertainty(
            X=cat
        )
        return cat[torch.argmax(uncertainty, dim=0)]

    def get_uncertainty(
            self,
            X: Tensor,
        ) -> Tensor:
        beta: float = self.state.convergence.confidence_level
        posterior: bp_gpytorch.GPyTorchPosterior = self.surrogate.posterior(X=X)
        return torch.sqrt(posterior.variance) * beta * 2


    ... # INSERT FUNCTIONS FOR LOGIC HERE