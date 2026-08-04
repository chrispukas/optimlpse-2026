from typing import Tuple, List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from safebo_simpl.util import generics as su_safe
from safebo_simpl.util import params as su_prms
from safebo_simpl.util import objfuncs as su_objfuncs

import torch
from torch import Tensor

import botorch
from botorch import models as b_models
from botorch.posteriors import gpytorch as bp_gpytorch

@dataclass
class BOParams_GPsTR(
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
            self
            ) -> None:
        # Setting initial config for dynamic variables across a run
        self.dynamics.max_iterations = 30    
        self.data.negate = False
        
class GoOSE(su_safe.SafeBOAlgorithm):
    def __init__(
            self, 
            X: Tensor, 
            Y: Tensor,

            dtype: torch.dtype,
            device: torch.device,

            state: BOParams_GPsTR,
            objective_function: su_objfuncs.ObjectiveFunction,
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
            objective_function: su_objfuncs.ObjectiveFunction,
            **kwargs: Any
        ) -> Tensor:
        z_candidates: Tensor = self.sample_discrete(
            n=self.state.sampling.batch_size,
            dim=X.shape[0],
            scramble=True
        )
        pessimistic: Tensor = self.get_pessimistic_safe_subset(
            X=X
        )
        optimistic: Tensor = self.get_optimistic_safe_subset(
            Z=z_candidates,
            X=X,
        )
        next_candidates: Tensor = self.selection(
            X_select=pessimistic,
            Z_select=optimistic.squeeze(0),
            X=X,
        )
        return next_candidates


    def get_pessimistic_safe_subset(
            self,
            X: Tensor,
        ) -> Tensor:
        ucb_tensor: Tensor = self.surrogate.get_lcb(
            X=X, 
            beta=self.state.convergence.confidence_level
            ).squeeze(1)
        return X[torch.argmin(ucb_tensor, dim=0)]

    def get_optimistic_safe_subset(
            self,
            Z: Tensor,
            X: Tensor,
        ) -> Tensor:

        X_ucb_tensor: Tensor = self.surrogate.get_lcb(
                X=X, 
                beta=self.state.convergence.confidence_level
            )
                
        posterior_X: bp_gpytorch.GPyTorchPosterior = self.surrogate.posterior(X=X)
        mean_flat: Tensor = posterior_X.mean.flatten()

        gradients: Tensor = torch.linalg.norm(
            torch.autograd.grad(
                outputs=mean_flat,
                inputs=X,
                grad_outputs=torch.ones_like(mean_flat),
            )[0],
            ord=float("inf"),
            dim=1
        )

        eucl_distance_XZ: Tensor = torch.cdist(
            x1=X,
            x2=Z,
        )

        L_i: Tensor = torch.amax(gradients)
        safety: Tensor = X_ucb_tensor - L_i * eucl_distance_XZ
        mask: Tensor = (safety >= 0.0).any(dim=0)

        Z_lcb_tensor: Tensor = self.surrogate.get_lcb(
                X=Z, 
                beta=self.state.convergence.confidence_level
            ).squeeze(1)

        z_masked: Tensor = torch.where(mask, Z_lcb_tensor, float("inf"))
        return Z[torch.argmin(z_masked, dim=0)]

    def selection(
            self,

            X_select: Tensor,
            Z_select: Tensor,

            X: Tensor,
        ) -> Tensor:
        beta: float = self.state.convergence.confidence_level
        x_lcb: Tensor = self.surrogate.get_lcb(
            X=X_select.unsqueeze(0), 
            beta=beta
            )
        z_lcb: Tensor = self.surrogate.get_lcb(
            X=Z_select.unsqueeze(0), 
            beta=beta
            )

        if x_lcb.item() < z_lcb.item():
            return X_select
        
        eucl_distance: Tensor = torch.cdist(
            x1=X,
            x2=Z_select.unsqueeze(0)
        )
        return X[torch.argmin(input=eucl_distance, dim=0)]