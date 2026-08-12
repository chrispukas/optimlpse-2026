from typing import Tuple, List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from safebo_simpl.util import generics as su_safe

from safebo_simpl.objective_functions import ObjectiveFunction

from safebo_simpl.util.params import BOParams
from safebo_simpl.constraints import SurrogateConstraint, NonSurrogateConstraint, Constraint

import torch
from torch import Tensor

import numpy as np
import numpy.typing as npt

import botorch
from botorch import models as b_models
from botorch.posteriors import gpytorch as bp_gpytorch
        
class GoOSE(su_safe.SafeBOAlgorithm):
    def __init__(
            self, 
            X: Tensor, 
            Y: Tensor,

            dtype: torch.dtype,
            device: torch.device,

            state: BOParams,
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

        pessimistic: Tensor = self.get_pessimistic_safe_subset(
            X=X
        )
        def acqf_wrapper(
                Z: float | npt.NDArray[np.float32]
                ) -> float | npt.NDArray[np.float32]:
            penalty: float = float("inf")

            Z_t: Tensor = torch.tensor(Z, dtype=self.dtype, device=self.device)
            if Z_t.ndim == 1:
                Z_t: Tensor = Z_t.unsqueeze(0)

            # Ensures that the algorithm abides by constraints
            if self.state.constraints.is_available():
                if not self.state.constraints(X=Z):
                    return penalty

            

            optimistic: Tensor = self.get_optimistic_safe_subset(
                Z=Z_t,
                X=X,
            )
            if optimistic.ndim == 1:
                optimistic = optimistic.unsqueeze(0)

            lcb: Tensor = self.surrogate.get_lcb(X=optimistic, beta=self.state.convergence.confidence_level)
            return lcb.item()

        z_candidates: Tensor = self.de_sampler(
            n=self.state.sampling.batch_size,
            acq_func=acqf_wrapper,
            bounds=self.sanitize_bounds(self.state.data.bounds),
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

        eucl_distance_XZ: Tensor = torch.cdist(
            x1=X,
            x2=Z,
        )
        Z_lcb_tensor: Tensor = self.surrogate.get_lcb(
                X=Z, 
                beta=self.state.convergence.confidence_level
            ).squeeze(1)
        constraint_mask: Tensor = self.get_constraint_mask(
            X=X,
            eucl=eucl_distance_XZ,
            constraints=self.state.constraints.constraints,
        ).any(dim=0)

        z_masked: Tensor = torch.where(constraint_mask, Z_lcb_tensor, float("inf"))
        return Z[torch.argmin(z_masked, dim=0)]

    def get_constraint_mask[
        T_Constraint: Constraint
    ](
            self,
            X: Tensor,
            eucl: Tensor,
            constraints: List[T_Constraint],
        ) -> Tensor:

        constraint_mask: Tensor = torch.ones_like(
            eucl, 
            dtype=torch.bool, 
            device=self.device
            )
        for constraint in constraints:
            if not isinstance(constraint, SurrogateConstraint):
                continue

            L_i: Tensor = self.get_ith_lipscitz_constraint(
                X=X,
                surrogate_constraint=constraint,
            )

            u_i: Tensor = self.get_ith_lcb(
                X=X,
                surrogate_constraint=constraint,
            )

            safety: Tensor = (u_i - L_i * eucl) >= 0.
            constraint_mask: Tensor = constraint_mask & safety

        return constraint_mask
    def get_ith_lipscitz_constraint(
            self,
            X: Tensor,
            surrogate_constraint: SurrogateConstraint
        ) -> Tensor:

        with torch.enable_grad():
            X_grad: Tensor = X.clone().detach().requires_grad_(True)

            mean: Tensor = surrogate_constraint.surrogate.posterior(X=X_grad).mean.flatten()
            gradients: Tensor = torch.linalg.norm(
                torch.autograd.grad(
                    outputs=mean,
                    inputs=X_grad,
                    grad_outputs=torch.ones_like(mean),
                )[0],
                ord=float("inf"),
                dim=1
            )
            L_i: Tensor = torch.amax(gradients)
        return L_i.detach()
    def get_ith_lcb(
            self,
            X: Tensor,
            surrogate_constraint: SurrogateConstraint
            ) -> Tensor:
        return surrogate_constraint.surrogate.get_ucb(X=X, beta=self.state.convergence.confidence_level)

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