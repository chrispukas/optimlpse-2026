from typing import Tuple, List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from safebo_simpl.util import generics as su_safe

from safebo_simpl.objective_functions import ObjectiveFunction

from safebo_simpl.util.params import BOParams
from safebo_simpl.util.continuity import NormTensor
from safebo_simpl.constraints import SurrogateConstraint, NonSurrogateConstraint, Constraint
from safebo_simpl.util.math import LipschitzConstraints

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
        self.bounds: npt.NDArray[np.float32] = np.array([[0.0, 1.0] for _ in range(self.state.data.dimensions)], dtype=np.float32)

    def train(
            self
            ) -> None:
        super()._train(
            single_pass=self.forward, 
            metrics=False
            )

    def forward(
            self,
            X: Tensor,
            objective_function: ObjectiveFunction,
            **kwargs: Any
        ) -> Tensor:

        # Compute lipschitz constraints at the beginning of the loop (cached once for next computations)
        lipschitz: LipschitzConstraints = LipschitzConstraints(self.state.convergence.confidence_level)
        constraints: List[Tuple[Tensor, Tensor]] = lipschitz.get_constraint_list(X, self.state.constraints.constraints)

        # Calculate the pessimistic safe subset (independent from others)
        pessimistic: Tensor = self.get_pessimistic_safe_subset(
            X=X # returns a normalized t
        )
        def acqf_wrapper(
                Z: Tensor
                ) -> float | npt.NDArray[np.float32]:
            penalty: float = float("inf")

            # Ensures that the algorithm abides by constraints
            if not isinstance(self.objective_function.x_normtensor, NormTensor):
                raise ValueError("Normalization for x not set in the objective function!")
            if self.state.constraints.is_available():
                if not self.state.constraints(X=self.objective_function.x_normtensor.denormalize(X=Z)):
                    return penalty

            optimistic: Tensor = self.get_optimistic_safe_subset(
                Z=Z,
                X=X,
                constraints=constraints
            )
            if optimistic.ndim == 1:
                optimistic = optimistic.unsqueeze(0)

            lcb: Tensor = self.surrogate.get_lcb(X=optimistic, beta=self.state.convergence.confidence_level)
            return lcb.item()

        z_candidates: Tensor = self.de_sampler(
            n=self.state.sampling.batch_size,
            acq_func=acqf_wrapper,
            bounds=self.bounds,
        )
        optimistic: Tensor = self.get_optimistic_safe_subset(
            Z=z_candidates,
            X=X,
            constraints=constraints
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
        lcb_tensor: Tensor = self.surrogate.get_lcb(
            X=X, 
            beta=self.state.convergence.confidence_level
            ).squeeze(1)
        return X[torch.argmin(lcb_tensor, dim=0)]

    def get_optimistic_safe_subset(
            self,
            Z: Tensor,
            X: Tensor,
            constraints: List[Tuple[Tensor, Tensor]]
        ) -> Tensor:

        if Z.ndim == 1:
            Z = Z.unsqueeze(0)

        eucl_distance_XZ: Tensor = torch.cdist(
            x1=X,
            x2=Z,
        )
        Z_lcb_tensor: Tensor = self.surrogate.get_lcb(
                X=Z, 
                beta=self.state.convergence.confidence_level
            ).squeeze(1)
        constraint_mask: Tensor = self.get_constraint_mask(
            eucl=eucl_distance_XZ,
            constraints=constraints,
        ).any(dim=0)

        z_masked: Tensor = torch.where(constraint_mask, Z_lcb_tensor, float("inf"))
        return Z[torch.argmin(z_masked, dim=0)]

    def get_constraint_mask(
            self,
            eucl: Tensor,
            constraints: List[Tuple[Tensor, Tensor]],
        ) -> Tensor:

        constraint_mask: Tensor = torch.ones_like(
            eucl, 
            dtype=torch.bool, 
            device=self.device
            )
        for (L_i, u_i) in constraints:
            safety: Tensor = (u_i - L_i * eucl) >= 0.
            constraint_mask: Tensor = constraint_mask & safety

        return constraint_mask

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