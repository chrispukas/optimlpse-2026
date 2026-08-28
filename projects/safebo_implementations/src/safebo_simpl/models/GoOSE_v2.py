from typing import Any, Callable
from dataclasses import dataclass, field

from safebo_simpl.util import generics as su_safe

from safebo_simpl.objective_functions import ObjectiveFunction

from safebo_simpl.util.params import BOParams
from safebo_simpl.util.continuity import NormTensor
from safebo_simpl.constraints import SurrogateConstraint, NonSurrogateConstraint, Constraint
from safebo_simpl.util.s_math import LipschitzConstraints

import torch
from torch import Tensor

import numpy as np
import numpy.typing as npt

import botorch
from botorch import models as b_models
from botorch.posteriors import gpytorch as bp_gpytorch
        
class GoOSEV2(su_safe.SafeBOAlgorithm):
    def __init__(
            self,  
            X: Tensor, 
            Y: Tensor,
            dtype: torch.dtype,
            device: torch.device,
            state: BOParams,
            objective_function: ObjectiveFunction,
            ) -> None:
        super().__init__(X=X, Y=Y, dtype=dtype, device=device, state=state, objective_function=objective_function)

    def train(self) -> None:
        super()._train(single_pass=self.forward, metrics=False)

    # Room for improvement:
    # - Convex hull algorithms to find safe bounds of X to speed up differential evolution candidate selection
    # - Use rolling computed objective value to identify if the uncertainty sufficient to select a given candidate for d.e.

    def forward[T_Constraint: Constraint](self, X: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        conf_level: float = self.state.convergence.confidence_level
        batch_size: int = self.state.sampling.batch_size
        constraint_objects: list[T_Constraint] = self.state.constraints.constraints or []

        def x_de_reward_function(
                X: Tensor,
        ) -> Tensor:
            # bitmask to guarantee that this candidate is exploring in an uncertain point of the objective surrogate
            objective_uncertainty_mask: Tensor = self.objective_uncertainty_mask(X, conf_level=conf_level)
            
            # bitmask to guarantee that the x candidate is safe
            constraint_safe_mask: Tensor = self.constraint_safe_mask(X, conf_level=conf_level, constraint_objects=constraint_objects)
            valid_mask: Tensor = constraint_safe_mask & objective_uncertainty_mask

            if not valid_mask.any():
                print("EMPTY VALID MASK, obj_unc:", objective_uncertainty_mask.sum().item(), "safe:", constraint_safe_mask.sum().item())
                N: int = X.shape[0]
                return torch.full(size=(N, ), fill_value=float("inf"), device=self.device, dtype=self.dtype)

            reward: Tensor = self.surrogate.get_lcb(X, beta=conf_level).squeeze(-1)
            reward[~valid_mask] = float("inf") # Ensures that invalid candidates are unable to be picked
            return reward

        X_proposed: Tensor = self.de_sampler(de_samples_per_loop=8, batch_size=batch_size, 
                                             acq_func=x_de_reward_function, bounds=self.unit_bounds, vectorized=True,) # (N, dim)

        # Compute lipschitz constraints at the beginning of the loop (cached once for next computations)
        lipschitz_object: LipschitzConstraints = LipschitzConstraints(conf_level=conf_level)
        lipschitz_constraints: list[tuple[Tensor, Tensor]] = lipschitz_object.get_constraint_list(X_proposed, self.state.constraints.constraints)

        def z_de_reward_function(
                Z: Tensor
        ) -> Tensor:
            # bitmask to guarantee that this candidate is exploring in an uncertain point of the objective surrogate
            objective_uncertainty_mask: Tensor = self.objective_uncertainty_mask(Z, conf_level=conf_level) #(N, dim_z) D \ S_t

            # bitmask to check if Z lies outside the confidence region of the constraint surrogate (this suggests a safe set (?))
            constraint_uncertainty_mask: Tensor = self.constraint_uncertainty_mask(Z, conf_level=conf_level, constraint_objects=constraint_objects)

            # bitmask to enforce function smoothness
            lipschitz_bitmask: Tensor = self.get_lipschitz_bitmask(X=X_proposed, Z=Z, constraints=lipschitz_constraints) # (N_z, )
            valid_mask: Tensor = constraint_uncertainty_mask & lipschitz_bitmask & objective_uncertainty_mask # (N_z, )

            if not valid_mask.any():
                print("EMPTY VALID MASK, obj_unc:", objective_uncertainty_mask.sum().item(), "safe:", constraint_uncertainty_mask.sum().item())
                N: int = Z.shape[0]
                return torch.full(size=(N, ), fill_value=float("inf"), device=self.device, dtype=self.dtype)

            
            reward: Tensor = self.surrogate.get_lcb(Z, beta=conf_level).squeeze(-1)
            reward[~valid_mask] = float("inf") # Ensures that invalid candidates are unable to be picked
            return reward

        # Will propose N best Z candidates
        Z_proposed: Tensor = self.de_sampler(de_samples_per_loop=8, batch_size=batch_size, 
                                             acq_func=z_de_reward_function, bounds=self.unit_bounds, vectorized=True,) # (N, dim)

        X_proposed_lcb: Tensor = x_de_reward_function(X_proposed)
        Z_proposed_lcb: Tensor = z_de_reward_function(Z_proposed)

        X_min_idx: Tensor = torch.argmin(X_proposed_lcb)
        Z_min_idx: Tensor = torch.argmin(Z_proposed_lcb)

        X_min, X_min_lcb = X_proposed[X_min_idx], X_proposed_lcb[X_min_idx]
        Z_min, Z_min_lcb = Z_proposed[Z_min_idx], Z_proposed_lcb[Z_min_idx]

        if torch.isinf(X_min_lcb) and torch.isinf(Z_min_lcb):
            raise RuntimeError("Optimization failed: No valid or safe candidates found in X or Z proposals.")

        if X_min_lcb < Z_min_lcb:
            return X_min

        dist: Tensor = torch.cdist(x1=X_proposed, x2=Z_min.unsqueeze(0)).squeeze(-1)
        valid_X_mask: Tensor = torch.isfinite(X_proposed_lcb)
        dist[~valid_X_mask] = float("inf")
        
        return X_proposed[torch.argmin(dist, dim=0)]

    def objective_uncertainty_mask(self, X: Tensor, conf_level: float) -> Tensor:
        lcb: Tensor = self.surrogate.get_lcb(X=X, beta=conf_level).squeeze(-1) # (N_z, )
        ucb: Tensor = self.surrogate.get_ucb(X=X, beta=conf_level).squeeze(-1) # (N_z, )
        return (ucb - lcb > 0).squeeze(-1) # D \ S_t

    def constraint_safe_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint]) -> torch.Tensor:
        bitmask: Tensor = torch.ones((X.shape[0], ), dtype=torch.bool, device=X.device)
        for obj in constraint_objects:
            if not isinstance(obj, SurrogateConstraint):
                continue
            lcb: Tensor = obj.get_lcb(X=X, beta=conf_level).squeeze(-1)
            print("lcb: ", lcb)
            bitmask &= (lcb >= 0) # Guarantees safety
        return bitmask 
    
    def constraint_uncertainty_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint]) -> torch.Tensor:
        pessimistic_bitmask: Tensor = torch.ones((X.shape[0], ), dtype=torch.bool, device=X.device)
        optimistic_bitmask: Tensor = torch.clone(pessimistic_bitmask)
        for obj in constraint_objects:
            if not isinstance(obj, SurrogateConstraint):
                continue
            lcb: Tensor = obj.get_lcb(X=X, beta=conf_level).squeeze(-1)
            ucb: Tensor = obj.get_ucb(X=X, beta=conf_level).squeeze(-1)
            pessimistic_bitmask &= (lcb >= 0) # Guarantees uncertainty
            optimistic_bitmask &= (ucb >= 0)
        return optimistic_bitmask & ~pessimistic_bitmask

    def constraint_unsafe_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint]) -> torch.Tensor:
        bitmask: Tensor = torch.zeros((X.shape[0], ), dtype=torch.bool, device=X.device)
        for obj in constraint_objects:
            if not isinstance(obj, SurrogateConstraint):
                continue
            ucb: Tensor = obj.get_ucb(X=X, beta=conf_level).squeeze(-1)
            bitmask |= (ucb < 0) # Guarantees violations
        return bitmask 

    def get_lipschitz_bitmask(self, X: Tensor, Z: Tensor, constraints: list[tuple[Tensor, Tensor]]) -> Tensor:
        eucl_distance: Tensor = torch.cdist(x1=X, x2=Z)
        constraint_mask: Tensor = torch.ones_like(eucl_distance, dtype=torch.bool, device=self.device)
        for (L_i, u_i) in constraints:
            u_i_col: Tensor = u_i.reshape(-1, 1)
            safety: Tensor = (u_i_col - L_i * eucl_distance) >= 0.
            constraint_mask = constraint_mask & safety
        return constraint_mask.any(dim=0)