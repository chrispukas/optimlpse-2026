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
    def __init__(self, X: Tensor, Y: Tensor, 
                 dtype: torch.dtype, device: torch.device, state: BOParams, objective_function: ObjectiveFunction,) -> None:
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

        penalty_magnitude: float = 1.
        penalty_offset: float = 0

        def x_de_reward_function(
                X_de: Tensor,
        ) -> Tensor:
            # smooth penalty mask to guarantee that this candidate is exploring in an uncertain point of the objective surrogate
            objective_uncertainty_mask: Tensor = self.objective_uncertainty_mask(X_de, conf_level=conf_level)
            
            # smooth penalty mask to guarantee that the x candidate is safe
            constraint_safe_mask: Tensor = self.constraint_safe_mask(X_de, conf_level=conf_level, constraint_objects=constraint_objects, penalty_magnitude=penalty_magnitude)

            # smooth reward mask to fill in the safe regions
            reward_mask: Tensor = self.surrogate.get_lcb(X_de, conf_level).squeeze(-1)
            return torch.where(constraint_safe_mask == 0, -reward_mask, constraint_safe_mask + penalty_offset) + objective_uncertainty_mask

        rough_convexhull: Tensor = self.rough_convex_bounds(X)
        X_proposed: Tensor = self.de_sampler(de_samples_per_loop=8, batch_size=batch_size, 
                                             acq_func=x_de_reward_function, bounds=rough_convexhull, vectorized=True,) # (N, dim)

        # Compute lipschitz constraints at the beginning of the loop (cached once for next computations)
        lipschitz_object: LipschitzConstraints = LipschitzConstraints(conf_level=conf_level)
        lipschitz_constraints: list[tuple[Tensor, Tensor]] = lipschitz_object.get_constraint_list(X_proposed, self.state.constraints.constraints)

        def z_de_reward_function(
                Z: Tensor
        ) -> Tensor:
            # smooth penalty mask to guarantee that this candidate is exploring in an uncertain point of the objective surrogate
            objective_uncertainty_mask: Tensor = self.objective_uncertainty_mask(Z, conf_level=conf_level, penalty_magnitude=penalty_magnitude) #(N, dim_z) D \ S_t

            # smooth penalty mask to check if Z lies outside the confidence region of the constraint surrogate (this suggests a safe set (?))
            constraint_uncertainty_mask: Tensor = self.constraint_uncertainty_mask(Z, conf_level=conf_level, constraint_objects=constraint_objects, penalty_magnitude=penalty_magnitude)

            # smooth penalty mask to enforce function smoothness
            lipschitz_bitmask: Tensor = self.get_lipschitz_bitmask(X=X_proposed, Z=Z, constraints=lipschitz_constraints, penalty_magnitude=penalty_magnitude) # (N_z, )

            final: Tensor = constraint_uncertainty_mask + lipschitz_bitmask
            # smooth reward mask to fill in the safe regions
            reward_mask: Tensor = self.surrogate.get_lcb(Z, conf_level).squeeze(-1) # more negative is better
            return torch.where(final == 0, -reward_mask, final + penalty_offset) + objective_uncertainty_mask

        # Will propose N best Z candidates
        Z_proposed: Tensor = self.de_sampler(de_samples_per_loop=8, batch_size=batch_size, 
                                             acq_func=z_de_reward_function, bounds=self.unit_bounds, vectorized=True,) # (N, dim)


        if X.shape[1] == 2:
            plot_reward_planes_side_by_side(
                reward_function_1=x_de_reward_function,
                reward_function_2=z_de_reward_function,
                X_evaluated=X,
                vmax=1000.0, # Keeps the colors scaled to the objective variations
                device=self.device
            )


        X_min_idx = torch.argmin(x_de_reward_function(X_proposed))
        Z_min_idx = torch.argmin(z_de_reward_function(Z_proposed))

        X_min = X_proposed[X_min_idx]
        Z_min = Z_proposed[Z_min_idx]

        true_X_lcb = self.surrogate.get_lcb(X_min.unsqueeze(0), conf_level).squeeze(-1)
        true_Z_lcb = self.surrogate.get_lcb(Z_min.unsqueeze(0), conf_level).squeeze(-1)

        if true_X_lcb > true_Z_lcb:
            print("Z Selected!")
            return Z_min

        dist = torch.cdist(x1=X_proposed, x2=Z_min.unsqueeze(0)).squeeze(-1)        
        safe_X_mask = self.constraint_safe_mask(X_proposed, conf_level, constraint_objects, penalty_magnitude=1.0) == 0
        dist[~safe_X_mask] = float("inf")
        
        if torch.isinf(dist).all():
            return X_min

        print("X Selected!")
        return X_proposed[torch.argmin(dist, dim=0)]

    @staticmethod
    def _default_penalty_mask(X: Tensor) -> Tensor:
        N: int = X.shape[0]
        return torch.full((N, ), 0, dtype=X.dtype, device=X.device)

    @staticmethod
    def _validate_result(X: Tensor) -> Tensor:
        if torch.isneginf(X).all():
            raise ValueError("No valid constraint evaluations were found!")
        return X

    def objective_uncertainty_mask(self, X: Tensor, conf_level: float, penalty_magnitude: float = 1.) -> Tensor:
        penalty_mask: Tensor = self._default_penalty_mask(X=X)

        lcb: Tensor = self.surrogate.get_lcb(X=X, beta=conf_level).squeeze(-1) # (N_z, )
        ucb: Tensor = self.surrogate.get_ucb(X=X, beta=conf_level).squeeze(-1) # (N_z, )
        violation: Tensor = torch.clamp(-(ucb-lcb), 0.0)

        penalty_mask: Tensor = torch.maximum(penalty_mask, violation)
        
        return self._validate_result((penalty_mask) * penalty_magnitude) # D \ S_t
    
    def constraint_safe_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint], penalty_magnitude: float = 1.) -> torch.Tensor:
        penalty_mask: Tensor = self._default_penalty_mask(X=X)
        for obj in constraint_objects:
            if not isinstance(obj, SurrogateConstraint):
                continue
            lcb: Tensor = obj.get_lcb(X=X, beta=conf_level).squeeze(-1)
            violation: Tensor = torch.clamp(lcb, 0.) # Guarantees safety
            penalty_mask: Tensor = torch.maximum(penalty_mask, violation)

        return self._validate_result((penalty_mask) * penalty_magnitude)

    def constraint_uncertainty_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint], penalty_magnitude: float = 1.) -> torch.Tensor:
        pessimistic_bitmask: Tensor = self._default_penalty_mask(X=X)
        optimistic_bitmask: Tensor = torch.clone(pessimistic_bitmask)

        for obj in constraint_objects:
            if not isinstance(obj, SurrogateConstraint):
                continue
            lcb: Tensor = obj.get_lcb(X=X, beta=conf_level).squeeze(-1)
            ucb: Tensor = obj.get_ucb(X=X, beta=conf_level).squeeze(-1)

            pessimistic_bitmask: Tensor = torch.maximum(pessimistic_bitmask, torch.clamp(lcb, 0.)) # given that the lcb is clamped by a maximum
            optimistic_bitmask: Tensor = torch.maximum(optimistic_bitmask, torch.clamp(-ucb, 0.)) # given that ucb is clamped by a minimum

        return self._validate_result((pessimistic_bitmask + optimistic_bitmask) * penalty_magnitude)

    def get_lipschitz_bitmask(self, X: Tensor, Z: Tensor, constraints: list[tuple[Tensor, Tensor]], penalty_magnitude: float = 1.0) -> Tensor:
        penalty_mask: Tensor = self._default_penalty_mask(Z) # (N_z, )
        eucl_distance: Tensor = torch.cdist(x1=X, x2=Z) # (N_x, N_z)

        for (L_i, u_i) in constraints:
            u_i_col: Tensor = u_i.reshape(-1, 1)
            lipschitz: Tensor = u_i_col - L_i * eucl_distance
            violation: Tensor = torch.clamp(-lipschitz.max(dim=0).values, 0.0)
            penalty_mask: Tensor = torch.maximum(penalty_mask, violation)
        return penalty_mask * penalty_magnitude

    def rough_convex_bounds(self, X: Tensor, padding: float = 1.) -> Tensor:
        min_bounds: Tensor = torch.clamp(torch.amin(X, dim=0) - padding, 0.0, 1.0)
        max_bounds: Tensor = torch.clamp(torch.amax(X, dim=0) + padding, 0.0, 1.0)

        return torch.permute(torch.vstack((min_bounds, max_bounds)), (1, 0))






import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Callable
import numpy as np

def plot_reward_planes_side_by_side(
    reward_function_1: Callable[[Tensor], Tensor], 
    reward_function_2: Callable[[Tensor], Tensor], 
    X_evaluated: Tensor = None,
    resolution: int = 100, 
    vmax: float = 10.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
    title_1: str = "X Proposed Landscape (Exploitation)",
    title_2: str = "Z Proposed Landscape (Exploration)"
) -> None:
    x1 = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    x2 = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    
    X_grid = torch.stack((X1.flatten(), X2.flatten()), dim=-1)
    
    eval_batch_size = 1000
    z1_list, z2_list = [], []
    
    with torch.no_grad():
        for i in range(0, X_grid.shape[0], eval_batch_size):
            X_batch = X_grid[i:i + eval_batch_size].to(dtype=dtype) # Force dtype match
            z1_list.append(reward_function_1(X_batch))
            z2_list.append(reward_function_2(X_batch))
            
    Z_grid_1 = torch.cat(z1_list)
    Z_grid_2 = torch.cat(z2_list)
    
    Z1 = np.clip(Z_grid_1.view(resolution, resolution).cpu().numpy(), a_min=-float('inf'), a_max=vmax)
    Z2 = np.clip(Z_grid_2.view(resolution, resolution).cpu().numpy(), a_min=-float('inf'), a_max=vmax)
    
    X1_np = X1.cpu().numpy()
    X2_np = X2.cpu().numpy()
    
    plt.ion() 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    contour_1 = axes[0].contourf(X1_np, X2_np, Z1, levels=50, cmap="viridis")
    fig.colorbar(contour_1, ax=axes[0], label="Reward Value (Lower is Better)")
    axes[0].set_title(title_1)
    axes[0].set_xlabel("Feature 1 (Normalized)")
    axes[0].set_ylabel("Feature 2 (Normalized)")
    
    contour_2 = axes[1].contourf(X1_np, X2_np, Z2, levels=50, cmap="viridis")
    fig.colorbar(contour_2, ax=axes[1], label="Reward Value (Lower is Better)")
    axes[1].set_title(title_2)
    axes[1].set_xlabel("Feature 1 (Normalized)")
    
    if X_evaluated is not None:
        X_np = X_evaluated.cpu().numpy()
        for ax in axes:
            ax.scatter(X_np[:, 0], X_np[:, 1], c='red', edgecolors='white', label='Evaluated (X)', s=40)
            ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1.0)