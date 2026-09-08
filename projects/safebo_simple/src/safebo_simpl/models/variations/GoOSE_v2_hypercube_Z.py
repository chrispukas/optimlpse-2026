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


ran: bool = False
        
class GoOSEV2_HypercubeZ(su_safe.SafeBOAlgorithm):
    def __init__(self, X: Tensor, Y: Tensor, 
                 dtype: torch.dtype, device: torch.device, state: BOParams, objective_function: ObjectiveFunction,) -> None:
        super().__init__(X=X, Y=Y, dtype=dtype, device=device, state=state, objective_function=objective_function)

    def train(self) -> None:
        super()._train(single_pass=self.forward, metrics=False)

    # Room for improvement:
    # - Use rolling computed objective value to identify if the uncertainty sufficient to select a given candidate for d.e.

    def forward[T_Constraint: Constraint](self, X: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        conf_level: float = self.state.convergence.confidence_level
        batch_size: int = self.state.sampling.batch_size
        constraint_objects: list[T_Constraint] = self.state.constraints.constraints or []

        beta: float = 1. # Penalty magnitude (scaled by the current properties of the surrogate)
        alpha_0: float = 1. # Penalty margin (step to minimize marginal candidates)
        alpha_max: float = 0. #self.surrogate.get_lcb(X, conf_level).max().item()

        def x_de_reward_function(
                X_de: Tensor,
        ) -> Tensor:
            # smooth penalty mask to guarantee that this candidate is exploring in an uncertain point of the objective surrogate
            objective_uncertainty_mask: Tensor = self.objective_uncertainty_mask(X_de, conf_level=conf_level)
            
            # smooth penalty mask to guarantee that the x candidate is safe
            (safety_penalty_mask, safety_bitmask) = self.constraint_safe_mask(X_de, conf_level=conf_level, 
                                                                    constraint_objects=constraint_objects, penalty_magnitude=beta)

            penalty: Tensor = safety_penalty_mask
            # smooth reward mask to fill in the safe regions
            reward_mask: Tensor = self.surrogate.get_lcb(X_de, conf_level).squeeze(-1) #argmin the lcb
            return torch.where(safety_bitmask, reward_mask, penalty + alpha_max + alpha_0) - objective_uncertainty_mask
        
        X_proposed: Tensor = self.de_sampler(de_samples_per_loop=8, batch_size=batch_size, 
                                             acq_func=x_de_reward_function, bounds=self.unit_bounds, vectorized=True,) # (N, dim)

        # Compute lipschitz constraints at the beginning of the loop (cached once for next computations)
        lipschitz_object: LipschitzConstraints = LipschitzConstraints(conf_level=conf_level)
        lipschitz_constraints: list[tuple[Tensor, Tensor]] = lipschitz_object.get_constraint_list(X_proposed, self.state.constraints.constraints)

        def z_de_reward_function(
                Z: Tensor
        ) -> Tensor:
            # smooth penalty mask to guarantee that this candidate is exploring in an uncertain point of the objective surrogate
            objective_uncertainty_mask: Tensor = self.objective_uncertainty_mask(Z, conf_level=conf_level, penalty_magnitude=beta) #(N, dim_z) D \ S_t

            # smooth penalty mask to check if Z lies outside the confidence region of the constraint surrogate (this suggests a safe set (?))
            (constraint_uncertainty_mask, uncertainty_bitmask) = self.constraint_uncertainty_mask(Z, conf_level=conf_level, 
                                                                    constraint_objects=constraint_objects, penalty_magnitude=beta)

            # smooth penalty mask to enforce function smoothness
            lipschitz_penalty_mask: Tensor = self.get_lipschitz_bitmask(X=X_proposed, Z=Z, 
                                                                    constraints=lipschitz_constraints, penalty_magnitude=beta) # (N_z, )

            penalty: Tensor = constraint_uncertainty_mask + lipschitz_penalty_mask
            # smooth reward mask to fill in the safe regions
            reward_mask: Tensor = self.surrogate.get_lcb(Z, conf_level).squeeze(-1) # argmin the lcb
            return torch.where(uncertainty_bitmask, reward_mask, penalty + alpha_max + alpha_0) - objective_uncertainty_mask

        # Will propose N best Z candidates
        hypercube_bounds: Tensor = self.hypercube_bounds(X) # Bound the exploration candidates to a padded hypercube
        Z_proposed: Tensor = self.de_sampler(de_samples_per_loop=8, batch_size=batch_size, 
                                             acq_func=z_de_reward_function, bounds=hypercube_bounds, vectorized=True,) # (N, dim)

        global ran
        if False and not ran and X.shape[1] == 2:
            #ran = True
            plot_reward_planes_side_by_side(
                reward_function_1=x_de_reward_function,
                reward_function_2=z_de_reward_function,
                X_evaluated=X,
                vmax=1000.0, # Keeps the colors scaled to the objective variations
                device=self.device
            )
        

        if X_proposed.shape[0] == 0:
            X_proposed: Tensor = X # Fallback to the known safe candidate set

        X_min: Tensor = self._argmin_rewards(X_proposed, x_de_reward_function)
        Z_min: Tensor = self._argmin_rewards(Z_proposed, z_de_reward_function)

        true_X_lcb = self.surrogate.get_lcb(X_min.unsqueeze(0), conf_level).squeeze(-1)
        true_Z_lcb = self.surrogate.get_lcb(Z_min.unsqueeze(0), conf_level).squeeze(-1)

        if true_X_lcb < true_Z_lcb:
            return X_min

        X_dist: Tensor = self.dist_selection(X=X_proposed, Z=Z_min)
        return X_dist

    @staticmethod
    def _argmin_rewards(X: Tensor, reward_method: Callable[[Tensor], Tensor]) -> Tensor:
        return X[torch.argmin(reward_method(X))]
    @staticmethod
    def _default_penalty_mask(X: Tensor) -> Tensor:
        N: int = X.shape[0]
        return torch.full((N, ), 0, dtype=X.dtype, device=X.device)
    @staticmethod
    def _default_bitmask(X: Tensor, true: bool = True) -> Tensor:
        N: int = X.shape[0]
        return torch.full((N, ), int(true), dtype=torch.bool, device=X.device)
    @staticmethod
    def _validate_result(X: Tensor) -> Tensor:
        if torch.isneginf(X).all():
            raise ValueError("No valid constraint evaluations were found!")
        return X

    def objective_uncertainty_mask(self, X: Tensor, conf_level: float, penalty_magnitude: float = 1.) -> Tensor:
        penalty_mask: Tensor = self._default_penalty_mask(X=X)

        lcb: Tensor = self.surrogate.get_lcb(X=X, beta=conf_level).squeeze(-1) # (N_z, )
        ucb: Tensor = self.surrogate.get_ucb(X=X, beta=conf_level).squeeze(-1) # (N_z, )
        violation: Tensor = torch.clamp(0. - (ucb-lcb), 0.)

        penalty_mask: Tensor = torch.maximum(penalty_mask, violation)
        
        return self._validate_result((penalty_mask) * penalty_magnitude) # D \ S_t
    
    def constraint_safe_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint], penalty_magnitude: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Returns: penalty_mask, safety_bitmask
        """
        penalty_mask: Tensor = self._default_penalty_mask(X=X)
        bitmask: Tensor = self._default_bitmask(X=X)

        for constraint_surrogate in constraint_objects:
            if not isinstance(constraint_surrogate, SurrogateConstraint):
                continue
            lcb: Tensor = constraint_surrogate.get_lcb(X=X, beta=conf_level).squeeze(-1)
            this_bitmask: Tensor = (lcb >= 0.) # bitmask with AND operator
            bitmask &= this_bitmask

            violation: Tensor = torch.where(~this_bitmask, -lcb, torch.zeros_like(lcb)) # Guarantees safety
            penalty_mask: Tensor = torch.maximum(penalty_mask, violation) # stack on violations to ensure that any non-0 region becomes invalid

        return (self._validate_result((penalty_mask) * penalty_magnitude), bitmask)

    def constraint_uncertainty_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint], penalty_magnitude: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
        pessimistic_penalty_mask: Tensor = self._default_penalty_mask(X=X)
        optimistic_penalty_mask: Tensor = torch.clone(pessimistic_penalty_mask)

        single_uncertain: Tensor = self._default_bitmask(X, True)

        for constraint_surrogate in constraint_objects:
            if not isinstance(constraint_surrogate, SurrogateConstraint):
                continue
            lcb: Tensor = constraint_surrogate.get_lcb(X=X, beta=conf_level).squeeze(-1)
            ucb: Tensor = constraint_surrogate.get_ucb(X=X, beta=conf_level).squeeze(-1)

            this_uncertain: Tensor = (lcb < 0.) & (ucb > 0.)
            single_uncertain &= this_uncertain

            pessimistic_violation: Tensor = torch.where(~this_uncertain, lcb, torch.zeros_like(lcb)) # Guarantees safety
            optimistic_violation: Tensor = torch.where(~this_uncertain, -ucb, torch.zeros_like(ucb)) # Guarantees safety

            pessimistic_penalty_mask: Tensor = torch.maximum(pessimistic_penalty_mask, pessimistic_violation)
            optimistic_penalty_mask: Tensor = torch.maximum(optimistic_penalty_mask, optimistic_violation)

        return (self._validate_result((pessimistic_penalty_mask + optimistic_penalty_mask) * penalty_magnitude), single_uncertain)

    def get_lipschitz_bitmask(self, X: Tensor, Z: Tensor, constraints: list[tuple[Tensor, Tensor]], penalty_magnitude: float = 1.) -> Tensor:
        penalty_mask: Tensor = self._default_penalty_mask(Z) # (N_z, )
        eucl_distance: Tensor = torch.cdist(x1=X, x2=Z) # (N_x, N_z)

        for (L_i, u_i) in constraints:
            u_i_col: Tensor = u_i.reshape(-1, 1)
            lipschitz: Tensor = u_i_col - L_i * eucl_distance
            violation: Tensor = torch.clamp(-lipschitz.max(dim=0).values, 0.0)
            penalty_mask: Tensor = torch.maximum(penalty_mask, violation)
        return penalty_mask * penalty_magnitude

    def hypercube_bounds(self, X: Tensor, padding: float = 1.) -> Tensor:
        min_bounds: Tensor = torch.clamp(torch.amin(X, dim=0) - padding, 0.0, 1.0)
        max_bounds: Tensor = torch.clamp(torch.amax(X, dim=0) + padding, 0.0, 1.0)

        return torch.permute(torch.vstack((min_bounds, max_bounds)), (1, 0))

    def enforce_safety[T_Constraint: Constraint](self, X: Tensor, X_safe: Tensor, conf_level: float, constraint_objects: list[T_Constraint], penalty_magnitude: float) -> Tensor:
        (_, bitmask) = self.constraint_safe_mask(
                    X, 
                    conf_level=conf_level, 
                    constraint_objects=constraint_objects, 
                    penalty_magnitude=penalty_magnitude
                )
        safety: Tensor = X[bitmask]
        return safety

    def get_lcb_matrix[T_Constraint: Constraint](self, X: Tensor, beta: float, constraint_objects: list[T_Constraint]) -> Tensor:
        final: Tensor = torch.zeros_like(X, device=X.device, dtype=X.dtype)

        for constraint_surrogate in constraint_objects:
            if not isinstance(constraint_surrogate, SurrogateConstraint):
                continue
            lcb: Tensor = constraint_surrogate.get_lcb(X=X, beta=beta)
            final: Tensor = torch.hstack((final, lcb))
        return final


    def safety_fallback(self, X_prop: Tensor, X_safe: Tensor, beta: float) -> Tensor:
        (X_safe_lcb, X_safe_idx) = torch.min(self.surrogate.get_lcb(X_safe, beta=beta))
        (X_proposed_lcb, X_prop_idx) = torch.min(self.surrogate.get_lcb(X_prop, beta=beta))

        gamma: float = (X_safe_lcb / (X_safe_lcb - X_proposed_lcb)).item()

        X_safe_single: Tensor = X_safe[X_safe_idx]
        X_prop_single: Tensor = X_prop[X_prop_idx]

        return X_safe_single + gamma * (X_prop_single - X_safe_single)

    def dist_selection(self, X: Tensor, Z: Tensor) -> Tensor:
        dist: Tensor = torch.cdist(x1=X, x2=Z.unsqueeze(0)).squeeze()   
        return X[torch.argmin(dist, dim=0)]

    def recalculate_magnitude(self, X: Tensor) -> float:
        (_, std) = self.surrogate.get_properties(X=X)
        return torch.sqrt(std).min().item()




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