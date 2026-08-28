from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

import enum

from safebo_simpl.constraints import SurrogateConstraint

@dataclass
class LipschitzConstraints():
    conf_level: float = 2.

    def get_constraint_list(self, X: Tensor, surrogate_constraints: list[SurrogateConstraint]) -> list[tuple[Tensor, Tensor]]:
        final: list[tuple[Tensor, Tensor]] = []
        for constraint in surrogate_constraints:
            if not isinstance(constraint, SurrogateConstraint):
                continue
            final.append(self(X, constraint))
        return final

    def __call__(self, X: Tensor, surrogate_constraint: SurrogateConstraint) -> tuple[Tensor, Tensor]:
        if not isinstance(surrogate_constraint, SurrogateConstraint):
            return 
        return (
            self.get_ith_lipscitz_constraint(X=X, surrogate_constraint=surrogate_constraint), 
            self.get_ith_ucb(X=X, conf_level=self.conf_level, surrogate_constraint=surrogate_constraint),
            )

    def get_ith_lipscitz_constraint(self, X: Tensor, surrogate_constraint: SurrogateConstraint) -> Tensor:
        with torch.enable_grad():
            X_grad: Tensor = X.clone().detach().requires_grad_(True)

            mean: Tensor = surrogate_constraint.get_mean(X=X_grad).flatten()
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
    
    def get_ith_ucb(self, X: Tensor, conf_level: float, surrogate_constraint: SurrogateConstraint) -> Tensor:
        return surrogate_constraint.get_ucb(X=X, beta=conf_level)
