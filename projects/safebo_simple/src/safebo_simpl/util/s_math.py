import enum
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from safebo_simpl.constraints import SurrogateConstraint, Constraint

type LipschitzConstraintPair = tuple[Tensor, Tensor]

@dataclass
class LipschitzConstraints[T_Constraint: Constraint]():
    conf_level: float = 2.
    surrogate_constraints: Optional[list[T_Constraint]] = None

    def __call__(self, X: Tensor, surrogate_constraints: Optional[list[T_Constraint]] = surrogate_constraints) -> list[LipschitzConstraintPair]:
        """
            Inputs:
                X: input tensor of shape (n, dim)
                surrogate_constraints: list of constraint objects (either specified at function call, or at object creation)
            Returns:
                `list[tuple[Tensor, Tensor]]: list of pairs: (L_i, ucb_i)`
        """
        return self.get_constraint_list(X, surrogate_constraints=surrogate_constraints)

    def get_violation_mask(self, X1: Tensor, X2: Tensor, surrogate_constraints: Optional[list[T_Constraint]] = surrogate_constraints) -> Tensor:
        violation: Tensor = torch.zeros_like(X1, dtype=torch.bool)
        d: Tensor = torch.cdist(x1=X1, x2=X2)
        pairs: list[LipschitzConstraintPair] = self(X1)

        for (L_i, u_i) in pairs:
            violation |= (u_i.reshape(-1, 1) - L_i * d) < 0

        return violation.any(dim=1)

    def get_constraint_list(self, X: Tensor, surrogate_constraints: Optional[list[T_Constraint]] = surrogate_constraints) -> list[LipschitzConstraintPair]:
        """
            Inputs:
                X: input tensor of shape (n, dim)
                surrogate_constraints: list of constraint objects (either specified at function call, or at object creation)
            Returns:
                `list[tuple[Tensor, Tensor]]: list of pairs: (L_i, ucb_i)`
        """
        if not surrogate_constraints:
            raise ValueError("Constraint objects have not been specified correctly!")

        final: list[tuple[Tensor, Tensor]] = []
        for constraint in surrogate_constraints:
            if not isinstance(constraint, SurrogateConstraint):
                continue
            final.append(self.get_constraint_single(X, constraint))
        return final

    def get_constraint_single(self, X: Tensor, surrogate_constraint: SurrogateConstraint) -> LipschitzConstraintPair:
        """
            Inputs:
                X: input tensor of shape (n, dim)
                surrogate_constraint: single constraint object
            Returns: 
                `tuple[Tensor, Tensor]: a single pair: (L_i, ucb_i)`
        """
        if not isinstance(surrogate_constraint, SurrogateConstraint):
            return 
        return (
            self.get_ith_lipscitz_constraint(X=X, surrogate_constraint=surrogate_constraint), 
            self.get_ith_ucb(X=X, conf_level=self.conf_level, surrogate_constraint=surrogate_constraint),
            )
        
    def get_ith_lipscitz_constraint(self, X: Tensor, surrogate_constraint: SurrogateConstraint) -> Tensor:
        """
            Gradient-based approach, using the mean of the surrogate to calculate the i-th Lipschitz constant.

            Inputs:
                X: input tensor of shape (n, dim)
                surrogate_constraint: single constraint object
            Returns:
                Tensor: L_i
        """
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
        """
            Inputs:
                X: input tensor of shape (n, dim)
                conf_level: confidence interval
                surrogate_constraint: single constraint object
            Returns:
                Tensor: upper confidence boundary of shape (n, )
        """
        return surrogate_constraint.get_ucb(X=X, beta=conf_level)

    def get_ith_lcb(self, X: Tensor, conf_level: float, surrogate_constraint: SurrogateConstraint) -> Tensor:
        """
            Inputs:
                X: input tensor of shape (n, dim)
                conf_level: confidence interval
                surrogate_constraint: single constraint object
            Returns:
                Tensor: lower confidence boundary of shape (n, )
        """
        return surrogate_constraint.get_lcb(X=X, beta=conf_level)