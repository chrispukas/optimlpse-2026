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
class BOParams_Convergence_GPsTR(
    su_prms.BOParams_Convergence
    ):
    eta_1: float = 0.2
    eta_2: float = 0.8

    gamma_inc: float = 1.2
    gamma_red: float = 0.8

    delta_max: float = 1.5
    delta_min: float = 0.1

@dataclass
class BOParams_Dynamics_GPsTR(
    su_prms.BOParams_Dynamics
    ):
    delta_t: float = 0

@dataclass
class BOParams_GPsTR(
    su_prms.BOParams[
        su_prms.BOParams_Sampling,
        BOParams_Convergence_GPsTR,
        su_prms.BOParams_Constraints,
        su_prms.BOParams_Data,
        BOParams_Dynamics_GPsTR
        ]
    ):
    def __init__(
            self,
            ) -> None:
        super().__init__(
            convergence=BOParams_Convergence_GPsTR,
            dynamics=BOParams_Dynamics_GPsTR,
        )

        # Setting initial config for dynamic variables across a run
        self.dynamics.delta_t = (self.convergence.delta_max + self.convergence.delta_min) / 2.
        self.dynamics.max_iterations = 30    

        self.data.negate = False



class GPsTR(su_safe.SafeBOAlgorithm):
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

        x_prev: Tensor = X[-1, :]

        d_candidates: Tensor = self.sample_discrete(
            n=X.shape[0],
            dim=X.shape[1],
            scramble=True,
            center=True
        )
        d_new: Tensor = self.get_center_next(
            X=X, 
            D=d_candidates,
            delta_t=self.state.dynamics.delta_t,
        )
        accuracy_ratio: float = self.get_accuracy_ratio(
            X_prev=x_prev.unsqueeze(0),
            D_next=d_new.unsqueeze(0),
            objective_function=objective_function,
        )
        delta_t_new: float = self.get_next_candidates(
            accuracy_ratio=accuracy_ratio,
            delta_t=self.state.dynamics.delta_t,
        )

        self.state.dynamics.delta_t = delta_t_new

        next_candidates: Tensor = (x_prev + d_new)
        return next_candidates


    def get_center_next(
            self,
            X: Tensor,
            D: Tensor,
            delta_t: float,
        ) -> Tensor:
        lcb_XD: Tensor = self.surrogate.get_lcb(X=X+D, beta=self.state.convergence.confidence_level)
        eucl_mask: Tensor = (torch.linalg.norm(D, dim=1) <= delta_t)
        abs_XD_proposed: Tensor = torch.abs(X+D)

        # Ackley constraint bounds
        constraint_mask: Tensor = (abs_XD_proposed <= 5).all(dim=1)
        valid_mask: Tensor = constraint_mask & eucl_mask
        if not valid_mask.any():
            return torch.zeros_like(D[0, :])

        masked: Tensor = torch.where(valid_mask.unsqueeze(1), lcb_XD, float("inf"))
        d_min: Tensor = torch.argmin(input=masked, dim=0)

        if d_min.shape[0] > 1:
            d_min: Tensor = d_min[0, :]

        return D[d_min.squeeze()]

    def get_accuracy_ratio(
            self,
            X_prev: Tensor,
            D_next: Tensor,

            objective_function: su_objfuncs.ObjectiveFunction,
        ) -> float:

        new: Tensor = X_prev + D_next

        posterior_XD: bp_gpytorch.GPyTorchPosterior = self.posterior(X=new)
        posterior_X: bp_gpytorch.GPyTorchPosterior  = self.posterior(X=X_prev)
        
        a_num: Tensor = (objective_function.forward(new) - objective_function.forward(X_prev))
        a_denom: Tensor = posterior_XD.mean - posterior_X.mean
        
        ratio: Tensor = a_num/a_denom
        if ratio.shape[1] != 1:
            return 0
        return ratio.item()

    def get_next_candidates(
            self,
            accuracy_ratio: float,    
            delta_t: float,
        ) -> float:

        state_conv: BOParams_Convergence_GPsTR = self.state.convergence
        
        if accuracy_ratio < state_conv.eta_1:
            return max(delta_t * state_conv.gamma_red, state_conv.delta_min)
        if state_conv.eta_1 < accuracy_ratio < state_conv.eta_2:
            return delta_t
        else:
            return min(delta_t * state_conv.gamma_inc, state_conv.delta_max)
        

