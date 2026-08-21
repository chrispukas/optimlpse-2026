from typing import Tuple, List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from safebo_simpl.util import generics as su_safe
from safebo_simpl.util import params as su_prms

from safebo_simpl.objective_functions import ObjectiveFunction

from safebo_simpl.util.typing import AllowUndefined

import torch
from torch import Tensor

import botorch
from botorch import models as b_models
from botorch.posteriors import gpytorch as bp_gpytorch

import numpy as np
import numpy.typing as npt

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
        self.dynamics.delta_t = self.convergence.delta_min + 0.5 * (self.convergence.delta_max - self.convergence.delta_min)
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

        self.center: AllowUndefined[Tensor] = None

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
        ) -> AllowUndefined[Tensor]:

        if self.center is None:
            self.center: AllowUndefined[Tensor] = X[-1, :]

        def acqf_wrapper(D: float) -> float:
            penalty: float = float('inf')

            if self.center is None or \
                torch.linalg.norm(D, dim=1).item() > self.state.dynamics.delta_t: # Restricts to hypersphere
                return penalty

            XD: Tensor = self.center + D
            if self.state.constraints.is_available():
                constraint_mask: Tensor = self.state.constraints(X=XD)
                if not constraint_mask.item():
                    return penalty
            
            lcb: Tensor = self.surrogate.get_lcb(XD, beta=self.state.convergence.confidence_level)
            return lcb.item()

        
        dt_bds: npt.NDArray[np.float32] = np.array(
            [[-self.state.dynamics.delta_t, self.state.dynamics.delta_t] for _ in range(self.state.data.dimensions)] 
            , dtype=np.float32)
        d_candidate: Tensor = self.de_sampler(
            n=self.state.sampling.batch_size,
            acq_func=acqf_wrapper,
            bounds=dt_bds,
            maxiter=50,
        )
        accuracy_ratio: float = self.get_accuracy_ratio(
            X_prev=self.center.unsqueeze(0),
            D_next=d_candidate.unsqueeze(0),
            objective_function=objective_function,
        )
        success, delta_t_new = self.get_next_candidates(
            accuracy_ratio=accuracy_ratio,
            delta_t=self.state.dynamics.delta_t,
        )

        self.state.dynamics.delta_t = delta_t_new
        next_candidates: Tensor = (self.center + d_candidate)

        if success:
            self.center: AllowUndefined[Tensor] = next_candidates

        return next_candidates.unsqueeze(0)


    def get_center_next(
            self,
            D: Tensor,
            x_prev: Tensor,
            delta_t: float,
        ) -> Tensor:
        XD: Tensor = x_prev + D
        lcb_XD: Tensor = self.surrogate.get_lcb(X=XD, beta=self.state.convergence.confidence_level).squeeze()
        eucl_mask: Tensor = (torch.linalg.norm(D, dim=1) <= delta_t)

        # Ackley constraint bounds
        if self.state.constraints.is_available():
            constraint_mask: Tensor = self.state.constraints.get_constraint(X=XD)
            valid_mask: Tensor = constraint_mask & eucl_mask
        else:
            valid_mask: Tensor = eucl_mask
        if not valid_mask.any():
            return torch.zeros_like(D[0, :])

        masked: Tensor = torch.where(valid_mask, lcb_XD, float("inf"))
        d_min: Tensor = torch.argmin(input=masked)

        return D[d_min]

    def get_accuracy_ratio(
            self,
            X_prev: Tensor,
            D_next: Tensor,

            objective_function: ObjectiveFunction,
        ) -> float:

        new: Tensor = X_prev + D_next

        posterior_XD: bp_gpytorch.GPyTorchPosterior = self.surrogate.posterior(X=new)
        posterior_X: bp_gpytorch.GPyTorchPosterior  = self.surrogate.posterior(X=X_prev)
        
        a_num: Tensor = (objective_function.forward(new) - objective_function.forward(X_prev)).squeeze()
        a_denom: Tensor = ((posterior_XD.mean - posterior_X.mean)).squeeze()
        
        ratio: Tensor = a_num/a_denom
        return ratio.item()

    def get_next_candidates(
            self,
            accuracy_ratio: float,    
            delta_t: float,
        ) -> Tuple[bool, float]:

        state_conv: BOParams_Convergence_GPsTR = self.state.convergence
        
        if accuracy_ratio < state_conv.eta_1:
            return False, max(delta_t * state_conv.gamma_red, state_conv.delta_min)
        if state_conv.eta_1 < accuracy_ratio < state_conv.eta_2:
            return False, delta_t
        else:
            return True, min(delta_t * state_conv.gamma_inc, state_conv.delta_max)