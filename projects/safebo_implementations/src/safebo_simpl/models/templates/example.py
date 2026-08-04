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
class BOParams_Example(
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
                self,
            ) -> None:

        # Setting initial config for dynamic variables across a run
        self.dynamics.max_iterations = 30    
        self.data.negate = False

class Example(su_safe.SafeBOAlgorithm):
    def __init__(
            self, 
            X: Tensor, 
            Y: Tensor,

            dtype: torch.dtype,
            device: torch.device,

            state: BOParams_Example,
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
        raise NotImplementedError(f"Forward pass for {self.__class__.__name__} not implemented!")
        ... # INSERT LOGIC FOR SELECTING CANDIDATES HERE, AND RETURN NEXT CANDIDATES AS TYPE TENSOR


    ... # INSERT FUNCTIONS FOR LOGIC HERE