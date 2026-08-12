from typing import Any, Unpack, Callable

import torch
from torch import Tensor

from safebo_simpl.util import generics, params
from safebo_simpl.util import typing as su_typing
from botorch import posteriors

from safebo_simpl.constraints._parent import Constraint

class NonSurrogateConstraint(Constraint):
    def __init__(
            self,
            dtype: torch.dtype,
            device: torch.device,
            bounds: Tensor,

            **kwargs: Any,
            ) -> None:
        
        # Pre-define types to expose them to the IDE        
        self.dtype: torch.dtype
        self.device: torch.device
        self.bounds: Tensor = bounds

        super().__init__(
            dtype=dtype,
            device=device,
            **kwargs
        )
