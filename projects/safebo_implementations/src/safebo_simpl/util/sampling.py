import torch
from torch import Tensor

from safebo_simpl.util import bayesian_parameters as su_prms

def rand_initial(
        state: su_prms.BOParams
        ) -> Tensor:
    X: Tensor = torch.rand(
        size=[state.sampling.initial_candidates, state.data.dimensions],
        requires_grad=True
    )
    return X