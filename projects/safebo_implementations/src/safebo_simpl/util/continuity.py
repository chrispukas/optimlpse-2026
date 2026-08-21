
from dataclasses import dataclass

from torch import Tensor

class LipschitzConstraints():
    def __init__(
            self
            ) -> None:
        ...


@dataclass
class NormTensor():
    def __init__(
            self,
            X: Tensor,
            ) -> None:
        self.mean: Tensor = X.mean(dim=0).unsqueeze(0)
        self.std: Tensor = X.std(dim=0).unsqueeze(0)

    def normalize(
            self,
            X: Tensor,
    ) -> Tensor:
        return (X - self.mean) / self.std
    
    def denormalize(
            self,
            X: Tensor,
    ) -> Tensor:
        return (X * self.std) + self.mean
