
from dataclasses import dataclass
from torch import Tensor
import enum

class StandardizationType(enum.Enum):
    Default = 0
    ZScore = 1
    MinMax = 2


std_dict: dict[str, StandardizationType] = \
    {
        "zscore": StandardizationType.ZScore,
        "minmax": StandardizationType.MinMax,
    }

@dataclass
class NormTensor():
    def __init__(
            self,
            X: Tensor,
            method: StandardizationType | str
        ) -> None:
        if not isinstance(method, (StandardizationType, str)):
            raise ValueError(f"Method type {type(method)} not recognized!")
        self.method: StandardizationType = self._conv_to_enum(method) if isinstance(method, str) else method 

        match self.method:
            case StandardizationType.Default:
                ...
            case StandardizationType.ZScore:
                self.mean: Tensor = X.mean(dim=0).unsqueeze(0)
                self.std: Tensor = X.std(dim=0).unsqueeze(0)
            case StandardizationType.MinMax:
                self.min: Tensor = X.min(dim=0)[0].unsqueeze(0)
                self.max: Tensor = X.max(dim=0)[0].unsqueeze(0)
                self.range: Tensor = self.max - self.min
            case _:
                raise ValueError(f"Method: {method} is not valid.")

    def normalize(
            self,
            X: Tensor,
    ) -> Tensor:
        match self.method:
            case StandardizationType.Default:
                return X
            case StandardizationType.ZScore:
                return (X - self.mean) / self.std
            case StandardizationType.MinMax:
                return (X - self.min) / self.range
            case _:
                raise ValueError(f"Method {self.method} is not valid.")
    
    def denormalize(
            self,
            X: Tensor,
    ) -> Tensor:
        match self.method:
            case StandardizationType.Default:
                return X
            case StandardizationType.ZScore:
                return (X * self.std) + self.mean
            case StandardizationType.MinMax:
                return (X * self.range) + self.min
            case _:
                raise ValueError(f"Method {self.method} is not valid.")
    
    def _conv_to_enum(
            self,
            method: str
    ) -> StandardizationType:
        return std_dict.get(
            method, 
            StandardizationType.Default
            )
        
