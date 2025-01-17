from abc import ABC, abstractmethod

import torch
from torch import Tensor

from trajectories.objectives import ConvexQuadraticForm


class ParetoSet(ABC):
    @abstractmethod
    def __call__(self, w: Tensor) -> Tensor:
        pass


class CQFParetoSet(ParetoSet):
    def __init__(self, cqf: ConvexQuadraticForm):
        self.cqf = cqf

    def __call__(self, w: Tensor) -> Tensor:
        G = torch.stack([w_i * A_i for w_i, A_i in zip(w, self.cqf.As)]).sum(dim=0)
        b = torch.stack(
            [w_i * A_i @ u_i for w_i, A_i, u_i in zip(w, self.cqf.As, self.cqf.us)]
        ).sum(dim=0)
        return torch.linalg.lstsq(G, b).solution
