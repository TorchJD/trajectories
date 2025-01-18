from abc import ABC, abstractmethod

import torch
from torch import Tensor

from trajectories.objectives import ConvexQuadraticForm, ElementWiseQuadratic, Objective


class StrongParetoStationarySet(ABC):
    def __init__(self, objective: Objective):
        self.objective = objective

    def __call__(self, w: Tensor) -> Tensor:
        """
        Compute the optimal point corresponding to the scalarization of the objective by a vector of
        positive weights.

        :param w: The vector of weights. All of its coordinates must be (strictly) positive.
        """

        if (w.le(0.0)).any():
            raise ValueError(f"All coordinates of w must be (strictly) positive. Found w = {w}.")

        return self._compute(w)

    @abstractmethod
    def _compute(self, w: Tensor) -> Tensor:
        pass


class CQF_SPSS(StrongParetoStationarySet):
    def __init__(self, cqf: ConvexQuadraticForm):
        super().__init__(cqf)
        self.As = cqf.As
        self.us = cqf.us

    def _compute(self, ws: Tensor) -> Tensor:
        G = torch.stack([w * A for w, A in zip(ws, self.As)]).sum(dim=0)
        b = torch.stack([w * A @ u for w, A, u in zip(ws, self.As, self.us)]).sum(dim=0)
        return torch.linalg.lstsq(G, b).solution


class EWQ_SPSS(StrongParetoStationarySet):
    def __init__(self, ewq: ElementWiseQuadratic):
        super().__init__(ewq)
        self.n_values = ewq.n_values

    def _compute(self, w: Tensor) -> Tensor:
        return torch.zeros(self.n_values)
