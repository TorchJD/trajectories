from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Objective(ABC):
    def __init__(self, n_objectives: int):
        self.n_objectives = n_objectives

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the value of the objective function at x. It has to be a vector."""

    def __str__(self) -> str:
        """Return a string representation of the objective function."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n_objectives})"


class ElementWiseQuadratic(Objective):
    def __call__(self, x: Tensor) -> Tensor:
        if len(x) != self.n_objectives:
            raise ValueError("x must have the same length as the number of objectives.")
        return x**2


class QuadraticForm(Objective):
    def __init__(self, As: list[Tensor], us: list[Tensor]):
        if len(As) != len(us):
            raise ValueError("As and us must have the same length.")

        super().__init__(n_objectives=len(As))
        # TODO: if A is not positive definite, the objective is not convex. Think about what to do.
        #  Maybe just use A @ A^T instead of A.
        self.As = As
        self.us = us

    def __call__(self, x: Tensor) -> Tensor:
        objective_values = [self.quad(x, A, u) for A, u in zip(self.As, self.us)]
        return torch.stack(objective_values)

    @staticmethod
    def quad(x: Tensor, A: Tensor, u: Tensor):
        x_minus_u = x - u
        return x_minus_u @ A @ x_minus_u

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(As={self.As}, us={self.us})"
