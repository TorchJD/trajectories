from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Objective(ABC):
    def __init__(self, n_params: int, n_values: int):
        self.n_params = n_params
        self.n_values = n_values

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the value of the objective function at x. It has to be a vector."""

    @abstractmethod
    def jacobian(self, x: Tensor) -> Tensor:
        """
        Compute the value of the Jacobian of the objective function at x. It is a matrix of shape
        [n_values, n_params].
        """

    def __str__(self) -> str:
        """Return a string representation of the objective function."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n_values})"


class QuadraticForm(Objective):
    def __init__(self, As: list[Tensor], us: list[Tensor]):
        if len(As) != len(us):
            raise ValueError("As and us must have the same length.")

        if len(As) < 1:
            raise ValueError("As and us must have at least one element.")

        super().__init__(n_params=len(us[0]), n_values=len(As))
        # Note that if A is not PSD, the objective is not convex.
        self.As = As
        self.us = us

    def __call__(self, x: Tensor) -> Tensor:
        objective_values = [self.quad(x, A, u) for A, u in zip(self.As, self.us)]
        return torch.stack(objective_values)

    def jacobian(self, x: Tensor) -> Tensor:
        return torch.vstack([2 * (x - u) @ A for A, u in zip(self.As, self.us)])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(As={self.As}, us={self.us})"

    @staticmethod
    def quad(x: Tensor, A: Tensor, u: Tensor):
        x_minus_u = x - u
        return x_minus_u @ A @ x_minus_u


class ConvexQuadraticForm(QuadraticForm):
    def __init__(self, Bs: list[Tensor], us: list[Tensor]):
        self.Bs = Bs
        super().__init__(As=[B @ B.T for B in self.Bs], us=us)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Bs={self.Bs}, us={self.us})"


class ElementWiseQuadratic(Objective):
    def __init__(self, n_dim: int):
        super().__init__(n_params=n_dim, n_values=n_dim)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x) != self.n_values:
            raise ValueError("x must have the same length as the number of values.")
        return x**2

    def jacobian(self, x: Tensor) -> Tensor:
        return torch.diagonal(torch.stack([2 * x[0], 2 * x[1]]))
