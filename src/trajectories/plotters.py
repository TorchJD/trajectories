from abc import ABC, abstractmethod

import numpy as np
import torch
from matplotlib import cm as cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from numpy.lib._stride_tricks_impl import sliding_window_view

from trajectories.objectives import ConvexQuadraticForm, ElementWiseQuadratic
from trajectories.pareto_sets import CQFParetoSet, EWQParetoSet, ParetoSet


class Plotter(ABC):
    """Class that can modify a matplotlib Axes object."""

    @abstractmethod
    def __call__(self, ax: plt.Axes) -> None:
        pass


class MultiPlotter(Plotter):
    def __init__(self, plotters: list[Plotter]):
        self.plotters = plotters

    def __call__(self, ax: plt.Axes) -> None:
        for plotter in self.plotters:
            plotter(ax)


class PointPlotter(Plotter, ABC):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class OptimalPointPlotter(PointPlotter):
    def __call__(self, ax: plt.Axes) -> None:
        ax.scatter(self.x, self.y, marker="*", color="black", zorder=1000, s=50)


class InitialPointPlotter(PointPlotter):
    def __call__(self, ax: plt.Axes) -> None:
        ax.scatter(self.x, self.y, color="black", s=30)


class AxesPlotter(Plotter):
    def __call__(self, ax: plt.Axes) -> None:
        ax.axhline(y=0, color="black", linewidth=0.75, alpha=0.5)
        ax.axvline(x=0, color="black", linewidth=0.75, alpha=0.5)


class CirclePlotter(Plotter):
    def __init__(self, radius: float, color):
        self.radius = radius
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        circle = plt.Circle(
            (0, 0),
            self.radius,
            color=self.color,
            fill=False,
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
        )
        ax.add_patch(circle)


class ContourCirclesPlotter(MultiPlotter):
    def __init__(self):
        radiuses = [1.0, 2.5, 4, 5.5, 7, 8.5]
        colormap = cm.inferno_r
        norm = mcolors.Normalize(vmin=-1, vmax=max(radiuses))
        plotters = [CirclePlotter(radius, colormap(norm(radius))) for radius in radiuses]
        super().__init__(plotters)


class SegmentPlotter(Plotter):
    def __init__(self, xp: np.ndarray, yp: np.ndarray, color: str):
        self.xp = xp
        self.yp = yp
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        ax.plot(self.xp, self.yp, color=self.color, solid_capstyle="round", linewidth=1)


class PathPlotter(MultiPlotter):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        x_view = sliding_window_view(xs, window_shape=2)
        y_view = sliding_window_view(ys, window_shape=2)

        colors = _get_color_gradient("#FF0000", "#FFEE00", len(xs) - 1)
        plotters = [SegmentPlotter(xp, yp, color) for xp, yp, color in zip(x_view, y_view, colors)]
        super().__init__(plotters)


class ParamTrajPlotter(MultiPlotter):
    def __init__(self, X: np.ndarray):
        plotters = [PathPlotter(xs, ys) for xs, ys in zip(X[:, :, 0], X[:, :, 1])]
        plotters += [InitialPointPlotter(xs[0], ys[0]) for xs, ys in zip(X[:, :, 0], X[:, :, 1])]
        super().__init__(plotters)


class ParetoSetPlotter(Plotter, ABC):
    def __init__(self, pareto_set: ParetoSet):
        self.pareto_set = pareto_set


class LineParetoSetPlotter(ParetoSetPlotter):
    def __call__(self, ax: plt.Axes) -> None:
        ws_np = np.linspace([0, 1], [1, 0], 100)
        ws = torch.tensor(ws_np)
        xs = torch.stack([self.pareto_set(w) for w in ws])
        ax.plot(xs[:, 0], xs[:, 1], color="black", linewidth=2.5)


class SinglePointParetoSetPlotter(ParetoSetPlotter):
    def __call__(self, ax: plt.Axes) -> None:
        x = self.pareto_set(torch.tensor([0.5, 0.5]))
        OptimalPointPlotter(x[0].item(), x[1].item())(ax)


class EWQParamTrajPlotter(ParamTrajPlotter):
    def __init__(self, ewq: ElementWiseQuadratic, X: np.ndarray):
        super().__init__(X)
        background_plotters = [AxesPlotter(), ContourCirclesPlotter()]
        self.plotters = (
            background_plotters + self.plotters + [SinglePointParetoSetPlotter(EWQParetoSet(ewq))]
        )


class CQFParamTrajPlotter(ParamTrajPlotter):
    def __init__(self, cqf: ConvexQuadraticForm, X: np.ndarray):
        super().__init__(X)
        background_plotters = [AxesPlotter()]
        self.plotters = (
            background_plotters + [LineParetoSetPlotter(CQFParetoSet(cqf))] + self.plotters
        )


def _get_color_gradient(c1: str, c2: str, n: int) -> list[str]:
    """Given two hex colors, returns a color gradient with n colors."""

    assert n > 1
    c1_rgb = np.array(_hex_to_rgb(c1)) / 255
    c2_rgb = np.array(_hex_to_rgb(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return [
        "#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors
    ]


def _hex_to_rgb(hex_str: str) -> list[int]:
    """#FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]
