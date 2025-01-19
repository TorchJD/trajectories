from abc import ABC, abstractmethod

import numpy as np
from matplotlib import cm as cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from numpy.lib._stride_tricks_impl import sliding_window_view


class Plotter(ABC):
    """Abstract base class to modify a matplotlib Axes object."""

    @abstractmethod
    def __call__(self, ax: plt.Axes) -> None:
        pass


class MultiPlotter(Plotter):
    """Plotter applying several plotters."""

    def __init__(self, plotters: list[Plotter]):
        self.plotters = plotters

    def __call__(self, ax: plt.Axes) -> None:
        for plotter in self.plotters:
            plotter(ax)


class PointPlotter(Plotter, ABC):
    """Abstract plotter storing a single point."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class InitialPointPlotter(PointPlotter):
    """PointPlotter that can draw the initial point."""

    def __call__(self, ax: plt.Axes) -> None:
        ax.scatter(self.x, self.y, color="black", s=30)


class OptimalPointPlotter(PointPlotter):
    """PointPlotter that can draw the optimal point."""

    def __init__(self, x: float, y: float, color: str):
        super().__init__(x, y)
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        ax.scatter(
            self.x,
            self.y,
            marker="*",
            color=self.color,
            zorder=1000,
            s=50,
        )


class OptimalLinePlotter(Plotter):
    """
    Plotter that can draw a continuous path with uniform color linking the provided optimal points.
    """

    def __init__(self, points: np.ndarray, color: str):
        self.points = points
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        ax.plot(self.points[:, 0], self.points[:, 1], color=self.color, linewidth=2.5)


class AxesPlotter(Plotter):
    """Plotter that can draw the x=0 and y=0 axes."""

    def __call__(self, ax: plt.Axes) -> None:
        ax.axhline(y=0, color="black", linewidth=0.75, alpha=0.5)
        ax.axvline(x=0, color="black", linewidth=0.75, alpha=0.5)


class CirclePlotter(Plotter):
    """Plotter that can draw a circle."""

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
    """
    MultiPlotter that can draw several circles of different radii and colors, to make contour
    lines centered at zero.
    """

    def __init__(self):
        radiuses = [1.0, 2.5, 4, 5.5, 7, 8.5]
        colormap = cm.inferno_r
        norm = mcolors.Normalize(vmin=-1, vmax=max(radiuses))
        plotters = [CirclePlotter(radius, colormap(norm(radius))) for radius in radiuses]
        super().__init__(plotters)


class SegmentPlotter(Plotter):
    """Plotter that can draw a single segment of a given color."""

    def __init__(self, xp: np.ndarray, yp: np.ndarray, color: str):
        self.xp = xp
        self.yp = yp
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        ax.plot(self.xp, self.yp, color=self.color, solid_capstyle="round", linewidth=1)


class PathPlotter(MultiPlotter):
    """Plotter that can draw a path of segments with colors varying along a gradient."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        x_view = sliding_window_view(xs, window_shape=2)
        y_view = sliding_window_view(ys, window_shape=2)

        colors = PathPlotter._get_color_gradient("#FF0000", "#FFEE00", len(xs) - 1)
        plotters = [SegmentPlotter(xp, yp, color) for xp, yp, color in zip(x_view, y_view, colors)]
        super().__init__(plotters)

    @staticmethod
    def _get_color_gradient(c1: str, c2: str, n: int) -> list[str]:
        """Given two hex colors, returns a color gradient with n colors."""

        assert n > 1
        c1_rgb = np.array(PathPlotter._hex_to_rgb(c1)) / 255
        c2_rgb = np.array(PathPlotter._hex_to_rgb(c2)) / 255
        mix_pcts = [x / (n - 1) for x in range(n)]
        rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
        return [
            "#" + "".join([format(int(round(val * 255)), "02x") for val in item])
            for item in rgb_colors
        ]

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> list[int]:
        """#FFFFFF -> [255,255,255]"""
        # Pass 16 to the integer function for change of base
        return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]


class TrajPlotter(MultiPlotter):
    """Plotter that can draw a trajectory: initial point + path."""

    def __init__(self, x0s: np.array, x1s: np.array):
        plotters = [PathPlotter(x0s, x1s), InitialPointPlotter(x0s[0], x1s[0])]
        super().__init__(plotters)


class MultiTrajPlotter(MultiPlotter):
    """Plotter that can draw several trajectories (one for each initial point)."""

    def __init__(self, M: np.ndarray):
        plotters = [TrajPlotter(x0s, x1s) for x0s, x1s in zip(M[:, :, 0], M[:, :, 1])]
        super().__init__(plotters)


class SetPlotter(Plotter):
    """
    Plotter that can represent an optimal set.

    If the provided array of optimal points contains a single point, the set will be represented by
    a star. Otherwise, it will be represented as a connected line plot. This does not necessarily
    work for all optimal sets, but it should be fine for those that are convex.
    """

    def __init__(self, points: np.ndarray, color: str):
        self.points = points
        self.color = color

        if len(points) == 1:
            self.plotter = OptimalPointPlotter(points[0, 0], points[0, 1], color=color)
        else:
            self.plotter = OptimalLinePlotter(points, color=color)

    def __call__(self, ax: plt.Axes) -> None:
        self.plotter(ax)


class SPSPlotter(SetPlotter):
    """Plotter that can represent the Strong Pareto stationary set: black SetPlotter"""

    def __init__(self, sps_points: np.ndarray):
        super().__init__(points=sps_points, color="black")


class PFPlotter(SetPlotter):
    """Plotter that can represent the Pareto front: green SetPlotter"""

    def __init__(self, pf_points: np.ndarray):
        super().__init__(points=pf_points, color="green")


class HeatmapPlotter(Plotter):
    """
    Plotter that can draw a heatmap with the given values extending between the provided
    coordinates.
    """

    def __init__(
        self, values: np.ndarray, x0_min: float, x0_max: float, x1_min: float, x1_max: float
    ):
        self.values = values
        self.x0_min = x0_min
        self.x0_max = x0_max
        self.x1_min = x1_min
        self.x1_max = x1_max

    def __call__(self, ax: plt.Axes) -> None:
        ax.imshow(
            self.values.T,
            origin="lower",
            cmap="PRGn",
            aspect="auto",
            vmin=-1,
            vmax=1,
            extent=(self.x0_min, self.x0_max, self.x1_min, self.x1_max),
        )
