from pathlib import Path

import numpy as np
from matplotlib import cm as cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from numpy.lib._stride_tricks_impl import sliding_window_view


def plot_parameters(
    X: np.ndarray,
    save_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(1, figsize=(2.5, 2.5))

    ax.axhline(y=0, color="black", linewidth=0.75, alpha=0.5)
    ax.axvline(x=0, color="black", linewidth=0.75, alpha=0.5)

    radiuses = [1.0, 2.5, 4, 5.5, 7, 8.5]
    colormap = cm.inferno_r
    norm = mcolors.Normalize(vmin=-1, vmax=max(radiuses))

    for radius in radiuses:
        color = colormap(norm(radius))
        circle = plt.Circle(
            (0, 0), radius, color=color, fill=False, linestyle="--", alpha=0.8, linewidth=1.5
        )
        ax.add_patch(circle)

    for params in X:
        x = np.array([param[0] for param in params])
        y = np.array([param[1] for param in params])
        colors = get_color_gradient("#FF0000", "#FFEE00", len(x) - 1)
        ax.scatter(x[0], y[0], color="black", s=30)

        x_view = sliding_window_view(x, window_shape=2)
        y_view = sliding_window_view(y, window_shape=2)

        for xs, ys, color in zip(x_view, y_view, colors):
            ax.plot(xs, ys, color=color, solid_capstyle="round", linewidth=1.5)

        # ax.scatter(x, y, color=colors, marker=".", alpha=0.5, s=10)

    ax.scatter(x=0, y=0, marker="*", color="black", zorder=1000, s=50)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.savefig(save_path, bbox_inches="tight")


def get_color_gradient(c1: str, c2: str, n: int) -> list[str]:
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_rgb(c1)) / 255
    c2_rgb = np.array(hex_to_rgb(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return [
        "#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors
    ]


def hex_to_rgb(hex_str: str) -> list[int]:
    """#FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]
