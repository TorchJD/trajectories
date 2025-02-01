"""
Plot the trajectories of an objective function in the parameter space.
Usage:
  plot_params <objective>

Arguments:
  <objective>         The key of the objective function.

Options:
  -h --help          Show this screen.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

from trajectories.constants import (
    INITIAL_POINTS,
    LATEX_NAMES,
    N_SAMPLES_SPSM,
    OBJECTIVES,
    SUBPLOT_LOCATIONS,
)
from trajectories.objectives import ElementWiseQuadratic, WithSPSMappingMixin
from trajectories.optimization import compute_gradient_cosine_similarities
from trajectories.paths import RESULTS_DIR, get_param_plots_dir, get_params_dir
from trajectories.plotters import (
    AxesPlotter,
    ContentLimAdjuster,
    ContourCirclesPlotter,
    HeatmapPlotter,
    LimAdjuster,
    MultiTrajPlotter,
    SPSPlotter,
    SquareBoxAspectSetter,
    TitleSetter,
    XAxisLabeller,
    XTicksClearer,
    YAxisLabeller,
    YTicksClearer,
)


def main():
    arguments = docopt(__doc__)
    objective_key = arguments["<objective>"]

    # Read metadata.json
    with open(RESULTS_DIR / objective_key / "metadata.json", "r") as f:
        metadata = json.load(f)

    params_dir = get_params_dir(objective_key)
    param_plots_dir = get_param_plots_dir(objective_key)
    param_plots_dir.mkdir(parents=True, exist_ok=True)

    # This seems to be the only way to make the font be Type1, which is the only font type supported
    # by ICML.
    plt.rcParams.update({"text.usetex": True})
    objective_key = metadata["objective_key"]
    objective = OBJECTIVES[objective_key]

    if objective.n_params != 2:
        raise ValueError("Can only plot param trajectories for objectives with 2 params.")

    initial_points = INITIAL_POINTS[objective_key]
    initial_points = np.stack([np.array(point) for point in initial_points])
    main_content = initial_points  # The content to which the axes must be adjusted

    n_samples_spsm = N_SAMPLES_SPSM[objective_key]
    common_plotter = SquareBoxAspectSetter()

    if isinstance(objective, WithSPSMappingMixin):
        sps_points = objective.sps_mapping.sample(n_samples_spsm, eps=1e-5).numpy()
        main_content = np.concatenate([main_content, sps_points])
        common_plotter += SPSPlotter(sps_points)

    adjust_plotter = ContentLimAdjuster(main_content)
    common_plotter += adjust_plotter

    if isinstance(objective, ElementWiseQuadratic):
        common_plotter += AxesPlotter()
        common_plotter += ContourCirclesPlotter()
        common_plotter += LimAdjuster(xlim=[-5.0, 5.0], ylim=[-5.0, 5.0])
    else:
        if objective.n_values == 2:
            similarities = compute_gradient_cosine_similarities(
                objective,
                x0_min=adjust_plotter.xlim[0],
                x0_max=adjust_plotter.xlim[1],
                x1_min=adjust_plotter.ylim[0],
                x1_max=adjust_plotter.ylim[1],
                n=200,
            )
            common_plotter += HeatmapPlotter(
                values=similarities.numpy() ** 3,
                x_min=adjust_plotter.xlim[0],
                x_max=adjust_plotter.xlim[1],
                y_min=adjust_plotter.ylim[0],
                y_max=adjust_plotter.ylim[1],
                vmin=-1,
                vmax=1,
                cmap="PiYG",
            )

    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_X = {key: np.load(params_dir / f"{key}.npy") for key in aggregator_keys}

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    save_path = param_plots_dir / "all.pdf"

    for aggregator_key, X in aggregator_to_X.items():
        i, j = SUBPLOT_LOCATIONS[aggregator_key]

        plotter = common_plotter + MultiTrajPlotter(X) + TitleSetter(LATEX_NAMES[aggregator_key])
        plotter += XAxisLabeller("$x_1$") if i == 1 else XTicksClearer()
        plotter += YAxisLabeller("$x_2$") if j == 0 else YTicksClearer()

        plotter(axes[i][j])

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
