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
import torch
import torch.nn.functional as F
from docopt import docopt
from torch import Tensor

from trajectories.constants import INITIAL_POINTS, N_SAMPLES_SPSM, OBJECTIVES
from trajectories.objectives import ElementWiseQuadratic, Objective, WithSPSMappingMixin
from trajectories.paths import RESULTS_DIR, get_param_plots_dir, get_params_dir
from trajectories.plotters import (
    AxesPlotter,
    ContourCirclesPlotter,
    HeatmapPlotter,
    MultiPlotter,
    MultiTrajPlotter,
    SPSPlotter,
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
    common_plotters = [AxesPlotter()]

    if isinstance(objective, WithSPSMappingMixin):
        sps_points = objective.sps_mapping.sample(n_samples_spsm, eps=1e-5).numpy()
        main_content = np.concatenate([main_content, sps_points])
        common_plotters.append(SPSPlotter(sps_points))

    if isinstance(objective, ElementWiseQuadratic):
        common_plotters.append(ContourCirclesPlotter())

    x0_min, x1_min = main_content.min(axis=0)
    x0_max, x1_max = main_content.max(axis=0)
    x0_range = x0_max - x0_min
    x1_range = x1_max - x1_min
    margin = 0.05
    xlim = [x0_min - margin * x0_range, x0_max + margin * x0_range]
    ylim = [x1_min - margin * x1_range, x1_max + margin * x1_range]

    if objective.n_values == 2:
        similarities = compute_gradient_cosine_similarities(
            objective,
            x0_min=xlim[0],
            x0_max=xlim[1],
            x1_min=ylim[0],
            x1_max=ylim[1],
            n=200,
        )
        heatmap_plotter = HeatmapPlotter(
            values=similarities.numpy(),
            x_min=xlim[0],
            x_max=xlim[1],
            y_min=ylim[0],
            y_max=ylim[1],
        )
        common_plotters.append(heatmap_plotter)

    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_X = {key: np.load(params_dir / f"{key}.npy") for key in aggregator_keys}

    for aggregator_key, X in aggregator_to_X.items():
        save_path = param_plots_dir / f"{aggregator_key}.pdf"
        fig, ax = plt.subplots(1, figsize=(2.5, 2.5))
        plotter = MultiPlotter([*common_plotters, MultiTrajPlotter(X)])
        plotter(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.savefig(save_path, bbox_inches="tight")


def compute_gradient_cosine_similarities(
    objective: Objective, x0_min: float, x0_max: float, x1_min: float, x1_max: float, n: int
) -> Tensor:
    if objective.n_values != 2:
        raise ValueError("Objective should have 2 values.")

    x0_len = x0_max - x0_min
    x0_start = x0_min + x0_len / (n * 2)
    x0_end = x0_max - x0_len / (n * 2)

    x1_len = x1_max - x1_min
    x1_start = x1_min + x1_len / (n * 2)
    x1_end = x1_max - x1_len / (n * 2)

    x0s = np.linspace(x0_start, x0_end, n, dtype=np.float32)
    x1s = np.linspace(x1_start, x1_end, n, dtype=np.float32)

    similarities = torch.zeros(n, n)
    for i, x0 in enumerate(x0s):
        for j, x1 in enumerate(x1s):
            x = torch.tensor([x0, x1])
            similarities[i][j] = compute_cosine_similarity(objective, x)

    return similarities


def compute_cosine_similarity(objective: Objective, x: Tensor) -> Tensor:
    J = objective.jacobian(x)
    return F.cosine_similarity(J[0].unsqueeze(0), J[1].unsqueeze(0), eps=1e-19).squeeze()
