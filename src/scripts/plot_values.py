"""
Plot the trajectories of an objective function in the value space.
Usage:
  plot_values <objective>

Arguments:
  <objective>         The key of the objective function.

Options:
  -h --help          Show this screen.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from docopt import docopt

from trajectories.constants import LATEX_NAMES, N_SAMPLES_SPSM, OBJECTIVES, SUBPLOT_LOCATIONS
from trajectories.objectives import WithSPSMappingMixin
from trajectories.optimization import compute_objectives_pf_distances
from trajectories.paths import RESULTS_DIR, get_value_plots_dir, get_values_dir
from trajectories.plotters import (
    AdjustToContentPlotter,
    EmptyPlotter,
    HeatmapPlotter,
    LabelAxesPlotter,
    MultiTrajPlotter,
    PFPlotter,
)


def main():
    arguments = docopt(__doc__)
    objective_key = arguments["<objective>"]

    # Read metadata.json
    with open(RESULTS_DIR / objective_key / "metadata.json", "r") as f:
        metadata = json.load(f)

    values_dir = get_values_dir(objective_key)
    value_plots_dir = get_value_plots_dir(objective_key)
    value_plots_dir.mkdir(parents=True, exist_ok=True)

    # This seems to be the only way to make the font be Type1, which is the only font type supported
    # by ICML.
    plt.rcParams.update({"text.usetex": True})
    objective_key = metadata["objective_key"]
    objective = OBJECTIVES[objective_key]

    if objective.n_values != 2:
        raise ValueError("Can only plot values trajectories for objectives with 2 values.")

    n_samples_spsm = N_SAMPLES_SPSM[objective_key]
    common_plotter = EmptyPlotter()
    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_Y = {key: np.load(values_dir / f"{key}.npy") for key in aggregator_keys}
    if len(aggregator_to_Y) == 0:
        return
    else:
        # The content to which the axes must be adjusted
        main_content = list(aggregator_to_Y.items())[0][1][:, 0, :]

    if isinstance(objective, WithSPSMappingMixin):
        sps_points = objective.sps_mapping.sample(n_samples_spsm, eps=1e-5)
        pf_points = torch.stack([objective(x) for x in sps_points])
        pf_points_array = pf_points.numpy()
        common_plotter += PFPlotter(pf_points_array)

        main_content = np.concatenate([main_content, pf_points_array])
        adjust_plotter = AdjustToContentPlotter(main_content)
        common_plotter += adjust_plotter
        distances = compute_objectives_pf_distances(
            pf_points=pf_points,
            y0_min=adjust_plotter.xlim[0],
            y0_max=adjust_plotter.xlim[1],
            y1_min=adjust_plotter.ylim[0],
            y1_max=adjust_plotter.ylim[1],
            n=200,
        )
        common_plotter += HeatmapPlotter(
            values=distances.numpy(),
            x_min=adjust_plotter.xlim[0],
            x_max=adjust_plotter.xlim[1],
            y_min=adjust_plotter.ylim[0],
            y_max=adjust_plotter.ylim[1],
            vmin=0,
            vmax=1,
            cmap="Reds",
        )

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    save_path = value_plots_dir / "all.pdf"

    for aggregator_key, Y in aggregator_to_Y.items():
        i, j = SUBPLOT_LOCATIONS[aggregator_key]
        plotter = common_plotter + MultiTrajPlotter(Y)
        if j == 0 and i == 1:
            plotter += LabelAxesPlotter("Objective $1$", "Objective $2$")
        elif j == 0:
            plotter += LabelAxesPlotter(None, "Objective $2$")
            axes[i][j].set_xticks([])
        elif i == 1:
            plotter += LabelAxesPlotter("Objective $1$", None)
            axes[i][j].set_yticks([])
        else:
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

        plotter(axes[i][j])
        axes[i][j].set_title(LATEX_NAMES[aggregator_key])
        axes[i][j].set_box_aspect(1)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
