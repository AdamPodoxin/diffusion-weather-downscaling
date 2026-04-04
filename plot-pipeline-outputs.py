import argparse
from pathlib import Path
import xarray as xr
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


CHANNEL_ORDER = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
COLOR_MAP = {
    "10m_u_component_of_wind": "winter",
    "10m_v_component_of_wind": "winter",
    "2m_temperature": "autumn",
    "mean_sea_level_pressure": "cool",
}


def parse_args():
    p = argparse.ArgumentParser("Visualize pipeline outputs")

    p.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/test_outputs/vanilla/normalized.pt"),
        help="Normalized test outputs file",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("plots/pipeline_outputs/vanilla.png"),
        help="Output image file",
    )
    p.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/test.zarr"),
        help="File containing ground truth HR output",
    )
    p.add_argument(
        "--pipeline",
        type=str,
        default="Vanilla",
        help="Pipeline name",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_test = xr.open_zarr(args.test_file)
    X_hr = ds_test["X_hr"]

    target = torch.from_numpy(X_hr.values).to(device)
    
    num_samples = target.shape[0]
    height = target.shape[2]
    width = target.shape[3]
    
    with torch.no_grad():
        target_means = target \
                    .mean(dim=(0, 2, 3)) \
                    .view(1, 4, 1, 1) \
                    .expand(num_samples, 4, height, width)
        target_stds = target \
                    .std(dim=(0, 2, 3)) \
                    .view(1, 4, 1, 1) \
                    .expand(num_samples, 4, height, width)

        target_normalized = (target - target_means) / target_stds

    predicted_normalized: torch.Tensor = torch.load(args.input)

    sample_index = int(args.sample)

    target_grid = make_grid(target_normalized.cpu()[sample_index])
    predicted_grid = make_grid(predicted_normalized.cpu()[sample_index])

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    fig.suptitle(f"{args.pipeline} pipeline outputs", fontsize=20)

    axs[0, 0].set_ylabel("Predicted", rotation=0, labelpad=34, fontsize=14)
    axs[1, 0].set_ylabel("Target", rotation=0, labelpad=34, fontsize=14)

    for channel_index, channel in enumerate(CHANNEL_ORDER):
        predicted_ax: Axes = axs[0, channel_index]
        target_ax: Axes = axs[1, channel_index]

        predicted_ax.set_title(channel)

        cmap = COLOR_MAP[channel]

        predicted_ax.imshow(predicted_grid[channel_index], cmap=cmap)
        target_ax.imshow(target_grid[channel_index], cmap=cmap)

    fig.savefig(args.output, bbox_inches="tight")
