import argparse
from pathlib import Path
import xarray as xr
import torch
from torch.nn.functional import interpolate
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
    X_lr = ds_test["X_lr"]

    target = torch.from_numpy(X_hr.values).to(device)
    lr = torch.from_numpy(X_lr.values).to(device)
    
    num_samples = target.shape[0]
    height = target.shape[2]
    width = target.shape[3]
    
    with torch.no_grad():
        means_base = target \
                    .mean(dim=(0, 2, 3)) \
                    .view(1, 4, 1, 1)
        stds_base = target \
                    .std(dim=(0, 2, 3)) \
                    .view(1, 4, 1, 1)

        target_means = means_base.expand(num_samples, 4, height, width)
        target_stds = stds_base.expand(num_samples, 4, height, width)

        target_normalized = (target - target_means) / target_stds
        
    height_lr = lr.shape[2]
    width_lr = lr.shape[3]
    
    with torch.no_grad():
        lr_means = means_base.expand(num_samples, 4, height_lr, width_lr)
        lr_stds = stds_base.expand(num_samples, 4, height_lr, width_lr)

        lr_normalized = (lr - lr_means) / lr_stds
        lr_interpolated = interpolate(lr_normalized, scale_factor=4)

    predicted_normalized: torch.Tensor = torch.load(args.input)

    sample_index = int(args.sample)

    target_grid = make_grid(target_normalized.cpu()[sample_index])
    lr_grid = make_grid(lr_interpolated.cpu()[sample_index])
    predicted_grid = make_grid(predicted_normalized.cpu()[sample_index])

    fig, axs = plt.subplots(3, 4, figsize=(16, 12))

    fig.suptitle(f"{args.pipeline} pipeline outputs", fontsize=20)

    axs[0, 0].set_ylabel("LR\ninterpolated", rotation=0, labelpad=40, fontsize=14)
    axs[1, 0].set_ylabel("Predicted", rotation=0, labelpad=40, fontsize=14)
    axs[2, 0].set_ylabel("Target", rotation=0, labelpad=40, fontsize=14)

    for channel_index, channel in enumerate(CHANNEL_ORDER):
        lr_ax: Axes = axs[0, channel_index]
        predicted_ax: Axes = axs[1, channel_index]
        target_ax: Axes = axs[2, channel_index]

        lr_ax.set_title(channel)

        cmap = COLOR_MAP[channel]

        lr_ax.imshow(lr_grid[channel_index], cmap=cmap)
        predicted_ax.imshow(predicted_grid[channel_index], cmap=cmap)
        target_ax.imshow(target_grid[channel_index], cmap=cmap)

    fig.savefig(args.output, bbox_inches="tight")
