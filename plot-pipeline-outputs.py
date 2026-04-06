import argparse
from pathlib import Path
import xarray as xr
import torch
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes

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

    lats = ds_test["latitude"].values
    lons = ds_test["longitude"].values
    
    X_hr = ds_test["X_hr"]
    X_lr = ds_test["X_lr"]

    target = torch.from_numpy(X_hr.values).to(device)
    lr = torch.from_numpy(X_lr.values).to(device)
    
    num_samples = target.shape[0]
    height, width = target.shape[2], target.shape[3]
    
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

    fig, axs = plt.subplots(
        nrows=3, ncols=4, 
        figsize=(15, 10), 
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    fig.suptitle(f"{args.pipeline} pipeline outputs", fontsize=20)

    # Row labels
    row_names = ["LR\nInterpolated", "Predicted", "Target"]
    for ax, row_name in zip(axs[:, 0], row_names):
        ax: GeoAxes = ax

        # Use a secondary axis or text for row labels because Cartopy axes 
        # behave differently with set_ylabel
        ax.text(
            x=-0.05, y=0.5,
            s=row_name,
            transform=ax.transAxes,
            va="center", ha="right",
            fontsize=14,
        )

    for channel_index, channel in enumerate(CHANNEL_ORDER):
        cmap = COLOR_MAP[channel]

        data_rows = [
            lr_interpolated[sample_index, channel_index].cpu().numpy(),
            predicted_normalized[sample_index, channel_index].cpu().numpy(),
            target_normalized[sample_index, channel_index].cpu().numpy()
        ]

        for row_index in range(3):
            ax: GeoAxes = axs[row_index, channel_index]
            
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
            
            im = ax.pcolormesh(lons, lats, data_rows[row_index], transform=ccrs.PlateCarree(), cmap=cmap)
            
            if row_index == 0:
                ax.set_title(channel, fontsize=12)

    fig.savefig(args.output, dpi=200)
    plt.close()
