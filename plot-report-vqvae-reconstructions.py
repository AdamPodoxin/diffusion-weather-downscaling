from pathlib import Path
import xarray as xr
import torch
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes

CHANNEL_ORDER = [
    "Standard\n10m_u_component_of_wind", 
    "Standard\nmean_sea_level_pressure", 
    "PSD\n10m_u_component_of_wind", 
    "PSD\nmean_sea_level_pressure"
]
COLOR_MAP = {
    "Standard\n10m_u_component_of_wind": "winter",
    "PSD\n10m_u_component_of_wind": "winter",
    "Standard\nmean_sea_level_pressure": "cool",
    "PSD\nmean_sea_level_pressure": "cool",
}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_test = xr.open_zarr("data/test.zarr")

    lats = ds_test["latitude"].values
    lons = ds_test["longitude"].values
    
    X_hr = ds_test["X_hr"].sel(channel=["10m_u_component_of_wind", "mean_sea_level_pressure"])

    target = torch.from_numpy(X_hr.values).to(device)
    
    num_samples = target.shape[0]
    height, width = target.shape[2], target.shape[3]
    
    with torch.no_grad():
        means_base = target \
                    .mean(dim=(0, 2, 3)) \
                    .view(1, 2, 1, 1)
        stds_base = target \
                    .std(dim=(0, 2, 3)) \
                    .view(1, 2, 1, 1)

        target_means = means_base.expand(num_samples, 2, height, width)
        target_stds = stds_base.expand(num_samples, 2, height, width)

        target_normalized = (target - target_means) / target_stds
        
    target_normalized = torch.cat([target_normalized, target_normalized], dim=1)

    standard_predicted: torch.Tensor = torch.load("evaluation/test_vqvae_reconstructions/vanilla/normalized.pt")[:, [0, 3], :, :]
    psd_predicted: torch.Tensor = torch.load("evaluation/test_vqvae_reconstructions/psd/normalized.pt")[:, [0, 3], :, :]

    predicted = torch.cat([standard_predicted, psd_predicted], dim=1)

    sample_index = 0

    fig, axs = plt.subplots(
        nrows=2, ncols=4, 
        figsize=(11, 5), 
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    fig.suptitle("Standard vs. PSD VQVAE reconstructions", fontsize=20)

    # Row labels
    row_names = ["Predicted", "Target"]
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
            predicted[sample_index, channel_index].cpu().numpy(),
            target_normalized[sample_index, channel_index].cpu().numpy()
        ]

        for row_index in range(2):
            ax: GeoAxes = axs[row_index, channel_index]
            
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
            
            im = ax.pcolormesh(lons, lats, data_rows[row_index], transform=ccrs.PlateCarree(), cmap=cmap)
            
            if row_index == 0:
                ax.set_title(channel, fontsize=12)

    fig.savefig("plots/report_vqvae_reconstructions/vqvae_reconstructions.png", dpi=200)
    plt.close()
