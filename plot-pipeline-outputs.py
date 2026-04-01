import sys
from pathlib import Path
import xarray as xr
from weather_downscaling_pipeline import WeatherLDMSuperResolutionPipeline
import torch
import torchvision
import matplotlib.pyplot as plt


DATA_PATH = Path("data")
TEST_PATH = DATA_PATH / "test.zarr"

SAMPLE_INDEX = 0

NUM_INFERENCE_STEPS = 500


if __name__ == "__main__":
    ds_test = xr.open_zarr(TEST_PATH)
    X_lr_test = ds_test["X_lr"]
    X_hr_test = ds_test["X_hr"]

    X_lr_test = X_lr_test.isel(sample=(slice(SAMPLE_INDEX, SAMPLE_INDEX + 1)))

    pipeline = WeatherLDMSuperResolutionPipeline()

    _, Y = pipeline(X_lr_test, NUM_INFERENCE_STEPS)

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    with torch.no_grad():
        for channel in range(4):
            axs[0, channel].imshow(torchvision.utils.make_grid(Y[0].cpu())[channel], cmap='Greys')
            axs[1, channel].imshow(torchvision.utils.make_grid(torch.from_numpy(X_hr_test[0].values))[channel], cmap='Greys')

    fig.savefig("pipeline-outputs.png")
