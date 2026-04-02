import sys
from pathlib import Path
from utils import get_4channel_vqvae, load_model_state_dict
import xarray as xr
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


DATA_PATH = Path("data")
TEST_PATH = DATA_PATH / "test.zarr"

if __name__ == "__main__":
    model_path = sys.argv[1]
    output_path = sys.argv[2]

    device = "cuda"

    vqvae = get_4channel_vqvae(device)
    vqvae.load_state_dict(load_model_state_dict(model_path))

    ds_test = xr.open_zarr(TEST_PATH)
    X_hr_test = ds_test["X_hr"]

    SAMPLE_INDEX = 2
    sample_hr = X_hr_test.isel(sample=range(SAMPLE_INDEX, SAMPLE_INDEX + 1))

    reduce_dims = ("sample", "longitude", "latitude")
    means_hr = sample_hr.mean(dim=reduce_dims)
    stds_hr = sample_hr.std(dim=reduce_dims)
    sample_hr_norm = (sample_hr - means_hr) / stds_hr
    sample_hr_norm_tensor = torch.from_numpy(sample_hr_norm.values).to(device)

    latents = vqvae.encode(sample_hr_norm_tensor).latents
    output = vqvae.decode(latents).sample

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    with torch.no_grad():
        for channel in range(4):
            axs[0, channel].imshow(torchvision.utils.make_grid(sample_hr_norm_tensor.cpu())[channel], cmap='Greys')
            axs[1, channel].imshow(torchvision.utils.make_grid(output.cpu())[channel], cmap='Greys')

    fig.savefig(output_path)

    print(torch.nn.functional.mse_loss(sample_hr_norm_tensor, output))