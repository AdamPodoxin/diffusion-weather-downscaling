from pathlib import Path
from utils import get_4channel_vqvae, load_model_state_dict
import xarray as xr
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


DATA_PATH = Path("data")
TEST_PATH = DATA_PATH / "test.zarr"

device = "cuda"

vqvae = get_4channel_vqvae(device)
vqvae.load_state_dict(load_model_state_dict("./models/vqvae-trained/vqvae-trained.pt"))

ds_test = xr.open_zarr(TEST_PATH)
X_lr_test = ds_test["X_lr"]
X_hr_test = ds_test["X_hr"]

SAMPLE_INDEX = 2
sample_lr = X_lr_test.isel(sample=range(SAMPLE_INDEX, SAMPLE_INDEX + 1))
sample_hr = X_hr_test.isel(sample=range(SAMPLE_INDEX, SAMPLE_INDEX + 1))

reduce_dims = ("sample", "longitude_lr", "latitude_lr")
means_lr = sample_lr.mean(dim=reduce_dims)
stds_lr = sample_lr.std(dim=reduce_dims)
sample_lr_norm = (sample_lr - means_lr) / stds_lr
sample_lr_norm_tensor = torch.from_numpy(sample_lr_norm.values).to(device)

latents = vqvae.encode(sample_lr_norm_tensor).latents
output = vqvae.decode(latents).sample

fig, axs = plt.subplots(2, 4, figsize=(10, 5))
with torch.no_grad():
    for channel in range(4):
        axs[0, channel].imshow(torchvision.utils.make_grid(sample_lr_norm_tensor.cpu())[channel], cmap='Greys')
        axs[1, channel].imshow(torchvision.utils.make_grid(output.cpu())[channel], cmap='Greys')

fig.savefig("vqvae-reconstructions-lr.png")

print(torch.nn.functional.mse_loss(sample_lr_norm_tensor, output))


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

fig.savefig("vqvae-reconstructions-hr.png")

print(torch.nn.functional.mse_loss(sample_hr_norm_tensor, output))