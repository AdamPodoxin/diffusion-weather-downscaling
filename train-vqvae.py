from pathlib import Path

from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline 
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import xarray as xr
from tqdm import tqdm


DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH / "train.zarr"

MODELS_PATH = Path("models")
VQVAE_PATH = MODELS_PATH / "vqvae"

BATCH_SIZE = 128
NUM_EPOCHS = 100

WIND_DIR_AND_PRESSURE_CHANNELS = \
[
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "mean_sea_level_pressure"
]


def z_normalize_tensor(tensor: torch.Tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor, mean, std


def normalize_across_channels(X: torch.Tensor, channel_dim=1):
    num_channels = X.shape[channel_dim]

    channel_means = [0 for _ in range(num_channels)]
    channel_stds = [0 for _ in range(num_channels)]
    
    X_normalized = torch.empty_like(X)

    for channel in range(num_channels):
        X_normalized[:, channel, :, :], mean, std = z_normalize_tensor(X[:, channel, :, :])
        
        channel_means[channel] = mean
        channel_stds[channel] = std

    return X_normalized, channel_means, channel_stds


def generate_batches(data: xr.DataArray, batch_size=32):
    for i in range(0, data.sizes["sample"], batch_size):
        batch = data.isel(sample=slice(i, i + batch_size))
        batch_tensor = torch.from_numpy(batch.values)
        yield batch_tensor


def train_vqvae(vqvae: VQModel, low_res_inputs: xr.DataArray, device: str):
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    num_batches = low_res_inputs.sizes["sample"] // BATCH_SIZE

    vqvae.train()

    for epoch in range(NUM_EPOCHS):
        print("\n", "Epoch", epoch + 1)
        
        batch_generator = generate_batches(low_res_inputs, batch_size=BATCH_SIZE)
        for X in tqdm(batch_generator, total=num_batches):
            X = X.to(device)
            X_normalized, _, _ = normalize_across_channels(X)

            output: DecoderOutput = vqvae(X)
            output_sample = output.sample
            output_sample_normalized, _, _ = normalize_across_channels(output_sample)

            loss = torch.nn.functional.mse_loss(output_sample_normalized, X_normalized)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Loss: {loss:.2e}")
        
        scheduler.step()
    
    torch.save(vqvae.state_dict(), VQVAE_PATH / "vqvae-trained.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)
    
    pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
    pipeline = pipeline.to(device)

    vqvae: VQModel = pipeline.vqvae

    ds_train = xr.open_zarr(TRAIN_PATH)
    X_lr = ds_train["X_lr"]
    X_lr_wind_dir_and_pressure = X_lr.sel(channel=WIND_DIR_AND_PRESSURE_CHANNELS)

    train_vqvae(vqvae, X_lr_wind_dir_and_pressure, device)