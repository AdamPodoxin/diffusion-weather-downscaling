from pathlib import Path

from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline 
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models import ModelMixin

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Optimizer

import xarray as xr
from tqdm import tqdm

import math


DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH / "train.zarr"
VAL_PATH = DATA_PATH / "val.zarr"

MODELS_PATH = Path("models")
VQVAE_PATH = MODELS_PATH / "vqvae"

# Batch size set to whatever GPU VRAM can handle
BATCH_SIZE = 96

NUM_EPOCHS = 10

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


def save_checkpoint(
        model: ModelMixin, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        optimizer: Optimizer, 
        path: Path):
    
    checkpoint = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)
    
    pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
    pipeline = pipeline.to(device)

    vqvae: VQModel = pipeline.vqvae

    ds_train = xr.open_zarr(TRAIN_PATH)
    X_lr_train = ds_train["X_lr"]
    X_lr_train_subset = X_lr_train.sel(channel=WIND_DIR_AND_PRESSURE_CHANNELS)

    ds_val = xr.open_zarr(VAL_PATH)
    X_lr_val = ds_val["X_lr"]
    X_lr_val_subset = X_lr_val.sel(channel=WIND_DIR_AND_PRESSURE_CHANNELS)
    
    loss_fn = torch.nn.functional.l1_loss
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    num_batches = X_lr_train_subset.sizes["sample"] // BATCH_SIZE

    best_val_loss = math.inf
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        vqvae.train()
        print("\nEpoch", epoch)

        avg_train_loss = 0
        
        batch_generator = generate_batches(X_lr_train_subset, batch_size=BATCH_SIZE)
        
        print("Training:")
        for X in tqdm(batch_generator, total=num_batches):
            X = X.to(device)
            X_normalized, _, _ = normalize_across_channels(X)

            output: DecoderOutput = vqvae(X_normalized)
            output_sample = output.sample
            output_sample_normalized, _, _ = normalize_across_channels(output_sample)

            loss = loss_fn(output_sample_normalized, X_normalized)
            avg_train_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        vqvae.eval()

        avg_train_loss /= num_batches

        num_val_batches = X_lr_val_subset.sizes["sample"] // BATCH_SIZE
        avg_val_loss = 0

        with torch.no_grad():
            val_batch_generator = generate_batches(X_lr_val_subset, batch_size=BATCH_SIZE)
            
            print("Validation:")
            for V in tqdm(val_batch_generator, total=num_val_batches):
                V = V.to(device)
                V_normalized, _, _ = normalize_across_channels(V)

                output_val: DecoderOutput = vqvae(V_normalized)
                output_sample_val = output_val.sample
                output_sample_val_normalized, _, _ = normalize_across_channels(output_sample_val)

                loss_val = loss_fn(output_sample_val_normalized, V_normalized)
                avg_val_loss += loss_val.item()

        avg_val_loss /= num_val_batches

        print(f"Average training loss: {avg_train_loss:.2f}")
        print(f"Average validation loss: {avg_val_loss:.2f}")

        torch.cuda.empty_cache()
        
        scheduler.step()

        save_checkpoint(
            model=vqvae,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            optimizer=optimizer,
            path=VQVAE_PATH / f"vqvae-trained-{epoch}.pt"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_train_loss
            best_epoch = epoch

            print("Saving epoch", epoch, "as best model")
            save_checkpoint(
                model=vqvae,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                optimizer=optimizer,
                path=VQVAE_PATH / "vqvae-trained.pt"
            )
    
    print("Epoch", best_epoch, f"had lowest validation loss {best_val_loss:.2f}")
