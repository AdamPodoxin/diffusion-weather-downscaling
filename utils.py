from pathlib import Path

import xarray as xr

from diffusers.models import ModelMixin

from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.unets.unet_2d import UNet2DModel

import torch
from torch.optim import Optimizer


def z_normalize_tensor(tensor: torch.Tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor, mean, std


def z_denormalize_tensor(normalized_tensor: torch.Tensor, 
                         mean: torch.Tensor, 
                         std: torch.Tensor):
    return normalized_tensor * std + mean


def normalize_across_channels(X: torch.Tensor, channel_dim=1):
    num_channels = X.shape[channel_dim]

    channel_means = [torch.tensor(0) for _ in range(num_channels)]
    channel_stds = [torch.tensor(0) for _ in range(num_channels)]
    
    X_normalized = torch.empty_like(X)

    for channel in range(num_channels):
        X_normalized[:, channel, :, :], mean, std = z_normalize_tensor(X[:, channel, :, :])
        
        channel_means[channel] = mean
        channel_stds[channel] = std

    return X_normalized, channel_means, channel_stds


def denormalize_across_channels(X_normalized: torch.Tensor, 
                                means: list[torch.Tensor],
                                stds: list[torch.Tensor],
                                channel_dim=1):
    
    num_channels = X_normalized.shape[channel_dim]

    X = torch.empty_like(X_normalized)
    
    for channel in range(num_channels):
        X[:, channel, :, :] = z_denormalize_tensor(
                                X_normalized[:, channel, :, :], 
                                means[channel],
                                stds[channel]
                                )
    
    return X



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


def load_model_state_dict(model_file_path: Path):
    model_dict = torch.load(model_file_path)
    return model_dict["model_state_dict"]


def get_4channel_vqvae(device: str):
    config: dict[str] = VQModel.load_config(
        "CompVis/ldm-super-resolution-4x-openimages", 
        subfolder="vqvae"
    )

    config["in_channels"] = 4
    config["out_channels"] = 4
    config["latent_channels"] = 4
    config["vq_embed_dim"] = 4

    vqvae = VQModel \
        .from_config(config) \
        .to(device)

    return vqvae


def get_4channel_unet(device: str, num_latents=3):
    config: dict[str] = UNet2DModel.load_config(
        "CompVis/ldm-super-resolution-4x-openimages", 
        subfolder="unet"
    )

    config["in_channels"] = 8 # 4 image channels, 4 latent channels
    config["out_channels"] = 4

    unet = UNet2DModel \
            .from_config(config) \
            .to(device)
    
    return unet
