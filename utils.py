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

    vqvae: VQModel = VQModel \
        .from_config(config) \
        .to(device)

    return vqvae


def get_4channel_unet(device: str, num_latents=3):
    # TODO: make this work from config too
    unet = UNet2DModel \
            .from_pretrained("CompVis/ldm-super-resolution-4x-openimages", subfolder="unet") \
            .to(device)
    
    original_input_layer = unet.conv_in
    unet.conv_in = torch.nn.Conv2d(
        in_channels=num_latents + 4,
        out_channels=original_input_layer.out_channels,
        kernel_size=original_input_layer.kernel_size,
        stride=original_input_layer.stride,
        padding=original_input_layer.padding,
    ).to(device)

    original_output_layer = unet.conv_out
    unet.conv_out = torch.nn.Conv2d(
        in_channels=original_output_layer.in_channels,
        out_channels=4,
        kernel_size=original_output_layer.kernel_size,
        stride=original_output_layer.stride,
        padding=original_output_layer.padding,
    ).to(device)

    return unet
