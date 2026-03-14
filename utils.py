from pathlib import Path

import xarray as xr

from diffusers.models import ModelMixin

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
