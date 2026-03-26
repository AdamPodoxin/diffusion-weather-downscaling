from pathlib import Path

import xarray as xr

from diffusers.models import ModelMixin

from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.unets.unet_2d import UNet2DModel

from peft import LoraConfig, get_peft_model

import torch
from torch.optim import Optimizer


def generate_batches(data: xr.DataArray, batch_size=32):
    for i in range(0, data.sizes["sample"], batch_size):
        batch = data.isel(sample=slice(i, i + batch_size))
        yield batch


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


def get_4channel_unet(device: str, num_latents=4):
    config: dict[str] = UNet2DModel.load_config(
        "CompVis/ldm-super-resolution-4x-openimages", 
        subfolder="unet"
    )

    config["in_channels"] = 4 + num_latents
    config["out_channels"] = 4

    unet = UNet2DModel \
            .from_config(config) \
            .to(device)
    
    return unet


def get_lora_unet(base_unet: UNet2DModel):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0", # Attention
            "conv1", "conv2", "conv_shortcut"   # ResNet Convolutions
        ],
        lora_dropout=0.05,
    )

    lora_unet = get_peft_model(base_unet, lora_config)
    return lora_unet
