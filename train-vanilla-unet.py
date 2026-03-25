from pathlib import Path

from diffusers.models.unets.unet_2d import UNet2DOutput

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import xarray as xr
from tqdm import tqdm

import math

from utils import (
    load_model_state_dict,
    generate_batches,
    normalize_across_channels,
    save_checkpoint,
    get_4channel_unet,
    get_lora_unet,
    get_4channel_vqvae,
)


# For CSIL
DATA_PATH = Path("/usr/shared/CMPT/scratch/alp11/data/cmpt420/project")
# DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH / "train.zarr"
VAL_PATH = DATA_PATH / "val.zarr"

MODELS_DIR = Path("models")
VQVAE_PATH = MODELS_DIR / "vqvae-trained" / "vqvae-trained.pt"
UNET_DIR = MODELS_DIR / "unet-trained-vanilla"

# Set to whatever GPU VRAM can handle, but must be factor of 
# number of training samples AND number of validation samples.
BATCH_SIZE = 100

NUM_EPOCHS = 50
SAVE_EVERY_EPOCH = False


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)

    vqvae = get_4channel_vqvae(device)

    # Need to use the fine-tuned VQVAE for generating latents
    vqvae_state_dict = load_model_state_dict(VQVAE_PATH)
    vqvae.load_state_dict(vqvae_state_dict)

    vqvae.eval()
    vqvae.requires_grad_(False)

    unet = get_lora_unet(get_4channel_unet(device))

    noise_scheduler = DDPMScheduler \
                .from_pretrained("CompVis/ldm-super-resolution-4x-openimages", subfolder="scheduler")

    ds_train = xr.open_zarr(TRAIN_PATH)
    X_lr_train = ds_train["X_lr"]
    X_hr_train = ds_train["X_hr"]

    ds_val = xr.open_zarr(VAL_PATH)
    X_lr_val = ds_val["X_lr"]
    X_hr_val = ds_val["X_hr"]

    loss_fn = torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam(
        [p for p in unet.parameters() if p.requires_grad], 
        lr=2e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    num_batches = X_lr_train.sizes["sample"] // BATCH_SIZE

    best_val_loss = math.inf
    best_epoch = 0

    def loop_logic(X: torch.Tensor, Y: torch.Tensor):
        X = X.to(device)
        X_normalized, _, _ = normalize_across_channels(X)

        Y = Y.to(device)
        Y_normalized, _, _ = normalize_across_channels(Y)

        with torch.no_grad():
            latents = vqvae.encode(Y_normalized).latents
        
        noise = torch.randn_like(latents)
        
        timesteps = torch.randint(
            low=0,
            high=noise_scheduler.config["num_train_timesteps"],
            size=(BATCH_SIZE,),
            device=device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        unet_input = torch.cat([noisy_latents, X_normalized], dim=1)

        unet_output: UNet2DOutput = unet(unet_input, timesteps)
        predicted_noise = unet_output.sample

        loss = loss_fn(predicted_noise.float(), noise.float(), reduction="mean")

        return loss


    for epoch in range(NUM_EPOCHS):
        unet.train()
        print("\nEpoch", epoch)

        avg_train_loss = 0
        
        lr_batch_generator = generate_batches(X_lr_train, batch_size=BATCH_SIZE)
        hr_batch_generator = generate_batches(X_hr_train, batch_size=BATCH_SIZE)
        
        print("Training:")
        for X, Y in tqdm(zip(lr_batch_generator, hr_batch_generator), total=num_batches):
            loss = loop_logic(X, Y)
            avg_train_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        unet.eval()

        avg_train_loss /= num_batches

        num_val_batches = X_lr_val.sizes["sample"] // BATCH_SIZE
        avg_val_loss = 0

        with torch.no_grad():
            lr_val_batch_generator = generate_batches(X_lr_val, batch_size=BATCH_SIZE)
            hr_val_batch_generator = generate_batches(X_hr_val, batch_size=BATCH_SIZE)
            
            print("Validation:")
            for V_lr, V_hr in tqdm(zip(lr_val_batch_generator, hr_val_batch_generator), total=num_val_batches):
                loss_val = loop_logic(V_lr, V_hr)
                avg_val_loss += loss_val.item()

        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")

        torch.cuda.empty_cache()
        
        scheduler.step()

        if SAVE_EVERY_EPOCH:
            save_checkpoint(
                model=unet,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                optimizer=optimizer,
                path=UNET_DIR / f"unet-trained-vanilla-{epoch}.pt"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

            print("Saving epoch", epoch, "as best model")
            save_checkpoint(
                model=unet,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                optimizer=optimizer,
                path=UNET_DIR / "unet-trained-vanilla.pt"
            )
    
    print("Epoch", best_epoch, f"had lowest validation loss {best_val_loss:.4f}")
