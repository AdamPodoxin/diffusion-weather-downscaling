from pathlib import Path

from diffusers.models.unets.unet_2d import UNet2DOutput

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import xarray as xr
import pandas as pd
from tqdm import tqdm

import math

from utils import (
    load_model_state_dict,
    generate_batches,
    save_checkpoint,
    get_4channel_unet,
    get_lora_unet,
    get_4channel_vqvae,
)


# For CSIL
# DATA_PATH = Path("/usr/shared/CMPT/scratch/alp11/data/cmpt420/project")
DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH / "train.zarr"
VAL_PATH = DATA_PATH / "val.zarr"

MODELS_DIR = Path("models")
VQVAE_PATH = MODELS_DIR / "vqvae-trained" / "vqvae-trained.pt"
UNET_DIR = MODELS_DIR / "unet-trained-vanilla"
LOSSES_DIR = UNET_DIR / "losses"

# Set to whatever GPU VRAM can handle, but must be factor of 
# number of training samples AND number of validation samples.
BATCH_SIZE = 50

NUM_EPOCHS = 10
SAVE_EVERY_EPOCH = False


if __name__ == "__main__":
    UNET_DIR.mkdir(exist_ok=True)
    LOSSES_DIR.mkdir(exist_ok=True)

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

    # Calculate mean and std per channel for normalization
    print("Calculating LR means and stds")
    with torch.no_grad():
        # Not completely ideal because loading entire dataset into VRAM,
        # but it should fit for our data so it's fine for now. 
        temp_train_tensor = torch.from_numpy(X_lr_train.to_numpy()).to(device)
        height = temp_train_tensor.shape[2]
        width = temp_train_tensor.shape[3]
        
        # Calculate per each channel (dim 1), and
        # expand to size of batch tensor. 
        lr_means = temp_train_tensor \
                .mean(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1) \
                .expand(BATCH_SIZE, 4, height, width)
        
        lr_stds = temp_train_tensor \
                .std(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1) \
                .expand(BATCH_SIZE, 4, height, width)

    # Calculate mean and std per channel for normalization
    print("Calculating HR means and stds")
    with torch.no_grad():
        # Not completely ideal because loading entire dataset into VRAM,
        # but it should fit for our data so it's fine for now. 
        temp_train_tensor = torch.from_numpy(X_hr_train.to_numpy()).to(device)
        height = temp_train_tensor.shape[2]
        width = temp_train_tensor.shape[3]
        
        # Calculate per each channel (dim 1), and
        # expand to size of batch tensor. 
        hr_means = temp_train_tensor \
                .mean(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1) \
                .expand(BATCH_SIZE, 4, height, width)
        
        hr_stds = temp_train_tensor \
                .std(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1) \
                .expand(BATCH_SIZE, 4, height, width)

    del temp_train_tensor
    torch.cuda.empty_cache()

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
        X_normalized = (X - lr_means) / lr_stds

        Y = Y.to(device)
        Y_normalized = (Y - hr_means) / hr_stds

        with torch.no_grad():
            # Use VQVAE to compute "ideal" latents
            latents = vqvae.encode(Y_normalized).latents
        
        noise = torch.randn_like(latents)
        
        timesteps = torch.randint(
            low=0,
            high=noise_scheduler.config["num_train_timesteps"],
            size=(BATCH_SIZE,),
            device=device
        ).long()

        # Add noise to "ideal" latents to simulate various
        # levels that UNet will encounter during inference. 
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        unet_input = torch.cat([noisy_latents, X_normalized], dim=1)
        unet_output: UNet2DOutput = unet(unet_input, timesteps)
        predicted_noise = unet_output.sample

        loss = loss_fn(predicted_noise.float(), noise.float(), reduction="mean")

        return loss


    train_losses = [math.inf for _ in range(NUM_EPOCHS)]
    val_losses = [math.inf for _ in range(NUM_EPOCHS)]

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
        train_losses[epoch] = avg_train_loss

        num_val_batches = X_lr_val.sizes["sample"] // BATCH_SIZE
        avg_val_loss = 0

        with torch.no_grad():
            lr_val_batch_generator = generate_batches(X_lr_val, batch_size=BATCH_SIZE)
            hr_val_batch_generator = generate_batches(X_hr_val, batch_size=BATCH_SIZE)
            
            print("Validation:")
            for V_lr, V_hr in tqdm(zip(lr_val_batch_generator, hr_val_batch_generator), total=num_val_batches):
                loss_val = loop_logic(V_lr, V_hr)
                avg_val_loss += loss_val.item()

        avg_val_loss /= num_val_batches
        val_losses[epoch] = avg_val_loss

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

        loss_df = pd.DataFrame({
            "epoch": [epoch],
            "train loss": [avg_train_loss], 
            "validation loss": [avg_val_loss]
        })
        loss_df.to_csv(LOSSES_DIR / f"epoch-{epoch:03}.csv", index=False)

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

    losses_df = pd.DataFrame(
        data={
            "epoch": [epoch for epoch in range(NUM_EPOCHS)],
            "train loss": train_losses,
            "validation loss": val_losses,
        }
    )
    losses_df.to_csv(LOSSES_DIR / "losses.csv", index=False)
