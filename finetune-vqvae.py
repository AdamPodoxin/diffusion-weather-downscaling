from pathlib import Path

from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import xarray as xr
from tqdm import tqdm

import math

from utils import (
    generate_batches, 
    normalize_across_channels, 
    save_checkpoint,
)


DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH / "train.zarr"
VAL_PATH = DATA_PATH / "val.zarr"

MODELS_DIR = Path("models")
VQVAE_DIR = MODELS_DIR / "vqvae-finetuned"

# Set to whatever GPU VRAM can handle, but 
# must be factor of number of samples.
BATCH_SIZE = 96

NUM_EPOCHS = 50

WIND_DIR_AND_PRESSURE_CHANNELS = \
[
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "mean_sea_level_pressure"
]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)
    
    vqvae = VQModel \
                .from_pretrained("CompVis/ldm-super-resolution-4x-openimages", subfolder="vqvae") \
                .to(device)

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
            path=VQVAE_DIR / f"vqvae-finetuned-{epoch}.pt"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

            print("Saving epoch", epoch, "as best model")
            save_checkpoint(
                model=vqvae,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                optimizer=optimizer,
                path=VQVAE_DIR / "vqvae-finetuned.pt"
            )
    
    print("Epoch", best_epoch, f"had lowest validation loss {best_val_loss:.2f}")
