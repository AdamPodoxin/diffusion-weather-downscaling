from pathlib import Path

from diffusers.models.autoencoders.vae import DecoderOutput

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import xarray as xr
import pandas as pd
from tqdm import tqdm

import math

from utils import (
    generate_batches, 
    save_checkpoint,
    get_4channel_vqvae,
)


# For CSIL
DATA_PATH = Path("/usr/shared/CMPT/scratch/alp11/data/cmpt420/project")
# DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH / "train.zarr"
VAL_PATH = DATA_PATH / "val.zarr"

MODELS_DIR = Path("models")
VQVAE_DIR = MODELS_DIR / "vqvae-trained"
LOSSES_DIR = VQVAE_DIR / "losses"

# Set to whatever GPU VRAM can handle, but must be factor of 
# number of training samples AND number of validation samples.
BATCH_SIZE = 16

NUM_EPOCHS = 20

SAVE_EVERY_EPOCH = False


if __name__ == "__main__":
    VQVAE_DIR.mkdir(exist_ok=True)
    LOSSES_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)
    
    vqvae = get_4channel_vqvae(device)

    reduce_dims = ("sample", "latitude", "longitude")

    ds_train = xr.open_zarr(TRAIN_PATH)
    X_hr_train = ds_train["X_hr"]
    train_means = X_hr_train.mean(dim=reduce_dims)
    train_stds = X_hr_train.std(dim=reduce_dims)

    ds_val = xr.open_zarr(VAL_PATH)
    X_hr_val = ds_val["X_hr"]
    val_means = X_hr_val.mean(dim=reduce_dims)
    val_stds = X_hr_val.std(dim=reduce_dims)

    loss_fn = torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    num_batches = X_hr_train.sizes["sample"] // BATCH_SIZE

    best_val_loss = math.inf
    best_epoch = 0

    def loop_logic(X: xr.DataArray, means: xr.DataArray, stds: xr.DataArray):
        X_normalized = (X - means) / stds
        X_tensor = torch.from_numpy(X_normalized.values).to(device)

        # Output is already normalized 
        output: DecoderOutput = vqvae(X_tensor)
        output_sample = output.sample

        loss = loss_fn(output_sample, X_tensor)
        return loss


    train_losses = [math.inf for _ in range(NUM_EPOCHS)]
    val_losses = [math.inf for _ in range(NUM_EPOCHS)]

    for epoch in range(NUM_EPOCHS):
        vqvae.train()
        print("\nEpoch", epoch)

        avg_train_loss = 0
        
        batch_generator = generate_batches(X_hr_train, batch_size=BATCH_SIZE)
        
        print("Training:")
        for X in tqdm(batch_generator, total=num_batches):
            loss = loop_logic(X, train_means, train_stds)
            avg_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        vqvae.eval()

        avg_train_loss /= num_batches
        train_losses[epoch] = avg_train_loss

        num_val_batches = X_hr_val.sizes["sample"] // BATCH_SIZE
        avg_val_loss = 0

        with torch.no_grad():
            val_batch_generator = generate_batches(X_hr_val, batch_size=BATCH_SIZE)
            
            print("Validation:")
            for V in tqdm(val_batch_generator, total=num_val_batches):
                loss_val = loop_logic(V, val_means, val_stds)
                avg_val_loss += loss_val.item()

        avg_val_loss /= num_val_batches
        val_losses[epoch] = avg_val_loss

        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")

        torch.cuda.empty_cache()
        
        scheduler.step()

        if SAVE_EVERY_EPOCH:
            save_checkpoint(
                model=vqvae,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                optimizer=optimizer,
                path=VQVAE_DIR / f"vqvae-trained-{epoch}.pt"
            )
        
        loss_df = pd.DataFrame({
            "train loss": avg_train_loss, 
            "validation loss": avg_val_loss
        })
        loss_df.to_csv(LOSSES_DIR / f"epoch-{epoch:03}.csv")

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
                path=VQVAE_DIR / "vqvae-trained.pt"
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
