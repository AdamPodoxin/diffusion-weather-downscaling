import argparse
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
BATCH_SIZE = 20

NUM_EPOCHS = 20

SAVE_EVERY_EPOCH = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--continue-checkpoint", action="store_true", help="Continue from last checkpoint")

    args = parser.parse_args()

    continue_from_checkpoint: bool = args.continue_checkpoint

    if continue_from_checkpoint:
        checkpoint: dict = torch.load(VQVAE_DIR / "vqvae-trained.pt")
        epoch: int = checkpoint["epoch"]
        last_epoch = epoch
        print("Continuing from epoch", epoch)
    else:
        epoch = 0
        last_epoch = -1

    VQVAE_DIR.mkdir(exist_ok=True)
    LOSSES_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)
    
    vqvae = get_4channel_vqvae(device)

    reduce_dims = ("sample", "latitude", "longitude")

    ds_train = xr.open_zarr(TRAIN_PATH)
    X_hr_train = ds_train["X_hr"]

    # Calculate mean and std per channel for normalization
    print("Calculating dataset means and stds")
    with torch.no_grad():
        # Not completely ideal because loading entire dataset into VRAM,
        # but it should fit for our data so it's fine for now. 
        temp_train_tensor = torch.from_numpy(X_hr_train.to_numpy()).to(device)
        height = temp_train_tensor.shape[2]
        width = temp_train_tensor.shape[3]
        
        # Calculate per each channel (dim 1), and
        # expand to size of batch tensor. 
        means = temp_train_tensor \
                .mean(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1) \
                .expand(BATCH_SIZE, 4, height, width)
        
        stds = temp_train_tensor \
                .std(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1) \
                .expand(BATCH_SIZE, 4, height, width)

    del temp_train_tensor
    torch.cuda.empty_cache()

    ds_val = xr.open_zarr(VAL_PATH)
    X_hr_val = ds_val["X_hr"]

    loss_fn = torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=2e-4)

    if continue_from_checkpoint:
        vqvae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, last_epoch=last_epoch)

    num_batches = X_hr_train.sizes["sample"] // BATCH_SIZE

    if continue_from_checkpoint:
        best_val_loss: float = checkpoint["val_loss"]
        best_epoch: int = epoch
    else:
        best_val_loss = math.inf
        best_epoch = 0

    def loop_logic(X: torch.Tensor):
        X = X.to(device)
        X_normalized = (X - means) / stds

        # Output sample is already normalized, so don't need to re-normalize
        output: DecoderOutput = vqvae(X_normalized)
        output_sample = output.sample
        commit_loss = output.commit_loss

        recon_loss = loss_fn(output_sample, X_normalized)
        loss = recon_loss + commit_loss
        return loss


    train_losses = [math.inf for _ in range(NUM_EPOCHS)]
    val_losses = [math.inf for _ in range(NUM_EPOCHS)]

    if continue_from_checkpoint:
        # Load losses from files
        for e in range(epoch):
            loss_df = pd.read_csv(LOSSES_DIR / f"epoch-{e:03}.csv")
            train_losses[e] = loss_df["train loss"].iloc[0]
            val_losses[e] = loss_df["validation loss"].iloc[0]

    print("Starting training")
    for epoch in range(epoch, NUM_EPOCHS):
        vqvae.train()
        print("\nEpoch", epoch)

        avg_train_loss = 0
        
        batch_generator = generate_batches(X_hr_train, batch_size=BATCH_SIZE)
        
        print("Training:")
        for X in tqdm(batch_generator, total=num_batches):
            loss = loop_logic(X)
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
                loss_val = loop_logic(V)
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
