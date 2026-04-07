import argparse
import torch
from torch.nn.functional import mse_loss
import xarray as xr
import pandas as pd
from pathlib import Path
from isotropic_fpsd_loss import calculate_batch_dx, isotropic_psd_loss


def parse_args():
    p = argparse.ArgumentParser(description="Calculate pipeline output or VQVAE reconstruction test losses for evaluation")

    p.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/test_outputs/vanilla"),
        help="Directory containing pipeline test outputs",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/losses/vanilla.csv"),
        help="Output losses file",
    )
    p.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/test.zarr"),
        help="File containing ground truth HR output",
    )
    p.add_argument(
        "--max-lat",
        type=float,
        default=54.75,
        help="Maximum latitude of test data",
    )
    p.add_argument(
        "--min-lat",
        type=float,
        default=23.0,
        help="Minimum latitude of test data",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ds_test = xr.open_zarr(args.test_file)
    X_hr = ds_test["X_hr"]

    target = torch.from_numpy(X_hr.values).to(device)
    
    num_samples = target.shape[0]
    height = target.shape[2]
    width = target.shape[3]

    with torch.no_grad():
        target_means = target \
                    .mean(dim=(0, 2, 3)) \
                    .view(1, 4, 1, 1) \
                    .expand(num_samples, 4, height, width)
        target_stds = target \
                    .std(dim=(0, 2, 3)) \
                    .view(1, 4, 1, 1) \
                    .expand(num_samples, 4, height, width)

        target_normalized = (target - target_means) / target_stds

    channel_names = list(X_hr["channel"].values)

    predicted_denormalized: torch.Tensor = torch.load(Path(args.input) / "denormalized.pt").to(device)
    predicted_normalized: torch.Tensor = torch.load(Path(args.input) / "normalized.pt").to(device)

    with torch.no_grad():
        mse_losses_denormalized = [
            [
                mse_loss(
                    input=predicted_denormalized[i, c, :, :],
                    target=target[i, c, :, :],
                ).item()
                for c in range(len(channel_names))
            ]
            for i in range(num_samples)
        ]

    mse_losses_denormalized_df = pd.DataFrame([
        {
            "sample": i,
            "channel": channel_names[c],
            "normalized": "denormalized",
            "loss_type": "MSE",
            "loss_value": loss,
        }
        for i, losses in enumerate(mse_losses_denormalized)
        for c, loss in enumerate(losses)
    ])

    with torch.no_grad():
        mse_losses_normalized = [
            [
                mse_loss(
                    input=predicted_normalized[i, c, :, :],
                    target=target_normalized[i, c, :, :],
                ).item()
                for c in range(len(channel_names))
            ]
            for i in range(num_samples)
        ]

    mse_losses_normalized_df = pd.DataFrame([
        {
            "sample": i,
            "channel": channel_names[c],
            "normalized": "normalized",
            "loss_type": "MSE",
            "loss_value": loss,
        }
        for i, losses in enumerate(mse_losses_normalized)
        for c, loss in enumerate(losses)
    ])

    center_lat = (float(args.max_lat) + float(args.min_lat)) / 2.0
    center_lats = torch.ones(1).to(device) * center_lat
    dx = calculate_batch_dx(center_lats)

    with torch.no_grad():
        psd_losses_denormalized = [
            [
                isotropic_psd_loss(
                    pred=predicted_denormalized[i:(i + 1), c:(c + 1), :, :],
                    target=target[i:(i + 1), c:(c + 1), :, :],
                    dx=dx,
                ).item()
                for c in range(len(channel_names))
            ]
            for i in range(num_samples)
        ]

    psd_losses_denormalized_df = pd.DataFrame([
        {
            "sample": i,
            "channel": channel_names[c],
            "normalized": "denormalized",
            "loss_type": "PSD",
            "loss_value": loss,
        }
        for i, losses in enumerate(psd_losses_denormalized)
        for c, loss in enumerate(losses)
    ])

    with torch.no_grad():
        psd_losses_normalized = [
            [
                isotropic_psd_loss(
                    pred=predicted_normalized[i:(i + 1), c:(c + 1), :, :],
                    target=target[i:(i + 1), c:(c + 1), :, :],
                    dx=dx,
                ).item()
                for c in range(len(channel_names))
            ]
            for i in range(num_samples)
        ]

    psd_losses_normalized_df = pd.DataFrame([
        {
            "sample": i,
            "channel": channel_names[c],
            "normalized": "normalized",
            "loss_type": "PSD",
            "loss_value": loss,
        }
        for i, losses in enumerate(psd_losses_normalized)
        for c, loss in enumerate(losses)
    ])

    full_df = pd.concat([
        mse_losses_denormalized_df,
        mse_losses_normalized_df,
        psd_losses_denormalized_df,
        psd_losses_normalized_df,
    ])

    full_df.to_csv(args.output, index=False)
