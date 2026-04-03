import argparse
from pathlib import Path
import xarray as xr
from weather_downscaling_pipeline import WeatherLDMSuperResolutionPipeline
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Generate test outputs for evaluation")

    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/test.zarr"),
        help="The input test file",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/test_outputs/vanilla"),
        help="The output directory",
    )
    p.add_argument(
        "--vqvae",
        type=Path,
        default=Path("models/vqvae-trained-vanilla/vqvae-trained.pt"),
        help="Path to the VQVAE model to use",
    )
    p.add_argument(
        "--unet",
        type=Path,
        default=Path("models/unet-trained-vanilla/unet-trained-vanilla.pt"),
        help="Path to the UNet model to use",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of inference steps"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size (should ideally be factor of test set size)"
    )
    p.add_argument(
        "--noise-latents",
        action="store_true",
        help="If active, then use traditional LDM noise latents. Otherwise use the \"hack\" of interpolating then VQVAE encoding to get initial latents for better results."
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    ds_test = xr.open_zarr(args.input)
    X_lr_test = ds_test["X_lr"]

    if bool(args.noise_latents):
        print("Using traditional LDM noise latents")

    pipeline = WeatherLDMSuperResolutionPipeline(
        vqvae_path=Path(args.vqvae),
        unet_path=Path(args.unet),
        batch_size=int(args.batch_size),
        use_noise_latents=bool(args.noise_latents),
    )

    Y_denormalized, Y_normalized = pipeline(X_lr_test, args.num_steps)

    print("Saving to", str(output_dir))

    torch.save(
        obj=Y_denormalized,
        f=output_dir / "denormalized.pt"
    )
    torch.save(
        obj=Y_normalized,
        f=output_dir / "normalized.pt"
    )
