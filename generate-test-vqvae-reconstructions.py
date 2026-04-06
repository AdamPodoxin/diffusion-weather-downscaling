import argparse
from pathlib import Path
import xarray as xr
import torch
from diffusers.models.autoencoders.vae import DecoderOutput
from utils import get_4channel_vqvae, load_model_state_dict, generate_batches
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate VQVAE reconstructions for test data for evaluation")

    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/test.zarr"),
        help="The input test file",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/test_vqvae_reconstructions/vanilla"),
        help="The output directory",
    )
    p.add_argument(
        "--vqvae",
        type=Path,
        default=Path("models/vqvae-trained-vanilla/vqvae-trained.pt"),
        help="Path to the VQVAE model to use",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Batch size (should ideally be factor of test set size)"
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    batch_size = int(args.batch_size)

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    ds_test = xr.open_zarr(args.input)
    X_hr_test = ds_test["X_hr"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vqvae = get_4channel_vqvae(device)
    vqvae_state_dict = load_model_state_dict(args.vqvae)
    vqvae.load_state_dict(vqvae_state_dict)
    vqvae.eval()

    with torch.no_grad():
        temp_train_tensor = torch.from_numpy(X_hr_test.to_numpy()).to(device)
        height = temp_train_tensor.shape[2]
        width = temp_train_tensor.shape[3]
        
        means = temp_train_tensor \
                .mean(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1) \
                .expand(batch_size, 4, height, width)
        
        stds = temp_train_tensor \
                .std(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1) \
                .expand(batch_size, 4, height, width)

    del temp_train_tensor
    torch.cuda.empty_cache()

    num_batches = X_hr_test.sizes["sample"] // batch_size
    outputs_normalized = [None for _ in range(num_batches)]
    outputs_denormalized = [None for _ in range(num_batches)]

    batch_generator = generate_batches(X_hr_test, batch_size)

    for i, X in tqdm(enumerate(batch_generator), total=num_batches):
        with torch.no_grad():
            X = X.to(device)
            X_normalized = (X - means) / stds

            out: DecoderOutput = vqvae(X_normalized)
            outputs_normalized[i] = out.sample
            outputs_denormalized[i] = out.sample * stds + means

        torch.cuda.empty_cache()

    Y_normalized = torch.cat(outputs_normalized)
    del outputs_normalized
    torch.cuda.empty_cache()

    Y_denormalized = torch.cat(outputs_denormalized)
    del outputs_denormalized
    torch.cuda.empty_cache()

    print("Saving to", str(output_dir))

    torch.save(
        obj=Y_denormalized,
        f=output_dir / "denormalized.pt"
    )
    torch.save(
        obj=Y_normalized,
        f=output_dir / "normalized.pt"
    )
