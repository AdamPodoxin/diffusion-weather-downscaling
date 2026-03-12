from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline 
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput
import xarray as xr
import torch
from pathlib import Path
from tqdm import tqdm


DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH / "train.zarr"

MODELS_PATH = Path("models")
VQVAE_PATH = MODELS_PATH / "vqvae"

WIND_DIR_AND_PRESSURE_CHANNELS = \
[
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "mean_sea_level_pressure"
]


def train_vqvae(vqvae: VQModel, low_res_inputs: xr.DataArray, device: str):
    # The paper used cosine annealing LR scheduler
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=2e-4)

    vqvae.train()

    # TODO: temp, remove
    low_res_inputs = low_res_inputs.isel(sample=list(range(200)))

    for i, X in tqdm(enumerate(low_res_inputs), total=low_res_inputs.shape[0]):
        X_tensor = torch.from_numpy(X.values) \
                        .to(device) \
                        .unsqueeze(0)

        output: DecoderOutput = vqvae(X_tensor)
        output_sample = output.sample

        loss_fn = torch.nn.functional.mse_loss(output_sample, X_tensor)

        loss = loss_fn

        if i % 200 == 0:
            print("Batch", i, ": Loss:", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.save(vqvae.state_dict(), VQVAE_PATH / "vqvae-trained.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)
    
    pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
    pipeline = pipeline.to(device)

    vqvae: VQModel = pipeline.vqvae

    ds_train = xr.open_zarr(TRAIN_PATH)
    X_lr = ds_train["X_lr"]
    X_lr_wind_dir_and_pressure = X_lr.sel(channel=WIND_DIR_AND_PRESSURE_CHANNELS)

    train_vqvae(vqvae, X_lr_wind_dir_and_pressure, device)