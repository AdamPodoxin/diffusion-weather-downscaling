import xarray as xr
import xbatcher as xb
import xbatcher.loaders.torch
import torch
from torch.utils.data import DataLoader
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline 
import numpy
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path


DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH / "train.zarr"

WIND_DIR_AND_PRESSURE_CHANNELS = \
[
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "mean_sea_level_pressure"
]

MODEL_ID = "CompVis/ldm-super-resolution-4x-openimages"


def main():
    ds_train = xr.open_zarr(TRAIN_PATH)

    # Training input are low resolution samples.
    X_lr = ds_train["X_lr"]
    
    # Selecting three channels for training: 
    # two directions of wind, and air pressure.
    X_lr_wind_dir_and_pressure = X_lr.sel(channel=WIND_DIR_AND_PRESSURE_CHANNELS)

    # X_bgen = xb.BatchGenerator(
    #     ds=X_lr_wind_dir_and_pressure,
    #     input_dims={
    #         "sample": X_lr_wind_dir_and_pressure["sample"].shape[0],
    #         "channel": X_lr_wind_dir_and_pressure["channel"].shape[0],
    #         "latitude_lr": X_lr_wind_dir_and_pressure["latitude_lr"].shape[0],
    #         "longitude_lr": X_lr_wind_dir_and_pressure["longitude_lr"].shape[0],
    #     },
    #     preload_batch=False,
    # )

    # Training output targets are high resolution samples.
    X_hr = ds_train["X_hr"]
    X_hr_wind_dir_and_pressure = X_hr.sel(channel=WIND_DIR_AND_PRESSURE_CHANNELS)

    # y_bgen = xb.BatchGenerator(
    #     ds=X_hr_wind_dir_and_pressure,
    #         input_dims={
    #             "sample": X_hr_wind_dir_and_pressure["sample"].shape[0],
    #             "channel": X_hr_wind_dir_and_pressure["channel"].shape[0],
    #             "latitude": X_hr_wind_dir_and_pressure["latitude"].shape[0],
    #             "longitude": X_hr_wind_dir_and_pressure["longitude"].shape[0],
    #         },
    #         preload_batch=False,
    # )

    # dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
        
    #     # prefetch_factor=3,
    #     # num_workers=4,
    #     # persistent_workers=True,
    #     # multiprocessing_context="forkserver",
    # )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)

    pipeline = LDMSuperResolutionPipeline.from_pretrained(MODEL_ID)
    pipeline = pipeline.to(device)

    loss_fn = torch.nn.MSELoss()

    for X, Y in zip(X_lr_wind_dir_and_pressure, X_hr_wind_dir_and_pressure):
        X_tensor = torch.from_numpy(X.values)
        input_tensor = torch.cat((X_tensor,)).unsqueeze(0)
        
        output: numpy.ndarray = pipeline(input_tensor, num_inference_steps=100, eta=1, output_type="np").images[0]
        output_tensor = torch.from_numpy(output).permute(2, 0, 1)

        Y_tensor = torch.from_numpy(Y.values)
        
        print(loss_fn(output_tensor, Y_tensor))


if __name__ == "__main__":
    main()
