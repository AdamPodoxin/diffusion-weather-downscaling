import xarray as xr
import gcsfs
import xbatcher as xb
import xbatcher.loaders.torch
import torch
from torch.utils.data import DataLoader
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline 
import numpy
from torchvision.transforms.functional import to_pil_image
from PIL import Image


PATH_TRAIN = "gs://weather_bench_subset_hr_lr/train_split.zarr"
PATH_VAL = "gs://weather_bench_subset_hr_lr/val_split.zarr"
PATH_TEST = "gs://weather_bench_subset_hr_lr/test_split.zarr"

WIND_DIR_AND_PRESSURE_CHANNELS = \
[
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "mean_sea_level_pressure"
]

MODEL_ID = "CompVis/ldm-super-resolution-4x-openimages"


def load_data_from_google(path: str):
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper_train = fs.get_mapper(path)
    ds = xr.open_zarr(mapper_train)
    return ds


def main():
    ds_train = load_data_from_google(PATH_TRAIN)

    # Training input are low resolution samples.
    X_lr = ds_train["X_lr"]
    
    # Selecting three channels for training: 
    # two directions of wind, and air pressure.
    X_lr_wind_dir_and_pressure = X_lr.sel(channel=WIND_DIR_AND_PRESSURE_CHANNELS)

    X_bgen = xb.BatchGenerator(
        ds=X_lr_wind_dir_and_pressure,
        input_dims={
            "sample": len(X_lr_wind_dir_and_pressure["sample"]),
            "channel": len(X_lr_wind_dir_and_pressure["channel"]),
            "latitude_lr": len(X_lr_wind_dir_and_pressure["latitude_lr"]),
            "longitude_lr": len(X_lr_wind_dir_and_pressure["longitude_lr"]),
        },
        preload_batch=False,
    )

    # Training output targets are high resolution samples.
    X_hr = ds_train["X_hr"]
    X_hr_wind_dir_and_pressure = X_hr.sel(channel=WIND_DIR_AND_PRESSURE_CHANNELS)

    y_bgen = xb.BatchGenerator(
    ds=X_hr_wind_dir_and_pressure,
        input_dims={
            "sample": len(X_hr_wind_dir_and_pressure["sample"]),
            "channel": len(X_hr_wind_dir_and_pressure["channel"]),
            "latitude": len(X_hr_wind_dir_and_pressure["latitude"]),
            "longitude": len(X_hr_wind_dir_and_pressure["longitude"]),
        },
        preload_batch=False,
    )

    dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)

    train_dataloader = DataLoader(
        dataset,
        batch_size=None,
        prefetch_factor=3,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="forkserver",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)

    pipeline = LDMSuperResolutionPipeline.from_pretrained(MODEL_ID)
    pipeline = pipeline.to(device)

    loss_fn = torch.nn.MSELoss()

    X0 = X_lr_wind_dir_and_pressure.isel(sample=0).load()
    y0 = X_hr_wind_dir_and_pressure.isel(sample=0).load()

    X0_tensor = torch.from_numpy(X0.values)
    y0_tensor = torch.from_numpy(y0.values).permute(1, 2, 0)

    input_tensor = torch.cat((X0_tensor,)).unsqueeze(0)
    output: numpy.ndarray = pipeline(input_tensor, num_inference_steps=100, eta=1, output_type="np").images[0]
    output_tensor = torch.from_numpy(output)
    print(output)
    print("output shape:", output.shape)
    print("y0 shape:", y0_tensor.shape)

    print(loss_fn(output_tensor, y0_tensor))

    output_image: Image = to_pil_image(output_tensor.permute(2, 0, 1), "RGB")
    y0_image: Image = to_pil_image(y0_tensor.permute(2, 0, 1), "RGB")

    output_image.save("output.png")
    y0_image.save("y0.png")

    # TODO: fix this hanging
    for batch, (X, y) in enumerate(train_dataloader):
        input_tensor = torch.cat((X,)).unsqueeze(0)
        output: numpy.ndarray = pipeline(input_tensor, num_inference_steps=100, eta=1, output_type="np").images[0]
        print(output)



if __name__ == "__main__":
    main()
