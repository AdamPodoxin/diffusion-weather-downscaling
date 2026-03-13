import xarray as xr
from zarr.codecs import BloscCodec
from pathlib import Path
import gcsfs


PATH_TRAIN = "gs://weather_bench_subset_hr_lr/train_split.zarr"
PATH_VAL = "gs://weather_bench_subset_hr_lr/val_split.zarr"
PATH_TEST = "gs://weather_bench_subset_hr_lr/test_split.zarr"

DATA_PATH = Path("data")


def load_data_from_google(path: str):
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper_train = fs.get_mapper(path)
    ds = xr.open_zarr(mapper_train)
    return ds


def main():
    compressor = BloscCodec()

    encoding = {
        "sample": {"compressors": (compressor,)},
        "time": {"compressors": (compressor,)},
        "channel": {"compressors": (compressor,)},
        "latitude": {"compressors": (compressor,)},
        "longitude": {"compressors": (compressor,)},
        "latitude_lr": {"compressors": (compressor,)},
        "longitude_lr": {"compressors": (compressor,)},
        "X_hr": {"compressors": (compressor,)},
        "X_lr": {"compressors": (compressor,)},
    }

    local_train_path = DATA_PATH / "train.zarr"
    if not local_train_path.exists():
        print("Downloading train split...")
        ds_train = load_data_from_google(PATH_TRAIN)
        ds_train.to_zarr(local_train_path, mode="w", encoding=encoding)
    
    local_val_path = DATA_PATH / "val.zarr"
    if not local_val_path.exists():
        print("Downloading validation split...")
        ds_val = load_data_from_google(PATH_VAL)
        ds_val.to_zarr(local_val_path, mode="w", encoding=encoding)

    local_test_path = DATA_PATH / "test.zarr"
    if not local_test_path.exists():
        print("Downloading test split...")
        ds_test = load_data_from_google(PATH_TEST)
        ds_test.to_zarr(local_test_path, mode="w", encoding=encoding)


if __name__ == "__main__":
    main()