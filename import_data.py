import xarray as xr
import gcsfs
import zarr
import numpy as np
import pandas as pd


path_train = "gs://weather_bench_subset_hr_lr/train_split.zarr"
path_val = "gs://weather_bench_subset_hr_lr/val_split.zarr"
path_test = "gs://weather_bench_subset_hr_lr/test_split.zarr"

fs = gcsfs.GCSFileSystem(token="anon")
mapper_train = fs.get_mapper(path_train)
#mapper_val = fs.get_mapper(path_val)
#mapper_test = fs.get_mapper(path_test)

ds_train = xr.open_zarr(mapper_train) #training data set
#ds_val = xr.open_zarr(mapper_val) #validation data set
#ds_test = xr.open_zarr(mapper_test)# test data set

# Check contents of ds_train:
print(ds_train)
print(ds_train.X_hr)


# ds_--- are the datasets. To access individual samples, you can index along the 'sample' dimension. For example, to get the first sample:
sample_0_hr = ds_train["X_hr"].isel(sample=0)
sample_0_lr = ds_train["X_lr"].isel(sample=0)

#Each of these will be an xarray DataArray with dimensions (channel, latitude, longitude) and a coordinate 'time' 
# that gives the timestamp for that sample. You can check the variable names and dimensions like this:
print(sample_0_hr)
print(sample_0_lr)

#The channels correspond to: Channel=0='10m_u_component_of_wind', Channel=1='10m_v_component_of_wind', Channel=2='2m_temperature', Channel=3='mean_sea_level_pressure'. 
# You can access them by indexing the 'channel' dimension or by using the .sel method with the channel names if you have them as coordinates. For example:
sample_0_hr_u = sample_0_hr.isel(channel=0)  # 10m_u_component_of_wind
#or
sample_0_hr_u = sample_0_hr.sel(channel='10m_u_component_of_wind') 


#note that in these examples, the data is being loaded lazily (remotely), to actually load the data onto your machine do:

loaded_data = sample_0_hr.load()
#or if you only want to load a specific variable:
loaded_temp = sample_0_hr.sel(channel='2m_temperature').load()
