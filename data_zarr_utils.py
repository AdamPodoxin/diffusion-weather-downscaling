import xarray as xr
import gcsfs
import matplotlib.pyplot as plt
import zarr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
import pandas as pd




def show_with_coastlines(patch: xr.DataArray | xr.Dataset, var_name, cmap='jet'):
    """Display a data patch with coastlines overlay.

    The routine understands both high‑resolution patches that use
    ``latitude``/``longitude`` coords and low‑resolution ones whose
    coordinates were renamed to ``latitude_lr``/``longitude_lr``.

    Parameters
    ----------
    patch : xarray.DataArray or Dataset
        Array containing a ``channel`` dimension or a two‑dimensional
        ``latitude``/``longitude`` grid.  If a dataset is provided the
        variable named by ``var_name`` is selected internally.
    var_name : str
        Channel name (e.g. ``'2m_temperature'``) to plot.  Ignored if
        ``patch`` already has no ``channel`` dimension.
    cmap : str, optional
        Matplotlib colormap.
    """

    # --- Plot with coastlines overlay ---
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=proj)

    # pick the right spatial coordinate names
    if "latitude" in patch.coords:
        lat_name = "latitude"
        lon_name = "longitude"
    elif "latitude_lr" in patch.coords:
        lat_name = "latitude_lr"
        lon_name = "longitude_lr"
    else:
        raise ValueError("patch does not contain recognised latitude/longitude coords")

    # select variable if needed
    arr = patch
    if "channel" in patch.dims:
        arr = patch.sel(channel=var_name)

    # 1) draw the field using its real lon/lat coords
    mesh = ax.pcolormesh(
        arr[lon_name].values,
        arr[lat_name].values,
        arr.values,
        transform=proj,
        shading="auto",
        cmap=cmap,
    )

    # 2) set view to exactly the patch bounds
    ax.set_extent([
        arr[lon_name].min(), arr[lon_name].max(),
        arr[lat_name].min(), arr[lat_name].max(),
    ], crs=proj)

    # 3) add land/coast outlines on top
    ax.coastlines(resolution="110m", linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="none")  # (optional, no fill)

    plt.colorbar(mesh, ax=ax, shrink=0.8, label=f"{var_name} (K)")
    ax.set_title(f"{var_name} with coastlines\n time={str(patch.time.values)}")

    plt.show()


def sample_times(ds, years, n, seed=0):
    #sample n random times from ds.time that fall within the specified years
    rng = np.random.default_rng(seed)
    times = pd.DatetimeIndex(ds.time.values)
    mask = times.year.isin(list(years))
    eligible = times[mask]
    chosen = rng.choice(eligible, size=n, replace=False)
    return pd.DatetimeIndex(chosen).sort_values()


def extract_hr_tensor(ds_subset, t, lat_min, lat_max, lon_min, lon_max):
    """Return a high-resolution tensor for a single timestamp.

    Parameters
    ----------
    ds_subset : xarray.Dataset
        Dataset already limited to the desired variables (i.e. subset outside).
    t : datetime-like
        Timestamp to select.
    lat_min, lat_max, lon_min, lon_max : float
        Patch bounds in degrees east.  Note that ``latitude`` in the ERA5
        store runs from 90->-90 so we slice with ``lat_max`` first.

    Returns
    -------
    xarray.DataArray
        Array of shape ``(channel, latitude, longitude)`` with coordinate
        values preserved.
    """
    # slice at the requested time and spatial bounds
    patch = ds_subset.sel(
        time=t,
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )

    x_hr = patch.to_array(dim="channel").transpose("channel", "latitude", "longitude")
    return x_hr  # keeps coords: channel names + lat/lon values


def downsample_mean(x_hr, factor=4):
    """Coarsen an HR tensor by averaging non-overlapping blocks.

    Parameters
    ----------
    x_hr : xarray.DataArray
        Input with dims ``(channel, latitude, longitude)``.
    factor : int, optional
        Downsampling factor along each spatial axis (default 4).

    Returns
    -------
    xarray.DataArray
        Coarsened array with the same ``channel`` dim.
    """
    return x_hr.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()


# The original ``write_split_to_zarr_hr_lr`` below remains for backwards
# compatibility but has some shortcomings (it re-subsamples the dataset and
# doesn't provide any logging).  A newer helper follows.

def write_split_to_zarr_hr_lr(ds_subset, times, bounds, out_path, batch_size=50, debug=False):
    """
    Improved writer that operates on an already‑subsetted dataset and
    optionally prints diagnostics useful for tracking down the mysterious
    160×160 patch issue.

    Parameters
    ----------
    ds_subset : xarray.Dataset
        Must contain the desired variables already.
    times : sequence
    bounds : tuple
        ``(lat_min, lat_max, lon_min, lon_max)`` in degrees east.
    out_path : str
    batch_size : int, optional
    debug : bool, optional
        When True, prints the spatial size and ranges of each patch plus the
        resulting x_hr shape.
    """
    lat_min, lat_max, lon_min, lon_max = bounds

    fs = gcsfs.GCSFileSystem()
    mapper = fs.get_mapper(out_path)

    first = True
    sample_index = 0
    # reference coords to enforce identical grids when concatenating/writing
    ref_hr_lat = None
    ref_hr_lon = None
    ref_lr_lat = None
    ref_lr_lon = None

    for i in range(0, len(times), batch_size):
        batch_times = times[i : i + batch_size]
        batch_hr = []
        batch_lr = []

        for t in batch_times:
            patch = ds_subset.sel(
                time=t,
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max),
            )
            if debug:
                print(f"time {t}: patch lat {patch.latitude.size} lon {patch.longitude.size}")
                print(f"   lat range {patch.latitude.min().item()}..{patch.latitude.max().item()}")
                print(f"   lon range {patch.longitude.min().item()}..{patch.longitude.max().item()}")
            x_hr = patch.to_array(dim="channel").transpose("channel", "latitude", "longitude")
            if debug:
                print(f"   x_hr.shape = {x_hr.shape}")
            x_lr = downsample_mean(x_hr, factor=4)

            # Rename LR coords to prevent xarray from aligning/unioning
            # spatial dimensions when assembling the Dataset
            x_lr = x_lr.rename({"latitude": "latitude_lr", "longitude": "longitude_lr"})

            # On first sample capture the reference coordinate arrays so
            # subsequent samples are reindexed to the exact same grid.
            if ref_hr_lat is None:
                ref_hr_lat = x_hr.coords["latitude"].values
                ref_hr_lon = x_hr.coords["longitude"].values
                ref_lr_lat = x_lr.coords["latitude_lr"].values
                ref_lr_lon = x_lr.coords["longitude_lr"].values
            else:
                # reindex to the reference grids to avoid xarray/zarr creating
                # the union of slightly-different coords across samples
                x_hr = x_hr.reindex(latitude=ref_hr_lat, longitude=ref_hr_lon)
                x_lr = x_lr.reindex(latitude_lr=ref_lr_lat, longitude_lr=ref_lr_lon)
            x_hr = x_hr.expand_dims(sample=[sample_index])
            x_lr = x_lr.expand_dims(sample=[sample_index])
            x_hr = x_hr.assign_coords(time=("sample", [np.datetime64(t)]))
            x_lr = x_lr.assign_coords(time=("sample", [np.datetime64(t)]))
            batch_hr.append(x_hr)
            batch_lr.append(x_lr)
            sample_index += 1

        X_hr = xr.concat(batch_hr, dim="sample")
        X_lr = xr.concat(batch_lr, dim="sample")
        ds_out = xr.Dataset({"X_hr": X_hr, "X_lr": X_lr})

        if first:
            ds_out.to_zarr(mapper, mode="w", consolidated=False, zarr_version=2)
            first = False
        else:
            ds_out.to_zarr(mapper, mode="a", append_dim="sample", consolidated=False, zarr_version=2)

    zarr.consolidate_metadata(mapper)

