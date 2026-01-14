import glob
import io
import itertools
import json
import math
import os
import pickle
import re
import sys
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path

import bottleneck
import markdown
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
import zarr
from plotly.express.colors import sample_colorscale
from tqdm.notebook import tqdm

def wind_cartesian_to_polar(u, v):
    ws = (u**2 + v**2) ** 0.5
    wd = np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)

    return ws, wd


def wind_polar_to_cartesian(ws, wd):
    theta = np.deg2rad(wd)
    U = np.abs(ws)  # t.b.d. rotate ws<0
    u = -1 * U * np.sin(theta)
    v = -1 * U * np.cos(theta)

    return u, v


def xr_filter_spikes(
    da,
    kernel={},
    filter_bnds=slice(-np.inf, np.inf),
):
    """Despike mask and filter for xarray."""
    import numpy as np
    import xarray as xr
    from scipy.ndimage import generic_filter

    filter_1d = lambda x: (
        (np.ediff1d(x) < filter_bnds.start) | (np.ediff1d(x) >= filter_bnds.stop)
    ).any()

    kernel_nd = np.ones(tuple([kernel[n] if n in kernel else 1 for n in da.dims]))
    filter_nd = lambda x: generic_filter(
        x, filter_1d, footprint=kernel_nd, mode="reflect"
    )

    mask_nd = xr.apply_ufunc(
        filter_nd,
        da,
        input_core_dims=[[]],
        output_core_dims=[[]],
    ).astype(np.bool_)

    result_nd = da.where(~mask_nd)

    return result_nd, mask_nd



def xr_coords_set(dv):
    """collect coodinate sets"""
    from collections import defaultdict

    def get_coord_set(dv):
    
        from collections import defaultdict
    
        dv_set = defaultdict(list)
        for n in [(n[1].split("_")[0], n) for n in dv]:
            dv_set[n[0]].append(n[1])

        return dict(dv_set)    
    
    dv_set = defaultdict(list)
    for n in dv:
        k = None
        if isinstance(n, tuple):
            d = get_coord_set([n[-1]])
            k = list(d.keys())[0]
            v = n
        if isinstance(n, str):
            k = n.split("_")[0]
            v = n
        if not k == v:
            dv_set[k].append(v)

    return dict(dv_set)


def xr_reindex(ds):
    """makes indexes from coordinate sets"""
    ds = ds.set_index(
        {
            k: [n for n in v if len(ds[n].dims) == 1]
            for k, v in xr_coords_set(ds.coords).items()
            if len(v) > 1
        }
    )
    dc = {
        k: [n for n in v if len(ds[n].dims) == 1][0]
        for k, v in xr_coords_set(ds.coords).items()
        if len(v) == 1 and not k in ds.indexes
    }
    for k,v in dc.items():
        ds = ds.set_index({k:v}).rename_vars({k:v})
    return ds


def input_files(path_list, filename_pattern):
    """read production files"""
    fn = filename_pattern if isinstance(filename_pattern,list) else [filename_pattern]
    fp = path_list if isinstance(path_list,list) else [path_list]
    fp = [path for path in fp if os.path.exists(path)]

    fn_list = []
    for p in fp:
        for f in fn:
            fn_list.extend(
                glob.glob(
                    p + "**/" + f,
                    recursive=True,
                )
            )
    fn_list = sorted(fn_list)

    if not fp:
        print("Warning: File path does not exist!")
        fn_list, fn_dict = [], {}
    elif fn_list:
        # create groups
        fn_dict = defaultdict(list)
        for fn in fn_list:
            fn_dict[str(Path(fn).parent)].append(fn)
        fn_dict = dict(fn_dict)
    else:
        print("Warning: No files found!")
        fn_list, fn_dict = [], {}

    print(f"Found {len(fn_list)} files")

    return (fn_list, fn_dict)


def localstore(filepath, **kwargs):
    """local datastore handler (not for S3/HTTP)"""

    def test_zip(filepath):
        import sys
        import zipfile

        try:
            fz = zipfile.ZipFile(filepath)
            ret = fz.testzip()
            if ret is not None:
                return False
            else:
                return True
        except Exception as ex:
            # print("Exception:", ex)
            return False

    from pathlib import Path
    from urllib.parse import unquote, urlparse

    import fsspec
    import xarray as xr
    import zarr

    filepath = str(Path(filepath).resolve())  # fsspec required

    if test_zip(filepath):
        fileuri = unquote(f"zip::{str(Path(filepath).as_uri())}")
        return xr.open_dataset(fileuri, engine="zarr", **kwargs)
    else:
        return xr.open_dataset(filepath, engine="netcdf4", **kwargs)


def datastore(fn_dict):
    if not isinstance(fn_dict, dict):
        fn_dict = {None, fn_dict}

    # 1 of 2: Read and concatenate
    ds_dict = {}
    ga_dict = {}

    for fg, fns in tqdm(fn_dict.items()):
        ds_list = []
        for fn in fns:
            # netcdf lock workaround
            with localstore(fn, decode_timedelta=True) as dx:
                ds1 = dx
                dx.close()
            ds_list.append(ds1)
        ds = xr.concat(ds_list, dim="time")
        ds = ds.drop_duplicates("time", keep="last")
        try:
            ds = xr_reindex(ds)  # reindex, prepare for merge
        except:
            pass
        ds_dict[fg] = ds
        ga_dict[fg] = ds.attrs

    # 2 of 2: Combine, merge and/or join
    ds = xr.combine_by_coords(ds_dict.values(), combine_attrs="drop_conflicts")
    return ds, ga_dict, ds_dict


def plot_da(da, **kwargs):
    """a plot template"""
    import plotly.express as px

    px_opts = dict(
        range_color=[-2, 2],
        color_continuous_scale=px.colors.sequential.RdBu_r[1:-1],
    )
    px_opts = {**px_opts, **kwargs}

    fig = (
        px.imshow(
            da,
            origin="lower",
            aspect=None,
            **px_opts,
        )
        .update_layout(paper_bgcolor="lightgray", plot_bgcolor="black")
        .update_layout(width=1100, height=400)
        .update_layout(margin=dict(l=50, r=200, b=50, t=30, autoexpand=False))
    )
    return fig


def ds_plot(plot_ds, **kwargs):
    """convencience plot"""

    data_var = kwargs["data_var"] if "data_var" in kwargs else "ws"
    ycoord = kwargs["ycoord"] if "ycoord" in kwargs else ("cell_id", "cell_id")

    sel = (
        kwargs["sel"]
        if "sel" in kwargs
        else dict(
            channel_id=3,  # ...  model configuration (3=MESONH model)
            cell_mode=1,  # ... sampling mode (1=half-level)
            channel_mode=0 if data_var.startswith("w_") else 1,
        )
    )
    isel = (
        kwargs["isel"]
        if "isel" in kwargs
        else dict(
            cell_id=slice(13, None),  # ... exclude lowest levels
        )
    )
    px_defaults = dict(
        facet_col_wrap=1,
        facet_row_spacing=0.05,
        facet_col_spacing=0.03,
        facet_col="location",
    )
    px_opts = (
        {**px_defaults, **kwargs["px_opts"]} if "px_opts" in kwargs else px_defaults
    )

    # select dataarray
    da = plot_ds[data_var].sel(sel).isel(isel)

    # ... update the y-axis coord
    if isinstance(ycoord, tuple):
        da = da.dropna(ycoord[0], how="all")
        if "cell_z_bounds" in ycoord:
            da["cell_z_bounds"] = da["cell_z_bounds"].mean(
                [n for n in ["time", "location"] if n in da.dims]
            )
        da = da.set_index(dict([ycoord])).rename(dict([ycoord]))
        da = da.isel(**{ycoord[1]: ~da[ycoord[1]].isnull()})

    fig = plot_da(da, **px_opts)
    return fig    