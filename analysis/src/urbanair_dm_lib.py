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
import plotly.graph_objects as go
import xarray as xr
import zarr
from plotly.express.colors import sample_colorscale

from tqdm.notebook import tqdm

unslice = lambda x: ([x.start, x.stop] if isinstance(x, slice) else x)

def update_dicts(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1:
            update_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

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
        #print("Warning: No files found!")
        fn_list, fn_dict = [], {}

    print(f"Found {len(fn_list)} files")

    return (fn_list, fn_dict)
    

def get_zenodo_repository(repository_path, respository_or_record_id):
    import os
    import subprocess
    from contextlib import contextmanager
    from pathlib import Path
    # requires: pip install zenodo-get

    # in case a list of urls is passed
    if isinstance(respository_or_record_id, list):
        for rec in respository_or_record_id:
            try:
                # t.b.d. improve error catch
                get_zenodo_repository(repository_path, rec)
            except:
                return(False)
        return(True)

    # in case a url is passed
    respository_or_record_id = respository_or_record_id.split('.')[-1]
    
    # pip install zenodo-get
    cmd = ["zenodo_get", respository_or_record_id]

    if Path(repository_path).is_dir() and str(respository_or_record_id).isdigit():
        fp = Path(repository_path).resolve() / str(respository_or_record_id)
        fp.mkdir(parents=False, exist_ok=True)
    else:
        # t.b.d. use fsspec simple_cache
        fp = Path(respository_or_record_id)

    @contextmanager
    def change_directory(directory: str):
        original_cwd = os.getcwd()
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(original_cwd)

    if fp.is_dir():
        with change_directory(fp):
            subprocess.call(cmd)

def unpack_zenodo_repository(
    repository_path,
    respository_or_record_id,
    repository_file_pattern=["*_data.tar", "*.zarr.zip"],
    destination_path="../tmp/test/",
):
    import shutil
    import tarfile

    def unpack_or_copy(fp, destination_path, filename_pattern):
        # archives
        fn_list, fn_dict = input_files([str(fp)], filename_pattern)

        for fd, fns in fn_dict.items():
            for fn in fns:
                fname = Path(fn).name
                fpart = re.findall(r"^([A-Z]{3})\_(L\d).*?$", fname)
                if fpart:
                    fout = Path(destination_path).resolve()
                    if not ( fout.is_dir() or fout.exists() ):
                        return False
                    
                    fout.mkdir(parents=True, exist_ok=True)    
                    if fout.is_dir():
                        # print(fname)
                        # print((f"Copying files to:\n '{fout}'"))
                        if tarfile.is_tarfile(fn):
                            with tarfile.open(fn) as f:
                                f.extractall(path=fout, filter="data")
                        elif str(fname).endswith("zarr.zip"):
                            shutil.copy(fn, fout / Path(*fpart[0]) )  # this can be linking also

    # in case a list of urls is passed
    if isinstance(respository_or_record_id, list):
        for rec in respository_or_record_id:
            for fpat in repository_file_pattern:
                unpack_zenodo_repository(repository_path, rec, fpat, destination_path)
        return True

    # in case a url is passed
    respository_or_record_id = respository_or_record_id.split(".")[-1]

    if Path(repository_path).is_dir() and str(respository_or_record_id).isdigit():
        fp = Path(repository_path).resolve() / str(respository_or_record_id)
        fp.mkdir(parents=False, exist_ok=True)
        unpack_or_copy(fp, destination_path, repository_file_pattern)
    else:
        # t.b.d. use fsspec simple_cache instead of a fixed directory
        print(
            (
                f"Warning: destination '{repository_path}' does not exist, "
                f"or argument 'respository_or_record_id' is not valid."
            )
        )
    
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
        ds = xr.concat(ds_list, dim="time", coords='different', compat='equals', join='outer', data_vars='all')
        ds = ds.drop_duplicates("time", keep="last")
        try:
            ds = xr_reindex(ds)  # reindex, prepare for merge
        except:
            pass
        ds_dict[fg] = ds
        ga_dict[fg] = ds.attrs

    # 2 of 2: Combine, merge and/or join
    ds = xr.combine_by_coords(ds_dict.values(), combine_attrs="drop_conflicts", join='outer')
    return ds, ga_dict, ds_dict

def plot_da(da, **kwargs):
    """a plot template"""
    import plotly.express as px

    px_opts = dict(
        range_color=[-2, 2],
        color_continuous_scale=px.colors.sequential.RdBu_r[1:-1],
    )
    px_opts = {**px_opts, **kwargs}

    time_index = [ k.split('_')[0] == "time" for k in list(da.indexes) ] 
    if len(da.squeeze().dims) == 2 or len(da.squeeze().dims) == 3:       
        fig = px.imshow(
            da,
            origin="lower",
            aspect=None,
            **px_opts,
        )
    elif len(da.squeeze().dims) == 1 and any(time_index):
        # mods to make plotly accept datetime[ns]
        da = da.drop_vars([n for n in da.coords if not n in da.indexes]).squeeze()
        time_var = list(da.indexes)[[i for i, n in enumerate(time_index) if n][0]]
        da[time_var] = pd.to_datetime(da[time_var].dt.round(freq='1ms'))
        da[time_var] = da[time_var].dt.strftime('%Y-%m-%d %H:%M:%S')
        fig = px.line(da.to_dataframe().reset_index(),x=time_var,y=da.name).update_traces(
            line_color="magenta", line_width=2, line_dash="dash"
        )   
    else:
        fig = go.Figure()

    fig.update_layout(paper_bgcolor="lightgray", plot_bgcolor="black")
    fig.update_layout(width=1100, height=400)
    fig.update_layout(margin=dict(l=50, r=200, b=50, t=30, autoexpand=False))
    return fig


def plot_da_template(template, **kwargs):

    if template in ["wdir", "wind_direction", "DD"]:
        px_opts = dict(
            range_color=[0, 360],
            color_continuous_scale=[
                (0.0, "#D7191C"),
                (0.03125, "#D7191C"),
                (0.03125, "#E15248"),
                (0.09375, "#E15248"),
                (0.09375, "#EB8C74"),
                (0.15625, "#EB8C74"),
                (0.15625, "#F5C5A0"),
                (0.21875, "#F5C5A0"),
                (0.21875, "#FFFFCC"),
                (0.28125, "#FFFFCC"),
                (0.28125, "#DCF0B6"),
                (0.34375, "#DCF0B6"),
                (0.34375, "#B9E1A1"),
                (0.40625, "#B9E1A1"),
                (0.40625, "#96D28B"),
                (0.46875, "#96D28B"),
                (0.46875, "#74C476"),
                (0.53125, "#74C476"),
                (0.53125, "#6C9C7C"),
                (0.59375, "#6C9C7C"),
                (0.59375, "#647582"),
                (0.65625, "#647582"),
                (0.65625, "#5C4E88"),
                (0.71875, "#5C4E88"),
                (0.71875, "#54278F"),
                (0.78125, "#54278F"),
                (0.78125, "#773696"),
                (0.84375, "#773696"),
                (0.84375, "#9A459E"),
                (0.90625, "#9A459E"),
                (0.90625, "#BD54A5"),
                (0.96875, "#BD54A5"),
                (0.96875, "#D7191C"),
                (1.0, "#D7191C"),
            ],
        )
        pc_opts = dict(          
            coloraxis_colorbar=dict(
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"],
            )
        )
    elif template in ["ws", "wind_speed", "FF"]:

        def get_wind_colors(
            F=16.0,
            ticks=[0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 16.0],
            colors=[
                "rgb(8,81,156)",
                "rgb(49,130,189)",
                "rgb(150,150,150)",
                "rgb(82,82,82)",
                "rgb(197,27,125)",
                "rgb(142,1,82)",
            ],
        ):

            c1 = [n / F for n in ticks for i in range(2)][1:-1]
            c2 = [n for n in colors for i in range(2)]
            ccs = list(zip(c1, c2))

            # color lookup table
            ccs_lut = dict(list(zip(ticks, colors)))
            return ccs_lut, ccs

        ccs_lut, ccs = get_wind_colors()

        px_opts = dict(
            range_color=[list(ccs_lut.keys())[n] for n in [0, -1]],
            color_continuous_scale=ccs,
        )

        pc_opts = dict(          
            coloraxis_colorbar=dict(
                tickvals=list(ccs_lut.keys()),
            )
        )
    elif template in ['mask']:
        # mv_args, mv_opts
        px_opts = dict(
            range_color=[0, 1],
            color_continuous_scale=[
                list(*zip([i], [j]))
                for i, j in zip(
                    [0, 0.5, 0.5, 1],
                    [n for n in px.colors.colorbrewer.Paired[1:5:3] for k in range(2)],
                )
            ],
        )
        pc_opts = dict(
            coloraxis_colorbar=dict(tickvals=[0.25, 0.75], ticktext=["False", "True"])
        )  
    else:
        px_opts = dict()
        pc_opts = dict()

    if kwargs.get("reverse_colors", False):
        px_opts["color_continuous_scale"] = list(
            zip(
                [n[0] for n in px_opts["color_continuous_scale"]],
                reversed([n[1] for n in px_opts["color_continuous_scale"]]),
            )
        )
    return px_opts, pc_opts

def plot_da_dist(dv_dict, data, methods=["histogram"], **kwargs):

    def ecdf(a):
        # https://stackoverflow.com/a/65972406
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        y = cusum / cusum[-1]
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0.0)
        return x, y

    def fig_update_layout(fig):
        fig.update_layout(paper_bgcolor="lightgray")
        fig.update_layout(plot_bgcolor="white")
        fig.update_xaxes(gridcolor="lightgray", zerolinecolor="lightgray")
        fig.update_yaxes(gridcolor="lightgray", zerolinecolor="lightgray")
        fig.update_layout(legend={"itemsizing": "constant"})
        return fig

    bins = kwargs.get("bins", 100)
    if "y_range" in kwargs:
        y_range = kwargs.get("y_range", None)
        y_delta = kwargs.get("y_delta", 0.2)
        bins = np.arange(start=y_range[0], stop=y_range[1] + y_delta, step=y_delta)
        # bins = np.insert(bins, 0, -np.inf)
        # bins = np.insert(bins, bins.size, np.inf)

    fh = {}
    for key in data:
        x = data[key].values
        x = x[~np.isnan(x)]
        x = x.tolist()

        orientation = kwargs.get("orientation", "v")
        if orientation == "h":
            ax = ["y", "x"]
            ls = "vh"
            lm = dict(x=0.5, line_dash="dot", line_color="pink", line_width=1)
        else:
            ax = ["x", "y"]
            ls = "hv"
            lm = dict(y=0.5, line_dash="dot", line_color="lightgray", line_width=1)

        from plotly.graph_objs.layout import XAxis, YAxis
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.layout = go.Layout(
            xaxis2=XAxis(
                overlaying="x",
                side="top",
            ),
            yaxis2=YAxis(
                overlaying="y",
                side="right",
            ),
            xaxis3=XAxis(
                overlaying="x",
                side="top",
            ),
            yaxis3=YAxis(
                overlaying="y",
                side="right",
            ),
        )
        group = kwargs.get("group", "1")
        for method in methods:
            ntrace = len(fig["data"])
            if ntrace > 0:
                fax = dict(xaxis=f"x{ntrace+1}", yaxis=f"y{ntrace+1}")
                #sty = dict(marker_color=px.colors.colorbrewer.Paired[ntrace * 3 + 2])
                sty = dict(marker_color=px.colors.colorbrewer.Purples[-1])
            else:
                fax = dict(xaxis="x", yaxis="y")
                #sty = dict(marker_color=px.colors.colorbrewer.Paired[ntrace * 3 + 1])
                sty = dict(marker_color=px.colors.colorbrewer.Purples[-4])                

            lgd = dict(
                legendgroup=method,
                legendgrouptitle=dict(
                    text=method.capitalize()[0] + method[1:],  # font_size=12
                ),
                name=group,
            )
            if method in ["histogram"]:
                hist, bin_edges = np.histogram(x, bins=bins, density=True)
                xy = dict(zip(ax, [bin_edges, hist]))
                trace = go.Bar(**xy, **lgd, **fax, **sty, orientation=orientation)
                fig.add_trace(trace)
            elif method in ["density"]:
                hist, bin_edges = np.histogram(x, bins=bins, density=True)
                xy = dict(zip(ax, [bin_edges, hist / sum(hist)]))
                trace = go.Bar(**xy, **lgd, **fax, **sty, orientation=orientation)
                fig.add_trace(trace)
            elif method in ["PDF", "pdf"]:
                hist, bin_edges = np.histogram(x, bins=bins, density=True)
                xy = dict(zip(ax, [bin_edges, hist / sum(hist)]))
                trace = go.Scatter(
                    **xy,
                    **lgd,
                    **fax,
                    **sty,
                    line_shape=ls,
                    line_width=1,
                    fill="tonexty",
                )
                fig.add_trace(trace)
            elif method in ["CDF", "cdf"]:
                hist, bin_edges = np.histogram(x, bins=bins, density=True)
                cdf = np.cumsum(hist * np.diff(bin_edges))
                cdf_mid = cdf[np.argmin(np.abs(bin_edges)) - 1]
                xy = dict(zip(ax, [bin_edges, cdf]))
                trace = go.Scatter(
                    **xy,
                    **lgd,
                    **fax,
                    **sty,
                    line_shape=ls,
                    line_width=1.5,
                )
                fig.add_trace(trace)
                fig.update_layout(
                    {
                        f"{fax['xaxis'][0] + 'axis' + fax['xaxis'][1:]}": dict(
                            tickvals=[0, 0.25, 0.5, 0.75, 1]
                        )
                    }
                )
                fig.add_annotation(
                    text="",
                    xref=f"{fax["xaxis"]} domain",
                    yref=f"{fax["yaxis"]} domain",
                    x=cdf_mid,
                    y=1.0,
                    axref=f"{fax["xaxis"]} domain",
                    ayref=f"{fax["yaxis"]} domain",
                    ax=cdf_mid,
                    ay=0.5,
                    arrowhead=2,
                    arrowwidth=2,
                    arrowcolor="lightgray",
                )

        fig = fig_update_layout(fig)
        fig.update_yaxes(range=y_range)
        fig.update_xaxes(showline=False, showgrid=False)
        fig.update_layout(title_text=key, title_font_size=12)
        # fig.update_layout(legend_x=1.04)
        fig.update_layout(
            legend=dict(x=1.05)  # yref="container",yanchor="bottom",orientation="v",
        )
        fig.update_layout(
            legend=dict(x=1, y=0, xanchor="right", yanchor="bottom", orientation="h")
        )
        fig.update_layout(margin=dict(l=5, r=5, b=35, t=40))
        fig.update_layout(height=400, width=300)
        fh[key] = fig

    return fh

def plot_da_quantile(dv_dict, data, **kwargs):

    q_pairs = kwargs.get("q_pairs", [[0.025, 0.975], [0.25, 0.75], [0.5]])
    z_range = kwargs.get("z_range", [-5, 5])

    def pd_quantile(n):
        def _quantile(x):
            return x.quantile(n)

        _quantile.__name__ = f"quantile_{int(n*1000):03d}"
        return _quantile

    def fig_update_layout(fig):
        fig.update_layout(paper_bgcolor="lightgray")
        fig.update_layout(plot_bgcolor="white")
        fig.update_xaxes(gridcolor="lightgray", zerolinecolor="lightgray")
        fig.update_yaxes(gridcolor="lightgray", zerolinecolor="lightgray")
        fig.update_layout(legend={"itemsizing": "constant"})
        return fig

    fhd = {}
    for key in data.keys():
        z_name = f"{dv_dict[key]['z']['name']}"
        x_name = dv_dict[key]["x"]["name"]

        daq = data[key].to_dataframe(name=z_name).reset_index()  #
        daq = daq.dropna(subset=z_name)
        daq = daq.groupby(pd.Grouper(key=x_name, freq="10min")).agg(
            {f"{z_name}": [pd_quantile(q) for q_pair in q_pairs for q in q_pair]}
        )
        daq = daq.reset_index()
        daq[x_name] = daq[x_name].astype("datetime64[ms]")

        traces = []
        colors = px.colors.colorbrewer.Purples[3::2]
        for n, q_pair in enumerate(q_pairs):
            N = len(q_pair)
            for m, q in enumerate(q_pair):
                trace_name = f"Q {q}" if m < 1 else f"IQR {q_pair[0]} to {q_pair[1]}"
                y_name = (z_name, f"quantile_{int(q*1000):03d}")
                trace = go.Scatter(
                    x=daq.loc(axis=1)[x_name],
                    y=daq.loc(axis=1)[y_name],
                    mode="lines",
                    line_color=colors[n],
                    line_width=1 / N,
                    line_shape="vh",
                    name=trace_name,
                    legendgroup=str(n),
                    fill=None if (m < 1) else "tonexty",
                    showlegend=True if (m > 0 or N < 2) else False,
                )
                traces.append(trace)

        fig = go.Figure(traces)
        fig = fig_update_layout(fig)
        fig.update_yaxes(range=z_range)
        fig.update_layout(
            legend=dict(x=1, y=0, xanchor="right", yanchor="bottom", orientation="h")
        )
        fig.update_layout(
            title=dict(text=key, font_size=12, pad_b=40, yanchor="bottom")
        )
        fig.update_layout(margin=dict(l=5, r=15, b=5, t=40))
        fig.update_layout(height=400, width=800)

        fhd[key] = fig

    return fhd    

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

def plot_ds_windrose(
    ds,
    x="time",
    x_range=["2023-08-21 00:00:00", "2023-08-21 23:59:59"],
    x_delta="6h",
    y="cell_z_bounds",
    y_range=[200, "MLH"],
    a="ws",  # scalar
    a_scale=1,
    a_offset=0,
    b="wdir",  # direction
    b_scale=1,
    b_offset=0,
    ann1="",  # title annotation
    ann2="",  # title annotation
    **kwargs,
):

    polar_kwargs = {k: v for k, v in kwargs.items() if k in ["mbins"]}

    # validate wind directions
    cv = [n for n in ds.coords if not n in ds.indexes]
    ds = ds.reset_coords(cv).sel(**{x: slice(*x_range)}).set_coords(cv)

    y_min = y_range[0] if isinstance(y_range[0], (int, float)) else -np.inf
    y_max = y_range[1] if isinstance(y_range[1], (int, float)) else np.inf
    ds = ds.where((ds[y] >= y_min) & (ds[y] <= y_max), drop=True)

    wps_dict = {}
    col_wrap = 0
    for i0, g0 in ds.resample(**{x: x_delta}, offset="0h"):
        col_wrap += 1

        i1 = tuple(
            [
                ds[n].values.tolist() if n in ds else ""
                for n in ["station_lat", "station_id"]
            ]
        )
        g1 = g0

        t0 = pd.to_datetime(i0)
        t1 = t0 + pd.to_timedelta(x_delta)
        a1 = f"{i1[0]:.2f}\xb0N {i1[1]}" if i1[0] else ann1
        a1 = f"{a1}, " if len(a1) > 0 else ""
        a2 = f"{y_range[0]} to {y_range[1]} m agl"
        ix = f"{a1}{a2}"
        idx = f"{str(ix)}<br>{t0.strftime('%Y%m%dT%H')}/{t1.strftime('%Y%m%dT%H')}"
        ws = g1[a]
        wd = g1[b]

        ws = ws.values * a_scale + a_offset
        wd = wd.values * b_scale + b_offset

        wps = wind_polar_stats(ws, wd, **polar_kwargs)
        wps_dict[idx] = wps

    # print(wps_dict.keys())
    class reversor:
        def __init__(self, obj):
            self.obj = obj

        def __eq__(self, other):
            return other.obj == self.obj

        def __lt__(self, other):
            return other.obj < self.obj

    # reverse location, but not time
    wps_dict = dict(
        sorted(
            wps_dict.items(),
            key=lambda x: (reversor(x[0].split("<br>")[0]), (x[0].split("<br>")[-1])),
        )
    )

    fig_kwargs = dict(
        col_wrap=col_wrap,  # ds.sizes["station"],
        title="",
    )
    fig_kwargs = {
        **fig_kwargs,
        **{k: v for k, v in kwargs.items() if k not in ["mbins"]},
    }
    fig = wind_polar_plot(
        wps_dict,
        range_mode="row",
        **fig_kwargs,
    )
    return fig
  
    
def translate_channels(ds, a="arome", b="system"):

    fun_ds_to_df = lambda x: (
        x.to_dataframe()  # .isel(system=0, station=0, drop=True)
        .dropna(subset=dv)
        .unstack("bounds")
    )

    dv = ["cell_z_bounds"]
    dc = ["cell_id"]
    gb = ["station_id", "system_id", "cell_id"]

    ab_list = ds["channel_id"].attrs["flag_meaning"].split(" ")
    subset = {}
    isubset = {
        "channel_mode": 0,  # stare mode
        "cell_mode": 2,  # full bounds
    }

    # height boundaries, as dataarray
    da = ds.reset_coords(dv)[dv]
    da = da.unstack("channel").unstack("cell")
    da = da.sel(subset, drop=True).isel(isubset, drop=True).mean(dim="time")

    # extract a/b channels
    da1 = da.sel(channel_id=ab_list.index(a))
    da2 = da.sel(channel_id=ab_list.index(b))
    df1 = fun_ds_to_df(da1.copy())
    df2 = fun_ds_to_df(da2.copy())

    cid = []
    for ig, dg in df2.groupby(level=gb):
        col1 = ("cell_z_bounds", 0)
        col2 = ("cell_z_bounds", 1)
        v0 = dg[col1].tolist()[0]
        v1 = dg[col2].tolist()[0]
        # idx = np.logical_or(
        #    df1[col1].between(v0, v1, inclusive="neither"),
        #    df1[col2].between(v0, v1, inclusive="neither"),
        # )
        idx = (df1[[col1, col2]].mean(axis=1)).between(v0, v1, inclusive="neither")
        if any(idx):
            ix = list(
                dict.fromkeys(df1[idx].index.get_level_values(dc[0]).tolist()).keys()
            )
            il = [n for n in ig]
            iv = getattr(il[2], "tolist", lambda: il[2])()
            a_d = pd.Series(data=il[0:2] + [ix], index=gb).to_dict()
            b_d = pd.Series(data=il[0:2] + [iv], index=gb).to_dict()

            cid.append({a: a_d, b: b_d})

    # restructure
    df = pd.concat(
        [
            (
                pd.DataFrame(n)
                .T.rename_axis("channel_id")
                .set_index(["station_id", "system_id"], append=True)
                .reset_index("channel_id")
                .pivot(columns="channel_id")
            )
            for n in cid
        ]
    )

    # expand lists
    for n in list(df.columns):
        df = df.explode(n)

    # sort
    cols = [n for k in [a, b] for n in df.columns if k in n]
    df = df.reindex(cols, axis=1)

    # store names
    lut1 = df.rename_axis(axis=1, columns=["cell_id", "channel_id"])

    # more coordinates
    df1 = lut1.drop([("cell_id", a)], axis=1).droplevel(1, axis=1).reset_index()
    df2 = (
        ds.coords.to_dataset()
        .sel(isubset) # not isel
        .sel(channel_id=ab_list.index(b))
        .isel(time=0, time_delta=0)
        .to_dataframe()
        .reset_index()
    )
    lut2 = df1.merge(df2).set_index(lut1.index.names)
    lut2 = lut2.drop(["time", "time_delta"], axis=1).set_index('bounds',append=True)

    return (lut1, lut2, da)
    

def swap_channels(ds, df_cc):
    subset = dict()
    isubset = dict()
    cn = list(df_cc.columns.names)    
    gb = list(df_cc.index.names)
    dc = [cn[0]]

    a, b = df_cc.columns.get_level_values(cn[1]).to_list()

    dx = []
    for i0, gd0 in df_cc.groupby(
        by=[g for g in gb if not g in dc], sort=False, as_index=True
    ):
        ss = {**subset, **dict(zip([g for g in gb if not g in dc], [[i] for i in i0]))}
        dd = []
        for _, gd1 in gd0.groupby(by=[tuple(dc + [b])]):

            ai = gd1.loc[:, [tuple(dc + [a])]].values.ravel()
            bi = gd1.loc[:, [tuple(dc + [b])]].values.ravel()

            si = dict([tuple([*dc, [str(n) for n in ai.tolist()]])])
            ci = dict([tuple([*dc, ("cell", [str(n) for n in bi.tolist()])])])
            try:
                da = ds.sel(ss)
                for k, v in si.items():
                    da = da.where(da[k].isin(v), drop=True)
                da = da.assign_coords(**ci)
                dd.append(da)
            except:
                pass

        if dd:
            dx.append(xr.concat(dd, dim="cell"))

    ds = xr.merge(dx)
    ds = xr_reindex(ds)
    return ds

def swap_coords(ds, coords_lut, coord=('cell', 'cell_z_bounds')):
    ix = coords_lut.index.names
    idx = tuple([ ds[n].values.tolist() if n in ds.coords else [0] for n in ix ])
    dc = coords_lut.loc[ idx ,:][coord[1]]
    ds = xr_reindex(ds.assign_coords({coord[1] : (coord[0], dc )}))
    return(ds)
    
def dataset_to_datatree(*kargs, **kwargs):

    # dataset restructure and reduction needed for plots
    def _plot_ds(ds):
        # subset
        subset = {}
        isubset = {
            k: 0 for k in ["bounds", "time_delta", "attributes"] if k in ds.coords
        }

        # reduce to a 4-dimensional dataset
        ds = ds.sel(subset).isel(isubset)
        if "cell" in ds.indexes and "cell_id" in ds.indexes:
            if ds.indexes.is_multi("cell"):
                ds = ds.unstack(["cell"]).dropna("cell_id", how="all")
        ds = (
            ds.reset_index(["station", "system"])
            .stack(location=["station", "system"])
            .dropna("location", how="all")
            .dropna("time", how="all")
            .sortby("station_lat", ascending=False)
            .transpose(..., "time")
        )
        return ds

    # consolidated production file
    fd = kargs[0] if len(kargs) == 1 else None
    if isinstance(fd, list) and len(fd) == 2:
        repo_path, repo_file = fd
    else:
        repo_path = str(Path(fd).parent)
        repo_file = Path(fd).name

    # discover
    fn_list, fn_dict = input_files(repo_path, repo_file)
    pp = []    
    with xr.open_dataset(fn_list[0], **kwargs) as dx:
        for n in ["production_level", "production_profile"]:
            if n in dx.attrs:
                pp.append(dx.attrs[n])
        dx.close()
    print(f"dataset_to_datatree: {pp}")
    
    # fragile: custom per dataset group
    if "L2" in pp and "set(fr.paris,DWL,2022_2024)" in pp:
        # L2
        dt_dict = {}
        dx = xr_reindex(xr.open_dataset(fn_list[0], **kwargs))
        # split per station
        station_id = dx["station_id"].values.tolist()
        for k in station_id:
            ds = dx.sel(station_id=k)
            ds = _plot_ds(ds).isel(location=0)
            dt_dict[k[2:]] = ds
        dt = xr.DataTree.from_dict(dt_dict)
    elif "L2B" in pp and "set(fr.paris,ALC,2022_2024)" in pp:
        # L2B
        dt_dict = {}
        dx = xr_reindex(xr.open_dataset(fn_list[0], **kwargs))
        # split per station
        station_id = dx["station_id"].values.tolist()
        for k in station_id:
            ds = dx.sel(station_id=k)
            dt_dict[k[2:]] = ds
        dt = xr.DataTree.from_dict(dt_dict)        
    else:
        # L1, other
        return xr.DataTree()

    return dt

def datatree_to_dataarray(dtd, dvd, key):
    dad = {}

    # variable
    for ax in ["z"]:    
        if not ax in dvd[key]:
            print('Warning: not a 3-dimensional selection.')
            continue
            
        dad[key] = dtd[key][dvd[key][ax]["path"]][dvd[key][ax]["name"]]
        
        # variable subset
        if "sel" in dvd[key][ax]:
            dad[key] = dad[key].sel(dvd[key][ax]["sel"])
        if "isel" in dvd[key][ax]:
            dad[key] = dad[key].isel(dvd[key][ax]["isel"])

    # coordinates, slices
    for ax in ["x", "y"]:
        if not ax in dvd[key]:
            continue

        if not dvd[key][ax]["name"] in dad[key].coords:
            # print(f'DEBUG Expanding dims: {key}')
            dad[key] = dad[key].expand_dims({dvd[key][ax]["name"]: 1})

        if ax == 'x':
            dad[key] = dad[key].transpose(..., dvd[key][ax]["name"])            
            
        if "swap_coord" in dvd[key][ax]:
            scoord = dvd[key][ax]["swap_coord"]
            # ... update the y-axis coord
            if isinstance(scoord, tuple):
                da = dad[key].dropna(scoord[0], how="all")
                if "cell_z_bounds" in scoord:
                    da["cell_z_bounds"] = da["cell_z_bounds"].mean(
                        [n for n in ["time", "location"] if n in da.dims]
                    )
                da = da.set_index(dict([scoord])).rename(dict([scoord]))
                da = da.isel(**{scoord[1]: ~da[scoord[1]].isnull()})
                dad[key] = da
        if "sel" in dvd[key][ax]:
            dad[key] = dad[key].sel(
                {
                    dvd[key][ax]["name"]: dvd[key][ax]["sel"],
                }
            )
        if "isel" in dvd[key][ax]:
            dad[key] = dad[key].isel(
                {
                    dvd[key][ax]["name"]: dvd[key][ax]["isel"],
                }
            )
        if "resample" in dvd[key][ax]:
            dad[key] = (
                dad[key]
                .resample(
                    **{
                        dvd[key][ax]["name"]: dvd[key][ax]["resample"],
                    }
                )
                .mean()
            )

    return dad[key]



#### 

def wind_polar_stats(ws, wd, mbins=[0.0, 1.0, 2.0, 5.0, 10.0, 15.0, np.inf]):
    """direction-magnitude statistics from polar wind"""
    """Created by: Matthias Zeeman"""
    """Reference: Zeeman et al., 2021, 2022"""

    wbins = [0] + np.arange(11.25, 360, 22.5).tolist() + [360]
    wlabs = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
        "N",
    ]

    vhst = np.histogram2d(wd.flatten(), ws.flatten(), bins=[wbins, mbins])
    wps = (
        pd.DataFrame(vhst[0], index=wlabs, columns=mbins[:-1])
        .rename_axis(index="direction", columns="magnitude")
        .stack()
        .reset_index()
        .rename(columns={0: "count"})
    )
    # in percent
    wps["fraction"] = wps["count"] / wps["count"].sum() * 100

    # de-duplicate wind sectors, i.e., N
    wps = wps.groupby(["magnitude", "direction"], sort=False).sum()

    return wps.reset_index()


def wind_polar_plot(wps_dict, **kwargs):
    """plot wind polar statistics as windrose"""
    """Created by: Matthias Zeeman"""
    """Reference: Zeeman et al., 2021, 2022"""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def get_grid_panel(n, nrow, ncol):
        row = int(np.floor(n / ncol) + 1)
        col = int(np.remainder(n - (row - 1) * ncol, ncol) + 1)
        return row, col

    def get_wind_colors(
        F=16.0,
        ticks=[0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 16.0],
        colors=[
            "rgb(8,81,156)",
            "rgb(49,130,189)",
            "rgb(150,150,150)",
            "rgb(82,82,82)",
            "rgb(197,27,125)",
            "rgb(142,1,82)",
        ],
    ):

        c1 = [n / F for n in ticks for i in range(2)][1:-1]
        c2 = [n for n in colors for i in range(2)]
        ccs = list(zip(c1, c2))

        # color lookup table
        wps_cols = dict(list(zip(ticks, colors)))
        return wps_cols, ccs

    def get_range_max(wps_dict):
        # pre-compute ranges
        wps_sum = []
        for _, wps in wps_dict.items():
            for i, g in wps.groupby(["direction"], sort=False):
                wps_sum.append(np.ceil(g["fraction"].sum()))

        # compute range for the combined set of wind-rose panels
        wps_max = np.max(wps_sum)
        wps_range = [
            -5,
            (
                np.max([np.ceil(wps_max / 5.0) * 5.0, 5.0])
                if wps_max < 25
                else np.ceil(wps_max / 10.0) * 10.0
            ),
        ]

        return wps_max, wps_range

    if isinstance(wps_dict, dict):
        col_wrap = kwargs["col_wrap"] if "col_wrap" in kwargs else len(wps_dict)
        ncol = col_wrap
        nrow = int(np.ceil(len(wps_dict.keys()) / col_wrap))
    else:
        wps_dict = {"": wps_dict}
        col_wrap = kwargs["col_wrap"] if "col_wrap" in kwargs else len(wps_dict)
        nrow = 1
        ncol = col_wrap

    # figure config
    spec = [
        (
            [{"type": "polar"}] * ncol + [None]
            if n > 0
            else [{"type": "polar"}] * ncol + [{"type": "scene", "rowspan": nrow}]
        )
        for n in range(nrow)
    ]

    # custom or not
    if "ticks" in kwargs and "colors" in kwargs:
        wps_colors, wps_css = get_wind_colors(
            ticks=kwargs["ticks"], colors=kwargs["colors"], F=kwargs["ticks"][-1]
        )
    else:
        wps_colors, wps_css = get_wind_colors()

    # figure
    fig = make_subplots(
        rows=nrow,
        cols=ncol + 1,
        specs=spec,
        column_widths=[1] * ncol + [0.05 * ncol],
        vertical_spacing=0.07 / nrow,  # 0.05
        horizontal_spacing=0.07 / ncol,  # 0.07
    )

    n = 0
    for name, wps in wps_dict.items():
        row, col = get_grid_panel(n, nrow, ncol)

        # add polar plots
        for i, g in wps.groupby(["magnitude"], sort=False):
            m = i[0]
            fig.add_trace(
                go.Barpolar(
                    r=g["fraction"].copy(),
                    name=str(m),
                    marker_color=wps_colors[m],
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # loop counter
        n = n + 1

    # ranges
    n = 0
    wps_range_dict = {}
    _, wps_range_max = get_range_max(wps_dict)
    for name, wps in wps_dict.items():
        row, col = get_grid_panel(n, nrow, ncol)
        wps_max, wps_range = get_range_max({name: wps})
        wps_range_dict[n] = wps_range

        # loop counter
        n = n + 1

    # polar plot configuration
    fig_lay = fig.to_dict()["layout"].keys()
    for i, n in enumerate([x for x in fig_lay if x.startswith("polar")]):
        if "range_mode" in kwargs and i in list(wps_range_dict.keys()):
            wps_range = wps_range_dict[i]

        else:
            wps_range = wps_range_max

        lay = {
            n: {
                "angularaxis": {
                    "direction": "clockwise",
                    "rotation": 90,
                    "tickvals": list(range(0, 360, 45)),
                    "ticktext": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                },
                "radialaxis": {
                    "range": wps_range,
                    "tickmode": "array",
                    "tickvals": [0, 5, 10, 15, 20] + list(range(40, 100, 20)),
                    "tickangle": 0,
                    "tickfont": {"size": 14},
                },
            }
        }

        # update layout
        fig.update_layout(**lay)

    # add annotations (figure title)
    if "title" in kwargs:
        fig.add_annotation(
            {
                "font": {"size": 18},
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "text": kwargs["title"],
                "textangle": 0,
                "align": "left",
                "valign": "bottom",
                "x": 0.01,
                "xshift": 0,
                "xanchor": "left",
                "y": 1,
                "yshift": 30,
                "yanchor": "bottom",
            }
        )

    # add annotations (panel title)
    for n, k in enumerate(wps_dict.keys()):
        row, col = get_grid_panel(n, nrow, ncol)
        fig.add_annotation(
            {
                "font": {"size": 18},
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "text": "{}".format(k),
                "align": "left",
                "valign": "bottom",
                "textangle": 0,
                "x": 0.01 + (1 / ncol * (col - 1)),
                "xshift": -15 * (col - 1),
                "xanchor": "left",
                "y": 1 - (1 / nrow * (row - 1)),
                "yshift": -10 * (row - 1) - 10,
                "yanchor": "bottom",
            }
        )

    # add color axis, using an 'invisible' scene
    fig.add_trace(
        go.Mesh3d(x=[0], y=[0], z=[0], intensity=[1], coloraxis="coloraxis"),
        row=1,
        col=ncol + 1,
    )
    F = kwargs["F"] if "F" in kwargs else 16
    # F = 16
    fig.update_layout(
        coloraxis={
            "colorscale": wps_css,
            "cmin": 0,
            "cmax": F,
            "colorbar_title": (
                # "<span><i>&nbsp;U</i></span><br>"
                '<span style="font-size:85%">[m s<sup>-1</sup>]</span>'
            ),
            "colorbar": {
                "thickness": 24,
                "lenmode": "pixels",
                "len": nrow * 250,
                "x": 0.95,
                "y": 0.5,
                "yanchor": "middle",
                "xpad": 4,
                "xanchor": "left",
                "tickvals": [k * F for k, v in wps_css][:-1],
            },
        }
    )
    fig.update_layout(
        scene={
            k: dict(
                gridcolor="rgba(0, 0, 0, 0)",
                showbackground=False,
                zerolinecolor="rgba(0, 0, 0, 0)",
            )
            for k in ["xaxis", "yaxis", "zaxis"]
        }
    )
    margin = dict(l=40, r=40, b=40, t=90, pad=2, autoexpand=False)
    fig.update_layout(
        margin=go.layout.Margin(**margin),
        height=400 * nrow + 40 * (nrow - 1)+margin['t']+margin['b'],
        width=400 * ncol - 40 * (ncol - 1)+margin['l']+margin['r'],
        paper_bgcolor="lightgray",
        plot_bgcolor="black",
    )

    return fig


    
def hello_world():
    import pandas as pd
    return( 
        (
            f"<p>"
            f"<font size='4'>"
            f"<a href='https://github.com/z--m-n/urbanair-dm'>"
            f" &laquo; &#128517; &#128293; &raquo; <br>"
            f"{pd.Timestamp.now().floor('s').isoformat()}"
            f"</a>"
            f"</font>"
            f"</p>"
        )
    )