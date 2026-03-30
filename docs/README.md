# (urbanair-dm) Observation Examples and Data

Python and Jupyter notebook examples for UrbanAir Observations.

- Doppler lidar, keywords: wind field, turbulence, DWL, Streamline
- Ceilometer lidar, keywords: clouds, mixed layer height, ALC, CL61

## Setup

Set up your own jupyterlab environement, or use the supplied `environment.yml` (or `environment_optional.yml`) file to create the `urbanair312v1` environment:

```bash
mamba create -n urbanair312v1 python="3.12"
mamba env update -n urbanair312v1 -f environment.yml
conda activate urbanair312v1
```

Create a kernelspec for jupyter notebooks, in this case within the new environment:
```bash
conda activate urbanair312v1
python3 -m ipykernel install --user --name="urbanair312v1"
```

## Data / Analysis

Selected notebooks were made public:

- `urbanair-dm/analysis/notebooks/streamline_stats_mwe.ipynb` [link](https://github.com/z--m-n/urbanair-dm/blob/main/analysis/notebooks/streamline_stats_mwe.ipynb)
- `urbanair-dm/analysis/notebooks/ceilometer_stats_mwe.ipynb` [link](https://github.com/z--m-n/urbanair-dm/blob/main/analysis/notebooks/ceilometer_stats_mwe.ipynb)
- `urbanair-dm/analysis/notebooks/comparison_stats.ipynb` [link](https://github.com/z--m-n/urbanair-dm/blob/main/analysis/notebooks/comparison_stats.ipynb)

Instructions are included in these notebooks. Notebooks can be run within a python environment, for example,
```bash
conda activate urbanair312v1
jupyter notebook urbanair-dm/analysis/notebooks/ceilometer_stats_mwe.ipynb
```

Note that additional files may be needed:
```bash
urbanair-dm
└── analysis
   ├── conf
   │   └── environment.yml
   ├── notebooks
   │   ├── ceilometer_stats_mwe.ipynb
   │   ├── comparison_stats.ipynb
   │   └── streamline_stats_mwe.ipynb
   └── src
       └── urbanair_dm_lib.py
```

## Metadata

Example notebooks are currently not shared.

![](assets/metadb_locations.gif)
