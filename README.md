# mGV-input-data

Scripts to prepare the input data for mGV.

## Forcing

### WFDE5

[`forcing/WFDE5/download-wfde5.py`](forcing/WFDE5/download-wfde5.py) downloads WFDE5 from the
Copernicus Climate Data Store and writes daily forcing (prec, tair, psurf, vp, swdown, lwdown, wind)
on the 5 arcmin HydroSHEDS grid to a Zarr store. It needs [uv](https://docs.astral.sh/uv/), a
[CDS API key](https://cds.climate.copernicus.eu/how-to-api) and the HydroSHEDS DEM:

```sh
curl -LO https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_glo_dem_15s.zip
unzip hyd_glo_dem_15s.zip hyd_glo_dem_15s.tif
uv run forcing/WFDE5/download-wfde5.py                         # v3.1, 1979-2024 -> WFDE5_v3.1_5arcmin.zarr
uv run forcing/WFDE5/download-wfde5.py --version 2.1 --years 1990 1999  # other versions need --years
```

Interrupted runs continue where they stopped. See `--help` for all options.

## Routing

### HydroSHEDS

[`routing/HydroSHEDS/make-routing.py`](routing/HydroSHEDS/make-routing.py) derives the routing
parameters mGV reads (`downstream_id`, `downstream`, `cell_dist`, `cell_area`, `accumulation`) for
the grid of a domain file. It fills the depressions of the HydroSHEDS 15 arcsec DEM, derives D8 flow
directions and upscales them to the model grid with the IHU method of
[pyflwdir](https://deltares.github.io/pyflwdir/). It needs the same HydroSHEDS DEM as the forcing
script:

```sh
curl -LO https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_glo_dem_15s.zip
unzip hyd_glo_dem_15s.zip hyd_glo_dem_15s.tif
uv run routing/HydroSHEDS/make-routing.py --domain domain_Indus.nc --output routing_Indus.nc
uv run routing/HydroSHEDS/make-routing.py --domain domain_Indus.nc --mask-only --output routing_Indus.nc
uv run routing/HydroSHEDS/make-routing.py --bbox 104 8.5 107 11.5 --output routing_Mekong.nc  # 5 arcmin
```

All depressions are filled, so endorheic areas drain into a neighbouring basin. Use `--mask-only`
to route only the cells in the domain mask. See `--help` for all options.
