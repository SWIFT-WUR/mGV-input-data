# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "netCDF4",
#     "numpy",
#     "pyflwdir",
#     "rasterio",
#     "xarray",
# ]
# ///
"""Derive the mGV routing input from the HydroSHEDS 15 arcsec DEM.

mGV reads five arrays from the routing file: ``downstream_id``, ``downstream``,
``cell_dist``, ``cell_area`` and ``accumulation``. This script makes them for the
grid of a VIC/mGV domain file (or a lon/lat box):

1. the HydroSHEDS DEM is read for the domain plus a buffer (``--buffer``) so that
   flow paths near the domain edge are not determined by the edge;
2. all depressions are filled (priority flood, Wang & Liu 2006) and D8 flow
   directions are derived with pyflwdir. Outlets are the edges of the valid DEM
   (coast and window edge), so endorheic basins drain over their lowest pour
   point. Keeping depressions as sinks is not an option: the raw DEM has deep
   spurious pits along rivers (gorges, reservoirs) that break up the network;
3. the 15 arcsec flow directions are upscaled to the model grid with the
   iterative hydrography upscaling method (IHU, Eilander et al. 2021,
   https://doi.org/10.5194/hess-25-5287-2021), which follows the high resolution
   river paths;
4. the network is cut to the domain. Cells that drain out of the domain (or out
   of the mask with ``--mask-only``) become outlets;
5. the routing parameters are derived from the network:
     downstream_id  cell id, 1..N in (lat, lon) order
     downstream     id of the downstream cell; outlets drain to themselves
     cell_area      spherical cell area, R^2 dlon |sin(lat2) - sin(lat1)| [m2]
     cell_dist      great circle distance between the cell centre and the centre
                    of the downstream cell [m]; sqrt(cell_area) for outlets
     accumulation   upstream area within the domain, including the cell [m2]
   with R = 6371 km. The D8 directions are written as ``flow_direction``
   (1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE, 0 outlet) for inspection.

All cells with HydroSHEDS land data are part of the network, also outside the
domain mask, unless ``--mask-only`` is given. Use ``--mask-only`` when areas
outside the mask (e.g. filled endorheic basins) should not drain into it.

The HydroSHEDS v1 DEM (15 arcsec, ~1 GB) is not downloaded by this script. Get
it once and put hyd_glo_dem_15s.tif in the working directory (or pass --dem):
    curl -LO https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_glo_dem_15s.zip
    unzip hyd_glo_dem_15s.zip hyd_glo_dem_15s.tif

Usage:
    uv run make-routing.py --domain domain.nc --output routing.nc
    uv run make-routing.py --bbox 66 23 83 38 --output routing.nc   # 5 arcmin grid
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyflwdir
import rasterio
import rasterio.windows
import xarray as xr
from affine import Affine

log = logging.getLogger("make-routing")

EARTH_RADIUS = 6371.0e3  # m
DEM_RES = 1 / 240  # 15 arcsec
DEM_NODATA = -9999.0
D8_NODATA = 247
# D8 code -> (row, col) offset, rows from north to south
D8_OFFSETS = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}


def read_grid(args) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return the cell centre lats and lons of the model grid and the domain mask."""
    if args.domain is not None:
        with xr.open_dataset(args.domain) as ds:
            lat = ds["lat"].values.astype(np.float64)
            lon = ds["lon"].values.astype(np.float64)
            mask = ds["mask"].fillna(0).values > 0 if "mask" in ds else None
        return lat, lon, mask
    west, south, east, north = args.bbox
    res = args.resolution / 60
    lon = np.arange(west + res / 2, east, res)
    lat = np.arange(south + res / 2, north, res)
    return lat, lon, None


def grid_resolution(coords: np.ndarray) -> float:
    steps = np.diff(coords)
    res = float(np.abs(steps).mean())
    if not np.allclose(np.abs(steps), res, rtol=1e-3):
        raise ValueError("model grid must be regular")
    return res


def read_dem(path: Path, west: float, south: float, east: float, north: float):
    """Read the DEM for the bounds (clipped to the DEM) as float32 with DEM_NODATA."""
    with rasterio.open(path) as src:
        if not np.isclose(src.res[0], DEM_RES):
            raise ValueError(f"expected a 15 arcsec DEM, got {src.res}")
        b = src.bounds
        west, east = max(west, b.left), min(east, b.right)
        south, north = max(south, b.bottom), min(north, b.top)
        col0 = round((west - b.left) / DEM_RES)
        row0 = round((b.top - north) / DEM_RES)
        ncol = round((east - west) / DEM_RES)
        nrow = round((north - south) / DEM_RES)
        window = rasterio.windows.Window(col0, row0, ncol, nrow)
        dem = src.read(1, window=window).astype(np.float32)
        dem[dem == src.nodata] = DEM_NODATA
    transform = Affine(DEM_RES, 0, west, 0, -DEM_RES, north)
    return dem, transform, (west, south, east, north)


def cell_area(lat: np.ndarray, res: float) -> np.ndarray:
    """Spherical area [m2] of grid cells with centre latitude lat and size res (degrees)."""
    lat1 = np.deg2rad(lat - res / 2)
    lat2 = np.deg2rad(lat + res / 2)
    return EARTH_RADIUS**2 * np.deg2rad(res) * np.abs(np.sin(lat2) - np.sin(lat1))


def great_circle(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(h))


def downstream_index(d8: np.ndarray) -> np.ndarray:
    """2D (row, col) -> flat index of the downstream cell; -1 for outlets and nodata.

    Cells pointing out of the array or to a nodata cell are outlets."""
    nrow, ncol = d8.shape
    rows, cols = np.indices(d8.shape)
    ds_row, ds_col = rows.copy(), cols.copy()
    for code, (dr, dc) in D8_OFFSETS.items():
        sel = d8 == code
        ds_row[sel] += dr
        ds_col[sel] += dc
    has_ds = np.isin(d8, list(D8_OFFSETS))
    inside = (ds_row >= 0) & (ds_row < nrow) & (ds_col >= 0) & (ds_col < ncol)
    idx = np.full(d8.shape, -1, dtype=np.int64)
    sel = has_ds & inside
    idx[sel] = ds_row[sel] * ncol + ds_col[sel]
    idx[sel] = np.where(d8.ravel()[idx[sel]] == D8_NODATA, -1, idx[sel])
    return idx


def make_routing(args) -> xr.Dataset:
    lat, lon, mask = read_grid(args)
    res = grid_resolution(lon)
    if not np.isclose(grid_resolution(lat), res, rtol=1e-3):
        raise ValueError("model grid cells must be square in degrees")
    scale = round(res / DEM_RES)
    if not np.isclose(scale * DEM_RES, res, rtol=1e-4):
        raise ValueError(f"grid resolution {res} is not a multiple of 15 arcsec")
    res = scale * DEM_RES
    log.info("model grid: %d x %d cells of %.4g arcmin", lat.size, lon.size, res * 60)

    # Snap the domain to the DEM grid, add the buffer (whole model cells)
    west = round((lon.min() - res / 2) / DEM_RES) * DEM_RES
    north = round((lat.max() + res / 2) / DEM_RES) * DEM_RES
    nbuf = round(args.buffer / res)
    dem, transform, (bw, bs, be, bn) = read_dem(
        args.dem,
        west - nbuf * res,
        north - (lat.size + nbuf) * res,
        west + (lon.size + nbuf) * res,
        north + nbuf * res,
    )
    log.info("DEM window: %d x %d cells, %.3f-%.3f E, %.3f-%.3f N", *dem.shape[::-1], bw, be, bs, bn)

    log.info("filling depressions and deriving D8 flow directions")
    flw = pyflwdir.from_dem(dem, nodata=DEM_NODATA, transform=transform, latlon=True)
    del dem

    log.info("upscaling flow directions (IHU, scale factor %d)", scale)
    flw_c, _ = flw.upscale(scale, method="ihu")
    d8 = flw_c.to_array(ftype="d8")  # north up

    # Cut to the domain (rows north to south)
    row0 = round((bn - north) / res)
    col0 = round((west - bw) / res)
    d8 = d8[row0 : row0 + lat.size, col0 : col0 + lon.size].copy()
    if d8.shape != (lat.size, lon.size):
        raise ValueError("domain is (partly) outside the DEM")
    flip = lat[0] < lat[-1]  # model grid is south to north
    lat_n = lat[::-1] if flip else lat
    if args.mask_only and mask is not None:
        d8[~(mask[::-1] if flip else mask)] = D8_NODATA
    ds_idx = downstream_index(d8)
    valid = d8 != D8_NODATA
    d8[valid & (ds_idx < 0)] = 0
    log.info("network: %d cells, %d outlets", valid.sum(), (valid & (ds_idx < 0)).sum())

    # Ids are 1..N in the order of the model grid
    order = np.flipud if flip else (lambda a: a)
    ids = np.zeros(d8.shape, dtype=np.int32)
    ids_model = order(ids)
    ids_model[order(valid)] = np.arange(1, valid.sum() + 1, dtype=np.int32)
    ids = order(ids_model)
    downstream = np.where(ds_idx >= 0, ids.ravel()[ds_idx], ids)

    lat2d, lon2d = np.meshgrid(lat_n, lon, indexing="ij")
    area = cell_area(lat2d, res)
    ds_flat = np.where(ds_idx >= 0, ds_idx, np.arange(ds_idx.size).reshape(ds_idx.shape))
    dist = great_circle(lat2d, lon2d, lat2d.ravel()[ds_flat], lon2d.ravel()[ds_flat])
    dist = np.where(ds_idx >= 0, dist, np.sqrt(area))

    flw_d = pyflwdir.from_array(d8, ftype="d8", transform=Affine(res, 0, west, 0, -res, north), latlon=True)
    acc = flw_d.accuflux(np.where(valid, area, 0.0))

    def grid(a, fill, dtype):
        return order(np.where(valid, a, fill).astype(dtype))

    dims = ("lat", "lon")
    out = xr.Dataset(
        {
            "downstream_id": (dims, grid(ids, -1, np.int32), {"long_name": "id used for downstream cell"}),
            "downstream": (dims, grid(downstream, -1, np.int32), {"long_name": "downstream cell id"}),
            "cell_dist": (dims, grid(dist, np.nan, np.float32), {"long_name": "flow cell distance", "units": "m"}),
            "cell_area": (dims, grid(area, np.nan, np.float32), {"long_name": "cell area", "units": "m2"}),
            "accumulation": (
                dims,
                grid(acc, np.nan, np.float32),
                {"long_name": "flow accumulation from upstream area", "units": "m2"},
            ),
            "flow_direction": (
                dims,
                order(d8).astype(np.uint8),
                {
                    "long_name": "D8 flow direction",
                    "flag_values": np.array([0, 1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8),
                    "flag_meanings": "outlet east southeast south southwest west northwest north northeast",
                },
            ),
        },
        coords={
            "lat": ("lat", lat, {"units": "degrees_north", "long_name": "latitude of grid cell center"}),
            "lon": ("lon", lon, {"units": "degrees_east", "long_name": "longitude of grid cell center"}),
        },
        attrs={
            "title": "mGV routing parameters",
            "source": (
                f"HydroSHEDS v1 15 arcsec DEM ({Path(args.dem).name}), depression filling and D8, "
                f"IHU upscaling with pyflwdir {pyflwdir.__version__}"
            ),
            "history": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC: make-routing.py"
            + (f" --domain {args.domain}" if args.domain else f" --bbox {' '.join(map(str, args.bbox))}")
            + f" --buffer {args.buffer}"
            + (" --mask-only" if args.mask_only else ""),
        },
    )
    if mask is not None:
        n_mask = mask.sum()
        n_missing = (mask & (out["downstream_id"].values < 0)).sum()
        log.info("%d of %d domain mask cells are not in the network (no DEM data)", n_missing, n_mask)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--domain", type=Path, help="domain file with lat, lon (cell centres) and mask")
    grid.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"), help="lon/lat box")
    parser.add_argument("--resolution", type=float, default=5, help="grid size with --bbox [arcmin] (default 5)")
    parser.add_argument("--dem", type=Path, default=Path("hyd_glo_dem_15s.tif"), help="HydroSHEDS 15 arcsec DEM")
    parser.add_argument("--buffer", type=float, default=1.0, help="DEM buffer around the domain [degrees] (default 1)")
    parser.add_argument("--mask-only", action="store_true", help="only route cells in the domain mask")
    parser.add_argument("--output", type=Path, required=True, help="output netCDF file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    out = make_routing(args)
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    for v in ("downstream_id", "downstream"):
        encoding[v]["_FillValue"] = -1
    for v in ("cell_dist", "cell_area", "accumulation"):
        encoding[v]["_FillValue"] = np.float32(1e20)
    encoding["flow_direction"]["_FillValue"] = np.uint8(D8_NODATA)
    out.to_netcdf(args.output, encoding=encoding)
    log.info("written %s", args.output)


if __name__ == "__main__":
    main()
