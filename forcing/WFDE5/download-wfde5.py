# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cdsapi",
#     "dask",
#     "netCDF4",
#     "numpy",
#     "pandas",
#     "rasterio",
#     "scipy",
#     "xarray",
#     "xarray-regrid[accel]",
#     "zarr>=3",
# ]
# ///
"""Download WFDE5 from the Climate Data Store and add it to a 5 arcmin Zarr store.

Reproduces the processing of the cdo-made ``{var}_WFDE5_v2.0_5arcmin_f32_{year}.nc``
files, for WFDE5 (default v3.1, 1979-2024) from the CDS dataset
``derived-near-surface-meteorological-variables``:

1. hourly 0.5 degree data are averaged to daily values (00-23 UTC). Hours missing
   at the start of a month are replaced with the last hours of that month (as was
   done for 1979-01-01 00-06 UTC in v2.0);
2. prec = Rainf + Snowf (CRU+GPCC), the other variables use the CRU reference;
3. regridding to the 5 arcmin grid with xarray-regrid: conservative for prec,
   bilinear for the rest (periodic in longitude);
4. land cells (5 arcmin cells with HydroSHEDS DEM data) without data are filled
   by inverse-distance weighting of the 4 nearest cells with data (as
   ``cdo setmisstodis``), all other cells are set to missing;
5. tair, psurf and vp are downscaled to the HydroSHEDS DEM (15 arcsec, averaged
   over each 5 arcmin cell) from the WFDE5 grid-point altitude:
       dz     = DEM - ASurf
       tair   = Tair - 0.0065 dz
       r      = Qair / (1 - Qair)
       Tv_avg = (Tair + tair) / 2 * (1 + 0.61 r)
       psurf  = PSurf exp(-dz / (29.3 Tv_avg))
       vp     = r / (0.622 + r) psurf
6. units: prec mm (daily sum), tair degC, psurf and vp kPa, radiation W m-2, wind m s-1.

The store is int16 with scale/offset and Blosc (zstd) compression, written with
the layout described in store.py. Years are appended in order; years
already in the store are skipped, so an interrupted run continues where it
stopped.

Monthly downloads run concurrently (--workers) and are reduced to daily means
right away: only the daily 0.5 degree data are kept in --work-dir until the year
is in the store. The next year(s) are downloaded while a year is written.

The HydroSHEDS v1 DEM (15 arcsec, ~1 GB) is not downloaded by this script. Get
it once and put hyd_glo_dem_15s.tif in the working directory (or pass --dem):
    curl -LO https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_glo_dem_15s.zip
    unzip hyd_glo_dem_15s.zip hyd_glo_dem_15s.tif

Usage:
    uv run download-wfde5.py [--version 3.1] [--years 1979 2024] [--workers 8] [--output STORE]
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
import warnings
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import xarray_regrid  # noqa: F401  (registers the .regrid accessor)
import zarr
from rasterio.windows import Window
from scipy.spatial import cKDTree

from store import (
    DIMS,
    VARIABLES,
    encode,
    extend_store,
    init_store,
    iter_blocks,
    load_progress,
    locate_year,
    save_progress,
)

logger = logging.getLogger("download-wfde5")
warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")

NC_LOCK = threading.Lock()

DATASET = "derived-near-surface-meteorological-variables"
DEFAULT_VERSION = "3.1"
DEFAULT_YEARS = {"3.1": (1979, 2024)}  # --years is required for other versions

# WFDE5 name: (CDS variable, reference dataset)
SOURCES = {
    "Tair": ("near_surface_air_temperature", "cru"),
    "Qair": ("near_surface_specific_humidity", "cru"),
    "PSurf": ("surface_air_pressure", "cru"),
    "Wind": ("near_surface_wind_speed", "cru"),
    "LWdown": ("surface_downwelling_longwave_radiation", "cru"),
    "SWdown": ("surface_downwelling_shortwave_radiation", "cru"),
    "Rainf": ("rainfall_flux", "cru_and_gpcc"),
    "Snowf": ("snowfall_flux", "cru_and_gpcc"),
}
ASURF = ("grid_point_altitude", "cru")
CONSERVATIVE = {"Prec"}  # all other fields are regridded bilinearly
FILL_NEIGHBOURS = 4  # cdo setmisstodis default
LAPSE_RATE = 0.0065  # K m-1

# 0.5 degree WFDE5 grid and 5 arcmin target grid (cell centres, south to north)
SRC_LAT = -89.75 + 0.5 * np.arange(360)
SRC_LON = -179.75 + 0.5 * np.arange(720)
LAT = -56 + (np.arange(1680) + 0.5) / 12
LON = -180 + (np.arange(4320) + 0.5) / 12
NLAT, NLON = len(LAT), len(LON)
TARGET = xr.Dataset(coords={"lat": LAT, "lon": LON})

VAR_ATTRS = {
    "lwdown": {"units": "W m-2", "long_name": "surface downwelling longwave radiation"},
    "prec": {"units": "mm", "long_name": "total precipitation"},
    "psurf": {"units": "kPa", "long_name": "near surface air pressure"},
    "swdown": {"units": "W m-2", "long_name": "surface downwelling shortwave radiation"},
    "tair": {"units": "degC", "long_name": "near surface air temperature"},
    "vp": {"units": "kPa", "long_name": "near surface vapour pressure"},
    "wind": {"units": "m s-1", "long_name": "near surface wind speed"},
}
CRS_ATTRS = {
    "grid_mapping_name": "latitude_longitude",
    "longitude_of_prime_meridian": 0.0,
    "semi_major_axis": 6378137.0,
    "inverse_flattening": 298.257223563,
    "crs_wkt": 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]',
}
GLOBAL_ATTRS_TO_COPY = ("Conventions", "contact", "comment", "reference", "licence", "dataset_version")


# --- download -----------------------------------------------------------------


def download_dir(work: Path, tag: str) -> Path:
    """A new, empty folder for one download attempt.

    Never reused: on the NTFS (fuseblk) data disk, removing a folder right after
    its files were open can silently fail.
    """
    (work / "download").mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=work / "download"))


def retrieve(request: dict, tmp: Path) -> Path:
    """Download one CDS request into ``tmp`` and return the netCDF file in it."""
    target = tmp / "download"
    cdsapi.Client(quiet=True, progress=False).retrieve(DATASET, request, str(target))
    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as z:
            z.extractall(tmp)
        target.unlink()
    files = list(tmp.glob("*.nc")) or [target]
    if len(files) != 1:
        raise ValueError(f"{tmp.name}: expected one netCDF file, got {[f.name for f in files]}")
    return files[0]


def read_source(path: Path, name: str) -> tuple[xr.DataArray, dict]:
    """Variable ``name`` (loaded into memory) and the global attributes of a CDS file.

    netCDF4/HDF5 is not thread-safe and xarray only locks reading the data, not the
    metadata, so the whole file is read under one lock.
    """
    with NC_LOCK, xr.open_dataset(path) as ds:
        da = ds[name].load()
        attrs = dict(ds.attrs)
    if not (np.allclose(da.lat, SRC_LAT) and np.allclose(da.lon, SRC_LON)):
        raise ValueError(f"{path.name}: unexpected grid")
    return da, attrs


def daily_means(path: Path, name: str, year: int, month: int) -> np.ndarray:
    """Hourly month -> daily means (days, lat, lon) float32."""
    da, _ = read_source(path, name)
    hours = pd.DatetimeIndex(da.time.values)
    data = da.values
    start = pd.Timestamp(year, month, 1)
    expected = pd.date_range(start, start + pd.offsets.MonthBegin(), freq="h", inclusive="left")
    if not hours.equals(expected):
        k = len(expected) - len(hours)
        if k <= 0 or not hours.equals(expected[k:]):
            raise ValueError(f"{path.name}: time steps are not the hours of {year}-{month:02d}")
        logger.warning("%s: first %d hours missing, replaced with the last %d hours of the month", path.name, k, k)
        data = np.concatenate([data[-k:], data])
    return data.reshape(-1, 24, *data.shape[1:]).mean(1, dtype=np.float64).astype(np.float32)


def month_file(work: Path, name: str, year: int, month: int) -> Path:
    return work / "daily" / str(year) / f"{name}_{year}{month:02d}.npy"


def fetch_month(work: Path, version: str, name: str, year: int, month: int, attempts: int = 5) -> None:
    """Download one variable-month and store its daily means."""
    out = month_file(work, name, year, month)
    if out.exists():
        return
    variable, reference = SOURCES[name]
    request = {
        "product": "wfde5",
        "variable": [variable],
        "reference_dataset": reference,
        "version": version.replace(".", "_"),
        "year": [str(year)],
        "month": [f"{month:02d}"],
    }
    tag = f"{name}_{year}{month:02d}"
    tic = time.perf_counter()
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            time.sleep(60 * (attempt - 1))
            # The CDS returns the cached result of an identical request, even when
            # that is broken (e.g. AccessDenied on the cache object, for hours). An
            # unused key makes it compute the request again.
            request["nocache"] = uuid.uuid4().hex
        tmp = download_dir(work, tag)
        try:
            daily = daily_means(retrieve(request, tmp), name, year, month)
            break
        except Exception:
            if attempt == attempts:
                raise
            logger.exception("%s: attempt %d failed, retrying", tag, attempt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, daily)
    os.replace(tmp, out)
    logger.info("  %s done in %.0f s", tag, time.perf_counter() - tic)


def fetch_asurf(work: Path, version: str) -> tuple[np.ndarray, dict]:
    """WFDE5 grid-point altitude (lat, lon) and the source's global attributes."""
    out, attrs_file = work / "ASurf.npy", work / "source_attrs.json"
    if not out.exists():
        variable, reference = ASURF
        tmp = download_dir(work, "ASurf")
        try:
            path = retrieve({"product": "wfde5", "variable": [variable], "reference_dataset": reference, "version": version.replace(".", "_")}, tmp)
            da, attrs = read_source(path, "ASurf")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        attrs_file.write_text(json.dumps({k: str(v) for k, v in attrs.items()}, indent=1))
        np.save(out, da.values.astype(np.float32))
    return np.load(out), json.loads(attrs_file.read_text())


# --- static fields ----------------------------------------------------------------


def load_dem(path: Path, cache: Path) -> np.ndarray:
    """HydroSHEDS DEM averaged over the 5 arcmin cells (nodata ignored), south to north."""
    if cache.exists():
        return np.load(cache)
    logger.info("Averaging %s to 5 arcmin", path.name)
    with rasterio.open(path) as src:
        f = round(1 / 12 / src.res[0])
        if src.shape != (NLAT * f, NLON * f) or not src.transform.almost_equals(rasterio.Affine(1 / 12 / f, 0, -180, 0, -1 / 12 / f, 84)):
            raise ValueError(f"{path}: expected a global 84N-56S grid aligned with 5 arcmin")
        dem = np.empty((NLAT, NLON), np.float32)
        rows = 60
        for r0 in range(0, NLAT, rows):
            a = src.read(1, window=Window(0, r0 * f, NLON * f, rows * f))
            valid = (a != src.nodata).reshape(rows, f, NLON, f)
            total = np.where(valid, a.reshape(rows, f, NLON, f), 0).sum((1, 3), dtype=np.float64)
            count = valid.sum((1, 3))
            with np.errstate(invalid="ignore"):
                dem[r0 : r0 + rows] = total / count
    dem = dem[::-1].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, dem)
    return dem


# --- processing -------------------------------------------------------------------


def regrid(data: np.ndarray, conservative: bool) -> np.ndarray:
    """(time, 0.5 deg lat, lon) -> (time, 5 arcmin lat, lon)."""
    da = xr.DataArray(data, dims=DIMS, coords={"time": np.arange(len(data)), "lat": SRC_LAT, "lon": SRC_LON})
    if conservative:
        out = da.regrid.conservative(TARGET, latitude_coord="lat")
    else:
        # Periodic in longitude, so the outermost target columns are interpolated as well.
        da = xr.concat([da.isel(lon=[-1]).assign_coords(lon=[SRC_LON[-1] - 360]), da, da.isel(lon=[0]).assign_coords(lon=[SRC_LON[0] + 360])], "lon")
        out = da.regrid.linear(TARGET)
    return out.transpose(*DIMS).values


def unit_vectors(flat_index: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(LAT[flat_index // NLON])
    lon = np.deg2rad(LON[flat_index % NLON])
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


class GapFiller:
    """Regridded grid -> values of the land cells, like ``cdo setmisstodis``.

    Land cells without data get the inverse-distance weighted mean of the
    nearest cells with data. The weights depend only on where data is missing,
    which is fixed per regridding method, so they are computed once.
    """

    def __init__(self, mask: np.ndarray):
        self.cells = np.flatnonzero(mask)
        self.tables: dict[bytes, tuple] = {}

    def table(self, valid: np.ndarray) -> tuple:
        key = hashlib.sha1(np.packbits(valid)).digest()
        if key not in self.tables:
            source = np.flatnonzero(valid)
            gaps = np.flatnonzero(~valid[self.cells])
            dist, nearest = cKDTree(unit_vectors(source)).query(unit_vectors(self.cells[gaps]), k=FILL_NEIGHBOURS)
            weights = 1 / dist
            weights /= weights.sum(1, keepdims=True)
            self.tables[key] = gaps, source[nearest], weights
            logger.info("  %d land cells without data are filled from their %d nearest neighbours", len(gaps), FILL_NEIGHBOURS)
        return self.tables[key]

    def __call__(self, grid: np.ndarray) -> np.ndarray:
        flat = grid.reshape(len(grid), -1)
        valid = np.isfinite(flat[0])
        if not (np.isfinite(flat) == valid).all():
            raise ValueError("cells with data change over time")
        gaps, source, weights = self.table(valid)
        out = flat[:, self.cells].astype(np.float64)
        out[:, gaps] = np.einsum("tnk,nk->tn", flat[:, source], weights)
        return out


def derive(f: dict[str, np.ndarray], dz: np.ndarray) -> dict[str, np.ndarray]:
    """Land-cell values of the source fields -> store variables (store units)."""
    tair = f["Tair"] - LAPSE_RATE * dz
    r = f["Qair"] / (1 - f["Qair"])  # mixing ratio
    tv_avg = (f["Tair"] + tair) / 2 * (1 + 0.61 * r)
    psurf = f["PSurf"] * np.exp(-dz / (29.3 * tv_avg))
    vp = r / (0.622 + r) * psurf
    return {
        "lwdown": f["LWdown"],
        "prec": f["Prec"] * 86400,
        "psurf": psurf * 1e-3,
        "swdown": f["SWdown"],
        "tair": tair - 273.15,
        "vp": vp * 1e-3,
        "wind": f["Wind"],
    }


# --- store ------------------------------------------------------------------------


def year_coords(year: int) -> xr.Dataset:
    return xr.Dataset(
        coords={
            "time": pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"),
            "lat": ("lat", LAT, {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude", "axis": "Y"}),
            "lon": ("lon", LON, {"units": "degrees_east", "standard_name": "longitude", "long_name": "longitude", "axis": "X"}),
        }
    )


def global_attrs(source_attrs: dict, version: str) -> dict:
    attrs = {k: source_attrs[k] for k in GLOBAL_ATTRS_TO_COPY if k in source_attrs}
    attrs["title"] = f"{source_attrs.get('title', 'WFDE5')} downscaled to HydroSHEDS 5arc-min grid"
    attrs["institution"] = f"WU-DES;{source_attrs.get('institution', '')}"
    attrs["history"] = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}: WFDE5 v{version} hourly data from the Copernicus "
        f"Climate Data Store ({DATASET}) averaged to daily values; prec = Rainf + Snowf (CRU+GPCC); regridded to "
        "5 arcmin with xarray-regrid (prec conservative, other variables bilinear); land cells (with HydroSHEDS DEM data) without data filled "
        f"by inverse-distance weighting of the {FILL_NEIGHBOURS} nearest cells; tair, psurf and vp downscaled to the "
        "HydroSHEDS 15s DEM (5 arcmin mean) with a 6.5 K/km lapse rate, the hypsometric equation and a constant "
        "mixing ratio; stored as int16 scale/offset with Blosc compression by download-wfde5.py"
    )
    return attrs


def prepare_store(store: Path, year: int, source_attrs: dict, version: str) -> int:
    """Create or extend the store for ``year``; returns the year's first time index."""
    coords = year_coords(year)
    if not store.exists():
        logger.info("%d: creating %s", year, store)
        init_store(store, coords, global_attrs(source_attrs, version), {v: a | {"grid_mapping": "crs"} for v, a in VAR_ATTRS.items()}, CRS_ATTRS)
        return 0
    with xr.open_zarr(store, consolidated=False) as existing:
        for dim, values in (("lat", LAT), ("lon", LON)):
            if not np.allclose(existing[dim].values, values):
                raise ValueError(f"{dim} of {store} does not match the 5 arcmin grid")
        store_time = existing.indexes["time"]
    t_start = locate_year(store_time, coords.indexes["time"])
    if t_start is None:
        logger.info("%d: appending to %s", year, store)
        extend_store(store, coords)
        t_start = len(store_time)
    return t_start


def write_year(store: Path, year: int, work: Path, version: str, source_attrs: dict, filler: GapFiller, dz: np.ndarray) -> None:
    tic = time.perf_counter()
    daily = {name: np.concatenate([np.load(month_file(work, name, year, m)) for m in range(1, 13)]) for name in SOURCES}
    daily["Prec"] = daily.pop("Rainf") + daily.pop("Snowf")
    nt = len(year_coords(year).time)
    if any(len(a) != nt for a in daily.values()):
        raise ValueError(f"{year}: expected {nt} days")

    t_start = prepare_store(store, year, source_attrs, version)
    group = zarr.open_group(store, mode="r+", use_consolidated=False)
    grid = np.empty((24, NLAT * NLON), np.float32)
    for src_t, dst_t, _ in iter_blocks(t_start, nt, NLAT, NLAT):
        fields = {name: filler(regrid(a[src_t], name in CONSERVATIVE)) for name, a in daily.items()}
        for var, values in derive(fields, dz).items():
            array = group[var]
            g = grid[: len(values)]
            g.fill(np.nan)
            g[:, filler.cells] = values
            encoded, n_clipped = encode(g.reshape(-1, NLAT, NLON), None, float(array.attrs["scale_factor"]), float(array.attrs["add_offset"]))
            if n_clipped:
                logger.warning("%d %s: %d values outside the int16 range were clipped", year, var, n_clipped)
            array[dst_t] = encoded
        logger.info("  %d: days %d-%d written", year, src_t.start + 1, src_t.stop)

    progress = load_progress(store)
    progress[str(year)] = list(VARIABLES)
    save_progress(store, progress)
    zarr.consolidate_metadata(store)
    shutil.rmtree(work / "daily" / str(year))
    logger.info("%d: in store (%.0f s)", year, time.perf_counter() - tic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=DEFAULT_VERSION, help="WFDE5 version, e.g. 2.1 (default: %(default)s)")
    parser.add_argument("--output", type=Path, help="Zarr store (default: WFDE5_v{VERSION}_5arcmin.zarr)")
    parser.add_argument("--work-dir", type=Path, default=Path("cds_work"), help="DEM cache, and per version downloads and daily 0.5 degree data (default: %(default)s)")
    parser.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"), help="default: the full period of v3.1 (1979 2024); required for other versions")
    parser.add_argument("--dem", type=Path, default=Path("hyd_glo_dem_15s.tif"), help="HydroSHEDS DEM, see above for the download (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent CDS requests (default: %(default)s)")
    parser.add_argument("--prefetch-years", type=int, default=1, help="years downloaded ahead of the one being written (default: %(default)s)")
    args = parser.parse_args()
    if args.years is None:
        if args.version not in DEFAULT_YEARS:
            parser.error(f"--years is required for WFDE5 v{args.version}")
        args.years = DEFAULT_YEARS[args.version]
    if args.output is None:
        args.output = Path(f"WFDE5_v{args.version}_5arcmin.zarr")
    # Downloads and daily data depend on the version, the DEM average does not.
    work = args.work_dir / f"v{args.version}"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    progress = load_progress(args.output)
    years = [y for y in range(args.years[0], args.years[1] + 1) if set(progress.get(str(y), [])) < set(VARIABLES)]
    if not years:
        logger.info("All years are in the store")
        return

    # Leftovers of an interrupted run (nothing else uses this folder)
    shutil.rmtree(work / "download", ignore_errors=True)
    dem = load_dem(args.dem, args.work_dir / "dem_5min.npy")
    mask = np.isfinite(dem)  # land cells
    asurf, source_attrs = fetch_asurf(work, args.version)
    filler = GapFiller(mask)
    dz = dem[mask].astype(np.float64) - filler(regrid(asurf[None], conservative=False))[0]

    pool = ThreadPoolExecutor(args.workers)
    pending: dict[int, list[Future]] = {}

    def submit(year: int) -> None:
        pending[year] = [pool.submit(fetch_month, work, args.version, name, year, m) for m in range(1, 13) for name in SOURCES]

    try:
        for year in years[: 1 + args.prefetch_years]:
            submit(year)
        for i, year in enumerate(years):
            logger.info("%d: waiting for downloads", year)
            for future in pending.pop(year):
                future.result()
            if i + 1 + args.prefetch_years < len(years):
                submit(years[i + 1 + args.prefetch_years])
            write_year(args.output, year, work, args.version, source_attrs, filler, dz)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    logger.info("Done")


if __name__ == "__main__":
    main()
