"""Layout and encoding of the 5 arcmin forcing store (Zarr v3).

All variables live in one store, with dimensions (time, lat, lon):

- chunks of 24 time x 240 lat x 240 lon;
- stored as int16 with CF ``scale_factor``/``add_offset`` (NaN -> -32768);
- compressed with Blosc (zstd, byte shuffle).

The scale factors are powers of two, so every stored value (including zero,
e.g. zero precipitation) decodes exactly, in float32 as well as float64. The
scale/offset is fixed per variable (not derived from the data) so appended
years share the same encoding. Values outside the range are clipped.

Progress is tracked per (year, variable) in STORE.progress.json, so an
interrupted run resumes where it stopped.
"""

import json
import os
from pathlib import Path

import dask.array
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

DIMS = ("time", "lat", "lon")
CHUNKS = {"time": 24, "lat": 240, "lon": 240}

# Minimum physical value (source units) and power-of-two scale factor. The
# int16 range [-32767, 32767] then covers [vmin, vmin + 65534 * scale].
# Non-negative quantities start at 0 (so 0 is exact), tair is centred on 0.
VARIABLES = {
    "lwdown": (0.0, 2**-5),  # W m-2,   0 .. 2048,   step 0.031
    "prec": (0.0, 2**-5),  # mm,        0 .. 2048,   step 0.031
    "psurf": (0.0, 2**-8),  # kPa,      0 .. 256,    step 0.0039
    "swdown": (0.0, 2**-5),  # W m-2,   0 .. 2048,   step 0.031
    "tair": (-256.0, 2**-7),  # degC,   -256 .. 256, step 0.0078
    "vp": (0.0, 2**-11),  # kPa,        0 .. 32,     step 0.00049
    "wind": (0.0, 2**-9),  # m s-1,     0 .. 128,    step 0.002
}

INT16_FILL = np.int16(-32768)
INT16_MAX = 32767
COMPRESSORS = (BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),)
TIME_UNITS = "days since 1970-01-01"


def scale_offset(var: str) -> tuple[float, float]:
    vmin, scale = VARIABLES[var]
    return scale, vmin + INT16_MAX * scale


def lazy_template(coords: xr.Dataset, var_attrs: dict) -> xr.Dataset:
    """Dataset of lazy (never computed) data variables, used to write metadata only."""
    shape = tuple(coords.sizes[d] for d in DIMS)
    chunks = tuple(CHUNKS[d] for d in DIMS)
    return xr.Dataset(
        {
            var: (DIMS, dask.array.empty(shape, dtype="float32", chunks=chunks), var_attrs.get(var, {}))
            for var in VARIABLES
        },
        coords=coords.coords,
    )


def init_store(store: Path, coords: xr.Dataset, global_attrs: dict, var_attrs: dict, crs_attrs: dict) -> None:
    """Write the store's metadata and coordinates (data is filled later)."""
    ds = lazy_template(coords, var_attrs)
    ds.attrs = global_attrs
    if crs_attrs:
        ds["crs"] = xr.Variable((), np.int32(0), crs_attrs)

    encoding = {"time": {"units": TIME_UNITS, "calendar": "standard", "dtype": "int32"}}
    for var in VARIABLES:
        scale, offset = scale_offset(var)
        encoding[var] = {
            "dtype": "int16",
            "scale_factor": scale,
            "add_offset": offset,
            "_FillValue": INT16_FILL,
            "fill_value": INT16_FILL,  # zarr-level fill: all-ocean chunks are not stored
            "chunks": tuple(CHUNKS[d] for d in DIMS),
            "compressors": COMPRESSORS,
        }
    ds.to_zarr(store, mode="w-", encoding=encoding, compute=False, consolidated=False, zarr_format=3)


def extend_store(store: Path, coords: xr.Dataset) -> None:
    """Grow the time dimension and write the new time values (data is filled later)."""
    ds = lazy_template(coords, {}).drop_vars(["lat", "lon"])
    # Appending replaces the global attributes with those of ds: keep the existing ones.
    ds.attrs = dict(zarr.open_group(store, mode="r", use_consolidated=False).attrs)
    # safe_chunks=False: nothing is computed here, the writes are aligned with the chunks.
    ds.to_zarr(store, append_dim="time", compute=False, consolidated=False, safe_chunks=False)


def locate_year(store_time: pd.DatetimeIndex, time: pd.DatetimeIndex) -> int | None:
    """Start index of ``time`` in the store, or None if it still has to be appended."""
    if time[0] in store_time:
        start = store_time.get_loc(time[0])
        if not store_time[start : start + len(time)].equals(time):
            raise ValueError(f"Store has different time steps for {time[0]:%Y}")
        return start
    if time[0] < store_time[-1]:
        raise ValueError(f"{time[0]:%Y} lies before the end of the store: years must be appended in order")
    step = store_time[-1] - store_time[-2]
    if time[0] - store_time[-1] != step:
        raise ValueError(f"Appending {time[0]:%Y} would leave a gap after {store_time[-1]:%Y-%m-%d}")
    return None


def iter_blocks(t_start: int, nt: int, nlat: int, lat_rows: int):
    """(source time, store time, lat) slices, aligned with the store's chunk grid."""
    t, t_end = t_start, t_start + nt
    while t < t_end:
        t_next = min((t // CHUNKS["time"] + 1) * CHUNKS["time"], t_end)
        for lat0 in range(0, nlat, lat_rows):
            yield slice(t - t_start, t_next - t_start), slice(t, t_next), slice(lat0, min(lat0 + lat_rows, nlat))
        t = t_next


def encode(data: np.ndarray, src_fill, scale: float, offset: float) -> tuple[np.ndarray, int]:
    """float -> int16 (round, clip, NaN/fill -> INT16_FILL), in float32 to save memory.

    Rounds (data - vmin) / scale before shifting by INT16_MAX: with vmin = 0 and a
    power-of-two scale that is exact in float32, unlike subtracting the offset first.
    """
    invalid = ~np.isfinite(data)
    if src_fill is not None:
        invalid |= data == src_fill
    vmin = offset - INT16_MAX * scale
    q = np.subtract(data, np.float32(vmin), dtype=np.float32)
    q /= np.float32(scale)
    np.rint(q, out=q)
    q -= np.float32(INT16_MAX)
    q[invalid] = 0
    n_clipped = int(np.count_nonzero(np.abs(q) > INT16_MAX))
    np.clip(q, -INT16_MAX, INT16_MAX, out=q)
    out = q.astype(np.int16)
    out[invalid] = INT16_FILL
    return out, n_clipped


def progress_file(store: Path) -> Path:
    return store.with_name(store.name + ".progress.json")


def load_progress(store: Path) -> dict[str, list[str]]:
    """{year: [variables written]}; empty if the store does not exist (yet)."""
    if not store.exists() or not progress_file(store).exists():
        return {}
    return json.loads(progress_file(store).read_text())


def save_progress(store: Path, progress: dict[str, list[str]]) -> None:
    tmp = progress_file(store).with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=1))
    os.replace(tmp, progress_file(store))
