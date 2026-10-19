# CDS pipeline vs. existing WFDES store (1979)

Comparison of `download-wfde5.py` (then `cds_to_zarr.py`) output with `WFDES_v2.0_5arcmin.zarr` (cdo-made v2.0 files), for
January and July 1979 (62 days, 2,288,857 land cells). Both sides use the same int16 encoding, so
p99 = 0 means identical stored values.

## Pipeline (`download-wfde5.py`)

- WFDE5 hourly 0.5° from the CDS, averaged to daily means (00–23 UTC); prec = Rainf + Snowf (CRU+GPCC).
- Regridding to 5′ with xarray-regrid: conservative for prec, bilinear for everything else (incl. Tair, PSurf, ASurf, which were nearest neighbour in the original files).
- Land cells without data filled by inverse-distance weighting of the 4 nearest cells (as `cdo setmisstodis`), masked with `vic_glo_msk_5m.nc` (the script now uses the cells with HydroSHEDS DEM data instead).
- tair, psurf and vp downscaled with the HydroSHEDS 15″ DEM averaged per 5′ cell. The original DEM was confirmed to be this mean (median difference 2 cm).

## Results

Comparisons:

- **Method**: v2.0 data through the new pipeline, minus WFDES. This isolates the processing.
- **Version**: v3.1 minus v2.0, both through the same pipeline. This isolates the data.
- **Total**: v3.1 through the new pipeline, minus WFDES. This is what the new store will differ by.

| var (units) | Method | Version | Total |
|---|---|---|---|
| tair (°C) | MAE 0.27, p99 1.7 | MAE 0.82, bias −0.12, p99 6.8 | MAE 0.90, p99 7.0 |
| psurf (kPa) | MAE 0.021, p99 0.20 | MAE 0.029, bias −0.013, p99 0.47 | MAE 0.042, p99 0.53 |
| vp (kPa) | MAE 0.0003, p99 0.003 | MAE 0.037 (3.3 %), p99 0.22 | MAE 0.037, p99 0.22 |
| prec (mm) | identical on WFDE5 land | MAE 0.15 (6.9 %), bias ≈ 0, p99 3.4 | MAE 0.15, p99 3.4 |
| lwdown (W m⁻²) | p99 0 | MAE 4.3 (1.4 %), bias −0.5, p99 30 | MAE 4.3, p99 30 |
| swdown (W m⁻²) | p99 0 | MAE 4.7 (2.8 %), bias −1.1, p99 37 | MAE 4.7, p99 37 |
| wind (m s⁻¹) | p99 0 | identical | identical |

## Conclusions

- **The method reproduces the original processing.** prec, lwdown, swdown, wind and vp are identical except for a few gap-filled coastal cells (up to 11 W m⁻² or 20 mm, from small differences between this fill and `setmisstodis`).
- **tair and psurf differ only through the switch to bilinear regridding.** Tair, PSurf and ASurf are now bilinear instead of nearest neighbour, which gives tair an MAE of 0.27 K, with rare outliers up to 30 K in gap-filled coastal cells. Setting these three back to nearest neighbour would reproduce the original.
- **Almost all differences in the new store come from v3.1 vs v2.0.** They are already present in the 0.5° source data:
  - Tair changes most (MAE 0.8 K; 1 % of values by ≥ 7 K).
  - Precipitation is redistributed (6.9 % MAE) with an unchanged total.
  - Shortwave radiation is about 1 W m⁻² lower on average.
  - Wind is unchanged.
- **Limitation:** these are 2 months of 1979 only, so trends and seasonal variation of the version differences are not covered.
