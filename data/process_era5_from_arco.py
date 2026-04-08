"""
ERA5 + ENTLN Processing Pipeline  —  ARCO-ERA5 edition
Streams ERA5 data directly from Google Cloud Storage (no local NC files needed).
Loads the entire case time range in ONE batch fetch, then iterates in memory.
crwc + cswc are loaded from local CDS-downloaded NC files (not in ARCO).

Output:
  local_processed_data/ERA5/{case_name}/1_hours/{timestamp}.nc

Variable coverage:
  ARCO single-level  (17):        cape, cin, kx, totot, d2m, tcwv, tciw, tclw,
                                  cp, crr, tp, vimd, t2m, msl, sp, cbh, hcc
  ARCO pressure-level (7 × 37):   u, v, w, t, q, ciwc, clwc
  Derived  (1 × 37):              r  (relative humidity, from q + t)
  CDS local (2 × 37):             crwc, cswc  (run download_era5_crwc_cswc.py first)

  Total pressure-level vars: 10 × 37 = 370  ✓

Karin Pitlik
"""

import xarray as xr
import numpy as np
import scipy.io
import netCDF4 as nc4
import os
import pandas as pd
from datetime import datetime
import gcsfs

# ============================================================
#  CONFIGURATION
# ============================================================

CASES = [
    ('Case1_Nov_2022_23_25', '2022-11-23', '2022-11-25'),
    ('Case2_Jan_2023_11_16', '2023-01-11', '2023-01-16'),
    ('Case3_Mar_2023_13_15', '2023-03-13', '2023-03-15'),
    ('Case4_Apr_2023_09_13', '2023-04-09', '2023-04-13'),
    ('Case5_Jan_2024_26_31', '2024-01-26', '2024-01-31'),
]

MIN_LAT = 27.296
MAX_LAT = 36.598
MIN_LON = 27.954
MAX_LON = 39.292

BASE_PATH   = '/home/ubuntu/Desktop'
ENTLN_PATH  = os.path.join(BASE_PATH, 'local_raw_data/ENTLN')
OUTPUT_PATH = os.path.join(BASE_PATH, 'local_processed_data/ERA5')

ARCO_ZARR_PATH = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900, 925,
    950, 975, 1000
]

# ARCO long name → short name
ARCO_PRESSURE_VARS = {
    'u_component_of_wind':                 'u',
    'v_component_of_wind':                 'v',
    'vertical_velocity':                   'w',
    'temperature':                         't',
    'specific_humidity':                   'q',
    'specific_cloud_ice_water_content':    'ciwc',
    'specific_cloud_liquid_water_content': 'clwc',
}

# Local CDS files for variables missing from ARCO (short_name → filename stem)
CDS_PRESSURE_VARS = {
    'crwc': 'specific_rain_water_content',   # filename: specific_rain_water_content.nc
    'cswc': 'specific_snow_water_content',   # filename: specific_snow_water_content.nc
}

ARCO_SINGLE_VARS = {
    'convective_available_potential_energy':     'cape',
    'convective_inhibition':                     'cin',
    'k_index':                                   'kx',
    'total_totals_index':                        'totot',
    '2m_dewpoint_temperature':                   'd2m',
    'total_column_water_vapour':                 'tcwv',
    'total_column_cloud_ice_water':              'tciw',
    'total_column_cloud_liquid_water':           'tclw',
    'convective_precipitation':                  'cp',
    'convective_rain_rate':                      'crr',
    'total_precipitation':                       'tp',
    'vertically_integrated_moisture_divergence': 'vimd',
    '2m_temperature':                            't2m',
    'mean_sea_level_pressure':                   'msl',
    'surface_pressure':                          'sp',
    'cloud_base_height':                         'cbh',
    'high_cloud_cover':                          'hcc',
}


# ============================================================
#  OPEN ARCO-ERA5 STORE (once at startup)
# ============================================================

def open_arco_store():
    """Open the ARCO-ERA5 Zarr store with anonymous GCS access."""
    print("Opening ARCO-ERA5 Zarr store (anonymous)...")
    fs = gcsfs.GCSFileSystem(token='anon')
    store = gcsfs.GCSMap(ARCO_ZARR_PATH, gcs=fs)
    ds = xr.open_zarr(store, consolidated=True)
    print(f"  ARCO store opened. Available variables: {len(list(ds.data_vars))}")
    print(f"  Time range: {pd.Timestamp(ds.time.values[0])} → {pd.Timestamp(ds.time.values[-1])}")
    return ds


# ============================================================
#  DERIVE RELATIVE HUMIDITY FROM q AND t
# ============================================================

def derive_relative_humidity(t_da, q_da, plevel_hpa):
    """
    Compute relative humidity (%) from temperature (K), specific humidity (kg/kg),
    and pressure level (hPa) using Bolton's formula.

    Parameters
    ----------
    t_da       : xarray.DataArray  temperature in Kelvin
    q_da       : xarray.DataArray  specific humidity in kg/kg
    plevel_hpa : float             pressure level in hPa

    Returns
    -------
    xarray.DataArray  relative humidity in %
    """
    T_c = t_da - 273.15                                          # K → °C
    e_sat = 6.112 * np.exp(17.67 * T_c / (T_c + 243.5))         # hPa, Bolton 1980
    p = float(plevel_hpa)
    e = q_da * p / (0.622 + 0.378 * q_da)                       # hPa
    rh = 100.0 * e / e_sat
    rh = rh.clip(0.0, 100.0)
    return rh


# ============================================================
#  LOAD crwc + cswc FROM LOCAL CDS NC FILES
# ============================================================

def load_cds_pressure_batch(case_name, start_dt, end_dt):
    """
    Load crwc and cswc from local CDS-downloaded NC files for a case.
    Returns a dict: { 'crwc': DataArray(time, level, lat, lon),
                      'cswc': DataArray(time, level, lat, lon) }
    or an empty dict if files are not found.
    """
    pressure_folder = os.path.join(BASE_PATH, 'local_raw_data', case_name, 'ERA5_pressure')
    result = {}

    for short_name, file_stem in CDS_PRESSURE_VARS.items():
        nc_path = os.path.join(pressure_folder, f'{file_stem}.nc')
        if not os.path.exists(nc_path):
            print(f"  WARNING: {short_name} not found at {nc_path} — will be missing from output")
            continue

        ds = xr.open_dataset(nc_path)

        # Rename valid_time → time if needed
        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})

        # Find the variable (CDS may name it differently)
        var_name = None
        for v in ds.data_vars:
            if short_name in v.lower() or file_stem.split('_')[1] in v.lower():
                var_name = v
                break
        if var_name is None:
            var_name = list(ds.data_vars)[0]   # fallback: take first variable

        # Find pressure dimension
        plevel_dim = None
        for dim in ['pressure_level', 'level', 'plev']:
            if dim in ds.dims:
                plevel_dim = dim
                break

        if plevel_dim is None:
            print(f"  WARNING: No pressure dimension in {nc_path} — skipping {short_name}")
            ds.close()
            continue

        # Crop to domain and case time window
        t_slice = slice(start_dt, end_dt - pd.Timedelta(hours=1))
        da = (
            ds[var_name]
            .sel(
                time=t_slice,
                latitude=slice(MAX_LAT, MIN_LAT),
                longitude=slice(MIN_LON, MAX_LON),
            )
            .load()
        )
        result[short_name] = (da, plevel_dim)
        ds.close()
        print(f"  Loaded local CDS {short_name}: shape {da.shape}")

    return result


# ============================================================
#  BATCH-LOAD ERA5 FOR AN ENTIRE CASE  (one GCS fetch)
# ============================================================

def load_case_from_arco(arco_ds, start_dt, end_dt):
    """
    Load ALL ERA5 variables for the case time window in a single batch.
    Crops to the domain and pulls into RAM once — much faster than per-hour fetches.

    Returns
    -------
    sl_batch  : xarray.Dataset  single-level vars,  dims (time, latitude, longitude)
    pl_batch  : xarray.Dataset  pressure-level vars, dims (time, level, latitude, longitude)
    plevel_dim: str             name of the pressure dimension in pl_batch
    """
    # ── Identify available variables ────────────────────────────────────────
    avail_single   = [k for k in ARCO_SINGLE_VARS   if k in arco_ds.data_vars]
    avail_pressure = [k for k in ARCO_PRESSURE_VARS if k in arco_ds.data_vars]

    # ERA5 time slice (inclusive on both ends)
    t_slice = slice(start_dt, end_dt - pd.Timedelta(hours=1))

    print(f"  Fetching single-level  vars from ARCO: {len(avail_single)} vars ...")
    sl_batch = (
        arco_ds[avail_single]
        .sel(
            time=t_slice,
            latitude=slice(MAX_LAT, MIN_LAT),
            longitude=slice(MIN_LON, MAX_LON),
        )
        .load()   # ← ONE network fetch for the whole case
    )
    print(f"  Single-level fetch done. Shape example: {sl_batch[avail_single[0]].shape}")

    # Determine pressure dimension name
    plevel_dim = None
    for dim in ['level', 'pressure_level', 'plev']:
        if dim in arco_ds.dims:
            plevel_dim = dim
            break

    pl_batch = None
    if avail_pressure and plevel_dim is not None:
        print(f"  Fetching pressure-level vars from ARCO: {len(avail_pressure)} vars × {len(PRESSURE_LEVELS)} levels ...")
        pl_batch = (
            arco_ds[avail_pressure]
            .sel(
                time=t_slice,
                **{plevel_dim: PRESSURE_LEVELS},
                latitude=slice(MAX_LAT, MIN_LAT),
                longitude=slice(MIN_LON, MAX_LON),
            )
            .load()   # ← ONE network fetch for the whole case
        )
        print(f"  Pressure-level fetch done. Shape example: {pl_batch[avail_pressure[0]].shape}")
    else:
        print("  WARNING: No pressure-level dimension found — skipping pressure vars")

    return sl_batch, pl_batch, plevel_dim


# ============================================================
#  EXTRACT ONE TIMESTEP FROM THE BATCH (pure in-memory)
# ============================================================

def extract_timestep(sl_batch, pl_batch, plevel_dim, cds_batch, timestamp):
    """
    Slice a single hour from the pre-loaded batch datasets.
    Flattens pressure-level vars to e.g. u_500, ciwc_850.
    Derives relative humidity (r_*) from q and t.
    Merges crwc + cswc from local CDS files.
    Returns an xarray.Dataset with dims (latitude, longitude) only.
    """
    t = pd.Timestamp(timestamp)
    all_vars = {}

    # ── Single-level ────────────────────────────────────────────────────────
    sl_slice = sl_batch.sel(time=t)
    for long_name, short_name in ARCO_SINGLE_VARS.items():
        if long_name in sl_slice:
            all_vars[short_name] = sl_slice[long_name].squeeze(drop=True)

    # ── ARCO Pressure-level + derived RH ────────────────────────────────────
    if pl_batch is not None and plevel_dim is not None:
        pl_slice = pl_batch.sel(time=t)
        for long_name, short_name in ARCO_PRESSURE_VARS.items():
            if long_name not in pl_slice:
                continue
            var_all_levels = pl_slice[long_name]
            for plevel in PRESSURE_LEVELS:
                new_name = f"{short_name}_{int(plevel)}"
                try:
                    arr = var_all_levels.sel({plevel_dim: plevel})
                    all_vars[new_name] = arr.drop_vars(plevel_dim, errors='ignore').squeeze(drop=True)
                except KeyError:
                    pass

        # ── Derive r (relative humidity) from q + t ──────────────────────
        if 'temperature' in pl_slice and 'specific_humidity' in pl_slice:
            t_all = pl_slice['temperature']
            q_all = pl_slice['specific_humidity']
            for plevel in PRESSURE_LEVELS:
                rh_name = f"r_{int(plevel)}"
                try:
                    t_lev = t_all.sel({plevel_dim: plevel}).drop_vars(plevel_dim, errors='ignore').squeeze(drop=True)
                    q_lev = q_all.sel({plevel_dim: plevel}).drop_vars(plevel_dim, errors='ignore').squeeze(drop=True)
                    all_vars[rh_name] = derive_relative_humidity(t_lev, q_lev, plevel)
                except KeyError:
                    pass

    # ── CDS local vars: crwc + cswc ──────────────────────────────────────────
    for short_name, (da, cds_plevel_dim) in cds_batch.items():
        try:
            da_t = da.sel(time=t)
        except KeyError:
            continue
        for plevel in PRESSURE_LEVELS:
            new_name = f"{short_name}_{int(plevel)}"
            try:
                arr = da_t.sel({cds_plevel_dim: plevel})
                all_vars[new_name] = arr.drop_vars(cds_plevel_dim, errors='ignore').squeeze(drop=True)
            except KeyError:
                pass

    if not all_vars:
        return None
    return xr.Dataset(all_vars)


# ============================================================
#  LOAD ENTLN DATA
# ============================================================

def load_entln_mat(mat_path):
    """Load ENTLN _struct.mat → DataFrame with lat, lon, UTC columns."""
    S = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    key = [k for k in S.keys() if not k.startswith('_')][0]
    entln = S[key]

    lats    = np.array(entln.lat).flatten().astype(float)
    lons    = np.array(entln.lon).flatten().astype(float)
    utc_raw = np.array(entln.UTC).flatten()
    utc_strs = [str(u).strip() for u in utc_raw]
    utc = pd.to_datetime(utc_strs, format='%d-%b-%Y %H:%M:%S', errors='coerce')

    df = pd.DataFrame({'lat': lats, 'lon': lons, 'UTC': utc})
    return df.dropna(subset=['UTC'])


# ============================================================
#  BIN ENTLN TO ERA5 GRID
# ============================================================

def bin_entln_to_grid(df_entln, lats_1d, lons_1d):
    """Bin lightning pulses onto the ERA5 lat/lon grid."""
    nlat, nlon = len(lats_1d), len(lons_1d)

    if df_entln.empty:
        return np.zeros((nlat, nlon), dtype=np.float32)

    dlat = abs(float(lats_1d[1]) - float(lats_1d[0])) / 2.0
    dlon = abs(float(lons_1d[1]) - float(lons_1d[0])) / 2.0

    lats_sorted = np.sort(lats_1d)
    lat_edges = np.concatenate([
        [lats_sorted[0] - dlat],
        lats_sorted[:-1] + np.diff(lats_sorted) / 2,
        [lats_sorted[-1] + dlat]
    ])
    lon_edges = np.concatenate([
        [lons_1d[0] - dlon],
        lons_1d[:-1] + np.diff(lons_1d) / 2,
        [lons_1d[-1] + dlon]
    ])

    N, _, _ = np.histogram2d(
        df_entln.lat.values, df_entln.lon.values,
        bins=[lat_edges, lon_edges]
    )
    if lats_1d[0] > lats_1d[-1]:
        N = N[::-1, :]
    return N.astype(np.float32)


# ============================================================
#  SAVE HOURLY NC
# ============================================================

def save_hourly_nc(outfile, era5_slice, lightning_grid, lats, lons, t_start):
    """Save one hourly NC file: all ERA5 variables + ENTLN pulse count."""
    if os.path.exists(outfile):
        os.remove(outfile)

    nlat, nlon = len(lats), len(lons)

    with nc4.Dataset(outfile, 'w') as ds:
        ds.createDimension('lat', nlat)
        ds.createDimension('lon', nlon)

        lat_v = ds.createVariable('lat', 'f4', ('lat',))
        lon_v = ds.createVariable('lon', 'f4', ('lon',))
        lat_v[:] = lats;  lat_v.units = 'degrees_north'
        lon_v[:] = lons;  lon_v.units = 'degrees_east'

        for vname in era5_slice.data_vars:
            data = era5_slice[vname].values
            if data.ndim != 2:
                continue
            v = ds.createVariable(vname, 'f4', ('lat', 'lon'), fill_value=np.nan)
            v[:] = data.astype(np.float32)
            v.units     = era5_slice[vname].attrs.get('units', '')
            v.long_name = era5_slice[vname].attrs.get('long_name', vname)

        ln_v = ds.createVariable('entln_count', 'f4', ('lat', 'lon'), fill_value=np.nan)
        ln_v[:] = lightning_grid
        ln_v.long_name = 'ENTLN lightning pulse count'
        ln_v.units     = 'count'

        ds.description   = 'ERA5 (ARCO) + ENTLN lightning counts'
        ds.created_by    = 'Karin Pitlik'
        ds.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ds.timestamp_utc = str(t_start)


# ============================================================
#  PROCESS ONE CASE
# ============================================================

def process_case(case_name, start_str, end_str, arco_ds):
    print(f"\n{'='*60}")
    print(f"PROCESSING: {case_name}")
    print(f"{'='*60}")

    out_folder = os.path.join(OUTPUT_PATH, case_name, '1_hours')
    os.makedirs(out_folder, exist_ok=True)

    start_dt = pd.Timestamp(start_str)
    end_dt   = pd.Timestamp(end_str) + pd.Timedelta(days=1)
    case_times = pd.date_range(start=start_dt, end=end_dt, freq='1h', inclusive='left')

    # Check which files already exist — skip the whole case if all done
    missing_times = []
    for t in case_times:
        tstr     = t.strftime('%Y-%m-%d_%H_%M_%S')
        tend_str = (t + pd.Timedelta(hours=1)).strftime('%Y-%m-%d_%H_%M_%S')
        outfile  = os.path.join(out_folder, f'{tstr}_{tend_str}.nc')
        if not os.path.exists(outfile):
            missing_times.append(t)

    print(f"  Timesteps total: {len(case_times)} | to process: {len(missing_times)} | already done: {len(case_times) - len(missing_times)}")
    if not missing_times:
        print("  All files exist — skipping case.")
        return

    # ── Load ENTLN ──────────────────────────────────────────────────────────
    entln_file = os.path.join(ENTLN_PATH, f'ENTLN_pulse_{case_name}_struct.mat')
    if not os.path.exists(entln_file):
        print(f"  WARNING: ENTLN file not found — lightning will be zeros")
        df_entln_full = pd.DataFrame(columns=['lat', 'lon', 'UTC'])
    else:
        df_entln_full = load_entln_mat(entln_file)
        print(f"  ENTLN loaded: {len(df_entln_full):,} pulses")

    # ── ONE batch fetch from ARCO for the whole case ─────────────────────
    sl_batch, pl_batch, plevel_dim = load_case_from_arco(arco_ds, start_dt, end_dt)

    lats = sl_batch.latitude.values
    lons = sl_batch.longitude.values
    print(f"  Domain grid: {len(lats)} lats × {len(lons)} lons")

    # ── Load crwc + cswc from local CDS files ────────────────────────────
    cds_batch = load_cds_pressure_batch(case_name, start_dt, end_dt)
    if cds_batch:
        print(f"  CDS local vars loaded: {list(cds_batch.keys())}")
    else:
        print("  No CDS local vars found (crwc/cswc will be absent). Run download_era5_crwc_cswc.py first.")

    # ── Iterate over hours in memory ─────────────────────────────────────
    n_ok = n_skip = 0
    for t_start in missing_times:
        t_end    = t_start + pd.Timedelta(hours=1)
        tstr     = t_start.strftime('%Y-%m-%d_%H_%M_%S')
        tend_str = t_end.strftime('%Y-%m-%d_%H_%M_%S')
        outfile  = os.path.join(out_folder, f'{tstr}_{tend_str}.nc')

        try:
            era5_slice = extract_timestep(sl_batch, pl_batch, plevel_dim, cds_batch, t_start)
        except Exception as e:
            print(f"  ERROR {tstr}: {e}")
            n_skip += 1
            continue

        if era5_slice is None:
            print(f"  SKIP {tstr}: no ERA5 vars")
            n_skip += 1
            continue

        mask_t = (df_entln_full['UTC'] >= t_start) & (df_entln_full['UTC'] < t_end)
        lightning_grid = bin_entln_to_grid(df_entln_full[mask_t], lats, lons)

        save_hourly_nc(outfile, era5_slice, lightning_grid, lats, lons, t_start)
        n_ok += 1
        print(f"  Saved: {tstr} | lightning: {mask_t.sum():4d} pulses | ERA5 vars: {len(era5_slice.data_vars)}")

    print(f"  Done: {case_name} — {n_ok} saved, {n_skip} errors")


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    arco_ds = open_arco_store()

    for case_name, start_str, end_str in CASES:
        process_case(case_name, start_str, end_str, arco_ds)

    print("\n\nAll cases processed!")
