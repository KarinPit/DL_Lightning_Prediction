"""
append_cases_to_zarr.py — Append missing case date ranges directly to ERA5.zarr + Lightning.zarr
================================================================================================
Fetches Case1 (Nov 2022) and Case5 (Jan 2024) from ARCO-ERA5 and appends them
directly to the Zarr stores — no NC file intermediate step needed.

Why: ERA5.zarr only has 2023. Cases 1 and 5 are Nov 2022 and Jan 2024.
     This lets us replicate the full 5-case training setup in the Zarr pipeline.

ENTLN sources:
  Case1 (Nov 2022) → Total_DATA_Seasons_2017_2022.mat (covers 2017–2022)
  Case5 (Jan 2024) → ENTLN_pulse_Year2024_struct.mat  (if available, else zeros)

Usage:
    python3 data/append_cases_to_zarr.py
"""

import gcsfs
import numpy as np
import pandas as pd
import scipy.io
import xarray as xr

# ── Paths ─────────────────────────────────────────────────────────────────────
BUCKET         = '/home/ec2-user/thesis-bucket'
ERA5_ZARR      = f'{BUCKET}/Zarr/ERA5.zarr'
LIGHTNING_ZARR = f'{BUCKET}/Zarr/Lightning.zarr'
ENTLN_DIR      = f'{BUCKET}/Lightning_Data/ENTLN'

ARCO_ZARR_PATH = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

# ── Cases to append (not in 2023 Zarr) ────────────────────────────────────────
MISSING_CASES = [
    ('Case1_Nov_2022_23_25', '2022-11-23', '2022-11-25'),
    ('Case5_Jan_2024_26_31', '2024-01-26', '2024-01-31'),
]

# ── Domain ────────────────────────────────────────────────────────────────────
MIN_LAT, MAX_LAT = 27.296, 36.598
MIN_LON, MAX_LON = 27.954, 39.292
TIME_CHUNK = 24

# ── ARCO variable maps ────────────────────────────────────────────────────────
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

ARCO_PRESSURE_VARS = {
    'u_component_of_wind':                 'u',
    'v_component_of_wind':                 'v',
    'vertical_velocity':                   'w',
    'temperature':                         't',
    'specific_humidity':                   'q',
    'specific_cloud_ice_water_content':    'ciwc',
    'specific_cloud_liquid_water_content': 'clwc',
}

ALL_PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900, 925,
    950, 975, 1000,
]


# ── ENTLN helpers ─────────────────────────────────────────────────────────────

def load_entln_mat(mat_path):
    """Load a .mat ENTLN file → DataFrame with lat, lon, UTC.
    Supports both scipy-readable structs and MATLAB table format (via pymatreader).
    """
    print(f"  Loading ENTLN: {mat_path}")
    from pymatreader import read_mat
    data = read_mat(mat_path)
    key  = [k for k in data.keys() if not k.startswith('_')][0]
    tbl  = data[key]
    print(f"  Fields: {list(tbl.keys())}")

    # Try common lat/lon field names
    lat_key = next((k for k in tbl if 'lat' in k.lower()), None)
    lon_key = next((k for k in tbl if 'lon' in k.lower() or 'long' in k.lower()), None)
    utc_key = next((k for k in tbl if 'utc' in k.lower() or 'time' in k.lower() or 'date' in k.lower()), None)

    if not lat_key or not lon_key or not utc_key:
        raise KeyError(f"Could not find lat/lon/UTC fields. Available: {list(tbl.keys())}")

    lats = np.array(tbl[lat_key]).flatten().astype(float)
    lons = np.array(tbl[lon_key]).flatten().astype(float)
    utc  = pd.to_datetime(
        [str(u).strip() for u in np.array(tbl[utc_key]).flatten()],
        format='%d-%b-%Y %H:%M:%S', errors='coerce'
    )
    df = pd.DataFrame({'lat': lats, 'lon': lons, 'UTC': utc}).dropna(subset=['UTC'])
    df = df[(df.lat != 0) | (df.lon != 0)]
    print(f"  Loaded {len(df):,} pulses  ({df.UTC.min().date()} → {df.UTC.max().date()})")
    return df


def find_entln_for_case(case_name, year):
    """Find the best available ENTLN mat file for this case's year."""
    import os
    cases_dir = f'{ENTLN_DIR}/Cases'
    candidates = [
        f'{cases_dir}/ENTLN_pulse_{case_name}.mat',           # e.g. ENTLN_pulse_Case1_Nov_2022_23_25.mat
        f'{ENTLN_DIR}/ENTLN_pulse_Year{year}_struct.mat',
        f'{ENTLN_DIR}/ENTLN_pulse_{case_name}_struct.mat',
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"  Found ENTLN: {path}")
            return path
    print(f"  Searched:")
    for p in candidates:
        print(f"    {p}")
    return None


def bin_to_grid(df, lats_1d, lons_1d):
    nlat, nlon = len(lats_1d), len(lons_1d)
    if df.empty:
        return np.zeros((nlat, nlon), dtype=np.float32)
    dlat = abs(float(lats_1d[1]) - float(lats_1d[0])) / 2.0
    dlon = abs(float(lons_1d[1]) - float(lons_1d[0])) / 2.0
    lats_s = np.sort(lats_1d)
    lat_edges = np.concatenate([[lats_s[0]-dlat], lats_s[:-1]+np.diff(lats_s)/2, [lats_s[-1]+dlat]])
    lon_edges = np.concatenate([[lons_1d[0]-dlon], lons_1d[:-1]+np.diff(lons_1d)/2, [lons_1d[-1]+dlon]])
    N, _, _ = np.histogram2d(df.lat.values, df.lon.values, bins=[lat_edges, lon_edges])
    if lats_1d[0] > lats_1d[-1]:
        N = N[::-1, :]
    return N.astype(np.float32)


# ── ARCO fetch + flatten ───────────────────────────────────────────────────────

def fetch_from_arco(arco_ds, start_str, end_str):
    """
    Fetch ERA5 data from ARCO for a date range and return as xr.Dataset
    with flattened variable names (e.g. u_500, ciwc_200) matching ERA5.zarr format.
    """
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str) + pd.Timedelta(hours=23)  # inclusive
    t_slice = slice(start, end)

    plevel_dim = next((d for d in ['level', 'pressure_level', 'plev']
                       if d in arco_ds.dims), None)

    avail_single   = [k for k in ARCO_SINGLE_VARS   if k in arco_ds.data_vars]
    avail_pressure = [k for k in ARCO_PRESSURE_VARS if k in arco_ds.data_vars]

    all_datasets = []

    # ── Single-level ──────────────────────────────────────────────────────────
    print(f"  Fetching {len(avail_single)} single-level vars from ARCO...")
    sl = (
        arco_ds[avail_single]
        .sel(time=t_slice,
             latitude=slice(MAX_LAT, MIN_LAT),
             longitude=slice(MIN_LON, MAX_LON))
        .load()
        .rename({'latitude': 'lat', 'longitude': 'lon'})
    )
    # Rename long names to short names
    rename_map = {k: v for k, v in ARCO_SINGLE_VARS.items() if k in sl}
    sl = sl.rename(rename_map)
    all_datasets.append(sl)
    print(f"  Single-level done: {list(sl.data_vars)[:4]}...")

    # ── Pressure-level (one variable at a time) ───────────────────────────────
    if avail_pressure and plevel_dim:
        print(f"  Fetching {len(avail_pressure)} pressure vars × {len(ALL_PRESSURE_LEVELS)} levels...")
        flat_vars = {}
        for long_name in avail_pressure:
            short_name = ARCO_PRESSURE_VARS[long_name]
            print(f"    {short_name}...", end=' ', flush=True)
            da = (
                arco_ds[long_name]
                .sel(time=t_slice,
                     **{plevel_dim: ALL_PRESSURE_LEVELS},
                     latitude=slice(MAX_LAT, MIN_LAT),
                     longitude=slice(MIN_LON, MAX_LON))
                .load()
            )
            # Flatten: u at level 500 → u_500
            for lev in ALL_PRESSURE_LEVELS:
                flat_name = f"{short_name}_{int(lev)}"
                flat_vars[flat_name] = (
                    da.sel({plevel_dim: lev})
                    .drop_vars(plevel_dim, errors='ignore')
                    .rename({'latitude': 'lat', 'longitude': 'lon'})
                    .astype(np.float32)
                )
            print("done")

        pl_ds = xr.Dataset(flat_vars)
        all_datasets.append(pl_ds)

    result = xr.merge(all_datasets)
    result = result.chunk({'time': TIME_CHUNK, 'lat': -1, 'lon': -1})
    print(f"  Fetched {len(result.data_vars)} variables, {len(result.time)} timesteps")
    return result


# ── Main append logic ──────────────────────────────────────────────────────────

def append_case(case_name, start_str, end_str, arco_ds):
    print(f"\n{'='*60}")
    print(f"Appending: {case_name}  ({start_str} → {end_str})")
    print(f"{'='*60}")

    year = int(start_str[:4])

    # Check if already in Zarr
    ds_existing = xr.open_zarr(ERA5_ZARR)
    existing_times = pd.DatetimeIndex(ds_existing.time.values)
    ds_existing.close()
    case_start = pd.Timestamp(start_str)
    case_end   = pd.Timestamp(end_str) + pd.Timedelta(hours=23)
    already_there = ((existing_times >= case_start) & (existing_times <= case_end)).any()
    if already_there:
        print(f"  Already in ERA5.zarr — skipping ERA5 append.")
    else:
        # ── ERA5 ─────────────────────────────────────────────────────────────
        era5_ds = fetch_from_arco(arco_ds, start_str, end_str)
        print(f"  Appending to {ERA5_ZARR}...")
        era5_ds.to_zarr(ERA5_ZARR, append_dim='time')
        print(f"  ERA5 append done ✓")

    # ── Lightning ─────────────────────────────────────────────────────────────
    import zarr as zarr_lib

    ds_light = xr.open_zarr(LIGHTNING_ZARR)
    existing_ltimes = pd.DatetimeIndex(ds_light.time.values)
    lats = ds_light.lat.values
    lons = ds_light.lon.values
    ds_light.close()

    light_already = ((existing_ltimes >= case_start) & (existing_ltimes <= case_end)).any()

    # Find ENTLN file
    entln_path = find_entln_for_case(case_name, year)
    if entln_path:
        df_entln = load_entln_mat(entln_path)
        df_entln = df_entln[(df_entln.UTC >= case_start) & (df_entln.UTC <= case_end)]
        print(f"  ENTLN filtered to case: {len(df_entln):,} pulses")
    else:
        print(f"  WARNING: No ENTLN file found for {case_name} — lightning will be zeros")
        df_entln = pd.DataFrame(columns=['lat', 'lon', 'UTC'])

    # Bin hour by hour
    case_times = pd.date_range(start=case_start, end=case_end, freq='1h')
    grids = []
    for t in case_times:
        mask = (df_entln.UTC >= t) & (df_entln.UTC < t + pd.Timedelta(hours=1))
        grids.append(bin_to_grid(df_entln[mask], lats, lons))

    data = np.stack(grids, axis=0)

    if light_already:
        # Timestamps exist — overwrite in-place (handles zeros from a failed previous run)
        print(f"  Already in Lightning.zarr — overwriting in-place with real ENTLN data...")
        indices = np.where((existing_ltimes >= case_start) & (existing_ltimes <= case_end))[0]
        store = zarr_lib.open(LIGHTNING_ZARR, mode='r+')
        for i, zarr_idx in enumerate(indices):
            store['entln_count'][zarr_idx] = grids[i]
        total_lightning = (data > 0).sum()
        print(f"  Overwritten {len(indices)} timesteps  ({total_lightning} non-zero grid cells) ✓")
    else:
        # New timestamps — append
        ds_light_new = xr.Dataset(
            {'entln_count': (['time', 'lat', 'lon'], data)},
            coords={'time': case_times.values, 'lat': lats, 'lon': lons}
        ).chunk({'time': TIME_CHUNK, 'lat': -1, 'lon': -1})
        print(f"  Appending {len(case_times)} lightning timesteps to {LIGHTNING_ZARR}...")
        ds_light_new.to_zarr(LIGHTNING_ZARR, append_dim='time')
        print(f"  Lightning append done ✓")


def verify():
    print(f"\n{'='*60}")
    print("Verification")
    for zarr_path, label in [(ERA5_ZARR, 'ERA5'), (LIGHTNING_ZARR, 'Lightning')]:
        ds = xr.open_zarr(zarr_path)
        times = pd.DatetimeIndex(ds.time.values)
        print(f"  [{label}] {len(times)} timesteps  "
              f"{times.min().date()} → {times.max().date()}")
        # Check case dates are present + lightning is non-zero
        for name, start, end in MISSING_CASES:
            mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end) + pd.Timedelta(hours=23))
            status = '✓' if mask.any() else '✗ MISSING'
            if label == 'Lightning' and mask.any():
                max_val = ds['entln_count'].isel(time=np.where(mask)[0]).values.max()
                status += f'  max_count={max_val:.0f}' + (' ✓' if max_val > 0 else ' ✗ ALL ZEROS!')
            print(f"    {name}: {mask.sum()} timesteps {status}")
        ds.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Opening ARCO-ERA5 store...")
    fs      = gcsfs.GCSFileSystem(token='anon')
    store   = gcsfs.GCSMap(ARCO_ZARR_PATH, gcs=fs)
    arco_ds = xr.open_zarr(store, consolidated=True)
    print(f"ARCO opened. Time range: {pd.Timestamp(arco_ds.time.values[0])} → "
          f"{pd.Timestamp(arco_ds.time.values[-1])}")

    for case_name, start_str, end_str in MISSING_CASES:
        append_case(case_name, start_str, end_str, arco_ds)

    verify()
    print("\nDone! ERA5.zarr and Lightning.zarr now include all 5 cases.")
    print("Update ZARR_CONFIG in experiment.py to include the new date ranges.")
