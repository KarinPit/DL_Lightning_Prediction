"""
ERA5 + ENTLN Processing Pipeline — Full Year Edition
Processes the complete year 2023, month by month, streaming from ARCO-ERA5.
Fetches BATCH_DAYS days at a time to stay within RAM limits.

Output:
  local_processed_data/ERA5/Year2023_full/1_hours/{timestamp}.nc

Variable coverage:
  ARCO single-level  (17):        cape, cin, kx, totot, d2m, tcwv, tciw, tclw,
                                  cp, crr, tp, vimd, t2m, msl, sp, cbh, hcc
  ARCO pressure-level (7 × 37):   u, v, w, t, q, ciwc, clwc

  Total: 276 vars  (259 pressure-level + 17 single-level)

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

YEAR = 2023

# ← Lower this if still getting Killed (try 2 or 1)
BATCH_DAYS = 1

MIN_LAT = 27.296
MAX_LAT = 36.598
MIN_LON = 27.954
MAX_LON = 39.292

BASE_PATH   = '/home/ubuntu/Desktop'
OUTPUT_PATH = os.path.join(BASE_PATH, 'local_processed_data/ERA5')
CASE_NAME   = f'Year{YEAR}_full'

ENTLN_FULLYEAR_FILE = os.path.join(
    BASE_PATH, 'thesis-bucket/Raw_Data/ENTLN', f'ENTLN_pulse_Year{YEAR}_struct.mat'
)

ARCO_ZARR_PATH = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900, 925,
    950, 975, 1000
]

ARCO_PRESSURE_VARS = {
    'u_component_of_wind':                 'u',
    'v_component_of_wind':                 'v',
    'vertical_velocity':                   'w',
    'temperature':                         't',
    'specific_humidity':                   'q',
    'specific_cloud_ice_water_content':    'ciwc',
    'specific_cloud_liquid_water_content': 'clwc',
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
#  OPEN ARCO-ERA5 STORE
# ============================================================

def open_arco_store():
    print("Opening ARCO-ERA5 Zarr store (anonymous)...")
    fs = gcsfs.GCSFileSystem(token='anon')
    store = gcsfs.GCSMap(ARCO_ZARR_PATH, gcs=fs)
    ds = xr.open_zarr(store, consolidated=True)
    print(f"  ARCO store opened. Variables: {len(list(ds.data_vars))}")
    return ds


# ============================================================
#  LOAD FULL-YEAR ENTLN
# ============================================================

def load_all_entln():
    if not os.path.exists(ENTLN_FULLYEAR_FILE):
        raise FileNotFoundError(f"ENTLN file not found: {ENTLN_FULLYEAR_FILE}")

    print(f"  Loading full-year ENTLN: {ENTLN_FULLYEAR_FILE}")
    S = scipy.io.loadmat(ENTLN_FULLYEAR_FILE, squeeze_me=True, struct_as_record=False)
    key = [k for k in S.keys() if not k.startswith('_')][0]
    entln = S[key]

    lats     = np.array(entln.lat).flatten().astype(float)
    lons     = np.array(entln.lon).flatten().astype(float)
    utc_raw  = np.array(entln.UTC).flatten()
    utc_strs = [str(u).strip() for u in utc_raw]
    utc = pd.to_datetime(utc_strs, format='%d-%b-%Y %H:%M:%S', errors='coerce')

    df = pd.DataFrame({'lat': lats, 'lon': lons, 'UTC': utc}).dropna(subset=['UTC'])
    print(f"  ENTLN loaded: {len(df):,} pulses  ({df.UTC.min().date()} → {df.UTC.max().date()})")
    return df


# ============================================================
#  BIN ENTLN TO GRID
# ============================================================

def bin_entln_to_grid(df_entln, lats_1d, lons_1d):
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
    N, _, _ = np.histogram2d(df_entln.lat.values, df_entln.lon.values,
                              bins=[lat_edges, lon_edges])
    if lats_1d[0] > lats_1d[-1]:
        N = N[::-1, :]
    return N.astype(np.float32)


# ============================================================
#  SAVE HOURLY NC
# ============================================================

def save_hourly_nc(outfile, era5_slice, lightning_grid, lats, lons, t_start):
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
        ds.description   = f'ERA5 (ARCO) full-year {YEAR} + ENTLN lightning counts'
        ds.created_by    = 'Karin Pitlik'
        ds.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ds.timestamp_utc = str(t_start)


# ============================================================
#  EXTRACT ONE TIMESTEP (pure in-memory slice)
# ============================================================

def extract_timestep(sl_batch, pl_data, plevel_dim, timestamp):
    """
    sl_batch : xr.Dataset  single-level, dims (time, lat, lon)
    pl_data  : dict  short_name → DataArray(time, level, lat, lon)
    """
    t = pd.Timestamp(timestamp)
    all_vars = {}

    # Single-level
    sl_slice = sl_batch.sel(time=t)
    for long_name, short_name in ARCO_SINGLE_VARS.items():
        if long_name in sl_slice:
            all_vars[short_name] = sl_slice[long_name].squeeze(drop=True)

    # Pressure-level — one var at a time already in pl_data
    if pl_data and plevel_dim is not None:
        for short_name, da in pl_data.items():
            da_t = da.sel(time=t)
            for plevel in PRESSURE_LEVELS:
                try:
                    arr = da_t.sel({plevel_dim: plevel})
                    all_vars[f"{short_name}_{int(plevel)}"] = \
                        arr.drop_vars(plevel_dim, errors='ignore').squeeze(drop=True)
                except KeyError:
                    pass

    return xr.Dataset(all_vars) if all_vars else None


# ============================================================
#  PROCESS ONE MONTH  —  BATCH_DAYS days at a time
# ============================================================

def process_month(month_dt, arco_ds, df_entln_all, out_folder):
    if month_dt.month == 12:
        month_end = pd.Timestamp(f'{month_dt.year + 1}-01-01')
    else:
        month_end = pd.Timestamp(f'{month_dt.year}-{month_dt.month + 1:02d}-01')

    month_times = pd.date_range(start=month_dt, end=month_end, freq='1h', inclusive='left')

    missing_times = [
        t for t in month_times
        if not os.path.exists(os.path.join(
            out_folder,
            f'{t.strftime("%Y-%m-%d_%H_%M_%S")}_{(t + pd.Timedelta(hours=1)).strftime("%Y-%m-%d_%H_%M_%S")}.nc'
        ))
    ]

    if not missing_times:
        print(f"  {month_dt.strftime('%Y-%m')}: all done — skipping")
        return 0

    print(f"\n  {month_dt.strftime('%Y-%m')}: {len(missing_times)}/{len(month_times)} to process")

    avail_single   = [k for k in ARCO_SINGLE_VARS   if k in arco_ds.data_vars]
    avail_pressure = [k for k in ARCO_PRESSURE_VARS if k in arco_ds.data_vars]
    plevel_dim     = next((d for d in ['level', 'pressure_level', 'plev']
                           if d in arco_ds.dims), None)

    # Split into BATCH_DAYS-day chunks
    batch_size = BATCH_DAYS * 24
    batches = [missing_times[i:i + batch_size]
               for i in range(0, len(missing_times), batch_size)]

    lats = lons = None
    n_ok = 0

    for b_idx, batch in enumerate(batches):
        b_start = batch[0]
        b_end   = batch[-1] + pd.Timedelta(hours=1)
        t_slice = slice(b_start, b_end - pd.Timedelta(hours=1))

        print(f"    [{b_idx+1}/{len(batches)}] {b_start.date()} → {batch[-1].date()} "
              f"({len(batch)}h)", flush=True)

        # ── Fetch single-level (all vars together — small) ───────────────────
        print(f"      single-level ...", flush=True)
        sl_batch = (
            arco_ds[avail_single]
            .sel(time=t_slice,
                 latitude=slice(MAX_LAT, MIN_LAT),
                 longitude=slice(MIN_LON, MAX_LON))
            .load()
        )
        if lats is None:
            lats = sl_batch.latitude.values
            lons = sl_batch.longitude.values

        # ── Fetch pressure-level ONE VARIABLE AT A TIME to save RAM ──────────
        # Each fetch: 1 var × 37 levels × BATCH_DAYS×24h × 37×46 ≈ 40 MB max
        pl_data = {}   # short_name → DataArray(time, level, lat, lon)
        if avail_pressure and plevel_dim:
            for long_name in avail_pressure:
                short_name = ARCO_PRESSURE_VARS[long_name]
                print(f"      pressure {short_name} ...", flush=True)
                da = (
                    arco_ds[[long_name]]
                    .sel(time=t_slice,
                         **{plevel_dim: PRESSURE_LEVELS},
                         latitude=slice(MAX_LAT, MIN_LAT),
                         longitude=slice(MIN_LON, MAX_LON))
                    [long_name]
                    .load()
                )
                pl_data[short_name] = da

        print(f"      processing {len(batch)} hours ...", flush=True)

        # ── Iterate hours in memory ──────────────────────────────────────────
        for t_start in batch:
            t_end    = t_start + pd.Timedelta(hours=1)
            tstr     = t_start.strftime('%Y-%m-%d_%H_%M_%S')
            tend_str = t_end.strftime('%Y-%m-%d_%H_%M_%S')
            outfile  = os.path.join(out_folder, f'{tstr}_{tend_str}.nc')

            try:
                era5_slice = extract_timestep(sl_batch, pl_data, plevel_dim, t_start)
            except Exception as e:
                print(f"    ERROR {tstr}: {e}")
                continue

            if era5_slice is None:
                continue

            mask_t = (df_entln_all['UTC'] >= t_start) & (df_entln_all['UTC'] < t_end)
            lightning_grid = bin_entln_to_grid(df_entln_all[mask_t], lats, lons)
            save_hourly_nc(outfile, era5_slice, lightning_grid, lats, lons, t_start)
            n_ok += 1

            n_lightning = int(mask_t.sum())
            if n_lightning > 0 or t_start.hour == 0:
                print(f"    {tstr} | ⚡ {n_lightning:4d} | {len(era5_slice.data_vars)} vars")

        # Free RAM before next batch
        del sl_batch, pl_data

    print(f"  {month_dt.strftime('%Y-%m')} done: {n_ok} saved")
    return n_ok


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    out_folder = os.path.join(OUTPUT_PATH, CASE_NAME, '1_hours')
    os.makedirs(out_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Full-year ERA5 processing: {YEAR}  (BATCH_DAYS={BATCH_DAYS})")
    print(f"Output → {out_folder}")
    print(f"{'='*60}\n")

    df_entln_all = load_all_entln()
    arco_ds      = open_arco_store()

    total_saved = 0
    for month in range(1, 13):
        month_dt = pd.Timestamp(f'{YEAR}-{month:02d}-01')
        total_saved += process_month(month_dt, arco_ds, df_entln_all, out_folder)

    print(f"\n{'='*60}")
    print(f"Full year {YEAR} complete: {total_saved} files saved")
    print(f"{'='*60}")
