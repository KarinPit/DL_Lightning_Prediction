"""
ERA5 + ENTLN Processing Pipeline
Reads ERA5 single-level + pressure-level NC files (per case) + ENTLN _struct.mat files,
bins lightning to ERA5 grid, saves hourly paired NC files.

ERA5 folder structure:
  local_raw_data/{case_name}/ERA5/
      data_stream-oper_stepType-instant.nc
      data_stream-oper_stepType-accum.nc
      data_stream-oper_stepType-avg.nc
  local_raw_data/{case_name}/ERA5_pressure/
      (one or more NC files with pressure-level variables)

ENTLN files:
  local_raw_data/ENTLN/ENTLN_pulse_{case_name}_struct.mat

Output:
  local_processed_data/ERA5/{case_name}/1_hours/{timestamp}.nc

Karin Pitlik
"""

import xarray as xr
import numpy as np
import scipy.io
import netCDF4 as nc4
import os
import pandas as pd
from datetime import datetime

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

DROP_VARS = ['expver', 'number']

# ============================================================
#  LOAD ERA5 SINGLE-LEVEL DATA
# ============================================================

def load_era5(era5_folder):
    """Load and merge the three ERA5 step-type NC files for one case."""
    ds_list = []
    for fname in [
        'data_stream-oper_stepType-instant.nc',
        'data_stream-oper_stepType-accum.nc',
        'data_stream-oper_stepType-avg.nc',
    ]:
        fpath = os.path.join(era5_folder, fname)
        if os.path.exists(fpath):
            ds = xr.open_dataset(fpath)
            drop = [v for v in DROP_VARS if v in ds]
            ds = ds.drop_vars(drop)
            print(f"  Loaded {fname}: {list(ds.data_vars)}")
            ds_list.append(ds)

    if not ds_list:
        raise FileNotFoundError(f"No ERA5 NC files found in: {era5_folder}")

    ds_merged = xr.merge(ds_list, join='outer')

    if 'valid_time' in ds_merged.coords:
        ds_merged = ds_merged.rename({'valid_time': 'time'})

    print(f"  ERA5 single-level variables: {list(ds_merged.data_vars)}")
    print(f"  ERA5 time: {pd.Timestamp(ds_merged.time.values[0])} --> {pd.Timestamp(ds_merged.time.values[-1])}")
    return ds_merged


# ============================================================
#  LOAD ERA5 PRESSURE-LEVEL DATA
# ============================================================

def load_era5_pressure(pressure_folder):
    """
    Load ERA5 pressure-level NC files for one case.
    Flattens each (variable, pressure_level) pair into a named 2D variable
    e.g. u_500, v_700, w_850, t_500, r_700 etc.
    Returns an xarray Dataset with only (time, latitude, longitude) dimensions.
    """
    if not os.path.exists(pressure_folder):
        print(f"  No pressure-level folder found: {pressure_folder} — skipping")
        return None

    nc_files = [f for f in os.listdir(pressure_folder) if f.endswith('.nc')]
    if not nc_files:
        print(f"  No NC files in pressure folder: {pressure_folder} — skipping")
        return None

    ds_list = []
    for fname in nc_files:
        fpath = os.path.join(pressure_folder, fname)
        ds = xr.open_dataset(fpath)
        drop = [v for v in DROP_VARS if v in ds]
        ds = ds.drop_vars(drop)
        ds_list.append(ds)
        print(f"  Loaded pressure file {fname}: {list(ds.data_vars)}")

    ds_merged = xr.merge(ds_list, join='outer')

    if 'valid_time' in ds_merged.coords:
        ds_merged = ds_merged.rename({'valid_time': 'time'})

    # Find the pressure level dimension name
    plevel_dim = None
    for dim in ['pressure_level', 'level', 'plev']:
        if dim in ds_merged.dims:
            plevel_dim = dim
            break

    if plevel_dim is None:
        print("  WARNING: No pressure_level dimension found in pressure-level data")
        return None

    pressure_levels = ds_merged[plevel_dim].values
    print(f"  Pressure levels: {pressure_levels} hPa")
    print(f"  Pressure variables: {list(ds_merged.data_vars)}")

    # Flatten: create one 2D variable per (variable, pressure_level) pair
    flat_vars = {}
    for vname in ds_merged.data_vars:
        for plevel in pressure_levels:
            new_name = f"{vname}_{int(plevel)}"
            flat_vars[new_name] = ds_merged[vname].sel({plevel_dim: plevel}).drop_vars(plevel_dim, errors='ignore')

    ds_flat = xr.Dataset(flat_vars)
    print(f"  Flattened pressure variables: {list(ds_flat.data_vars)}")
    return ds_flat


# ============================================================
#  SUBSET TO DOMAIN
# ============================================================

def subset_era5_to_domain(ds):
    """Crop ERA5 dataset to Israel bounding box."""
    return ds.sel(
        latitude=slice(MAX_LAT, MIN_LAT),
        longitude=slice(MIN_LON, MAX_LON)
    )


# ============================================================
#  LOAD ENTLN DATA
# ============================================================

def load_entln_mat(mat_path):
    """
    Load ENTLN _struct.mat file (exported from MATLAB as simple struct).
    Returns DataFrame with columns: lat, lon, UTC (datetime)
    """
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
    nlat = len(lats_1d)
    nlon = len(lons_1d)

    if df_entln.empty:
        return np.zeros((nlat, nlon), dtype=np.float32)

    dlat = abs(float(lats_1d[1]) - float(lats_1d[0])) / 2.0
    dlon = abs(float(lons_1d[1]) - float(lons_1d[0])) / 2.0

    lats_sorted = np.sort(lats_1d)
    lat_edges = np.concatenate([
        [lats_sorted[0]  - dlat],
        lats_sorted[:-1] + np.diff(lats_sorted) / 2,
        [lats_sorted[-1] + dlat]
    ])
    lon_edges = np.concatenate([
        [lons_1d[0]  - dlon],
        lons_1d[:-1] + np.diff(lons_1d) / 2,
        [lons_1d[-1] + dlon]
    ])

    N, _, _ = np.histogram2d(
        df_entln.lat.values,
        df_entln.lon.values,
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

    nlat = len(lats)
    nlon = len(lons)

    with nc4.Dataset(outfile, 'w') as ds:
        ds.createDimension('lat', nlat)
        ds.createDimension('lon', nlon)

        lat_v = ds.createVariable('lat', 'f4', ('lat',))
        lon_v = ds.createVariable('lon', 'f4', ('lon',))
        lat_v[:] = lats
        lon_v[:] = lons
        lat_v.units = 'degrees_north'
        lon_v.units = 'degrees_east'

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

        ds.description   = 'ERA5 atmospheric variables + ENTLN lightning counts on ERA5 grid'
        ds.created_by    = 'Karin Pitlik'
        ds.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ds.timestamp_utc = str(t_start)


# ============================================================
#  PROCESS ONE CASE
# ============================================================

def process_case(case_name, start_str, end_str):
    print(f"\n{'='*55}")
    print(f"PROCESSING: {case_name}")
    print(f"{'='*55}")

    era5_folder = os.path.join(BASE_PATH, 'local_raw_data', case_name, 'ERA5')
    if not os.path.exists(era5_folder):
        print(f"  WARNING: ERA5 folder not found: {era5_folder} — skipping")
        return

    # Load single-level ERA5
    era5_ds = load_era5(era5_folder)

    # Load pressure-level ERA5 (optional)
    pressure_folder = os.path.join(BASE_PATH, 'local_raw_data', case_name, 'ERA5_pressure')
    era5_pressure_ds = load_era5_pressure(pressure_folder)

    out_folder = os.path.join(OUTPUT_PATH, case_name, '1_hours')
    os.makedirs(out_folder, exist_ok=True)

    entln_file = os.path.join(ENTLN_PATH, f'ENTLN_pulse_{case_name}_struct.mat')
    if not os.path.exists(entln_file):
        print(f"  WARNING: ENTLN struct file not found — lightning will be zeros")
        df_entln_full = pd.DataFrame(columns=['lat', 'lon', 'UTC'])
    else:
        df_entln_full = load_entln_mat(entln_file)
        print(f"  ENTLN loaded: {len(df_entln_full):,} pulses")

    era5_sub = subset_era5_to_domain(era5_ds)
    lats = era5_sub.latitude.values
    lons = era5_sub.longitude.values

    # Subset pressure-level data to same domain
    if era5_pressure_ds is not None:
        era5_pressure_sub = subset_era5_to_domain(era5_pressure_ds)
    else:
        era5_pressure_sub = None

    start_dt = pd.Timestamp(start_str)
    end_dt   = pd.Timestamp(end_str) + pd.Timedelta(days=1)

    era5_times = pd.DatetimeIndex(era5_sub.time.values)
    mask = (era5_times >= start_dt) & (era5_times < end_dt)
    case_times = era5_times[mask]

    if len(case_times) == 0:
        print(f"  WARNING: No ERA5 timesteps found for {start_str} to {end_str}")
        return

    print(f"  ERA5 timesteps in case: {len(case_times)}")

    for t_start in case_times:
        t_end = t_start + pd.Timedelta(hours=1)

        era5_slice = era5_sub.sel(time=t_start)

        # Merge pressure-level variables into the slice
        if era5_pressure_sub is not None:
            try:
                pressure_slice = era5_pressure_sub.sel(time=t_start)
                era5_slice = xr.merge([era5_slice, pressure_slice])
            except Exception as e:
                print(f"  WARNING: Could not merge pressure data for {t_start}: {e}")

        mask_t = (df_entln_full['UTC'] >= t_start) & (df_entln_full['UTC'] < t_end)
        df_hour = df_entln_full[mask_t]

        lightning_grid = bin_entln_to_grid(df_hour, lats, lons)

        tstr     = t_start.strftime('%Y-%m-%d_%H_%M_%S')
        tend_str = t_end.strftime('%Y-%m-%d_%H_%M_%S')
        outfile  = os.path.join(out_folder, f'{tstr}_{tend_str}.nc')

        save_hourly_nc(outfile, era5_slice, lightning_grid, lats, lons, t_start)
        print(f"  Saved: {tstr} | lightning pulses: {len(df_hour)}")

    print(f"  Done: {case_name}")
    era5_ds.close()


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    for case_name, start_str, end_str in CASES:
        process_case(case_name, start_str, end_str)

    print("\n\nAll cases processed!")
