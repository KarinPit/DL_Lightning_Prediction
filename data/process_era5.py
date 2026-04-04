"""
ERA5 + ENTLN Processing Pipeline
Reads ERA5 NC files (per case) + ENTLN _struct.mat files,
bins lightning to ERA5 grid, saves hourly paired NC files.

ERA5 folder structure:
  local_raw_data/{case_name}/ERA5/
      data_stream-oper_stepType-instant.nc
      data_stream-oper_stepType-accum.nc
      data_stream-oper_stepType-avg.nc

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

# Israel bounding box
MIN_LAT = 27.296
MAX_LAT = 36.598
MIN_LON = 27.954
MAX_LON = 39.292

BASE_PATH   = '/home/ubuntu/Desktop'
ENTLN_PATH  = os.path.join(BASE_PATH, 'local_raw_data/ENTLN')
OUTPUT_PATH = os.path.join(BASE_PATH, 'local_processed_data/ERA5')

# ERA5 metadata variables to drop
DROP_VARS = ['expver', 'number']

# ============================================================
#  LOAD ERA5 DATA (per case)
# ============================================================

def load_era5(era5_folder):
    """Load and merge the three ERA5 step-type NC files for one case"""
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

    # Newer ERA5 downloads use 'valid_time' — normalize to 'time'
    if 'valid_time' in ds_merged.coords:
        ds_merged = ds_merged.rename({'valid_time': 'time'})

    print(f"  ERA5 variables : {list(ds_merged.data_vars)}")
    print(f"  ERA5 time      : {pd.Timestamp(ds_merged.time.values[0])} --> {pd.Timestamp(ds_merged.time.values[-1])}")
    print(f"  ERA5 lat       : {float(ds_merged.latitude.min()):.2f} --> {float(ds_merged.latitude.max()):.2f}")
    print(f"  ERA5 lon       : {float(ds_merged.longitude.min()):.2f} --> {float(ds_merged.longitude.max()):.2f}")
    return ds_merged


def subset_era5_to_domain(ds):
    """Crop ERA5 dataset to Israel bounding box"""
    return ds.sel(
        latitude=slice(MAX_LAT, MIN_LAT),   # ERA5 lat is descending
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

    # Handle string arrays
    utc_strs = [str(u).strip() for u in utc_raw]
    utc = pd.to_datetime(utc_strs, format='%d-%b-%Y %H:%M:%S', errors='coerce')

    df = pd.DataFrame({'lat': lats, 'lon': lons, 'UTC': utc})
    df = df.dropna(subset=['UTC'])
    return df

# ============================================================
#  BIN ENTLN TO ERA5 GRID
# ============================================================

def bin_entln_to_grid(df_entln, lats_1d, lons_1d):
    """
    Bin lightning pulses onto the ERA5 lat/lon grid.
    Returns 2D array (nlat x nlon) of pulse counts.
    """
    nlat = len(lats_1d)
    nlon = len(lons_1d)

    if df_entln.empty:
        return np.zeros((nlat, nlon), dtype=np.float32)

    dlat = abs(float(lats_1d[1]) - float(lats_1d[0])) / 2.0
    dlon = abs(float(lons_1d[1]) - float(lons_1d[0])) / 2.0

    # histogram2d needs ascending lat edges
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

    # Flip back to descending lat to match ERA5
    if lats_1d[0] > lats_1d[-1]:
        N = N[::-1, :]

    return N.astype(np.float32)

# ============================================================
#  SAVE HOURLY NC
# ============================================================

def save_hourly_nc(outfile, era5_slice, lightning_grid, lats, lons, t_start):
    """Save one hourly NC file: all ERA5 variables + ENTLN pulse count"""
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

        # ERA5 variables
        for vname in era5_slice.data_vars:
            data = era5_slice[vname].values
            if data.ndim != 2:
                continue
            v = ds.createVariable(vname, 'f4', ('lat', 'lon'), fill_value=np.nan)
            v[:] = data.astype(np.float32)
            v.units     = era5_slice[vname].attrs.get('units', '')
            v.long_name = era5_slice[vname].attrs.get('long_name', vname)

        # ENTLN lightning count
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

    # ERA5 folder for this case
    era5_folder = os.path.join(BASE_PATH, 'local_raw_data', case_name, 'ERA5')
    if not os.path.exists(era5_folder):
        print(f"  WARNING: ERA5 folder not found: {era5_folder} — skipping")
        return

    era5_ds = load_era5(era5_folder)

    # Output folder
    out_folder = os.path.join(OUTPUT_PATH, case_name, '1_hours')
    os.makedirs(out_folder, exist_ok=True)

    # Load ENTLN (_struct.mat exported from MATLAB)
    entln_file = os.path.join(ENTLN_PATH, f'ENTLN_pulse_{case_name}_struct.mat')
    if not os.path.exists(entln_file):
        print(f"  WARNING: ENTLN struct file not found: {entln_file} — lightning will be zeros")
        df_entln_full = pd.DataFrame(columns=['lat', 'lon', 'UTC'])
    else:
        df_entln_full = load_entln_mat(entln_file)
        print(f"  ENTLN loaded: {len(df_entln_full):,} pulses")

    # Crop ERA5 to Israel domain
    era5_sub = subset_era5_to_domain(era5_ds)
    lats = era5_sub.latitude.values   # descending
    lons = era5_sub.longitude.values  # ascending

    # Case time range
    start_dt = pd.Timestamp(start_str)
    end_dt   = pd.Timestamp(end_str) + pd.Timedelta(days=1)

    # Filter ERA5 to case period
    era5_times = pd.DatetimeIndex(era5_sub.time.values)
    mask = (era5_times >= start_dt) & (era5_times < end_dt)
    case_times = era5_times[mask]

    if len(case_times) == 0:
        print(f"  WARNING: No ERA5 timesteps found for {start_str} to {end_str}")
        print(f"  ERA5 covers: {era5_times[0]} to {era5_times[-1]}")
        return

    print(f"  ERA5 timesteps in case: {len(case_times)}")

    # Process hour by hour
    for t_start in case_times:
        t_end = t_start + pd.Timedelta(hours=1)

        era5_slice = era5_sub.sel(time=t_start)

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
