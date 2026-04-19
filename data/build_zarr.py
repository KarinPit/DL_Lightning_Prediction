"""
build_zarr.py — Build ERA5 and Lightning Zarr stores
=====================================================
Builds two clean, separate Zarr stores directly into the S3 bucket:

  ERA5.zarr       — all atmospheric variables, full time coverage
  Lightning.zarr  — lightning counts per grid cell per hour

No concept of "cases" here. Cases/splits are defined in experiment.py.
New data (new year, new lightning network) → just append to the same store.

Usage (run on the server):
    python build_zarr.py era5       # ERA5 NC files → ERA5.zarr
    python build_zarr.py lightning  # ENTLN mat files → Lightning.zarr
    python build_zarr.py all        # both

Karin Pitlik
"""

import os
import re
import sys
import glob
import shutil
import scipy.io
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from tqdm import tqdm

# ============================================================
#  PATHS  — everything lives in the mounted bucket
# ============================================================

BUCKET = '/home/ec2-user/thesis-bucket'

# Input: ENTLN mat files
ENTLN_DIR = os.path.join(BUCKET, 'Lightning_Data/ENTLN')

# Output: Zarr stores go directly into the bucket
ZARR_DIR       = os.path.join(BUCKET, 'Zarr')
ERA5_ZARR      = os.path.join(ZARR_DIR, 'ERA5.zarr')
LIGHTNING_ZARR = os.path.join(ZARR_DIR, 'Lightning.zarr')

TIME_CHUNK = 24


# ============================================================
#  HELPERS
# ============================================================

def parse_timestamp(fname):
    m = re.match(r'(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})', os.path.basename(fname))
    if m:
        return pd.Timestamp(datetime.strptime(m.group(1), '%Y-%m-%d_%H_%M_%S'))
    return None


def open_nc(fpath):
    ts = parse_timestamp(fpath)
    if ts is None:
        return None, None
    ds = xr.open_dataset(fpath, engine='netcdf4').squeeze(drop=True)
    ds = ds.expand_dims(time=[ts])
    return ds, ts


# ============================================================
#  CORE: split NC files → ERA5 Zarr + Lightning Zarr
# ============================================================

def build_era5_zarr(nc_files, era5_zarr_path, desc=''):
    """Read clean ERA5-only NC files and write to a Zarr store."""
    nc_files = sorted(
        [f for f in nc_files if parse_timestamp(f) is not None],
        key=parse_timestamp
    )
    n = len(nc_files)
    if n == 0:
        print(f'No NC files found for {desc}')
        return False

    print(f'\n{"="*55}')
    print(f'Building: {desc}')
    print(f'  {n} NC files → {era5_zarr_path}')
    print(f'  Time range: {parse_timestamp(nc_files[0])} → {parse_timestamp(nc_files[-1])}')

    if os.path.exists(era5_zarr_path):
        ans = input(f'\n  {os.path.basename(era5_zarr_path)} exists. Overwrite? [y/N]: ').strip().lower()
        if ans != 'y':
            return False
        shutil.rmtree(era5_zarr_path)

    os.makedirs(ZARR_DIR, exist_ok=True)

    BATCH = 720
    batches = [nc_files[i:i+BATCH] for i in range(0, n, BATCH)]
    initialized = False
    t_total = datetime.now()

    for b_idx, batch in enumerate(tqdm(batches, desc='ERA5 batches', unit='batch')):
        t0 = datetime.now()
        b_start = parse_timestamp(batch[0]).date()
        b_end   = parse_timestamp(batch[-1]).date()

        datasets = []
        for f in tqdm(batch, desc=f'  {b_start}→{b_end}', unit='file', leave=True):
            d, _ = open_nc(f)
            if d is not None:
                datasets.append(d)

        if not datasets:
            continue

        ds_batch = xr.concat(datasets, dim='time')
        for d in datasets:
            d.close()

        ds_batch = ds_batch.chunk({'time': TIME_CHUNK, 'lat': -1, 'lon': -1})

        if not initialized:
            ds_batch.to_zarr(era5_zarr_path, mode='w')
            initialized = True
        else:
            ds_batch.to_zarr(era5_zarr_path, append_dim='time')

        ds_batch.close()

    print(f'\n  Finished in {(datetime.now()-t_total).total_seconds()/60:.1f} min')
    _verify(era5_zarr_path, 'ERA5')
    return True


def _verify(zarr_path, label):
    ds = xr.open_zarr(zarr_path)
    print(f'\n  [{label}] {os.path.basename(zarr_path)}')
    print(f'    Variables : {len(ds.data_vars)}  → {sorted(ds.data_vars)[:5]}...')
    print(f'    Time steps: {len(ds.time)}')
    print(f'    Time range: {pd.Timestamp(ds.time.values[0])} → {pd.Timestamp(ds.time.values[-1])}')
    print(f'    Grid      : lat {float(ds.lat.min()):.2f}→{float(ds.lat.max()):.2f}  '
          f'lon {float(ds.lon.min()):.2f}→{float(ds.lon.max()):.2f}')
    ds.close()


# ============================================================
#  ERA5
# ============================================================

def build_era5():
    """Merge all ERA5 NC files from the bucket into one ERA5.zarr."""
    # Collect all NC files from all subdirs under ERA5/
    nc_files = []
    era5_root = os.path.join(BUCKET, 'ERA5')
    for root, dirs, files in os.walk(era5_root):
        # skip the crwc subdir
        dirs[:] = [d for d in dirs if d not in ('crwc', 'cswc')]
        for f in files:
            if f.endswith('.nc'):
                nc_files.append(os.path.join(root, f))

    print(f'ERA5: {len(nc_files)} NC files found under {era5_root}')
    build_era5_zarr(nc_files, ERA5_ZARR, desc='ERA5')


# ============================================================
#  LIGHTNING ZARR  —  built from mat files, not NC files
# ============================================================

def load_entln_mat(mat_path):
    """Load ENTLN mat file → DataFrame with lat, lon, UTC columns."""
    S = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    key = [k for k in S.keys() if not k.startswith('_')][0]
    e = S[key]
    lats = np.array(e.lat).flatten().astype(float)
    lons = np.array(e.lon).flatten().astype(float)
    utc  = pd.to_datetime(
        [str(u).strip() for u in np.array(e.UTC).flatten()],
        format='%d-%b-%Y %H:%M:%S', errors='coerce'
    )
    df = pd.DataFrame({'lat': lats, 'lon': lons, 'UTC': utc}).dropna(subset=['UTC'])
    # Drop null-island pulses
    df = df[(df.lat != 0) | (df.lon != 0)]
    return df


def bin_to_grid(df, lats_1d, lons_1d):
    """Bin lightning pulses onto a regular lat/lon grid."""
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


def build_lightning():
    """
    Read all ENTLN mat files from the bucket, bin pulses to the ERA5 grid,
    write Lightning.zarr with one entln_count(time, lat, lon) per hour.
    """
    # Get the ERA5 grid from Year2023 Zarr (must exist first)
    if not os.path.exists(ERA5_ZARR):
        print('ERROR: run "python build_zarr.py era5" first to get the grid.')
        return

    ds_grid = xr.open_zarr(ERA5_ZARR)
    lats = ds_grid.lat.values
    lons = ds_grid.lon.values
    times = pd.DatetimeIndex(ds_grid.time.values)
    ds_grid.close()

    # Find mat files
    mat_files = sorted(glob.glob(os.path.join(ENTLN_DIR, '*.mat')))
    if not mat_files:
        print(f'ERROR: no mat files found in {ENTLN_DIR}')
        return

    print(f'\n{"="*55}')
    print(f'Building Lightning.zarr')
    print(f'  Mat files: {[os.path.basename(f) for f in mat_files]}')
    print(f'  Grid: {len(lats)}×{len(lons)}, {len(times)} timesteps')
    print(f'  Output → {LIGHTNING_ZARR}')

    # Load all mat files into one DataFrame
    dfs = []
    for mat in mat_files:
        print(f'  Loading {os.path.basename(mat)}...')
        df = load_entln_mat(mat)
        print(f'    {len(df):,} pulses')
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print(f'  Total: {len(df_all):,} pulses')

    if os.path.exists(LIGHTNING_ZARR):
        ans = input(f'\n  Lightning.zarr exists. Overwrite? [y/N]: ').strip().lower()
        if ans != 'y':
            return
        shutil.rmtree(LIGHTNING_ZARR)

    os.makedirs(ZARR_DIR, exist_ok=True)

    # Bin hour by hour and write in monthly batches
    BATCH = 720
    batches = [times[i:i+BATCH] for i in range(0, len(times), BATCH)]
    initialized = False
    t_total = datetime.now()

    for b_idx, batch_times in enumerate(batches):
        print(f'\n  Batch {b_idx+1}/{len(batches)}: {batch_times[0].date()} → {batch_times[-1].date()}...',
              end='', flush=True)

        grids = []
        for t in batch_times:
            t_end = t + pd.Timedelta(hours=1)
            mask  = (df_all.UTC >= t) & (df_all.UTC < t_end)
            grids.append(bin_to_grid(df_all[mask], lats, lons))

        data = np.stack(grids, axis=0)   # (batch, lat, lon)
        ds_batch = xr.Dataset(
            {'entln_count': (['time', 'lat', 'lon'], data)},
            coords={'time': batch_times.values, 'lat': lats, 'lon': lons}
        ).chunk({'time': TIME_CHUNK, 'lat': -1, 'lon': -1})

        if not initialized:
            ds_batch.to_zarr(LIGHTNING_ZARR, mode='w')
            initialized = True
        else:
            ds_batch.to_zarr(LIGHTNING_ZARR, append_dim='time')

        print(f' done', flush=True)

    elapsed = (datetime.now() - t_total).total_seconds()
    print(f'\n  Finished in {elapsed/60:.1f} min')
    _verify(LIGHTNING_ZARR, 'Lightning')


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if mode == 'era5':
        build_era5()
    elif mode == 'lightning':
        build_lightning()
    elif mode == 'all':
        build_era5()
        build_lightning()
    else:
        print('Usage:')
        print('  python build_zarr.py era5       → ERA5.zarr       (all ERA5 NC files)')
        print('  python build_zarr.py lightning  → Lightning.zarr  (from ENTLN mat files)')
        print('  python build_zarr.py all        → both')
        print(f'\nAll outputs → {ZARR_DIR}')
