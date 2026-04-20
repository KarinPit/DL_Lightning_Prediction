"""
fix_lightning_zeros.py — Overwrite zero lightning values in Lightning.zarr
with real ENTLN data for a specific date range.

Use this when a case was appended with zeros (ENTLN file not found at the time).

Usage:
    python3 data/fix_lightning_zeros.py
"""

import numpy as np
import pandas as pd
import scipy.io
import xarray as xr
import zarr
import os

BUCKET         = '/home/ec2-user/thesis-bucket'
LIGHTNING_ZARR = f'{BUCKET}/Zarr/Lightning.zarr'
ENTLN_CASES_DIR = f'{BUCKET}/Lightning_Data/ENTLN/Cases'

# ── Cases to fix (date range + ENTLN filename) ────────────────────────────────
CASES_TO_FIX = [
    {
        'name':       'Case2_Jan_2023_11_16',
        'start':      '2023-01-11',
        'end':        '2023-01-16',
        'entln_file': f'{ENTLN_CASES_DIR}/ENTLN_pulse_Case2_Jan_2023_11_16_struct.mat',
    },
    {
        'name':       'Case3_Mar_2023_13_15',
        'start':      '2023-03-13',
        'end':        '2023-03-15',
        'entln_file': f'{ENTLN_CASES_DIR}/ENTLN_pulse_Case3_Mar_2023_13_15_struct.mat',
    },
    {
        'name':       'Case5_Jan_2024_26_31',
        'start':      '2024-01-26',
        'end':        '2024-01-31',
        'entln_file': f'{ENTLN_CASES_DIR}/ENTLN_pulse_Case5_Jan_2024_26_31_struct.mat',
    },
]


def load_entln(mat_path):
    print(f"  Loading ENTLN: {mat_path}")
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
    df = df[(df.lat != 0) | (df.lon != 0)]
    print(f"  {len(df):,} pulses  ({df.UTC.min().date()} → {df.UTC.max().date()})")
    return df


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


def fix_case(case):
    name       = case['name']
    start      = pd.Timestamp(case['start'])
    end        = pd.Timestamp(case['end']) + pd.Timedelta(hours=23)
    entln_file = case['entln_file']

    print(f"\n{'='*55}")
    print(f"Fixing: {name}  ({case['start']} → {case['end']})")

    if not os.path.exists(entln_file):
        print(f"  ERROR: ENTLN file not found: {entln_file}")
        print(f"  Available files in {ENTLN_CASES_DIR}:")
        for f in os.listdir(ENTLN_CASES_DIR):
            print(f"    {f}")
        return

    # Open Lightning.zarr to find the time indices
    ds = xr.open_zarr(LIGHTNING_ZARR)
    all_times = pd.DatetimeIndex(ds.time.values)
    lats      = ds.lat.values
    lons      = ds.lon.values
    ds.close()

    mask    = (all_times >= start) & (all_times <= end)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"  ERROR: No timestamps found for {name} in Lightning.zarr")
        return

    print(f"  Found {len(indices)} timestamps at Zarr indices {indices[0]}–{indices[-1]}")

    # Load ENTLN and filter to case period
    df_entln = load_entln(entln_file)
    df_entln = df_entln[(df_entln.UTC >= start) & (df_entln.UTC <= end)]
    print(f"  Filtered to case: {len(df_entln):,} pulses")

    # Bin hour by hour
    case_times = all_times[mask]
    grids = []
    for t in case_times:
        t_end = t + pd.Timedelta(hours=1)
        m = (df_entln.UTC >= t) & (df_entln.UTC < t_end)
        grids.append(bin_to_grid(df_entln[m], lats, lons))
        if m.sum() > 0:
            print(f"    {t}  ⚡ {m.sum()} pulses")

    data = np.stack(grids, axis=0)  # (n_times, lat, lon)
    total_lightning = (data > 0).sum()
    print(f"  Binned: {total_lightning} lightning grid cells across {len(grids)} hours")

    # Overwrite in-place using zarr store directly
    store = zarr.open(LIGHTNING_ZARR, mode='r+')
    for i, zarr_idx in enumerate(indices):
        store['entln_count'][zarr_idx] = grids[i]

    print(f"  ✓ Overwritten {len(indices)} timesteps in Lightning.zarr")

    # Verify
    ds_check = xr.open_zarr(LIGHTNING_ZARR)
    written = ds_check['entln_count'].isel(time=slice(int(indices[0]), int(indices[-1])+1)).values
    ds_check.close()
    print(f"  Verification: max lightning count = {written.max():.1f}  "
          f"({'✓ non-zero' if written.max() > 0 else '✗ still zero!'})")


if __name__ == '__main__':
    for case in CASES_TO_FIX:
        fix_case(case)
    print("\nDone.")
