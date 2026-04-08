"""
ENTLN TXT to MAT Converter
Reads hourly pulse txt files from All_Data folder,
filters to Israel domain, groups by case date range,
saves as struct .mat files compatible with process_era5.py

Karin Pitlik
"""

import os
import glob
import pandas as pd
import numpy as np
import scipy.io
from datetime import datetime

# ============================================================
#  CONFIGURATION
# ============================================================

BASE_PATH     = '/home/ubuntu/Desktop/local_raw_data'
ALL_DATA_PATH = os.path.join(BASE_PATH, 'ENTLN', 'All_Data')
OUTPUT_PATH   = os.path.join(BASE_PATH, 'ENTLN')

# Israel bounding box
MIN_LAT = 27.296
MAX_LAT = 36.598
MIN_LON = 27.954
MAX_LON = 39.292

# Cases to generate — add/extend as needed
CASES = [
    ('Jan2022_pulse', '2022-01-01', '2022-01-31'),
]

# ============================================================
#  READ ALL TXT FILES
# ============================================================

def read_all_txt(folder):
    """Read all pulse txt files from folder into one DataFrame."""
    all_files = sorted(glob.glob(os.path.join(folder, 'pulse-*.txt')))
    print(f"Found {len(all_files)} txt files in {folder}")

    dfs = []
    for fpath in all_files:
        try:
            df = pd.read_csv(
                fpath,
                header=None,
                names=['type', 'datetime_str', 'lat', 'lon',
                       'peak_current', 'num_sensors', 'ic_height',
                       'multiplicity', 'cg_multiplicity'],
            )
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {fpath}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No txt files found in {folder}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows loaded: {len(combined):,}")
    return combined


# ============================================================
#  PARSE AND FILTER
# ============================================================

def parse_and_filter(df):
    """Parse datetime, filter to Israel bounding box."""
    # Parse datetime: 20220131T225949.446 → datetime
    df['UTC'] = pd.to_datetime(
        df['datetime_str'].astype(str),
        format='%Y%m%dT%H%M%S.%f',
        errors='coerce'
    )
    df = df.dropna(subset=['UTC'])

    # Ensure numeric lat/lon
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])

    # Filter to Israel
    mask = (
        (df['lat'] >= MIN_LAT) & (df['lat'] <= MAX_LAT) &
        (df['lon'] >= MIN_LON) & (df['lon'] <= MAX_LON)
    )
    df_israel = df[mask].copy()
    print(f"  Rows in Israel domain: {len(df_israel):,} / {len(df):,}")
    return df_israel


# ============================================================
#  SAVE AS STRUCT MAT
# ============================================================

def save_mat(df_case, case_name):
    """Save case DataFrame as struct .mat file compatible with process_era5.py"""
    outfile = os.path.join(OUTPUT_PATH, f'ENTLN_pulse_{case_name}_struct.mat')

    # Format UTC as 'dd-mmm-yyyy HH:MM:SS' to match load_entln_mat()
    utc_strings = df_case['UTC'].dt.strftime('%d-%b-%Y %H:%M:%S').values

    entln = {
        'lat': df_case['lat'].values.astype(np.float64),
        'lon': df_case['lon'].values.astype(np.float64),
        'UTC': np.array(utc_strings, dtype=object),
    }

    scipy.io.savemat(outfile, {'entln': entln}, do_compression=True)
    print(f"  Saved: {outfile} ({len(df_case):,} pulses)")
    return outfile


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    # Load all txt data once
    df_all = read_all_txt(ALL_DATA_PATH)
    df_filtered = parse_and_filter(df_all)

    # Generate one mat file per case
    for case_name, start_str, end_str in CASES:
        print(f"\nProcessing case: {case_name} ({start_str} to {end_str})")
        start_dt = pd.Timestamp(start_str)
        end_dt   = pd.Timestamp(end_str) + pd.Timedelta(days=1)

        mask = (df_filtered['UTC'] >= start_dt) & (df_filtered['UTC'] < end_dt)
        df_case = df_filtered[mask]

        if df_case.empty:
            print(f"  WARNING: No data found for {case_name}")
            continue

        save_mat(df_case, case_name)

    print("\n\nDone!")
