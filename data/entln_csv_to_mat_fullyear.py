"""
ENTLN CSV → MAT Converter  —  Full Year Edition
Reads hourly pulse CSV files from the nested feed-MM-DD-YYYY/pulse/ structure,
filters to Israel domain, saves one struct .mat file for the full year.

Input structure:
  ENTLN_2023/
    feed-01-01-2023/
      pulse/
        pulse-01-01-2023-00.csv   ← one per hour (24 per day)
        pulse-01-01-2023-01.csv
        ...
    feed-01-02-2023/
      ...

Output:
  ENTLN_pulse_Year2023_struct.mat   (lat, lon, UTC fields)

Karin Pitlik
"""

import os
import glob
import pandas as pd
import numpy as np
import scipy.io

# ============================================================
#  CONFIGURATION  ← set these paths
# ============================================================

YEAR = 2023

# Folder containing all the unzipped feed-MM-DD-YYYY directories
ENTLN_ROOT = "/Users/karinpitlik/Downloads/ENTLN_2023"

# Where to save the output .mat file
OUTPUT_PATH = "/Users/karinpitlik/Downloads/ready"

# Israel bounding box
MIN_LAT = 27.296
MAX_LAT = 36.598
MIN_LON = 27.954
MAX_LON = 39.292

# Column names (no header in CSV — same layout as old TXT files)
CSV_COLUMNS = [
    "type",
    "datetime_str",
    "lat",
    "lon",
    "peak_current",
    "col5",
    "ic_height",
    "multiplicity",
    "cg_multiplicity",
]


# ============================================================
#  READ ALL CSV FILES
# ============================================================


def read_all_csvs(root_folder):
    """
    Walk all feed-*/pulse/ subdirectories and read every pulse CSV.
    Returns a combined DataFrame (unfiltered).
    """
    # Find all pulse CSV files recursively
    pattern = os.path.join(root_folder, "feed-*", "pulse", "pulse-*.csv")
    all_files = sorted(glob.glob(pattern))
    print(f"Found {len(all_files):,} CSV files under {root_folder}")

    if not all_files:
        raise FileNotFoundError(
            f"No CSV files found. Check ENTLN_ROOT path: {root_folder}"
        )

    dfs = []
    for i, fpath in enumerate(all_files):
        try:
            # Only read the 4 columns we need — handles files with 9 OR 17 columns
            # on_bad_lines='skip' silently drops rows with wrong field count
            df = pd.read_csv(
                fpath,
                header=None,
                usecols=[0, 1, 2, 3],
                names=["type", "datetime_str", "lat", "lon"],
                dtype={"datetime_str": str},
                on_bad_lines="skip",
            )
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {os.path.basename(fpath)}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  Read {i + 1:,} / {len(all_files):,} files ...")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows loaded: {len(combined):,}")
    return combined


# ============================================================
#  PARSE DATETIME AND FILTER TO ISRAEL
# ============================================================


def parse_and_filter(df):
    """
    Parse datetime string (20230101T001523.301 → Timestamp),
    filter to Israel bounding box, drop rows with invalid lat/lon.
    """
    print("  Parsing datetimes ...")
    df["UTC"] = pd.to_datetime(
        df["datetime_str"].astype(str).str.strip(),
        format="%Y%m%dT%H%M%S.%f",
        errors="coerce",
    )
    df = df.dropna(subset=["UTC"])

    # Numeric lat/lon
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # Drop clearly invalid rows (lat=0, lon=0 are null pulses)
    df = df[(df["lat"] != 0.0) | (df["lon"] != 0.0)]

    # Filter to Israel domain
    mask = (
        (df["lat"] >= MIN_LAT)
        & (df["lat"] <= MAX_LAT)
        & (df["lon"] >= MIN_LON)
        & (df["lon"] <= MAX_LON)
    )
    df_israel = df[mask].copy().reset_index(drop=True)
    print(f"  Rows in Israel domain: {len(df_israel):,} / {len(df):,}")
    return df_israel


# ============================================================
#  SAVE AS STRUCT .MAT
# ============================================================


def save_mat(df, year, output_dir):
    """
    Save DataFrame as a struct .mat file.
    UTC stored as 'DD-Mon-YYYY HH:MM:SS' strings to match load_entln_mat().
    """
    outfile = os.path.join(output_dir, f"ENTLN_pulse_Year{year}_struct.mat")

    utc_strings = df["UTC"].dt.strftime("%d-%b-%Y %H:%M:%S").values

    entln_struct = {
        "lat": df["lat"].values.astype(np.float64),
        "lon": df["lon"].values.astype(np.float64),
        "UTC": np.array(utc_strings, dtype=object),
    }

    scipy.io.savemat(outfile, {"entln": entln_struct}, do_compression=True)
    print(f"\n  Saved: {outfile}")
    print(f"  Total pulses in Israel: {len(df):,}")
    print(f"  Date range: {df['UTC'].min()} → {df['UTC'].max()}")
    return outfile


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"ENTLN CSV → MAT  |  Full year {YEAR}")
    print(f"{'='*55}")

    # Step 1: Read all CSV files
    df_raw = read_all_csvs(ENTLN_ROOT)

    # Step 2: Parse datetimes + filter to Israel
    df_israel = parse_and_filter(df_raw)

    # Step 3: Keep only the target year (in case any files bleed into adjacent years)
    df_year = df_israel[df_israel["UTC"].dt.year == YEAR].copy()
    print(f"  Pulses in year {YEAR}: {len(df_year):,}")

    # Step 4: Save .mat
    save_mat(df_year, YEAR, OUTPUT_PATH)

    print(f"\nDone! Copy the .mat file to the server:")
    print(
        f"  scp {OUTPUT_PATH}/ENTLN_pulse_Year{YEAR}_struct.mat "
        f"ubuntu@<server-ip>:/home/ubuntu/Desktop/local_raw_data/ENTLN/"
    )
