"""
CDS Download Script — crwc + cswc, full year 2023, week by week
Downloads specific_rain_water_content and specific_snow_water_content
at all 37 pressure levels, split into 7-day chunks to stay under CDS field limit.

1 var × 37 levels × 7 days × 24h = 6,216 fields per request  ✓ (well under limit)

Output: local_raw_data/Year2023_full/ERA5_pressure/{variable}_{year}_{month:02d}_{week}.nc
        (merged into monthly files at the end: {variable}_{year}_{month:02d}.nc)

Resume-safe: already-downloaded weeks are skipped via .done flags.

Karin Pitlik
"""

import cdsapi
import os
import zipfile
import calendar
import numpy as np
import xarray as xr
from datetime import date, timedelta

YEAR      = 2023
BASE_PATH = '/home/ubuntu/Desktop/local_raw_data'
OUT_DIR   = os.path.join(BASE_PATH, f'Year{YEAR}_full', 'ERA5_pressure')
os.makedirs(OUT_DIR, exist_ok=True)

MISSING_VARS = [
    'specific_rain_water_content',   # → crwc
    'specific_snow_water_content',   # → cswc
]

PRESSURE_LEVELS = [
    '1','2','3','5','7','10','20','30','50','70',
    '100','125','150','175','200','225','250','300',
    '350','400','450','500','550','600','650','700',
    '750','775','800','825','850','875','900','925',
    '950','975','1000',
]

TIMES = [f'{h:02d}:00' for h in range(24)]
AREA  = [36.598, 27.954, 27.296, 39.292]   # N, W, S, E

# Split year into 7-day chunks
def get_weekly_chunks(year):
    """Return list of (start_date, end_date) 7-day chunks covering the full year."""
    chunks = []
    d = date(year, 1, 1)
    year_end = date(year, 12, 31)
    while d <= year_end:
        chunk_end = min(d + timedelta(days=6), year_end)
        chunks.append((d, chunk_end))
        d = chunk_end + timedelta(days=1)
    return chunks

client = cdsapi.Client()

total_downloaded = 0
total_skipped    = 0
weekly_chunks    = get_weekly_chunks(YEAR)
print(f"Total weekly chunks: {len(weekly_chunks)}")

for chunk_start, chunk_end in weekly_chunks:
    # Build day list for this chunk (may span two months)
    day_list   = []
    month_list = set()
    d = chunk_start
    while d <= chunk_end:
        day_list.append(f'{d.day:02d}')
        month_list.add(d.month)
        d += timedelta(days=1)

    # Use start month for naming (chunks never span more than 2 months)
    chunk_label = f'{YEAR}_{chunk_start.month:02d}_{chunk_start.day:02d}'

    for variable in MISSING_VARS:
        done_flag = os.path.join(OUT_DIR, f'{variable}_{chunk_label}.done')
        out_nc    = os.path.join(OUT_DIR, f'{variable}_{chunk_label}.nc')

        if os.path.exists(done_flag) and os.path.exists(out_nc):
            total_skipped += 1
            continue

        print(f"\n  Downloading {variable}  {chunk_start} → {chunk_end}  ({len(day_list)} days) ...")

        # Build month-aware request (CDS needs explicit month/day lists)
        months_in_chunk = sorted(month_list)

        request = {
            'product_type':   ['reanalysis'],
            'variable':       [variable],
            'pressure_level': PRESSURE_LEVELS,
            'year':           [str(YEAR)],
            'month':          [f'{m:02d}' for m in months_in_chunk],
            'day':            sorted(set(day_list)),   # unique days across months
            'time':           TIMES,
            'data_format':    'netcdf',
            'download_format': 'zip',
            'area':           AREA,
        }

        zip_file = os.path.join(OUT_DIR, f'{variable}_{chunk_label}_download.zip')

        try:
            client.retrieve('reanalysis-era5-pressure-levels', request, target=zip_file)
        except Exception as e:
            print(f"  ERROR downloading {variable} {chunk_label}: {e}")
            if os.path.exists(zip_file):
                os.remove(zip_file)
            continue

        # Extract zip
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(OUT_DIR)
        os.remove(zip_file)

        # CDS extracts to a generic name — find and rename it
        possible_names = [
            'data_stream-oper_stepType-instant.nc',
            'data_stream-oper_stepType-accum.nc',
        ]
        renamed = False
        for generic in possible_names:
            generic_path = os.path.join(OUT_DIR, generic)
            if os.path.exists(generic_path):
                os.rename(generic_path, out_nc)
                renamed = True
                break

        if not renamed:
            # Last resort: grab any .nc not yet named with this variable
            nc_files = [f for f in os.listdir(OUT_DIR)
                        if f.endswith('.nc') and variable not in f]
            if nc_files:
                os.rename(os.path.join(OUT_DIR, nc_files[0]), out_nc)
                renamed = True

        if not renamed:
            print(f"  WARNING: Could not find extracted NC for {variable} {chunk_label}")
            continue

        open(done_flag, 'w').close()
        print(f"  Saved → {out_nc}")
        total_downloaded += 1

print(f'\n{"="*55}')
print(f'Downloads done: {total_downloaded} new, {total_skipped} skipped')
print(f'{"="*55}')

# ============================================================
#  MERGE WEEKLY CHUNKS INTO MONTHLY FILES
#  (process_era5_fullyear.py expects {variable}_{year}_{month:02d}.nc)
# ============================================================

print(f'\nMerging weekly chunks into monthly files ...')

for month in range(1, 13):
    month_str = f'{YEAR}_{month:02d}'

    for variable in MISSING_VARS:
        out_monthly = os.path.join(OUT_DIR, f'{variable}_{month_str}.nc')
        done_monthly = os.path.join(OUT_DIR, f'{variable}_{month_str}_merged.done')

        if os.path.exists(done_monthly) and os.path.exists(out_monthly):
            print(f"  SKIP merge {variable} {month_str} (already merged)")
            continue

        # Collect all weekly NC files for this variable + month
        weekly_files = []
        for chunk_start, chunk_end in weekly_chunks:
            if chunk_start.month != month and chunk_end.month != month:
                continue
            chunk_label = f'{YEAR}_{chunk_start.month:02d}_{chunk_start.day:02d}'
            nc_path = os.path.join(OUT_DIR, f'{variable}_{chunk_label}.nc')
            if os.path.exists(nc_path):
                weekly_files.append(nc_path)

        if not weekly_files:
            print(f"  WARNING: No weekly files found for {variable} {month_str} — skipping merge")
            continue

        print(f"  Merging {variable} {month_str}: {len(weekly_files)} chunks ...")
        ds_list = [xr.open_dataset(f) for f in weekly_files]

        # Rename valid_time if needed
        ds_list = [ds.rename({'valid_time': 'time'}) if 'valid_time' in ds.coords else ds
                   for ds in ds_list]

        # Select only this month's timestamps
        ds_merged = xr.concat(ds_list, dim='time').sortby('time')
        ds_merged = ds_merged.sel(
            time=ds_merged.time.dt.month == month
        )

        ds_merged.to_netcdf(out_monthly)
        for ds in ds_list:
            ds.close()

        open(done_monthly, 'w').close()
        print(f"  Merged → {out_monthly}  ({len(ds_merged.time)} timesteps)")

print(f'\nAll done! Monthly files ready in: {OUT_DIR}')
