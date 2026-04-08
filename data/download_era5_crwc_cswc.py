"""
CDS Download Script — crwc + cswc only (not available in ARCO-ERA5)
Downloads specific_rain_water_content and specific_snow_water_content
at all 37 pressure levels for each storm case.

Output: local_raw_data/{case_name}/ERA5_pressure/{variable}.nc
        local_raw_data/{case_name}/ERA5_pressure/{variable}.done  (skip flag)

Karin Pitlik
"""

import cdsapi
import os
import zipfile

BASE_PATH = '/home/ubuntu/Desktop/local_raw_data'

CASES = [
    {'name': 'Case1_Nov_2022_23_25', 'year': ['2022'], 'month': ['11'], 'day': ['23','24','25']},
    {'name': 'Case2_Jan_2023_11_16', 'year': ['2023'], 'month': ['01'], 'day': ['11','12','13','14','15','16']},
    {'name': 'Case3_Mar_2023_13_15', 'year': ['2023'], 'month': ['03'], 'day': ['13','14','15']},
    {'name': 'Case4_Apr_2023_09_13', 'year': ['2023'], 'month': ['04'], 'day': ['09','10','11','12','13']},
    {'name': 'Case5_Jan_2024_26_31', 'year': ['2024'], 'month': ['01'], 'day': ['26','27','28','29','30','31']},
]

# Only the two variables missing from ARCO
MISSING_VARS = [
    'specific_rain_water_content',   # crwc
    'specific_snow_water_content',   # cswc
]

# All 37 ERA5 pressure levels
PRESSURE_LEVELS = [
    '1','2','3','5','7','10','20','30','50','70',
    '100','125','150','175','200','225','250','300',
    '350','400','450','500','550','600','650','700',
    '750','775','800','825','850','875','900','925',
    '950','975','1000',
]

TIMES = [f'{h:02d}:00' for h in range(24)]
AREA  = [36.598, 27.954, 27.296, 39.292]   # N, W, S, E

client = cdsapi.Client()

for case in CASES:
    out_dir = os.path.join(BASE_PATH, case['name'], 'ERA5_pressure')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Case: {case['name']}")

    for variable in MISSING_VARS:
        done_flag   = os.path.join(out_dir, f'{variable}.done')
        renamed_nc  = os.path.join(out_dir, f'{variable}.nc')

        if os.path.exists(done_flag) and os.path.exists(renamed_nc):
            print(f"  SKIP {variable} (already downloaded)")
            continue

        print(f"  Downloading {variable} ...")

        request = {
            'product_type': ['reanalysis'],
            'variable':      [variable],
            'pressure_level': PRESSURE_LEVELS,
            'year':   case['year'],
            'month':  case['month'],
            'day':    case['day'],
            'time':   TIMES,
            'data_format':    'netcdf',
            'download_format': 'zip',
            'area':   AREA,
        }

        zip_file = os.path.join(out_dir, f'{variable}_download.zip')
        client.retrieve('reanalysis-era5-pressure-levels', request, target=zip_file)

        # Extract zip — CDS always names the file data_stream-oper_stepType-instant.nc
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(out_dir)
        os.remove(zip_file)

        # Rename to variable-specific name so files don't overwrite each other
        generic_nc = os.path.join(out_dir, 'data_stream-oper_stepType-instant.nc')
        if os.path.exists(generic_nc):
            os.rename(generic_nc, renamed_nc)
            print(f"  Saved → {renamed_nc}")
        else:
            # Some CDS responses use different step-type names — find any .nc file
            nc_files = [f for f in os.listdir(out_dir)
                        if f.endswith('.nc') and not f.startswith(variable)]
            if nc_files:
                os.rename(os.path.join(out_dir, nc_files[0]), renamed_nc)
                print(f"  Saved (renamed from {nc_files[0]}) → {renamed_nc}")
            else:
                print(f"  WARNING: Could not find extracted NC file for {variable}")
                continue

        # Write done flag
        open(done_flag, 'w').close()
        print(f"  Done: {variable}")

print('\n\nAll crwc + cswc downloads complete!')
