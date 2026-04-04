"""
ERA5 Pressure-Level Download Script
Downloads wind and vertical velocity at key pressure levels.
Output saved to: local_raw_data/{case_name}/ERA5_pressure/

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

PRESSURE_VARIABLES = [
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
    'temperature',
    'relative_humidity',
]

PRESSURE_LEVELS = ['500', '700', '850']

TIMES = [f'{h:02d}:00' for h in range(24)]
AREA = [36.598, 27.954, 27.296, 39.292]

client = cdsapi.Client()

for case in CASES:
    out_dir = os.path.join(BASE_PATH, case['name'], 'ERA5_pressure')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Downloading pressure levels: {case['name']}")

    request = {
        'product_type': ['reanalysis'],
        'variable': PRESSURE_VARIABLES,
        'pressure_level': PRESSURE_LEVELS,
        'year':  case['year'],
        'month': case['month'],
        'day':   case['day'],
        'time':  TIMES,
        'data_format': 'netcdf',
        'download_format': 'zip',
        'area': AREA,
    }

    zip_file = os.path.join(out_dir, 'era5_pressure_download.zip')
    client.retrieve('reanalysis-era5-pressure-levels', request, target=zip_file)

    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(out_dir)
    os.remove(zip_file)
    print(f"  Done: {case['name']}")

print('\n\nAll pressure-level cases downloaded!')
