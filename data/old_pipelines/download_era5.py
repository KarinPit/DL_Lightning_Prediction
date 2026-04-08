"""
ERA5 Download Script
Downloads ERA5 single-level variables for each storm case.
Output saved to: local_raw_data/{case_name}/ERA5/

Karin Pitlik
"""

import cdsapi
import os

# ============================================================
#  CONFIGURATION
# ============================================================

BASE_PATH = '/home/ubuntu/Desktop/local_raw_data'

CASES = [
    {
        'name': 'Case1_Nov_2022_23_25',
        'year':  ['2022'],
        'month': ['11'],
        'day':   ['23', '24', '25'],
    },
    {
        'name': 'Case2_Jan_2023_11_16',
        'year':  ['2023'],
        'month': ['01'],
        'day':   ['11', '12', '13', '14', '15', '16'],
    },
    {
        'name': 'Case3_Mar_2023_13_15',
        'year':  ['2023'],
        'month': ['03'],
        'day':   ['13', '14', '15'],
    },
    {
        'name': 'Case4_Apr_2023_09_13',
        'year':  ['2023'],
        'month': ['04'],
        'day':   ['09', '10', '11', '12', '13'],
    },
    {
        'name': 'Case5_Jan_2024_26_31',
        'year':  ['2024'],
        'month': ['01'],
        'day':   ['26', '27', '28', '29', '30', '31'],
    },
]

VARIABLES = [
    # Instability
    'convective_available_potential_energy',
    'convective_inhibition',
    'k_index',
    'total_totals_index',
    # Moisture
    '2m_dewpoint_temperature',
    'total_column_water_vapour',
    'total_column_cloud_ice_water',
    'total_column_cloud_liquid_water',
    # Precipitation
    'convective_precipitation',
    'convective_rain_rate',
    'total_precipitation',
    # Dynamics
    'mean_vertically_integrated_moisture_divergence',
    # Surface
    '2m_temperature',
    'mean_sea_level_pressure',
    'surface_pressure',
    'cloud_base_height',
    'high_cloud_cover',
]

TIMES = [f'{h:02d}:00' for h in range(24)]

# Israel bounding box [North, West, South, East]
AREA = [36.598, 27.954, 27.296, 39.292]

# ============================================================
#  DOWNLOAD LOOP
# ============================================================

client = cdsapi.Client()

for case in CASES:
    out_dir = os.path.join(BASE_PATH, case['name'], 'ERA5')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Downloading: {case['name']}")
    print(f"  Year:  {case['year']}")
    print(f"  Month: {case['month']}")
    print(f"  Days:  {case['day']}")
    print(f"  Output: {out_dir}")
    print(f"{'='*50}")

    request = {
        'product_type': ['reanalysis'],
        'variable': VARIABLES,
        'year':  case['year'],
        'month': case['month'],
        'day':   case['day'],
        'time':  TIMES,
        'data_format': 'netcdf',
        'download_format': 'zip',
        'area': AREA,
    }

    zip_file = os.path.join(out_dir, 'era5_download.zip')

    client.retrieve(
        'reanalysis-era5-single-levels',
        request,
        target=zip_file,
    )

    # Unzip into the ERA5 folder
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(out_dir)
    os.remove(zip_file)
    print(f"  Extracted to: {out_dir}")

    print(f"Done: {case['name']}")

print('\n\nAll cases downloaded!')
