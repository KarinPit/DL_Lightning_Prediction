import cdsapi
import os
import zipfile

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

PRESSURE_VARIABLES = [
    "relative_humidity",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_humidity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10",
    "20", "30", "50", "70", "100", "125",
    "150", "175", "200", "225", "250", "300",
    "350", "400", "450", "500", "550", "600",
    "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975", "1000"
]

TIMES = [f'{h:02d}:00' for h in range(24)]
AREA = [36.598, 27.954, 27.296, 39.292]

client = cdsapi.Client()

for case in CASES:
    for variable in PRESSURE_VARIABLES:
        out_dir = os.path.join(BASE_PATH, case['name'], 'ERA5_pressure')
        os.makedirs(out_dir, exist_ok=True)

        done_flag = os.path.join(out_dir, f'{variable}.done')
        renamed_nc = os.path.join(out_dir, f'{variable}.nc')

        # Skip if already successfully downloaded and renamed
        if os.path.exists(done_flag) and os.path.exists(renamed_nc):
            print(f"  Skipping {variable} | {case['name']} — already done")
            continue

        print(f"\n{'='*50}")
        print(f"Downloading: {case['name']} | {variable}")

        request = {
            'product_type': ['reanalysis'],
            'variable': [variable],
            'pressure_level': PRESSURE_LEVELS,
            'year':  case['year'],
            'month': case['month'],
            'day':   case['day'],
            'time':  TIMES,
            'data_format': 'netcdf',
            'download_format': 'zip',
            'area': AREA,
        }

        zip_file = os.path.join(out_dir, f'{variable}.zip')
        client.retrieve('reanalysis-era5-pressure-levels', request, target=zip_file)

        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(out_dir)
        os.remove(zip_file)

        # Rename the generic extracted file to include the variable name
        generic_nc = os.path.join(out_dir, 'data_stream-oper_stepType-instant.nc')
        if os.path.exists(generic_nc):
            os.rename(generic_nc, renamed_nc)
            print(f"  Renamed to: {variable}.nc")
        else:
            print(f"  WARNING: expected NC file not found after extraction for {variable}")

        open(done_flag, 'w').close()
        print(f"  Done: {variable} | {case['name']}")

print('\n\nAll cases downloaded!')
