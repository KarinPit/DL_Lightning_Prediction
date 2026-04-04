import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [ # Instability
            'convective_available_potential_energy',
            'convective_inhibition',
            'k_index',
            'total_totals_index',
            # Moisture
            '2m_dewpoint_temperature',
            'total_column_water_vapour',
            'total_column_cloud_ice_water',   # best lightning proxy!
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
            'high_cloud_cover',],
    "year": [
        "2022"
    ],
    "month": [
        "11"
    ],
    "day": [

        "22", "23", "24",
        "25", "26",
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [36.598, 27.954, 27.296, 39.292]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
