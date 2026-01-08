import os
import numpy as np
import netCDF4 as nc


# --- Configuration ---
THESIS_PATH = "/Users/karinpitlik/Desktop/DataScience/Thesis"
CASE_NAME = "Case1_Nov_2022_23_25"
PROCESSED_DIR_NAME = "proccesed"
RESOLUTION = "4by4"
INTERVAL = "3_hours"
ATM_VARIABLES = ["PREC_RATE", "LPI", "KI", "CAPE2D"]
TARGET_VARIABLE = "ENTLN"
OUTPUT_BASE_DIR = f"{THESIS_PATH}/{CASE_NAME}/Ens/Processed_Ready_ML/"


def get_aligned_files(start_date, end_date):
    """
    Returns a single list of filenames that exist in *all* directories
    (both Atmospheric variables and ENTLN) and are within the specified date range.
    This ensures synchronization across all features and targets.
    """

    # Start with the file list from the first variable
    first_var = ATM_VARIABLES[0]
    base_path = f"{THESIS_PATH}/{CASE_NAME}/Ens/Raw/{first_var}/{PROCESSED_DIR_NAME}/{RESOLUTION}/{INTERVAL}/"

    # Check if path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path not found: {base_path}")
        return [], []

    # Initialize the set with files from the first variable
    common_files = set(os.listdir(base_path))

    # Intersect with all other atmospheric variables
    for var in ATM_VARIABLES[1:]:
        path = f"{THESIS_PATH}/{CASE_NAME}/Ens/Raw/{var}/{PROCESSED_DIR_NAME}/{RESOLUTION}/{INTERVAL}/"
        if os.path.exists(path):
            # Keep only files that exist in the current variable's folder as well
            common_files.intersection_update(os.listdir(path))
        else:
            print(f"Warning: Path missing for {var}")
            return [], []

    # Intersect with the target variable (ENTLN)
    entln_path = f"{THESIS_PATH}/{TARGET_VARIABLE}/{TARGET_VARIABLE}_pulse_{CASE_NAME}/{RESOLUTION}/{INTERVAL}/"
    if os.path.exists(entln_path):
        common_files.intersection_update(os.listdir(entln_path))
    else:
        print(f"Error: ENTLN path missing: {entln_path}")
        return [], []

    # Filter by date and Sort
    final_files = []
    sorted_files = sorted(list(common_files))

    for filename in sorted_files:
        if not filename.endswith(".nc"):
            continue

        # Extract date
        file_time = filename[:19]
        if start_date <= file_time <= end_date:
            final_files.append(filename)

    # Split into Train and Test
    num_files = len(final_files)
    print(f"Found {num_files} perfectly aligned files.")

    if num_files == 0:
        return [], []

    # 70% Train, 30% Test split
    split_idx = int(np.ceil(num_files * 0.7))
    train_files = final_files[:split_idx]
    test_files = final_files[split_idx:]

    return train_files, test_files


def investigate_missing_dates(start_date, end_date):
    print(f"Investigating files between {start_date} and {end_date}...\n")

    # Helper function to check any folder
    def check_folder(name, path):
        if not os.path.exists(path):
            print(f"❌ {name}: Path NOT found!")
            return

        files = sorted([f for f in os.listdir(path) if f.endswith(".nc")])
        relevant = [f for f in files if start_date <= f[:19] <= end_date]

        if len(relevant) > 0:
            print(f"✅ {name}: Found {len(relevant)} files.")
            print(f"   🔹 First: {relevant[0]}")
            print(f"   🔸 Last:  {relevant[-1]}")  # Prints the end file
        else:
            print(f"⚠️ {name}: Found 0 files in range!")
            # Debug: Show what IS inside the folder
            if len(files) > 0:
                print(
                    f"   (Folder actually contains data from: {files[0][:19]} -> {files[-1][:19]})"
                )

    # 1. Check Atmospheric Variables
    for var in ATM_VARIABLES:
        path = f"{THESIS_PATH}/{CASE_NAME}/Ens/Raw/{var}/{PROCESSED_DIR_NAME}/{RESOLUTION}/{INTERVAL}/"
        check_folder(var, path)

    # 2. Check ENTLN (Lightning)
    entln_path = f"{THESIS_PATH}/{TARGET_VARIABLE}/{TARGET_VARIABLE}_pulse_{CASE_NAME}/{RESOLUTION}/{INTERVAL}/"
    check_folder("ENTLN", entln_path)


def get_all_file_path(train_files_list, test_files_list):
    """
    Creates full paths for ENTLN and all Atmospheric variables.
    """
    # Get ENTLN paths
    entln_path_dir = f"{THESIS_PATH}/{TARGET_VARIABLE}/{TARGET_VARIABLE}_pulse_{CASE_NAME}/{RESOLUTION}/{INTERVAL}/"
    entln_train_paths = [os.path.join(entln_path_dir, f) for f in train_files_list]
    entln_test_paths = [os.path.join(entln_path_dir, f) for f in test_files_list]

    model_train_paths = {}
    model_test_paths = {}

    # Get ATM paths
    for atm_var in ATM_VARIABLES:
        base_path = f"{THESIS_PATH}/{CASE_NAME}/Ens/Raw/{atm_var}/{PROCESSED_DIR_NAME}/{RESOLUTION}/{INTERVAL}/"
        model_train_paths[atm_var] = [
            os.path.join(base_path, f) for f in train_files_list
        ]
        model_test_paths[atm_var] = [
            os.path.join(base_path, f) for f in test_files_list
        ]

    return entln_train_paths, entln_test_paths, model_train_paths, model_test_paths


def get_reduction_method(var_name):
    """Decides whether to SUM or MEAN based on the variable name."""
    if "PREC_RATE" in var_name.upper():
        return np.sum, "sum"
    else:
        return np.mean, "mean"


# def get_netcdf_data(model_train_files, model_test_files):
#     """
#     Opens a specific .nc file, reads the data, reduces the ensemble dimension
#     (Sum for Rain, Max for others), and flattens it to a 1D array.
#     """

#     for atm_var in model_train_files.keys():
#         paths = model_train_files[atm_var]
#         for path in paths:
#             ds = nc.Dataset(path)
#             print(ds)


def get_netcdf_data(files_dict):
    """
    Iterates over a dictionary of file paths, reads the data,
    reduces the time dimension (Sum for Rain, Mean for others),
    and returns a dictionary of flattened arrays.
    """
    processed_data = {}
    print(f"--- Loading and Processing Data ---")

    for atm_var, paths in files_dict.items():
        print(f"Processing variable: {atm_var} ({len(paths)} files)")

        var_data_list = []

        for path in paths:
            try:
                with nc.Dataset(path, "r") as ds:
                    keys = list(ds.variables.keys())
                    print(keys)
                    # target_key = atm_var.lower()
                    # data = ds.variables[target_key][:]
                    # print(data)

            except Exception as e:
                print(f"Error reading {path}: {e}")
    #                 # 1. Find the variable name inside the file
    #                 # (The file might use lowercase 'prec_rate' while folder is 'PREC_RATE')
    #                 keys = list(ds.variables.keys())

    #                 # Try exact match or lowercase match
    #                 target_key = None
    #                 if atm_var in keys:
    #                     target_key = atm_var
    #                 else:
    #                     # Search for case-insensitive match (e.g., 'prec_rate' == 'PREC_RATE')
    #                     for key in keys:
    #                         if key.lower() == atm_var.lower():
    #                             target_key = key
    #                             break

    #                 if target_key:
    #                     # 2. Extract Data
    #                     data = ds.variables[target_key][:]

    #                     # 3. Apply Math (Reduction)
    #                     # Your data is (time, lon, lat). We want to collapse 'time' (axis 0).
    #                     if data.ndim > 2:
    #                         if "PREC" in atm_var.upper():
    #                             # Sum accumulated rain over the 180 time steps
    #                             data_reduced = np.sum(data, axis=0)
    #                         else:
    #                             # Mean for other variables (CAPE, etc.)
    #                             data_reduced = np.mean(data, axis=0)
    #                     else:
    #                         data_reduced = data  # Already 2D

    #                     # 4. Flatten (2D Map -> 1D Array of pixels)
    #                     # Replaces NaNs with 0 to prevent model errors
    #                     flat_data = np.nan_to_num(data_reduced.flatten(), nan=0.0)

    #                     var_data_list.append(flat_data)

    #         except Exception as e:
    #             print(f"❌ Error reading {path}: {e}")

    #     # Store the list of arrays for this variable
    #     processed_data[atm_var] = var_data_list

    # return processed_data


if __name__ == "__main__":
    start_date = "2022-11-23_03_00_00"
    end_date = "2022-11-25_00_00_00"

    train_files, test_files = get_aligned_files(start_date, end_date)
    entln_train_files, entln_test_files, model_train_files, model_test_files = (
        get_all_file_path(train_files, test_files)
    )

    X_train = get_netcdf_data(model_train_files)
    # print(X_train)

    # print(f"Train files: {len(train_files)}")
    # print(f"Test files: {len(test_files)}")
    # if len(train_files) > 0:
    #     print(f"\nFirst train file: {train_files[0]}")
    #     print(f"Last train file:  {train_files[-1]}")
    #     print(f"\nFirst test file: {test_files[0]}")
    #     print(f"Last test file:  {test_files[-1]}")
