import os
import numpy as np

# --- Configuration ---
THESIS_PATH = "/Users/karinpitlik/Desktop/DataScience/Thesis/"
CASE_NAME = "Case1_Nov_2022_23_25"
PROCESSED_DIR_NAME = "proccesed"
RESOLUTION = "4by4"
INTERVAL = "3_hours"
ATM_VARIABLES = ["PREC_RATE", "LPI", "KI", "CAPE2D"]
TARGET_VARIABLE = "ENTLN"


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


if __name__ == "__main__":
    start_date = "2022-11-23_03_00_00"
    end_date = "2022-11-25_00_00_00"

    train_files, test_files = get_aligned_files(start_date, end_date)

    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    if len(train_files) > 0:
        print(f"\nFirst train file: {train_files[0]}")
        print(f"Last train file:  {train_files[-1]}")
        print(f"\nFirst test file: {test_files[0]}")
        print(f"Last test file:  {test_files[-1]}")
