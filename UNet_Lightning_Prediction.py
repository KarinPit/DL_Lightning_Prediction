import os
import re
import xarray as xr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm

MAIN_PATH = "/Users/karinpitlik/Desktop/DataScience/Thesis"
CASES = [
    "Case1_Nov_2022_23_25",
    "Case2_Jan_2023_11_16",
    "Case3_Mar_2023_13_15",
    "Case4_Apr_2023_09_13",
    "Case5_Jan_2024_26_31",
    "Case6_Nov_2025_24_25",
]


######### Data proccessing functions #########


def extract_timestamp(filename, is_entln=False):
    """Extracts the date and time from the filename of the raw files"""

    if is_entln:
        match = re.search(
            r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})",
            filename,
        )
        if match:
            # extracts the date and time if exists in the file name
            ts = match.group(1)
            dt = datetime.strptime(ts, "%Y-%m-%d_%H_%M_%S")
            # make sure the time is rounded
            dt_rounded = dt.replace(second=0, microsecond=0)
            minute = (dt_rounded.minute // 10) * 10
            dt_rounded = dt_rounded.replace(minute=minute)
            return dt_rounded.strftime("%Y-%m-%d_%H/%M/%S")
    else:
        # match = re.search(
        #     r"(\d{4}-\d{2}-\d{2})_(\d{2})[\/:](\d{2})[\/:](\d{2})", filename
        # )
        match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})", filename)
        if match:
            # extracts the date and time if exists in the file name
            return f"{match.group(1)}_{match.group(2)}/{match.group(3)}/00"

    return None


def load_nc_layer(file_path, variable_name):
    """Loads the data from the .nc files and converts them into tensors"""
    ds = xr.open_dataset(file_path)
    target_var = next(
        (var for var in ds.variables if var.lower() == variable_name.lower()), None
    )

    # if variable name is not found, the ds is closed
    if target_var is None:
        ds.close()
        return None

    # make sure the data is 3 dimensional (variable, lat, long) and extract the vars's data
    data = ds.variables[target_var][:]
    if data.ndim == 3:
        data = data[0]

    # convert the data from the .nc file to tensor
    tensor = torch.from_numpy(np.array(data).astype(np.float32))
    ds.close()
    return tensor



if __name__ == "__main__":
    # case configuration
    case = CASES[5]
    atm_params = ["KI", "CAPE2D", "LPI", "PREC_RATE"]
    space_res = "24by24"
    time_res = "1_hours"

    # path configuration
    wrf_path = f"{MAIN_PATH}/{case}/Ens/Raw/"
    entln_path = f"{MAIN_PATH}/{case}/ENTLN/{space_res}/{time_res}"
    tensor_path = f"{MAIN_PATH}/{case}/Ens/Tensors"

    # check if case tensors exist
    tensors_exist = False
    if os.path.exists(tensor_path):
        files = os.listdir(tensor_path)
        if "X_final_2880.pt" in files and "Y_final_2880.pt" in files:
            tensors_exist = True

    if tensors_exist:
        pass
    else:
        # This will store all the available timestamps for later intersection
        all_keys_sets = []

        # Loop over the files in the ENTLN folder and extract all existing timestamps
        entln_map = {}
        if os.path.exists(entln_path):
            for file in os.listdir(entln_path):
                if file.endswith(".nc"):
                    ts = extract_timestamp(file, is_entln=True)
                    if ts:
                        entln_map[ts] = file

        # Loop over the files in each atm parameter's folder and extract all existing timestamps
        all_param_times = []
        param_maps = {}

        for param in atm_params:
            data_folder = os.path.join(
                wrf_path, param, "proccesed", space_res, time_res
            )
            if not os.path.exists(data_folder):
                continue

            param_maps[param] = {}
            current_param_times = set()

            # go through all the atm param's files and extract the timestamp (ts)
            for file in os.listdir(data_folder):
                if file.endswith(".nc"):
                    ts = extract_timestamp(file)
                    if ts:
                        ens_id = file.split("_", 1)[0]
                        ens_id = f"{int(ens_id):02d}" if ens_id.isdigit() else "00"

                        # Each timestamp has a few ensemble ids! so the ensemble id is also part of the key (to avoid ts over writing previous tss)
                        combined_key = f"{ens_id}_{ts}"
                        param_maps[param][combined_key] = file
                        current_param_times.add(ts)

            all_param_times.append(current_param_times)

        # Intersection of the ts in the ENTLN set and the ts of the atm parameter set
        all_keys_sets = all_param_times + [set(entln_map.keys())]
        common_timestamps = set.intersection(*all_keys_sets)
        print(f"Found {len(common_timestamps)} common timestamps.")

        # List of available ensemble ids
        ens_list = [f"{i:02d}" for i in range(11)]

        all_x_samples = []
        all_y_samples = []

        # This loop goes over all available timestamps, convert the .nc file to a tensor, and append the tensor to the x or y list
        for ts in tqdm(sorted(common_timestamps)):
            for ens_id in ens_list:
                combined_key = f"{ens_id}_{ts}"
                # Check if the ens id is available in all atm parameters
                if all(combined_key in param_maps[p] for p in atm_params):
                    current_tensors = []
                    for param in atm_params:
                        file_name = param_maps[param][combined_key]
                        file_path = os.path.join(wrf_path, param, "proccesed", space_res, time_res, file_name)
                        t = load_nc_layer(file_path, param)
                        if t is not None:
                            current_tensors.append(t)

                    if len(current_tensors) == len(atm_params):
                        x_tensor = torch.stack(
                            current_tensors, dim=0
                        )  # Create a [4, 249, 249] tensor

                        y_file_name = entln_map[ts]
                        y_file_path = os.path.join(entln_path, y_file_name)
                        y_tensor = load_nc_layer(
                            y_file_path, "ildn"
                        )  # Create a [1, 249, 249] tensor

                        # All tensors are added to the lists in order to unite them to two tensors (one for x and one for y)
                        if y_tensor is not None:
                            all_x_samples.append(x_tensor)
                            all_y_samples.append(y_tensor.unsqueeze(0))

        print(f"Total X samples: {len(all_x_samples)}")

        # Unite the tensors in the lists to one tensor: X_tensor shape- [2880, 4, 249, 249] , y_tensor shape- [2880, 1, 249, 249]
        if len(all_x_samples) > 0:
            final_x_tensor = torch.stack(all_x_samples, dim=0)
            final_y_tensor = torch.stack(all_y_samples, dim=0)

            print("-" * 30)
            print(f"Final X shape: {final_x_tensor.shape}")
            print(f"Final Y shape: {final_y_tensor.shape}")

            # Save tensors to disk to avoid memory usage
            if not os.path.exists(tensor_path):
                os.makedirs(tensor_path)

            torch.save(final_x_tensor, os.path.join(tensor_path, "X_final_2880.pt"))
            torch.save(final_y_tensor, os.path.join(tensor_path, "Y_final_2880.pt"))
        else:
            print(
                "No samples were created. Check if timestamps match between Ens and ENTLN."
            )
