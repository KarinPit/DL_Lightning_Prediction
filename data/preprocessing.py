import os
import re
from datetime import datetime

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm


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


def mean_std_norm(train_X):
    """calculate mean and standard deviation of a given train dataset"""
    means = []
    stds = []

    for c in range(train_X.shape[1]):
        mean = train_X[:, c, :, :].mean()
        std = train_X[:, c, :, :].std()
        means.append(mean)
        stds.append(std)
    return means, stds


def build_and_save_tensors(
    wrf_path, entln_path, tensor_path, atm_params, space_res, time_res
):
    """Build input/target tensors from raw files and save them to disk."""
    all_keys_sets = []

    # Loop over the files in the ENTLN folder and extract all existing timestamps
    entln_map = {}
    if os.path.exists(entln_path):
        for file in os.listdir(entln_path):
            if file.endswith(".nc"):
                ts = extract_timestamp(file, is_entln=True)
                if ts:
                    entln_map[ts] = file

    # Loop over the files in each atmospheric parameter folder and extract timestamps
    all_param_times = []
    param_maps = {}

    for param in atm_params:
        data_folder = os.path.join(wrf_path, param, "proccesed", space_res, time_res)
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Missing data folder for parameter '{param}': {data_folder}")

        param_maps[param] = {}
        current_param_times = set()

        for file in os.listdir(data_folder):
            if file.endswith(".nc"):
                ts = extract_timestamp(file)
                if ts:
                    ens_id = file.split("_", 1)[0]
                    ens_id = f"{int(ens_id):02d}" if ens_id.isdigit() else "00"

                    # Keep ensemble id in the key so different members do not overwrite.
                    combined_key = f"{ens_id}_{ts}"
                    param_maps[param][combined_key] = file
                    current_param_times.add(ts)

        all_param_times.append(current_param_times)

    # Intersection of the timestamps in the ENTLN set and the atmospheric parameter set
    all_keys_sets = all_param_times + [set(entln_map.keys())]
    common_timestamps = set.intersection(*all_keys_sets)
    print(f"Found {len(common_timestamps)} common timestamps.")

    ens_list = [f"{i:02d}" for i in range(11)]
    all_x_samples = []
    all_y_samples = []
    sample_groups = []

    for ts in tqdm(sorted(common_timestamps)):
        for ens_id in ens_list:
            combined_key = f"{ens_id}_{ts}"
            if all(combined_key in param_maps[p] for p in atm_params):
                current_tensors = []
                for param in atm_params:
                    file_name = param_maps[param][combined_key]
                    file_path = os.path.join(
                        wrf_path, param, "proccesed", space_res, time_res, file_name
                    )
                    tensor = load_nc_layer(file_path, param)
                    if tensor is not None:
                        current_tensors.append(tensor)

                if len(current_tensors) == len(atm_params):
                    x_tensor = torch.stack(current_tensors, dim=0)

                    y_file_name = entln_map[ts]
                    y_file_path = os.path.join(entln_path, y_file_name)
                    y_tensor = load_nc_layer(y_file_path, "ildn")

                    if y_tensor is not None:
                        all_x_samples.append(x_tensor)
                        all_y_samples.append(y_tensor.unsqueeze(0))
                        sample_groups.append(ts)

    print(f"Total X samples: {len(all_x_samples)}")

    if len(all_x_samples) == 0:
        print("No samples were created. Check if timestamps match between Ens and ENTLN.")
        return False

    final_x_tensor = torch.stack(all_x_samples, dim=0)
    final_y_tensor = torch.stack(all_y_samples, dim=0)

    print("-" * 30)
    print(f"Final X shape: {final_x_tensor.shape}")
    print(f"Final Y shape: {final_y_tensor.shape}")

    if not os.path.exists(tensor_path):
        os.makedirs(tensor_path)

    torch.save(final_x_tensor, os.path.join(tensor_path, "X_final.pt"))
    torch.save(final_y_tensor, os.path.join(tensor_path, "Y_final.pt"))
    torch.save(sample_groups, os.path.join(tensor_path, "sample_groups.pt"))
    return True
