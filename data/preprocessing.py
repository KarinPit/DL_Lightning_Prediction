import os
import re
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
from config.constants import MAIN_PATH

NORMALIZATION_EPS = 1e-6


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


def get_param_source_names(param_name, with_subparams):
    """Return the folder/file source names for a configured parameter."""
    return with_subparams.get(param_name, [param_name.lower()])


def mean_std_norm(train_X):
    """Calculate robust mean/std values that will not produce NaNs in normalization."""
    means = []
    stds = []

    for c in range(train_X.shape[1]):
        channel_tensor = torch.nan_to_num(
            train_X[:, c, :, :].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mean = channel_tensor.mean()
        std = channel_tensor.std()

        if not torch.isfinite(mean):
            mean = torch.tensor(
                0.0, dtype=channel_tensor.dtype, device=channel_tensor.device
            )
        if not torch.isfinite(std) or std.abs().item() < NORMALIZATION_EPS:
            std = torch.tensor(
                NORMALIZATION_EPS,
                dtype=channel_tensor.dtype,
                device=channel_tensor.device,
            )
        elif mean.abs().item() < NORMALIZATION_EPS:
            mean = torch.tensor(
                0.0, dtype=channel_tensor.dtype, device=channel_tensor.device
            )

        means.append(mean)
        stds.append(std)
    return means, stds


def build_and_save_tensors(
    wrf_path,
    entln_path,
    tensor_path,
    atm_params,
    space_res,
    time_res,
    case_config,
):
    """Build input/target tensors from raw files and save them to disk."""
    ens_list = [f"{i:02d}" for i in range(11)]
    all_x_samples = []
    all_y_samples = []
    sample_groups = []
    expected_channels = case_config.expected_input_channels

    for case in case_config.train_case_names:
        case_wrf_path = os.path.join(MAIN_PATH,'Processed_Data', case)
        case_entln_path = os.path.join(MAIN_PATH,'Processed_Data', case, "ENTLN", space_res, time_res)

        # Loop over the files in the ENTLN folder and extract all existing timestamps
        entln_map = {}
        if os.path.exists(case_entln_path):
            for file in os.listdir(case_entln_path):
                if file.endswith(".nc"):
                    ts = extract_timestamp(file, is_entln=True)
                    if ts:
                        entln_map[ts] = file

        # Loop over the files in each atmospheric parameter folder and extract timestamps
        all_param_times = []
        param_maps = {}

        for param in atm_params:
            param_maps[param] = {}
            source_names = get_param_source_names(param, case_config.with_subparams)
            current_param_keys = None

            for source_name in source_names:
                data_folder = os.path.join(case_wrf_path, param, space_res, time_res)
                if not os.path.exists(data_folder):
                    raise FileNotFoundError(
                        f"Missing data folder for parameter '{param}' in case "
                        f"'{case}' source '{source_name}': {data_folder}"
                    )

                source_file_map = {}

                for file in os.listdir(data_folder):
                    if file.endswith(".nc"):
                        ts = extract_timestamp(file)
                        if ts:
                            ens_id = file.split("_", 1)[0]
                            ens_id = f"{int(ens_id):02d}" if ens_id.isdigit() else "00"
                            combined_key = f"{ens_id}_{ts}"
                            source_file_map[combined_key] = os.path.join(
                                data_folder, file
                            )

                source_keys = set(source_file_map.keys())
                current_param_keys = (
                    source_keys
                    if current_param_keys is None
                    else current_param_keys.intersection(source_keys)
                )

                for combined_key in source_keys:
                    param_maps[param].setdefault(combined_key, {})[source_name] = (
                        source_file_map[combined_key]
                    )

            current_param_times = {
                key.split("_", 1)[1] for key in current_param_keys or set()
            }
            all_param_times.append(current_param_times)

        all_keys_sets = all_param_times + [set(entln_map.keys())]
        common_timestamps = set.intersection(*all_keys_sets)
        print(f"Found {len(common_timestamps)} common timestamps for case '{case}'.\n")

        for ts in tqdm(sorted(common_timestamps), desc=case):
            for ens_id in ens_list:
                combined_key = f"{ens_id}_{ts}"
                if all(combined_key in param_maps[p] for p in atm_params):
                    current_tensors = []
                    for param in atm_params:
                        source_names = get_param_source_names(
                            param, case_config.with_subparams
                        )
                        source_paths = param_maps[param][combined_key]

                        if param in case_config.with_subparams:
                            for source_name in source_names:
                                tensor = load_nc_layer(
                                    source_paths[source_name], source_name
                                )
                                if tensor is None:
                                    current_tensors = []
                                    break
                                current_tensors.append(tensor)
                        else:
                            tensor = load_nc_layer(source_paths[source_names[0]], param)
                            if tensor is None:
                                current_tensors = []
                                break
                            current_tensors.append(tensor)

                        if not current_tensors:
                            break

                    if len(current_tensors) == expected_channels:
                        x_tensor = torch.stack(current_tensors, dim=0)

                        y_file_name = entln_map[ts]
                        y_file_path = os.path.join(case_entln_path, y_file_name)
                        y_tensor = load_nc_layer(y_file_path, "ildn")

                        if y_tensor is not None:
                            all_x_samples.append(x_tensor)
                            all_y_samples.append(y_tensor.unsqueeze(0))
                            sample_groups.append(f"{case}__{ts}")

    print(f"Total WRF samples: {len(all_x_samples)}")

    if len(all_x_samples) == 0:
        print(
            "No samples were created. Check if timestamps match between Ens and ENTLN."
        )
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


def build_and_save_tensors_era5(tensor_path, atm_params, case_config):
    """
    Build input/target tensors from ERA5 processed NC files.

    Each hourly NC file contains all ERA5 variables + entln_count on the same grid.
    No ensemble members — ERA5 is a single reanalysis.

    When case_config.lookback_hours > 1, consecutive hours T-2, T-1, T are stacked
    as input channels so the model sees temporal evolution.  Any gap in the hourly
    sequence causes the triplet to be skipped.

    Input channels:  n_vars * lookback_hours  (e.g. 9 vars × 3 hours = 27)
    Output channels: 1  (binary lightning at hour T)

    Expected file path:
        thesis-bucket/Processed_Data/ERA5/{case}/1_hours/{timestamp}.nc
    """
    all_x_samples = []
    all_y_samples = []
    sample_groups = []

    lookback = getattr(case_config, 'lookback_hours', 1)
    n_vars   = len(atm_params)
    era5_base = os.path.join(MAIN_PATH, 'thesis-bucket', 'Processed_Data', 'ERA5')

    for case in case_config.train_case_names:
        case_era5_path = os.path.join(era5_base, case, case_config.time_res)

        if not os.path.exists(case_era5_path):
            print(f"WARNING: ERA5 folder not found for {case}: {case_era5_path}")
            continue

        # ── Sort files and parse timestamps ───────────────────────────────────
        raw_files = sorted([f for f in os.listdir(case_era5_path) if f.endswith('.nc')])

        valid_files = []
        timestamps  = []
        for fname in raw_files:
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})', fname)
            if ts_match:
                dt = datetime.strptime(ts_match.group(1), '%Y-%m-%d_%H_%M_%S')
                valid_files.append(fname)
                timestamps.append(dt)

        print(f"Case {case}: {len(valid_files)} hourly ERA5 files  |  lookback={lookback}h")

        skipped_gap = 0

        for i in tqdm(range(lookback - 1, len(valid_files)), desc=case):
            # ── Check all lookback hours are consecutive ───────────────────
            window_times = timestamps[i - lookback + 1 : i + 1]
            consecutive = all(
                window_times[j + 1] - window_times[j] == timedelta(hours=1)
                for j in range(len(window_times) - 1)
            )
            if not consecutive:
                skipped_gap += 1
                continue

            # ── Load variables for each hour in the window ─────────────────
            # Order: T-(lookback-1), ..., T-1, T  (chronological)
            all_hour_tensors = []
            valid = True
            for t_idx in range(i - lookback + 1, i + 1):
                fpath = os.path.join(case_era5_path, valid_files[t_idx])
                for param in atm_params:
                    tensor = load_nc_layer(fpath, param)
                    if tensor is None:
                        valid = False
                        break
                    all_hour_tensors.append(tensor)
                if not valid:
                    break

            expected_channels = lookback * n_vars
            if not valid or len(all_hour_tensors) != expected_channels:
                continue

            # ── Load ENTLN target at hour T (last in window) ───────────────
            fpath_t = os.path.join(case_era5_path, valid_files[i])
            y_tensor = load_nc_layer(fpath_t, 'entln_count')
            if y_tensor is None:
                continue

            x_tensor = torch.stack(all_hour_tensors, dim=0)  # [lookback*n_vars, H, W]
            y_binary  = (y_tensor > 0).float()

            all_x_samples.append(x_tensor)
            all_y_samples.append(y_binary.unsqueeze(0))
            sample_groups.append(
                f"{case}__{timestamps[i].strftime('%Y-%m-%d_%H_%M_%S')}"
            )

        if skipped_gap:
            print(f"  Skipped {skipped_gap} samples due to gaps in the hourly sequence.")

    print(f"Total ERA5 samples: {len(all_x_samples)}")

    if len(all_x_samples) == 0:
        print("No samples created. Check ERA5 processed files exist and variable names match.")
        return False

    final_x = torch.stack(all_x_samples, dim=0)
    final_y = torch.stack(all_y_samples, dim=0)

    print(f"Final X shape: {final_x.shape}  "
          f"({lookback} hours × {n_vars} vars = {lookback * n_vars} channels)")
    print(f"Final Y shape: {final_y.shape}")

    os.makedirs(tensor_path, exist_ok=True)
    torch.save(final_x, os.path.join(tensor_path, "X_final.pt"))
    torch.save(final_y, os.path.join(tensor_path, "Y_final.pt"))
    torch.save(sample_groups, os.path.join(tensor_path, "sample_groups.pt"))
    return True
