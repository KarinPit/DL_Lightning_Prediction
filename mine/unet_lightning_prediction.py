# -*- coding: utf-8 -*-

import re
from tqdm import tqdm
import os
from datetime import datetime
import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.metrics import MeanSquaredError, AUC
from tensorflow.keras.regularizers import L2
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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


def get_optimizer(s, learning_rate=0.001, **kwargs):
    if s == "Adam":
        o = optimizers.Adam(learning_rate=learning_rate)
    elif s == "SGD":
        # learning_rate = 0.001 # from sobash
        momentum = 0.99
        nesterov = True
        o = optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov, **kwargs
        )
    return o


def baseline_model(
    input_dim=None,
    name=None,
    numclasses=None,
    neurons=[16, 16],
    kernel_regularizer=None,
    optimizer_name="Adam",
    dropout=0,
    batch_normalize=False,
    learning_rate=0.01,
):

    model = tf.keras.models.Sequential(name=name)
    model.add(tf.keras.Input(shape=input_dim))
    for n in neurons:
        model.add(
            tf.keras.layers.Dense(
                n, activation="relu", kernel_regularizer=kernel_regularizer
            )
        )
        model.add(Dropout(rate=dropout))
        if batch_normalize:
            model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(numclasses, activation="sigmoid"))

    # Compile model with optimizer and loss function. MSE is same as brier_score.
    loss = "binary_crossentropy"  # in HWT_mode, I used categorical_crossentropy
    optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        run_eagerly=None,
        metrics=[MeanSquaredError(), AUC(), "accuracy"],
    )

    return model


if __name__ == "__main__":
    # available cases and atm params
    available_cases = [
        "Case1_Nov_2022_23_25",
        "Case2_Jan_2023_11_16",
        "Case3_Mar_2023_13_15",
        "Case4_Apr_2023_09_13",
        "Case5_Jan_2024_26_31",
        "Case6_Nov_2025_24_25",
    ]
    case_name = available_cases[5]
    atm_params = ["KI", "CAPE2D", "LPI", "PREC_RATE", "FLUX_UP", "WMAX_LAYER"]
    resolution = "24by24"
    time_interv = "1_hours"

    # data and output paths
    main_path = "/Users/karinpitlik/Desktop/DataScience/Thesis"
    main_folder = f"{main_path}/{case_name}/Ens/Raw/"
    entln_folder = f"{main_path}/{case_name}/ENTLN/{resolution}/{time_interv}"
    tensor_save_path = f"{main_path}/{case_name}/Ens/Tensors/{resolution}/{time_interv}"
    best_model_path = os.path.join(tensor_save_path, "best_lightning_model.pth")

    # check if case tensors exist
    tensors_exist = False
    if os.path.exists(tensor_save_path):
        files = os.listdir(tensor_save_path)
        if "X_final_2880.pt" in files and "Y_final_2880.pt" in files:
            tensors_exist = True

    if tensors_exist:
        ########### MODEL TRAINING CONFIGURATION ###########

        # # TODO CHANGE THE HYPERPARAMETERS

        batchnorm = False
        batchsize = 1024
        clobber = False
        debug = False
        dropout = 0.0
        epochs = 30
        labels = []
        fhr = list(range(1, 49))
        fits = None
        flash = 10
        folds = None
        kfold = 5
        idate = None
        ifile = None
        learning_rate = 0.001
        model = "HRRR"
        neurons = [16, 16]
        nfits = 10
        nprocs = 0
        optimizer_name = "Adam"
        reg_penalty = 0.01
        savedmodel = None
        seed = None

        #####################################################

        # DATA PREPROCESSING FOR MODEL
        print("X and y tensors available. Skipping tensor creation and saving!")

        X = torch.load(os.path.join(tensor_save_path, "X_final_2880.pt"))
        y = torch.load(os.path.join(tensor_save_path, "Y_final_2880.pt"))

        # convert torch tensors to numpy
        sample_indices = np.arange(X.shape[0])
        train_idx, val_idx = train_test_split(
            sample_indices, test_size=0.2, random_state=42
        )

        X_train_maps = X[train_idx]
        X_val_maps = X[val_idx]
        y_train_maps = y[train_idx]
        y_val_maps = y[val_idx]

        N, C, H, W = X.shape
        Ntr, C, H, W = X_train_maps.shape
        X_train = X_train_maps.permute(0, 2, 3, 1).reshape(-1, C)
        X_val = X_val_maps.permute(0, 2, 3, 1).reshape(-1, C)

        y_train = y_train_maps.permute(0, 2, 3, 1).reshape(-1, 1)
        y_val = y_val_maps.permute(0, 2, 3, 1).reshape(-1, 1)

        y_train = (y_train > 0).float()
        y_val = (y_val > 0).float()

        # TODO DO A MEAN AND STDEV NORMALIZATION
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0)

        eps = 1e-8
        std = np.where(std < eps, 1.0, std)

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        # convert to numpy for keras
        X_train = X_train.cpu().numpy().astype("float32")
        X_val = X_val.cpu().numpy().astype("float32")
        y_train = y_train.cpu().numpy().astype("float32")
        y_val = y_val.cpu().numpy().astype("float32")

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0)

        #####################################################

        # start training loop

        model = baseline_model(
            input_dim=(X_train.shape[1],),
            numclasses=1,
            neurons=neurons,
            name=f"fit_model",
            kernel_regularizer=L2(l2=reg_penalty),
            optimizer_name=optimizer_name,
            dropout=dropout,
            learning_rate=learning_rate,
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            class_weight=None,
            sample_weight=None,
            batch_size=batchsize,
            epochs=epochs,
            verbose=2,
        )

        # ####################################################

        # VISUALIZATION

        coords_nc = xr.open_dataset(
            "/Users/karinpitlik/Desktop/DataScience/Thesis/Case6_Nov_2025_24_25/Ens/Raw/CAPE2D/proccesed/24by24/1_hours/00_2025-11-24_00_00_00_2025-11-24_01_00_00.nc"
        )
        xlat = coords_nc["xlat"][:]
        xlong = coords_nc["xlong"][:]

        for idx in range(10):
            x_map = X_val_maps[idx]
            y_true_map = y_val_maps[idx][0]
            C, H, W = x_map.shape

            x_tab = (
                x_map.permute(1, 2, 0).reshape(-1, C).cpu().numpy().astype("float32")
            )

            # normalize using the training stats
            x_tab = (x_tab - mean) / std
            x_tab = np.nan_to_num(x_tab, nan=0.0)

            y_pred = model.predict(x_tab, verbose=0)
            y_pred = y_pred.reshape(H, W)

            y_pred[y_pred < 0.005] = np.nan
            y_true_map[y_true_map == 0] = np.nan

            fig, axes = plt.subplots(
                1, 2, figsize=(18, 5), subplot_kw={"projection": ccrs.PlateCarree()}
            )

            ax0, ax1 = axes

            mesh0 = ax0.pcolormesh(
                xlong,
                xlat,
                y_pred * 100,
                cmap="YlOrRd",
                shading="auto",
                transform=ccrs.PlateCarree(),
            )

            mesh1 = ax1.pcolormesh(
                xlong,
                xlat,
                y_true_map.cpu().numpy(),
                cmap="YlOrRd",
                shading="auto",
                transform=ccrs.PlateCarree(),
            )

            for ax in [ax0, ax1]:
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS)

            ax0.set_title("Model Prediction")
            ax1.set_title("Ground Truth")

            plt.colorbar(mesh0, ax=ax0, label="Lightning probability (%)")
            plt.colorbar(mesh1, ax=ax1, label="# real lightning")

            fig.suptitle("Lightning Prediction vs Ground Truth")
            plt.tight_layout()
            plt.show()

    else:
        # This will store all the available timestamps for later intersection
        all_keys_sets = []

        # Loop over the files in the ENTLN folder and extract all existing timestamps
        entln_map = {}
        if os.path.exists(entln_folder):
            for file in os.listdir(entln_folder):
                if file.endswith(".nc"):
                    ts = extract_timestamp(file, is_entln=True)
                    if ts:
                        entln_map[ts] = file

        # Loop over the files in each atm parameter's folder and extract all existing timestamps
        param_maps = {}
        all_param_times = []

        for param in atm_params:
            data_folder = os.path.join(
                main_folder, param, "proccesed", resolution, time_interv
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
        valid_pairs = []

        # This loop goes over all available timestamps, convert the .nc file to a tensor, and append the tensor to the x or y list
        for ts in tqdm(sorted(common_timestamps)):
            for ens_id in ens_list:
                combined_key = f"{ens_id}_{ts}"
                if all(combined_key in param_maps[p] for p in atm_params):
                    valid_pairs.append(combined_key)

                # Check if the ens id is available in all atm parameters
                if all(combined_key in param_maps[p] for p in atm_params):
                    current_tensors = []
                    for param in atm_params:
                        file_name = param_maps[param][combined_key]
                        file_path = os.path.join(
                            main_folder,
                            param,
                            "proccesed",
                            resolution,
                            time_interv,
                            file_name,
                        )
                        t = load_nc_layer(file_path, param)
                        if t is not None:
                            current_tensors.append(t)

                    if len(current_tensors) == len(atm_params):
                        x_tensor = torch.stack(
                            current_tensors, dim=0
                        )  # Create a [4, 249, 249] tensor

                        y_file_name = entln_map[ts]
                        y_file_path = os.path.join(entln_folder, y_file_name)
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
            if not os.path.exists(tensor_save_path):
                os.makedirs(tensor_save_path)

            torch.save(
                final_x_tensor, os.path.join(tensor_save_path, "X_final_2880.pt")
            )
            torch.save(
                final_y_tensor, os.path.join(tensor_save_path, "Y_final_2880.pt")
            )
        else:
            print(
                "No samples were created. Check if timestamps match between Ens and ENTLN."
            )

        print(f"Total valid (ensemble, timestamp) pairs: {len(valid_pairs)}")
        print(valid_pairs[:10])
