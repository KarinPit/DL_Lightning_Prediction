import os
import re
import xarray as xr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, Subset, random_split
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


######### Deep learning Classes #########


class DoubleConv(nn.Module):
    """(convolution => Batch normalization => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # if a pixel is missing (creating a size mismatch) - fix by padding
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (Down sampling)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Bottleneck

        # Decoder (Up sampling)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with Skip Connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


######### Deep learning functions #########


def calc_csi(binary_preds, true_y):
    """Critical Success Index (CSI)"""
    true_pos = np.sum((binary_preds == 1) & (true_y == 1))
    false_neg = np.sum((binary_preds == 0) & (true_y == 1))
    false_pos = np.sum((binary_preds == 1) & (true_y == 0))

    denominator = true_pos + false_neg + false_pos
    if denominator == 0:
        return np.nan

    return true_pos / denominator


def calc_far(binary_preds, true_y):
    """False Alarm Ratio (FAR)"""
    true_pos = np.sum((binary_preds == 1) & (true_y == 1))
    false_pos = np.sum((binary_preds == 1) & (true_y == 0))

    denominator = true_pos + false_pos
    if denominator == 0:
        return np.nan

    return false_pos / denominator


def calc_pod(binary_preds, true_y):
    """Probability Of Detection (POD)"""
    true_pos = np.sum((binary_preds == 1) & (true_y == 1))
    false_neg = np.sum((binary_preds == 0) & (true_y == 1))

    denominator = true_pos + false_neg
    if denominator == 0:
        return np.nan

    return true_pos / denominator


def calc_bs(pred_probs, true_y):
    """Brier Score (BS)"""
    return np.mean((pred_probs - true_y) ** 2)


def calc_bss(pred_probs, true_y):
    """Brier Skill Score (BSS) relative to climatology"""
    bs = calc_bs(pred_probs, true_y)

    climatology = np.mean(true_y)
    bs_ref = np.mean((climatology - true_y) ** 2)

    if bs_ref == 0:
        return np.nan

    return 1 - (bs / bs_ref)


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


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs,
    device,
    descision_threshold,
    save_path,
):
    """Train the U-Net model"""
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0

        print(f"Starting Epoch {epoch}...")

        # go through all x and y pairs in the current batch
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # perform backprop on the loss and store the given loss
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)

            # calculate contigency metrics
            probs = torch.sigmoid(logits)
            pred_binary = (probs > descision_threshold).int()

        # model evaluation- calculate loss on the validation set
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss = criterion(logits, yb)

                total_val_loss += v_loss.item() * xb.size(0)

    return train_losses, val_losses


# TODO- implement optimizer function (optional?)

##############################################


if __name__ == "__main__":
    # case configuration
    case = CASES[5]
    atm_params = ["KI", "CAPE2D", "LPI", "PREC_RATE"]
    space_res = "24by24"
    time_res = "1_hours"

    # paths configuration
    wrf_path = f"{MAIN_PATH}/{case}/Ens/Raw/"
    entln_path = f"{MAIN_PATH}/{case}/ENTLN/{space_res}/{time_res}"
    tensor_path = f"{MAIN_PATH}/{case}/Ens/Tensors"

    # DL model configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=4, n_classes=1).to(device)
    pos_weight = torch.tensor([5000.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    batch_size = 4

    # check if case tensors exist
    tensors_exist = False
    if os.path.exists(tensor_path):
        files = os.listdir(tensor_path)
        if "X_final.pt" in files and "Y_final.pt" in files:
            tensors_exist = True

    if tensors_exist:
        print("X and y tensors available. Skipping tensor creation and saving!")

        X = torch.load(os.path.join(tensor_path, "X_final.pt"))
        y = torch.load(os.path.join(tensor_path, "Y_final.pt"))

        # TODO- implement training and evaluation loop
        # TODO- save the best model weights
        # TODO- codex for VSCode?

        # create tensor dataset and split to training and validation sets
        train_size = int(0.8 * len(X))
        val_size = len(X) - train_size

        # split indices first
        train_dataset, val_dataset = random_split(range(len(X)), [train_size, val_size])
        train_idx = train_dataset.indices

        # compute stats
        train_X = X[train_idx]
        means, stds = mean_std_norm(train_X)

        # normalize
        for c in range(X.shape[1]):
            X[:, c, :, :] = (X[:, c, :, :] - means[c]) / (stds[c] + 1e-8)

        # NOW create datasets
        full_dataset = TensorDataset(X, y)
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_dataset.indices)

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
                        file_path = os.path.join(
                            wrf_path, param, "proccesed", space_res, time_res, file_name
                        )
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

            torch.save(final_x_tensor, os.path.join(tensor_path, "X_final.pt"))
            torch.save(final_y_tensor, os.path.join(tensor_path, "Y_final.pt"))
        else:
            print(
                "No samples were created. Check if timestamps match between Ens and ENTLN."
            )
