import os
import re
import netCDF4 as nc
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F


############ Data proccessing functions ############


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
        match = re.search(
            r"(\d{4}-\d{2}-\d{2})_(\d{2})[\/:](\d{2})[\/:](\d{2})", filename
        )
        if match:
            # extracts the date and time if exists in the file name
            return f"{match.group(1)}_{match.group(2)}/{match.group(3)}/00"

    return None


def load_nc_layer(file_path, variable_name):
    """Loads the data from the .nc files and converts them into tensors"""
    ds = nc.Dataset(file_path)
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


############ Deep learning functions ############


def calculate_f1(logits, targets, threshold=0.5):
    """Calculates the f1 scores of the model's map to see how similar it is to the ground truth map"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1.item()


def visualize_prediction(model, dataset, device, idx=0):
    """Creates a 3 subplot plot where the left map is a variable's values,
    and the right is the ground truth map
    """
    model.eval()
    with torch.no_grad():
        x, y_true = dataset[idx]
        x_in = x.unsqueeze(0).to(device)

        logits = model(x_in)
        y_pred = torch.sigmoid(logits).cpu().squeeze().numpy()
        y_true = y_true.squeeze().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        # model's prediction
        im1 = axes[0].imshow(y_pred, cmap="magma")
        axes[0].set_title("Model Prediction")
        plt.colorbar(im1, ax=axes[0])

        # ground truth
        im2 = axes[1].imshow(y_true, cmap="magma")
        axes[1].set_title("Ground Truth (ENTLN)")
        plt.colorbar(im2, ax=axes[1])

        plt.show()


def train_model(
    model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path
):
    """Train the U-Net model"""
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss, total_train_f1 = 0.0, 0.0

        print(f"Starting Epoch {epoch}...")

        # model training- perform backprop and loss calculation
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)
            total_train_f1 += calculate_f1(logits, yb) * xb.size(0)

        # model evaluation- calculate loss on the validation set
        model.eval()
        total_val_loss, total_val_f1 = 0.0, 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss = criterion(logits, yb)

                total_val_loss += v_loss.item() * xb.size(0)
                total_val_f1 += calculate_f1(logits, yb) * xb.size(0)

        # calculate average loss and f1 scores
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_train_f1 = total_train_f1 / len(train_loader.dataset)
        avg_val_f1 = total_val_f1 / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_f1s.append(avg_train_f1)
        val_f1s.append(avg_val_f1)

        print(
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )
        print(f"         | Train F1: {avg_train_f1:.4f} | Val F1: {avg_val_f1:.4f}")

        # save the best model's weights to a file for future use
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(
                model.state_dict(), os.path.join(save_path, "best_lightning_model.pth")
            )
            print(f"(!) Saved new best model based on Validation F1: {best_val_f1:.4f}")

    return train_losses, train_f1s, val_losses, val_f1s


############ Deep Learning Classes ############


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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
    def __init__(self, n_channels=4, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (Down scalling)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Bottleneck

        # Decoder (Up scalling)
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


if __name__ == "__main__":
    # available cases and atm params
    available_cases = ["Case1_Nov_2022_23_25", "Case2_Jan_2023_11_16"]
    case_name = available_cases[1]
    atm_params = ["KI", "CAPE2D", "LPI", "PREC_RATE"]
    resolution = "4by4"

    # data and output paths
    main_folder = f"/Users/karinpitlik/Desktop/DataScience/Thesis/{case_name}/Ens/Raw"
    entln_folder = f"/Users/karinpitlik/Desktop/DataScience/Thesis/ENTLN/ENTLN_pulse_{case_name}/{resolution}/10_minutes/"
    tensor_save_path = (
        f"/Users/karinpitlik/Desktop/DataScience/Thesis/{case_name}/Ens/Tensors"
    )
    best_model_path = os.path.join(tensor_save_path, "best_lightning_model.pth")

    # check if case tensors exist
    tensors_exist = False
    if os.path.exists(tensor_save_path):
        files = os.listdir(tensor_save_path)
        if "X_final_2880.pt" in files and "Y_final_2880.pt" in files:
            tensors_exist = True

    if tensors_exist:
        print("X and y tensors available. Skipping tensor creation and saving!")

        X = torch.load(os.path.join(tensor_save_path, "X_final_2880.pt"))
        y = torch.load(os.path.join(tensor_save_path, "Y_final_2880.pt"))

        # perform a min-max normalization on the X data so it will contain only 0 to 1 values
        for c in range(X.shape[1]):
            channel_min = X[:, c, :, :].min()
            channel_max = X[:, c, :, :].max()
            X[:, c, :, :] = (X[:, c, :, :] - channel_min) / (
                channel_max - channel_min + 1e-6
            )

        # create tensor dataset and split to training and validation sets
        full_dataset = TensorDataset(X, y)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # model configuration
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = UNet(n_channels=4, n_classes=1).to(device)
        # lightning occurences have large weight
        pos_weight = torch.tensor([5000.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        # check if previous model weights exist, if not training will start from scratch
        best_model_exist = False
        if os.path.exists(best_model_path):
            print(f"Found existing model at {best_model_path}. Loading weights...")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Model loaded successfully! You are starting from your best point.")

        else:
            print("No saved model found. Starting training from scratch.")

        # start training loop
        num_epochs = 10
        train_losses, train_f1s, val_losses, val_f1s = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            num_epochs,
            device,
            tensor_save_path,
        )

        # loss graph
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss", color="blue", linestyle="--")
        plt.plot(val_losses, label="Val Loss", color="red")
        plt.title("Loss Convergence")
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.legend()
        plt.grid(True)

        # f1 graph
        plt.subplot(1, 2, 2)
        plt.plot(train_f1s, label="Train F1", color="blue", linestyle="--")
        plt.plot(val_f1s, label="Val F1", color="red")
        plt.title("F1-Score (Accuracy) Comparison")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        print("Visualizing results from both datasets...")

        # visualize random slices from the model and the ground truth
        indices_to_check = [0, 10, 20]

        for i in indices_to_check:
            visualize_prediction(model, train_dataset, device, idx=i)

        for i in indices_to_check:
            visualize_prediction(model, val_dataset, device, idx=i)

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
            data_folder = os.path.join(main_folder, param)
            if not os.path.exists(data_folder):
                continue

            param_maps[param] = {}
            current_param_times = set()

            # go through all the atm param's files and extract the timestamp (ts)
            for file in os.listdir(data_folder):
                if file.endswith(".nc"):
                    ts = extract_timestamp(file)
                    if ts:
                        ens_match = re.search(r"_(\d{1,2})_", file)
                        ens_id = f"{int(ens_match.group(1)):02d}" if ens_match else "00"

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
        for ts in sorted(common_timestamps):
            for ens_id in ens_list:
                combined_key = f"{ens_id}_{ts}"

                # Check if the ens id is available in all atm parameters
                if all(combined_key in param_maps[p] for p in atm_params):
                    current_tensors = []
                    for param in atm_params:
                        file_name = param_maps[param][combined_key]
                        file_path = os.path.join(main_folder, param, file_name)
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
