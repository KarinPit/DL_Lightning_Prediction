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


# Data proccessing functions
def extract_timestamp(filename, is_entln=False):
    """Extracts the date and time from the filename of the raw files"""

    if is_entln:
        match = re.search(
            r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})",
            filename,
        )
        if match:
            # matches format of the entln date and time to the atm parameters date and time format
            ts = match.group(1)
            dt = datetime.strptime(ts, "%Y-%m-%d_%H_%M_%S")
            dt_rounded = dt.replace(second=0, microsecond=0)
            minute = (dt_rounded.minute // 10) * 10
            dt_rounded = dt_rounded.replace(minute=minute)
            return dt_rounded.strftime("%Y-%m-%d_%H/%M/%S")
    else:
        match = re.search(
            r"(\d{4}-\d{2}-\d{2})_(\d{2})[\/:](\d{2})[\/:](\d{2})", filename
        )
        if match:
            return f"{match.group(1)}_{match.group(2)}/{match.group(3)}/00"  # מאפסים שניות ליתר ביטחון

    return None


def load_nc_layer(file_path, variable_name):
    """Loads the data from the .nc files and converts them into tensors"""
    ds = nc.Dataset(file_path)
    target_var = next(
        (var for var in ds.variables if var.lower() == variable_name.lower()), None
    )

    if target_var is None:
        ds.close()
        return None

    data = ds.variables[target_var][:]
    if data.ndim == 3:
        data = data[0]

    # convert the data from the .nc file to tensor
    tensor = torch.from_numpy(np.array(data).astype(np.float32))
    ds.close()
    return tensor


# Deep learning functions


def calculate_f1(logits, targets, threshold=0.1):
    """מחשב כמה המפה שהמודל יצר דומה למפת האמת"""
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
    """מציג השוואה בין הקלט, החיזוי והאמת"""
    model.eval()
    with torch.no_grad():
        x, y_true = dataset[idx]
        x_in = x.unsqueeze(0).to(device)

        logits = model(x_in)
        y_pred = torch.sigmoid(logits).cpu().squeeze().numpy()
        y_true = y_true.squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # מציגים שכבה אחת מהקלט (למשל CAPE - ערוץ 3)
        axes[0].imshow(x[3].numpy(), cmap="jet")
        axes[0].set_title("Input (CAPE)")

        # החיזוי של המודל
        im1 = axes[1].imshow(y_pred, cmap="magma")
        axes[1].set_title("Model Prediction")
        plt.colorbar(im1, ax=axes[1])

        # האמת מהשטח
        im2 = axes[2].imshow(y_true, cmap="magma")
        axes[2].set_title("Ground Truth (ENTLN)")
        plt.colorbar(im2, ax=axes[2])

        plt.show()


def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    train_losses = []
    train_f1s = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_f1 = 0.0, 0.0

        print(f"Starting Epoch {epoch}...")

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_f1 += calculate_f1(logits, yb) * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        avg_f1 = total_f1 / len(train_loader.dataset)

        train_losses.append(avg_loss)
        train_f1s.append(avg_f1)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | F1 (Accuracy): {avg_f1:.4f}")

    return train_losses, train_f1s


def visualize_prediction(model, dataset, device, idx=0):
    """מציג השוואה בין הקלט, מפת ההסתברויות והאמת מהשטח"""
    model.eval()
    with torch.no_grad():
        x, y_true = dataset[idx]
        # מוסיפים מימד Batch ושולחים ל-MPS
        x_in = x.unsqueeze(0).to(device)

        # הרצת המודל
        logits = model(x_in)
        # הפיכה להסתברות (0 עד 1)
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()
        y_true = y_true.squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 1. הצגת קלט (למשל LPI או CAPE)
        # נבחר להציג את ה-LPI (ערוץ 2) כי הוא לרוב הכי קשור לברקים
        im0 = axes[0].imshow(x[2].numpy(), cmap="viridis")
        axes[0].set_title(f"Input Feature (LPI) - Sample {idx}")
        plt.colorbar(im0, ax=axes[0])

        # 2. מפת ההסתברויות (החיזוי של המודל)
        # נשתמש ב-vmin/vmax כדי לוודא שהסקלה היא תמיד 0 עד 1
        im1 = axes[1].imshow(probs, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Predicted Probability (0-1)")
        plt.colorbar(im1, ax=axes[1])

        # 3. האמת מהשטח (ENTLN)
        im2 = axes[2].imshow(y_true, cmap="gray")
        axes[2].set_title("Ground Truth (Actual Lightning)")
        axes[2].imshow(y_true, cmap="gray", vmin=0, vmax=1)
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()


# Deep Learning Classes


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

        # --- התיקון כאן: חישוב הפרשי הגדלים ---
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # הוספת Padding אם חסר פיקסל (משמאל, מימין, מלמעלה, מלמטה)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # עכשיו הגדלים זהים ואפשר לחבר!
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs הם ה-logits מהמודל
        # targets הם ה-labels (0 או 1)

        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt זו ההסתברות שהמודל צדק

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (הצד היורד)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Bottleneck

        # Decoder (הצד העולה)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # שכבת פלט - מוציאה ערוץ 1 (ENTLN)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path עם Skip Connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    case_name = "Case1_Nov_2022_23_25"
    atm_params = ["KI", "CAPE2D", "LPI", "PREC_RATE"]
    resolution = "4by4"

    main_folder = f"/Users/karinpitlik/Desktop/DataScience/Thesis/{case_name}/Ens/Raw"
    entln_folder = f"/Users/karinpitlik/Desktop/DataScience/Thesis/ENTLN/ENTLN_pulse_{case_name}/{resolution}/10_minutes/"
    tensor_save_path = (
        f"/Users/karinpitlik/Desktop/DataScience/Thesis/{case_name}/Ens/Tensors"
    )

    tensors_exist = False
    if os.path.exists(tensor_save_path):
        files = os.listdir(tensor_save_path)
        if "X_final_2880.pt" in files and "Y_final_2880.pt" in files:
            tensors_exist = True

    if tensors_exist:
        print("X and y tensors available. Skipping tensor creation and saving!")

        X = torch.load(os.path.join(tensor_save_path, "X_final_2880.pt"))
        y = torch.load(os.path.join(tensor_save_path, "Y_final_2880.pt"))

        # נרמול גלובלי לכל ערוץ (4 ערוצים)
        for c in range(X.shape[1]):
            channel_min = X[:, c, :, :].min()
            channel_max = X[:, c, :, :].max()
            X[:, c, :, :] = (X[:, c, :, :] - channel_min) / (
                channel_max - channel_min + 1e-8
            )

        print("Normalization complete. All values are between 0 and 1.")

        full_dataset = TensorDataset(X, y)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        # train_dataset, val_dataset = random_split(
        #     full_dataset, [100, len(full_dataset) - 100]
        # )  # Run model on smaller portion of the data to make sure it runs correctly

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        model = UNet(n_channels=4, n_classes=1).to(device)

        # נניח שרק 0.1% מהפיקסלים הם ברקים, ניתן להם משקל של פי 100
        pos_weight = torch.tensor([5000.0]).to(device)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = nn.BCEWithLogitsLoss()

        # alpha=0.95 אומר שאנחנו נותנים משקל גבוה מאוד למחלקה החיובית (ברקים)
        # gamma=2 אומר שאנחנו "משתיקים" חזק פיקסלים שהמודל כבר בטוח בהם
        criterion = FocalLoss(alpha=0.95, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 20
        train_losses, train_f1s = train_model(
            model, train_loader, optimizer, criterion, num_epochs, device
        )

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(train_f1s)
        plt.title("F1-Score (Similarity)")
        plt.show()

        # נבחר דגימה אקראית מתוך ה-Validation set
        # עדיף לבדוק כמה אינדקסים (למשל 0, 10, 50) כדי לראות מקרים שונים
        print("Visualizing results...")
        for i in [0, 10, 20]:
            visualize_prediction(model, val_dataset, device, idx=i)

    else:
        all_keys_sets = (
            []
        )  # This will store all the available timestamps for later intersection

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

            for file in os.listdir(data_folder):
                if file.endswith(".nc"):
                    ts = extract_timestamp(file)
                    if ts:
                        ens_match = re.search(r"_(\d{1,2})_", file)
                        ens_id = f"{int(ens_match.group(1)):02d}" if ens_match else "00"

                        # Each timestamp has a number of ensemble ids! so the ensemble id is also part of the key (to avoid ts overide previous ens ts)
                        combined_key = f"{ens_id}_{ts}"
                        param_maps[param][combined_key] = file
                        current_param_times.add(ts)

            all_param_times.append(current_param_times)

        # Intersection of the ts in the ENTLN ts set and the atm parameter ts set
        all_keys_sets = all_param_times + [set(entln_map.keys())]
        common_timestamps = set.intersection(*all_keys_sets)
        print(f"Found {len(common_timestamps)} common timestamps.")

        # List of available ens ids
        ens_list = [f"{i:02d}" for i in range(11)]

        all_x_samples = []
        all_y_samples = []

        # This loop goes over all available timestamps and converts the .nc files to tensors
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

        # Unite the tensors in the lists to: X_tensor shape- [2880, 4, 249, 249] , y_tensor shape- [2880, 1, 249, 249]
        if len(all_x_samples) > 0:
            final_x_tensor = torch.stack(all_x_samples, dim=0)
            final_y_tensor = torch.stack(all_y_samples, dim=0)

            print("-" * 30)
            print(f"Final X shape: {final_x_tensor.shape}")
            print(f"Final Y shape: {final_y_tensor.shape}")
            print(
                f"Total memory: {final_x_tensor.element_size() * final_x_tensor.nelement() / 1e6:.2f} MB"
            )

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
