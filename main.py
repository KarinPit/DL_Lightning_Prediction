import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset, DataLoader, random_split

from data.preprocessing import build_and_save_tensors, mean_std_norm
from models.unet import UNet
from training.train import train_model

MAIN_PATH = "/Users/karinpitlik/Desktop/DataScience/Thesis"
CASES = [
    "Case1_Nov_2022_23_25",
    "Case2_Jan_2023_11_16",
    "Case3_Mar_2023_13_15",
    "Case4_Apr_2023_09_13",
    "Case5_Jan_2024_26_31",
    "Case6_Nov_2025_24_25",
]


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
    batch_size = 50
    decision_threshold = 0.9

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

        # get indices of training and validation datasets after splitting
        train_size = int(0.8 * len(X))
        val_size = len(X) - train_size

        # split dataset into train and val and get indices of each
        index_train_subset, index_val_subset = random_split(
            range(len(X)), [train_size, val_size]
        )
        train_idx = index_train_subset.indices
        val_idx = index_val_subset.indices

        # compute mean and std of each feature
        train_X = X[train_idx]
        means, stds = mean_std_norm(train_X)

        # normalize by mean and std
        for c in range(X.shape[1]):
            X[:, c, :, :] = (X[:, c, :, :] - means[c]) / (stds[c] + 1e-8)

        full_dataset = TensorDataset(X, y)
        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
        )

        # run train and evaluation
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=num_epochs,
            device=device,
            decision_threshold=decision_threshold,  # change threshold
        )

        # TODO- save the best model weights

    else:
        build_and_save_tensors(
            wrf_path=wrf_path,
            entln_path=entln_path,
            tensor_path=tensor_path,
            atm_params=atm_params,
            space_res=space_res,
            time_res=time_res,
        )
