import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset, DataLoader

from data.preprocessing import build_and_save_tensors, mean_std_norm
from models.unet import UNet
from training.train import train_model
from training.visualization import inspect_probability_maps

MAIN_PATH = "/Users/karinpitlik/Desktop/DataScience/Thesis"
CASES = [
    "Case1_Nov_2022_23_25",
    "Case2_Jan_2023_11_16",
    "Case3_Mar_2023_13_15",
    "Case4_Apr_2023_09_13",
    "Case5_Jan_2024_26_31",
    "Case6_Nov_2025_24_25",
]


def set_seed(seed_value):
    """Set random seeds for reproducible experiments."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_experiment_tag(use_seed, seed_value):
    """Return a stable tag for seeded experiments and random runs."""
    if use_seed:
        return f"seed_{seed_value}"
    return "random"


def split_indices_by_group(groups, train_fraction=0.8, seed=None):
    """Split indices so all samples from the same group stay together."""
    unique_groups = np.array(sorted(set(groups)))
    if len(unique_groups) < 2:
        raise ValueError(
            "Need at least two unique groups for a train/validation split. "
            "Regenerate tensors with richer grouping metadata or provide more data."
        )

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    train_group_count = int(train_fraction * len(shuffled_groups))
    train_group_count = min(max(train_group_count, 1), len(shuffled_groups) - 1)

    train_groups = set(shuffled_groups[:train_group_count])
    val_groups = set(shuffled_groups[train_group_count:])

    train_idx = [idx for idx, group in enumerate(groups) if group in train_groups]
    val_idx = [idx for idx, group in enumerate(groups) if group in val_groups]
    return train_idx, val_idx


def load_saved_tensors(tensor_path):
    """Load saved input, target, and grouping tensors from disk."""
    X = torch.load(os.path.join(tensor_path, "X_final.pt"))
    y = torch.load(os.path.join(tensor_path, "Y_final.pt"))
    groups_path = os.path.join(tensor_path, "sample_groups.pt")
    sample_groups = (
        torch.load(groups_path)
        if os.path.exists(groups_path)
        else [f"sample_{idx}" for idx in range(len(X))]
    )

    if not os.path.exists(groups_path):
        print(
            "Warning: sample_groups.pt not found. Falling back to sample-level split, "
            "which is less safe than timestamp-level splitting."
        )

    return X, y, sample_groups


if __name__ == "__main__":
    # set to True when you want to train the model, and false when only evaluating it
    # also change use_seed to false when random results are required
    to_train = False
    use_seed = True
    seed_value = 42

    # case configuration
    case = CASES[5]
    atm_params = ["KI", "CAPE2D", "LPI", "PREC_RATE", "FLUX_UP", "WMAX_LAYER"]
    space_res = "24by24"
    time_res = "1_hours"
    expected_input_channels = len(atm_params)

    # paths configuration
    wrf_path = f"{MAIN_PATH}/{case}/Ens/Raw/"
    entln_path = f"{MAIN_PATH}/{case}/ENTLN/{space_res}/{time_res}"
    tensor_path = f"{MAIN_PATH}/{case}/Ens/Tensors"
    experiment_tag = get_experiment_tag(use_seed, seed_value)
    weights_save_path = os.path.join(tensor_path, f"unet_weights_{experiment_tag}.pth")

    if use_seed:
        set_seed(seed_value)
        print(f"Using fixed seed: {seed_value}")
    else:
        print("Using random training without a fixed seed.")
        if not to_train:
            raise ValueError(
                "Evaluation-only mode without a seed is not reliable. "
                "Set use_seed=True so the validation split matches the saved weights."
            )

    # DL model configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([5000.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    num_epochs = 10
    batch_size = 50
    decision_threshold = 0.95

    # check if case tensors exist
    tensors_exist = False

    if os.path.exists(tensor_path):
        files = os.listdir(tensor_path)
        if "X_final.pt" in files and "Y_final.pt" in files:
            tensors_exist = True

    if tensors_exist:
        print("X and y tensors available. Skipping tensor creation and saving!")

        X, y, sample_groups = load_saved_tensors(tensor_path)

        if X.shape[1] != expected_input_channels:
            print(
                "Saved tensors do not match the configured atmospheric parameters. "
                f"Expected {expected_input_channels} input channels from atm_params, "
                f"but found {X.shape[1]} in X_final.pt. Rebuilding tensors."
            )
            build_and_save_tensors(
                wrf_path=wrf_path,
                entln_path=entln_path,
                tensor_path=tensor_path,
                atm_params=atm_params,
                space_res=space_res,
                time_res=time_res,
            )
            X, y, sample_groups = load_saved_tensors(tensor_path)

        model = UNet(n_channels=X.shape[1], n_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # get indices of training and validation datasets after splitting
        train_loader_generator = None
        if use_seed:
            train_loader_generator = torch.Generator().manual_seed(seed_value)

        train_idx, val_idx = split_indices_by_group(
            sample_groups,
            train_fraction=0.8,
            seed=seed_value if use_seed else None,
        )
        print(
            f"Group-aware split: {len(set(sample_groups[i] for i in train_idx))} train groups, "
            f"{len(set(sample_groups[i] for i in val_idx))} val groups."
        )

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
            generator=train_loader_generator,
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
        )

        if to_train:
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
            # save the model's state for future runs
            torch.save(model.state_dict(), weights_save_path)

        else:
            if not os.path.exists(weights_save_path):
                raise FileNotFoundError(
                    f"No saved weights found at {weights_save_path}. "
                    "Train the model first with the same seed settings."
                )

            model.load_state_dict(torch.load(weights_save_path, map_location=device))
            model.eval()

            inspect_probability_maps(
                model,
                val_loader,
                device,
                output_dir="training/visualizations",
                batch_index=0,
                sample_index=0,
                input_channel=1,
                decision_threshold=0.98,
                thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                prefix=f"val_{experiment_tag}",
                require_lightning=True,
                lightning_occurrence_index=0,
                channel_names=atm_params,
            )

    else:
        build_and_save_tensors(
            wrf_path=wrf_path,
            entln_path=entln_path,
            tensor_path=tensor_path,
            atm_params=atm_params,
            space_res=space_res,
            time_res=time_res,
        )
