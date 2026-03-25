import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset, DataLoader

from config.constants import MAIN_PATH
from config.experiment import CASE_CONFIG, MODEL_CONFIG, RUN_CONFIG
from data.preprocessing import build_and_save_tensors, mean_std_norm
from models.unet import UNet
from training.train import train_model
from training.visualization import (
    inspect_probability_maps,
    plot_original_maps_for_loader_sample,
    inspect_geo_probability_map,
)


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
    """Split indices so all samples from the same group stay together.
    Notice that the groups are generated in data/preprocessing- they are the different timestamps available in the data
    """
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


def save_tensor_stats_report(
    X, y, tensor_path, experiment_tag, channel_names=None, groups=None
):
    """Save a text report with post-normalization tensor statistics."""
    report_path = os.path.join(
        tensor_path, f"tensor_stats_{experiment_tag}_post_normalization.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Post-normalization tensor statistics\n\n")
        file.write(f"X shape: {tuple(X.shape)}\n")
        file.write(f"y shape: {tuple(y.shape)}\n\n")
        if groups is not None:
            file.write(f"train groups: {groups[0]}\n\n")
            file.write(f"val groups: {groups[1]}\n\n")

        file.write("X overall stats\n")
        file.write(f"mean={X.mean().item():.6f}\n")
        file.write(f"std={X.std().item():.6f}\n")
        file.write(f"min={X.min().item():.6f}\n")
        file.write(f"max={X.max().item():.6f}\n\n")

        file.write("X per-channel stats\n")
        for channel_index in range(X.shape[1]):
            channel_tensor = X[:, channel_index, :, :]
            channel_label = (
                channel_names[channel_index]
                if channel_names is not None and channel_index < len(channel_names)
                else f"channel_{channel_index}"
            )
            file.write(
                f"{channel_label}: "
                f"mean={channel_tensor.mean().item():.6f}, "
                f"std={channel_tensor.std().item():.6f}, "
                f"min={channel_tensor.min().item():.6f}, "
                f"max={channel_tensor.max().item():.6f}\n"
            )

        file.write("\ny overall stats\n")
        file.write(f"mean={y.mean().item():.6f}\n")
        file.write(f"std={y.std().item():.6f}\n")
        file.write(f"min={y.min().item():.6f}\n")
        file.write(f"max={y.max().item():.6f}\n")

    print(f"Saved tensor stats report to: {report_path}")
    return report_path


def clip_normalized_tensor(X, clip_value):
    """Clip normalized input tensors to a symmetric range."""
    if clip_value is None:
        return X

    clipped_X = torch.clamp(X, min=-clip_value, max=clip_value)
    print(f"Clipped normalized X values to [-{clip_value}, {clip_value}].")
    return clipped_X


if __name__ == "__main__":
    run_config = RUN_CONFIG
    model_config = MODEL_CONFIG
    case_config = CASE_CONFIG

    # paths configuration
    experiment_tag = get_experiment_tag(run_config.use_seed, run_config.seed_value)
    tensor_path = os.path.join(
        MAIN_PATH,
        case_config.dataset_name,
        "Ens",
        "Tensors",
        case_config.space_res,
    )
    weights_save_path = os.path.join(tensor_path, f"unet_weights_{experiment_tag}.pth")

    if run_config.use_seed:
        set_seed(run_config.seed_value)
        print(f"Using fixed seed: {run_config.seed_value}")
    else:
        print("Using random training without a fixed seed.")
        if not run_config.to_train:
            raise ValueError(
                "Evaluation-only mode without a seed is not reliable. "
                "Set use_seed=True so the validation split matches the saved weights."
            )

    # DL model configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([model_config.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # check if case tensors exist
    tensors_exist = False

    if os.path.exists(tensor_path):
        files = os.listdir(tensor_path)
        if "X_final.pt" in files and "Y_final.pt" in files:
            tensors_exist = True

    if tensors_exist:
        print("X and y tensors available. Skipping tensor creation and saving!")

        X, y, sample_groups = load_saved_tensors(tensor_path)

        if X.shape[1] != case_config.expected_input_channels:
            print(
                "Saved tensors do not match the configured atmospheric parameters. "
                f"Expected {case_config.expected_input_channels} input channels from atm_params, "
                f"but found {X.shape[1]} in X_final.pt. Rebuilding tensors."
            )
            build_and_save_tensors(
                wrf_path=None,
                entln_path=None,
                tensor_path=tensor_path,
                atm_params=case_config.atm_params,
                space_res=case_config.space_res,
                time_res=case_config.time_res,
                case_config=case_config,
            )
            X, y, sample_groups = load_saved_tensors(tensor_path)

        model = UNet(n_channels=X.shape[1], n_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)

        # get indices of training and validation datasets after splitting
        train_loader_generator = None
        if run_config.use_seed:
            train_loader_generator = torch.Generator().manual_seed(
                run_config.seed_value
            )

        train_idx, val_idx = split_indices_by_group(
            sample_groups,
            train_fraction=model_config.train_fraction,
            seed=run_config.seed_value if run_config.use_seed else None,
        )

        train_groups = set(sample_groups[i] for i in train_idx)
        val_groups = set(sample_groups[i] for i in val_idx)
        print(
            f"Group-aware split: {len(train_groups)} train groups, "
            f"{len(val_groups)} val groups."
        )

        if run_config.plot_raw_tensors:
            X_raw = X.clone()
            y_raw = y.clone()

        # compute mean and std of each feature
        train_X = X[train_idx]
        means, stds = mean_std_norm(train_X)

        # normalize by mean and std while keeping the tensor finite
        for c in range(X.shape[1]):
            channel_tensor = torch.nan_to_num(
                X[:, c, :, :].float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            safe_mean = means[c]
            safe_std = stds[c]
            normalized_channel = (channel_tensor - safe_mean) / safe_std
            X[:, c, :, :] = torch.nan_to_num(
                normalized_channel,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        if model_config.clip_after_normalization:
            X = clip_normalized_tensor(X, model_config.normalization_clip_value)

        save_tensor_stats_report(
            X=X,
            y=y,
            tensor_path=tensor_path,
            experiment_tag=experiment_tag,
            channel_names=case_config.input_channel_names,
            groups=[train_groups, val_groups],
        )

        full_dataset = TensorDataset(X, y)
        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=model_config.batch_size,
            shuffle=True,
            generator=train_loader_generator,
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=model_config.batch_size,
            shuffle=False,
        )

        if run_config.to_train:
            # run train and evaluation
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=model_config.num_epochs,
                device=device,
                decision_threshold=model_config.decision_threshold,  # change threshold.
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

            inspection_result = inspect_probability_maps(
                model,
                val_loader,
                device,
                output_dir="training/visualizations",
                batch_index=1,
                sample_index=0,
                input_channel=1,
                decision_threshold=model_config.decision_threshold,
                thresholds=model_config.visualization_thresholds,
                prefix=f"val_{experiment_tag}",
                require_lightning=True,
                lightning_occurrence_index=0,
                channel_names=case_config.input_channel_names,
                sample_metadata=sample_groups,
            )

            inspect_geo_probability_map(
                model,
                val_loader,
                device,
                output_dir="training/visualizations",
                batch_index=1,
                sample_index=0,
                input_channel=1,
                decision_threshold=model_config.decision_threshold,
                thresholds=model_config.visualization_thresholds,
                prefix=f"val_{experiment_tag}",
                require_lightning=True,
                lightning_occurrence_index=0,
                channel_names=case_config.input_channel_names,
                sample_metadata=sample_groups,
                min_lat=CASE_CONFIG.min_lat,
                max_lat=CASE_CONFIG.max_lat,
                min_lon=CASE_CONFIG.min_lon,
                max_lon=CASE_CONFIG.max_lon,
            )

            if run_config.plot_raw_tensors:
                plot_original_maps_for_loader_sample(
                    X_raw,
                    y_raw,
                    val_loader,
                    inspection_result=inspection_result,
                    output_dir="training/visualizations",
                    prefix="raw",
                    channel_names=case_config.input_channel_names,
                    sample_metadata=sample_groups,
                )

    else:
        build_and_save_tensors(
            wrf_path=None,
            entln_path=None,
            tensor_path=tensor_path,
            atm_params=case_config.atm_params,
            space_res=case_config.space_res,
            time_res=case_config.time_res,
            case_config=case_config,
        )
