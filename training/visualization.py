import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch
from torch.utils.data import Subset


def _sanitize_batch(xb, yb):
    """Replace NaN/Inf values so visualization stays finite."""
    xb = torch.nan_to_num(xb.float(), nan=0.0, posinf=0.0, neginf=0.0)
    yb = torch.nan_to_num(yb.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return xb, yb


def _get_batch(data_loader, batch_index):
    """Return the requested batch from a data loader."""
    for current_batch_index, batch in enumerate(data_loader):
        if current_batch_index == batch_index:
            return batch[0], batch[1]
    raise IndexError(f"Batch index {batch_index} is out of range for this data loader.")


def _find_sample_with_lightning(data_loader, occurrence_index=0):
    """Return the nth sample whose target contains at least one lightning pixel."""
    found_count = 0

    for current_batch_index, batch in enumerate(data_loader):
        xb, yb = batch[0], batch[1]
        _, yb = _sanitize_batch(xb, yb)
        yb = (yb > 0).float()

        for current_sample_index in range(yb.shape[0]):
            has_lightning = torch.any(yb[current_sample_index, 0] > 0.5)
            if has_lightning:
                if found_count == occurrence_index:
                    return current_batch_index, current_sample_index
                found_count += 1

    raise ValueError(
        "No sample with lightning was found at the requested occurrence index "
        f"({occurrence_index})."
    )


def _threshold_summary(lightning_probs, background_probs, thresholds):
    """Create a simple threshold summary from lightning/background probabilities."""
    rows = []
    lightning_total = max(len(lightning_probs), 1)
    background_total = max(len(background_probs), 1)

    for threshold in thresholds:
        detected_lightning = np.sum(lightning_probs >= threshold) / lightning_total
        false_alarm_fraction = np.sum(background_probs >= threshold) / background_total
        rows.append(
            {
                "threshold": threshold,
                "lightning_detected_fraction": detected_lightning,
                "background_above_threshold_fraction": false_alarm_fraction,
            }
        )

    return rows


def _resolve_dataset_sample_index(dataset, relative_index):
    """Map a loader-relative index back to the original dataset index."""
    if isinstance(dataset, Subset):
        subset_index = dataset.indices[relative_index]
        return _resolve_dataset_sample_index(dataset.dataset, subset_index)
    return relative_index


def _resolve_sample_selection(
    data_loader,
    batch_index=0,
    sample_index=0,
    require_lightning=False,
    lightning_occurrence_index=0,
    sample_metadata=None,
):
    """Resolve loader and dataset indices for a selected sample."""
    if require_lightning:
        batch_index, sample_index = _find_sample_with_lightning(
            data_loader, occurrence_index=lightning_occurrence_index
        )

    loader_sample_index = batch_index * data_loader.batch_size + sample_index
    original_sample_index = None
    sample_label = None
    if loader_sample_index < len(data_loader.dataset):
        original_sample_index = _resolve_dataset_sample_index(
            data_loader.dataset, loader_sample_index
        )
        if sample_metadata is not None and original_sample_index < len(sample_metadata):
            sample_label = sample_metadata[original_sample_index]

    return {
        "batch_index": batch_index,
        "sample_index": sample_index,
        "loader_sample_index": loader_sample_index,
        "original_sample_index": original_sample_index,
        "sample_label": sample_label,
    }


def _channel_label(channel_index, channel_names=None):
    """Return a readable channel label."""
    if channel_names is not None and channel_index < len(channel_names):
        return f"Input Channel {channel_index}: {channel_names[channel_index]}"
    return f"Input Channel {channel_index}"


def _apply_geo_axis_style(axis, min_lon, max_lon, min_lat, max_lat):
    """Apply a consistent Cartopy styling to a geographic axis."""
    axis.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    axis.coastlines(resolution="10m")
    axis.add_feature(cfeature.BORDERS, linestyle=":")
    gridliner = axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gridliner.top_labels = False
    gridliner.right_labels = False


def _get_probability_colormap(probability_bounds=None, cmap_name="plasma"):
    """Return a discrete colormap and norm for probability visualization."""
    if probability_bounds is None:
        probability_bounds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    cmap = plt.get_cmap(cmap_name, len(probability_bounds) - 1)
    norm = mcolors.BoundaryNorm(probability_bounds, cmap.N)
    return probability_bounds, cmap, norm


def plot_original_sample_maps(
    X_original,
    y_original,
    output_dir="training/visualizations",
    original_sample_index=0,
    channel_names=None,
    sample_metadata=None,
    prefix="raw",
):
    """
    Save plots for one sample from the original, unnormalized tensors.

    This is useful for checking whether normalization introduced artifacts by
    comparing the raw maps to the processed visualization outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    if original_sample_index < 0 or original_sample_index >= len(X_original):
        raise IndexError(
            f"Original sample index {original_sample_index} is out of range for "
            f"{len(X_original)} samples."
        )

    x_sample = torch.nan_to_num(
        X_original[original_sample_index].detach().cpu().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y_sample = torch.nan_to_num(
        y_original[original_sample_index].detach().cpu().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    true_map = (y_sample[0] > 0).numpy().astype(np.float32)

    sample_label = None
    if sample_metadata is not None and original_sample_index < len(sample_metadata):
        sample_label = sample_metadata[original_sample_index]

    figure_path = os.path.join(
        output_dir, f"{prefix}_sample{original_sample_index}_raw_maps.png"
    )
    summary_path = os.path.join(
        output_dir, f"{prefix}_sample{original_sample_index}_raw_summary.txt"
    )

    panel_count = x_sample.shape[0] + 1
    fig, axes = plt.subplots(1, panel_count, figsize=(4.8 * panel_count, 5))
    if panel_count == 1:
        axes = [axes]

    for channel_index in range(x_sample.shape[0]):
        channel_map = x_sample[channel_index].numpy()
        label = _channel_label(channel_index, channel_names)
        im = axes[channel_index].imshow(channel_map, cmap="viridis")
        axes[channel_index].set_title(label)
        plt.colorbar(im, ax=axes[channel_index], fraction=0.046, pad=0.04)

    im_true = axes[-1].imshow(true_map, cmap="Blues")
    axes[-1].set_title("True Lightning")
    plt.colorbar(im_true, ax=axes[-1], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"Original dataset sample index: {original_sample_index}\n")
        if sample_label is not None:
            file.write(f"Sample timestamp/group: {sample_label}\n")
        if channel_names is not None:
            file.write(
                "Input channels: "
                + ", ".join(
                    f"{idx} ({name})"
                    for idx, name in enumerate(channel_names[: x_sample.shape[0]])
                )
                + "\n"
            )
        else:
            file.write(f"Input channel count: {x_sample.shape[0]}\n")

        file.write("\nRaw channel stats\n")
        for channel_index in range(x_sample.shape[0]):
            channel_map = x_sample[channel_index].numpy()
            label = _channel_label(channel_index, channel_names)
            file.write(
                f"{label} -> min={channel_map.min():.6f}, "
                f"max={channel_map.max():.6f}, "
                f"mean={channel_map.mean():.6f}, "
                f"std={channel_map.std():.6f}\n"
            )

        file.write("\nTarget stats\n")
        file.write(
            f"Lightning pixels: {int(true_map.sum())} | "
            f"Total pixels: {true_map.size}\n"
        )

        file.write("\nSaved files\n")
        file.write(f"raw_maps={figure_path}\n")

    print(f"Saved raw map figure to: {figure_path}")
    print(f"Saved raw summary to: {summary_path}")
    if sample_label is not None:
        print(f"Raw sample timestamp/group: {sample_label}")

    return {
        "figure_path": figure_path,
        "summary_path": summary_path,
        "original_sample_index": original_sample_index,
        "sample_label": sample_label,
    }


def plot_original_maps_for_loader_sample(
    X_original,
    y_original,
    data_loader,
    inspection_result=None,
    model=None,
    device=None,
    output_dir="training/visualizations",
    batch_index=0,
    sample_index=0,
    input_channel=0,
    decision_threshold=0.5,
    thresholds=None,
    prefix="raw",
    require_lightning=False,
    lightning_occurrence_index=0,
    channel_names=None,
    sample_metadata=None,
):
    """
    Save raw-map plots for the same sample selected from a loader.

    Use this after or alongside inspect_probability_maps() to compare the
    original tensors against the normalized model inputs for the exact sample.
    Parameters such as model/device/decision_threshold are accepted so this
    function can mirror inspect_probability_maps() calls without mismatch.
    If inspection_result is provided, its resolved sample selection is reused.
    """
    del model, device, decision_threshold
    if thresholds is None:
        thresholds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]

    if inspection_result is not None:
        selection = {
            "batch_index": inspection_result["batch_index"],
            "sample_index": inspection_result["sample_index"],
            "loader_sample_index": inspection_result.get("loader_sample_index"),
            "original_sample_index": inspection_result.get("original_sample_index"),
            "sample_label": inspection_result.get("sample_label"),
        }
    else:
        selection = _resolve_sample_selection(
            data_loader=data_loader,
            batch_index=batch_index,
            sample_index=sample_index,
            require_lightning=require_lightning,
            lightning_occurrence_index=lightning_occurrence_index,
            sample_metadata=sample_metadata,
        )

    if selection["original_sample_index"] is None:
        raise ValueError("Could not resolve the original dataset sample index.")

    result = plot_original_sample_maps(
        X_original=X_original,
        y_original=y_original,
        output_dir=output_dir,
        original_sample_index=selection["original_sample_index"],
        channel_names=channel_names,
        sample_metadata=sample_metadata,
        prefix=(
            f"{prefix}_batch{selection['batch_index']}_sample"
            f"{selection['sample_index']}"
        ),
    )
    result.update(selection)
    result["input_channel"] = input_channel
    result["thresholds"] = thresholds
    return result


def inspect_probability_maps(
    model,
    data_loader,
    device,
    output_dir="training/visualizations",
    batch_index=0,
    sample_index=0,
    input_channel=0,
    decision_threshold=0.5,
    thresholds=None,
    prefix="val",
    require_lightning=False,
    lightning_occurrence_index=0,
    channel_names=None,
    sample_metadata=None,
    probability_bounds=None,
):
    """
    Save diagnostic plots for one sample from a loader.

    Outputs:
    - all input channel maps
    - predicted probability map
    - true lightning map
    - probability map with true lightning contour
    - histogram of lightning-vs-background probabilities
    - simple threshold summary text file
    """
    if thresholds is None:
        thresholds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]

    os.makedirs(output_dir, exist_ok=True)

    selection = _resolve_sample_selection(
        data_loader=data_loader,
        batch_index=batch_index,
        sample_index=sample_index,
        require_lightning=require_lightning,
        lightning_occurrence_index=lightning_occurrence_index,
        sample_metadata=sample_metadata,
    )
    batch_index = selection["batch_index"]
    sample_index = selection["sample_index"]
    loader_sample_index = selection["loader_sample_index"]
    original_sample_index = selection["original_sample_index"]
    sample_label = selection["sample_label"]

    xb, yb = _get_batch(data_loader, batch_index)
    xb, yb = _sanitize_batch(xb, yb)
    yb = (yb > 0).float()

    if sample_index >= xb.shape[0]:
        raise IndexError(
            f"Sample index {sample_index} is out of range for batch size {xb.shape[0]}."
        )
    if input_channel >= xb.shape[1]:
        raise IndexError(
            f"Input channel {input_channel} is out of range for {xb.shape[1]} channels."
        )

    channel_label = _channel_label(input_channel, channel_names)

    with torch.no_grad():
        logits = model(xb.to(device))
        probs = torch.sigmoid(logits).detach().cpu()

    input_maps = [
        xb[sample_index, current_channel].detach().cpu().numpy()
        for current_channel in range(xb.shape[1])
    ]
    prob_map = probs[sample_index, 0].numpy()
    true_map = yb[sample_index, 0].detach().cpu().numpy()
    pred_binary_map = (prob_map >= decision_threshold).astype(np.float32)

    lightning_mask = true_map > 0.5
    background_mask = ~lightning_mask

    lightning_probs = prob_map[lightning_mask]
    background_probs = prob_map[background_mask]
    probability_bounds, probability_cmap, probability_norm = _get_probability_colormap(
        probability_bounds
    )

    figure_path = os.path.join(
        output_dir, f"{prefix}_batch{batch_index}_sample{sample_index}_maps.png"
    )
    hist_path = os.path.join(
        output_dir, f"{prefix}_batch{batch_index}_sample{sample_index}_hist.png"
    )
    summary_path = os.path.join(
        output_dir, f"{prefix}_batch{batch_index}_sample{sample_index}_summary.txt"
    )

    panel_count = xb.shape[1] + 4
    fig, axes = plt.subplots(1, panel_count, figsize=(4.8 * panel_count, 5))
    if panel_count == 1:
        axes = [axes]

    for current_channel, input_map in enumerate(input_maps):
        input_label = _channel_label(current_channel, channel_names)

        im = axes[current_channel].imshow(input_map, cmap="viridis")
        if current_channel == input_channel:
            input_label = f"{input_label} [selected]"
        axes[current_channel].set_title(input_label)
        plt.colorbar(im, ax=axes[current_channel], fraction=0.046, pad=0.04)

    im_prob = axes[-4].imshow(prob_map, cmap=probability_cmap, norm=probability_norm)
    axes[-4].set_title("Predicted Probabilities")
    plt.colorbar(
        im_prob,
        ax=axes[-4],
        fraction=0.046,
        pad=0.04,
        ticks=probability_bounds,
    )

    im_true = axes[-3].imshow(true_map, cmap="Blues")
    axes[-3].set_title("True Lightning")
    plt.colorbar(im_true, ax=axes[-3], fraction=0.046, pad=0.04)

    im_overlay = axes[-2].imshow(prob_map, cmap=probability_cmap, norm=probability_norm)
    axes[-2].contour(true_map, levels=[0.5], colors="cyan", linewidths=1)
    axes[-2].set_title("Probabilities + Truth Contour")
    plt.colorbar(
        im_overlay,
        ax=axes[-2],
        fraction=0.046,
        pad=0.04,
        ticks=probability_bounds,
    )

    im_binary = axes[-1].imshow(pred_binary_map, cmap="gray", vmin=0.0, vmax=1.0)
    axes[-1].contour(true_map, levels=[0.5], colors="red", linewidths=1)
    axes[-1].set_title(f"Pred >= {decision_threshold}")
    plt.colorbar(im_binary, ax=axes[-1], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if background_probs.size > 0:
        ax.hist(
            background_probs.flatten(),
            bins=50,
            alpha=0.6,
            label="No Lightning",
            color="gray",
        )
    if lightning_probs.size > 0:
        ax.hist(
            lightning_probs.flatten(),
            bins=50,
            alpha=0.6,
            label="Lightning",
            color="orange",
        )
    ax.set_title("Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    threshold_rows = _threshold_summary(lightning_probs, background_probs, thresholds)

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"Sample: batch={batch_index}, sample={sample_index}\n")
        file.write(f"Loader sample index: {loader_sample_index}\n")
        if original_sample_index is not None:
            file.write(f"Original dataset sample index: {original_sample_index}\n")
        if sample_label is not None:
            file.write(f"Sample timestamp/group: {sample_label}\n")
        if channel_names is not None:
            file.write(
                "Input channels: "
                + ", ".join(
                    f"{idx} ({name})"
                    for idx, name in enumerate(channel_names[: xb.shape[1]])
                )
                + "\n"
            )
        else:
            file.write(f"Input channel count: {xb.shape[1]}\n")
        file.write(f"Selected input channel: {channel_label}\n")
        file.write(f"Require lightning: {require_lightning}\n")
        if require_lightning:
            file.write(f"Lightning occurrence index: {lightning_occurrence_index}\n")
        file.write(f"Decision threshold shown in map: {decision_threshold}\n\n")

        file.write("Probability stats\n")
        file.write(
            f"Lightning pixels: {lightning_probs.size} | "
            f"Background pixels: {background_probs.size}\n"
        )

        if lightning_probs.size > 0:
            file.write(
                "Lightning probs -> "
                f"min={lightning_probs.min():.6f}, "
                f"max={lightning_probs.max():.6f}, "
                f"mean={lightning_probs.mean():.6f}, "
                f"median={np.median(lightning_probs):.6f}\n"
            )
        else:
            file.write("Lightning probs -> no lightning pixels in this sample\n")

        if background_probs.size > 0:
            file.write(
                "Background probs -> "
                f"min={background_probs.min():.6f}, "
                f"max={background_probs.max():.6f}, "
                f"mean={background_probs.mean():.6f}, "
                f"median={np.median(background_probs):.6f}\n"
            )
        else:
            file.write("Background probs -> no background pixels in this sample\n")

        file.write("\nThreshold summary\n")
        for row in threshold_rows:
            file.write(
                f"threshold={row['threshold']:.3f} | "
                f"lightning_detected_fraction={row['lightning_detected_fraction']:.4f} | "
                f"background_above_threshold_fraction={row['background_above_threshold_fraction']:.4f}\n"
            )

        file.write("\nSaved files\n")
        file.write(f"maps={figure_path}\n")
        file.write(f"histogram={hist_path}\n")

    print(f"Saved map figure to: {figure_path}")
    print(f"Saved histogram to: {hist_path}")
    print(f"Saved summary to: {summary_path}")
    if sample_label is not None:
        print(f"Sample timestamp/group: {sample_label}")
    print(
        f"Visualized batch={batch_index}, sample={sample_index}, input_channel={input_channel}"
    )

    if original_sample_index is not None:
        print(f"Original dataset sample index: {original_sample_index}")

    return {
        "figure_path": figure_path,
        "hist_path": hist_path,
        "summary_path": summary_path,
        "lightning_probs": lightning_probs,
        "background_probs": background_probs,
        "threshold_rows": threshold_rows,
        "batch_index": batch_index,
        "sample_index": sample_index,
        "loader_sample_index": loader_sample_index,
        "original_sample_index": original_sample_index,
        "sample_label": sample_label,
    }


def inspect_geo_probability_map(
    model,
    data_loader,
    device,
    output_dir="training/visualizations",
    batch_index=0,
    sample_index=0,
    input_channel=0,
    decision_threshold=0.5,
    thresholds=None,
    prefix="val",
    require_lightning=False,
    lightning_occurrence_index=0,
    channel_names=None,
    sample_metadata=None,
    min_lat=None,
    max_lat=None,
    min_lon=None,
    max_lon=None,
    probability_bounds=None,
):
    """
    Save diagnostic plots for one sample from a loader.

    Outputs:
    - all input channel maps
    - predicted probability map
    - true lightning map
    - probability map with true lightning contour
    - histogram of lightning-vs-background probabilities
    - simple threshold summary text file
    """
    if thresholds is None:
        thresholds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]

    os.makedirs(output_dir, exist_ok=True)

    selection = _resolve_sample_selection(
        data_loader=data_loader,
        batch_index=batch_index,
        sample_index=sample_index,
        require_lightning=require_lightning,
        lightning_occurrence_index=lightning_occurrence_index,
        sample_metadata=sample_metadata,
    )
    batch_index = selection["batch_index"]
    sample_index = selection["sample_index"]
    loader_sample_index = selection["loader_sample_index"]
    original_sample_index = selection["original_sample_index"]
    sample_label = selection["sample_label"]

    xb, yb = _get_batch(data_loader, batch_index)
    xb, yb = _sanitize_batch(xb, yb)
    yb = (yb > 0).float()

    if sample_index >= xb.shape[0]:
        raise IndexError(
            f"Sample index {sample_index} is out of range for batch size {xb.shape[0]}."
        )

    with torch.no_grad():
        logits = model(xb.to(device))
        probs = torch.sigmoid(logits).detach().cpu()

    prob_map = probs[sample_index, 0].numpy()
    true_map = yb[sample_index, 0].detach().cpu().numpy()
    pred_binary_map = (prob_map >= decision_threshold).astype(np.float32)

    lightning_mask = true_map > 0.5
    background_mask = ~lightning_mask

    lightning_probs = prob_map[lightning_mask]
    background_probs = prob_map[background_mask]
    probability_bounds, probability_cmap, probability_norm = _get_probability_colormap(
        probability_bounds
    )

    figure_path = os.path.join(
        output_dir, f"{prefix}_batch{batch_index}_sample{sample_index}_maps.png"
    )

    # plotting
    panel_count = 2
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(6.2 * panel_count, 5.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if panel_count == 1:
        axes = [axes]

    extent = [min_lon, max_lon, min_lat, max_lat]

    # probability map
    _apply_geo_axis_style(axes[-2], min_lon, max_lon, min_lat, max_lat)
    im_overlay = axes[-2].imshow(
        prob_map,
        cmap=probability_cmap,
        norm=probability_norm,
        extent=extent,
        origin="lower",
        transform=ccrs.PlateCarree(),
    )
    axes[-2].contour(
        true_map,
        levels=[0.5],
        colors="cyan",
        linewidths=1,
        extent=extent,
        origin="lower",
        transform=ccrs.PlateCarree(),
    )
    axes[-2].set_title("Probabilities + Truth Contour")
    plt.colorbar(
        im_overlay,
        ax=axes[-2],
        fraction=0.046,
        pad=0.04,
        ticks=probability_bounds,
    )

    # binary decision map
    _apply_geo_axis_style(axes[-1], min_lon, max_lon, min_lat, max_lat)
    im_binary = axes[-1].imshow(
        pred_binary_map,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        origin="lower",
        transform=ccrs.PlateCarree(),
    )
    axes[-1].contour(
        true_map,
        levels=[0.5],
        colors="red",
        linewidths=1,
        extent=extent,
        origin="lower",
        transform=ccrs.PlateCarree(),
    )
    axes[-1].set_title(f"Pred >= {decision_threshold}")
    plt.colorbar(im_binary, ax=axes[-1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    threshold_rows = _threshold_summary(lightning_probs, background_probs, thresholds)

    print(f"Saved map figure to: {figure_path}")

    if sample_label is not None:
        print(f"Sample timestamp/group: {sample_label}")
    print(
        f"Visualized batch={batch_index}, sample={sample_index}, input_channel={input_channel}"
    )

    if original_sample_index is not None:
        print(f"Original dataset sample index: {original_sample_index}")

    return {
        "figure_path": figure_path,
        "lightning_probs": lightning_probs,
        "background_probs": background_probs,
        "threshold_rows": threshold_rows,
        "batch_index": batch_index,
        "sample_index": sample_index,
        "loader_sample_index": loader_sample_index,
        "original_sample_index": original_sample_index,
        "sample_label": sample_label,
    }
