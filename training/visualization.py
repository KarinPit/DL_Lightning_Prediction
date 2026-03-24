import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _sanitize_batch(xb, yb):
    """Replace NaN/Inf values so visualization stays finite."""
    xb = torch.nan_to_num(xb.float(), nan=0.0, posinf=0.0, neginf=0.0)
    yb = torch.nan_to_num(yb.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return xb, yb


def _get_batch(data_loader, batch_index):
    """Return the requested batch from a data loader."""
    for current_batch_index, (xb, yb) in enumerate(data_loader):
        if current_batch_index == batch_index:
            return xb, yb
    raise IndexError(f"Batch index {batch_index} is out of range for this data loader.")


def _find_sample_with_lightning(data_loader, occurrence_index=0):
    """Return the nth sample whose target contains at least one lightning pixel."""
    found_count = 0

    for current_batch_index, (xb, yb) in enumerate(data_loader):
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

    if require_lightning:
        batch_index, sample_index = _find_sample_with_lightning(
            data_loader, occurrence_index=lightning_occurrence_index
        )

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

    channel_label = f"Input Channel {input_channel}"
    if channel_names is not None and input_channel < len(channel_names):
        channel_label = f"Input Channel {input_channel}: {channel_names[input_channel]}"

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
        input_label = f"Input Channel {current_channel}"
        if channel_names is not None and current_channel < len(channel_names):
            input_label = f"Input Channel {current_channel}: {channel_names[current_channel]}"

        im = axes[current_channel].imshow(input_map, cmap="viridis")
        if current_channel == input_channel:
            input_label = f"{input_label} [selected]"
        axes[current_channel].set_title(input_label)
        plt.colorbar(im, ax=axes[current_channel], fraction=0.046, pad=0.04)

    im_prob = axes[-4].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[-4].set_title("Predicted Probabilities")
    plt.colorbar(im_prob, ax=axes[-4], fraction=0.046, pad=0.04)

    im_true = axes[-3].imshow(true_map, cmap="Blues")
    axes[-3].set_title("True Lightning")
    plt.colorbar(im_true, ax=axes[-3], fraction=0.046, pad=0.04)

    im_overlay = axes[-2].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[-2].contour(true_map, levels=[0.5], colors="cyan", linewidths=1)
    axes[-2].set_title("Probabilities + Truth Contour")
    plt.colorbar(im_overlay, ax=axes[-2], fraction=0.046, pad=0.04)

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
        if channel_names is not None:
            file.write(
                "Input channels: "
                + ", ".join(
                    f"{idx} ({name})" for idx, name in enumerate(channel_names[: xb.shape[1]])
                )
                + "\n"
            )
        else:
            file.write(f"Input channel count: {xb.shape[1]}\n")
        file.write(f"Selected input channel: {channel_label}\n")
        file.write(f"Require lightning: {require_lightning}\n")
        if require_lightning:
            file.write(
                f"Lightning occurrence index: {lightning_occurrence_index}\n"
            )
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
    print(
        f"Visualized batch={batch_index}, sample={sample_index}, input_channel={input_channel}"
    )

    return {
        "figure_path": figure_path,
        "hist_path": hist_path,
        "summary_path": summary_path,
        "lightning_probs": lightning_probs,
        "background_probs": background_probs,
        "threshold_rows": threshold_rows,
        "batch_index": batch_index,
        "sample_index": sample_index,
    }
