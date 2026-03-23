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
):
    """
    Save diagnostic plots for one sample from a loader.

    Outputs:
    - input channel map
    - predicted probability map
    - true lightning map
    - probability map with true lightning contour
    - histogram of lightning-vs-background probabilities
    - simple threshold summary text file
    """
    if thresholds is None:
        thresholds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
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

    input_map = xb[sample_index, input_channel].detach().cpu().numpy()
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

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    im0 = axes[0].imshow(input_map, cmap="viridis")
    axes[0].set_title(f"Input Channel {input_channel}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Predicted Probabilities")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(true_map, cmap="Blues")
    axes[2].set_title("True Lightning")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].contour(true_map, levels=[0.5], colors="cyan", linewidths=1)
    axes[3].set_title("Probabilities + Truth Contour")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    im4 = axes[4].imshow(pred_binary_map, cmap="gray", vmin=0.0, vmax=1.0)
    axes[4].contour(true_map, levels=[0.5], colors="red", linewidths=1)
    axes[4].set_title(f"Pred >= {decision_threshold}")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

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

    return {
        "figure_path": figure_path,
        "hist_path": hist_path,
        "summary_path": summary_path,
        "lightning_probs": lightning_probs,
        "background_probs": background_probs,
        "threshold_rows": threshold_rows,
    }
