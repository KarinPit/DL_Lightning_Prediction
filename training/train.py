import numpy as np
import torch
from tqdm import tqdm


def _sanitize_batch(xb, yb):
    """Replace NaN/Inf values so loss and metrics stay finite."""
    xb = torch.nan_to_num(xb.float(), nan=0.0, posinf=0.0, neginf=0.0)
    yb = torch.nan_to_num(yb.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return xb, yb


def _safe_loss(loss):
    """Convert invalid losses into a finite fallback to keep training stable."""
    if torch.isnan(loss) or torch.isinf(loss):
        return None
    return loss


def _update_confusion_stats(binary_preds_np, yb_np, stats):
    """Accumulate contingency-table counts across an epoch."""
    stats["tp"] += np.sum((binary_preds_np == 1) & (yb_np == 1))
    stats["fn"] += np.sum((binary_preds_np == 0) & (yb_np == 1))
    stats["fp"] += np.sum((binary_preds_np == 1) & (yb_np == 0))


def _finalize_epoch_metrics(stats):
    """Convert accumulated epoch statistics into metrics."""
    csi_denom = stats["tp"] + stats["fn"] + stats["fp"]
    far_denom = stats["tp"] + stats["fp"]
    pod_denom = stats["tp"] + stats["fn"]

    csi = 0.0 if csi_denom == 0 else stats["tp"] / csi_denom
    far = 0.0 if far_denom == 0 else stats["fp"] / far_denom
    pod = 0.0 if pod_denom == 0 else stats["tp"] / pod_denom

    if stats["prob_count"] == 0:
        bs = 0.0
        bss = 0.0
    else:
        bs = stats["bs_sum"] / stats["prob_count"]
        climatology = stats["target_sum"] / stats["prob_count"]
        bs_ref = (
            stats["target_sq_sum"]
            - 2 * climatology * stats["target_sum"]
            + stats["prob_count"] * (climatology**2)
        ) / stats["prob_count"]
        bss = 0.0 if bs_ref == 0 else 1 - (bs / bs_ref)

    return {
        "csi": csi,
        "far": far,
        "pod": pod,
        "bs": float(bs),
        "bss": float(bss),
    }


def evaluate_model(model, data_loader, criterion, device, decision_threshold=0.5):
    """
    Evaluate model on a loader and return averaged loss + metrics for one epoch.
    Metrics:
        - CSI, FAR, POD use binary predictions
        - BS, BSS use probabilistic predictions
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    epoch_stats = {
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "bs_sum": 0.0,
        "prob_count": 0,
        "target_sum": 0.0,
        "target_sq_sum": 0.0,
    }

    with torch.no_grad():
        for xb, yb in tqdm(data_loader, desc="Validation", unit="batch", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            xb, yb = _sanitize_batch(xb, yb)
            yb = (yb > 0).float()

            logits = model(xb)
            loss = criterion(logits, yb)
            loss = _safe_loss(loss)
            if loss is None:
                continue

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Convert logits to probabilities for probabilistic metrics
            probs = torch.sigmoid(logits)

            # Convert probabilities to binary predictions for contingency metrics
            binary_preds = (probs > decision_threshold).int()

            # Move to CPU / NumPy for metric functions
            probs_np = probs.detach().cpu().numpy()
            binary_preds_np = binary_preds.detach().cpu().numpy()
            yb_np = yb.detach().cpu().numpy()
            _update_confusion_stats(binary_preds_np, yb_np, epoch_stats)
            epoch_stats["bs_sum"] += float(np.sum((probs_np - yb_np) ** 2))
            epoch_stats["prob_count"] += probs_np.size
            epoch_stats["target_sum"] += float(np.sum(yb_np))
            epoch_stats["target_sq_sum"] += float(np.sum(yb_np**2))

    avg_loss = total_loss / total_samples if total_samples > 0 else np.nan
    metric_values = _finalize_epoch_metrics(epoch_stats)

    return {
        "loss": avg_loss,
        "csi": metric_values["csi"],
        "far": metric_values["far"],
        "pod": metric_values["pod"],
        "bs": metric_values["bs"],
        "bss": metric_values["bss"],
    }


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs,
    device,
    decision_threshold=0.5,
):
    """
    Train the model and track epoch-level metrics.
    Returns:
        history: dict with train/val losses and metrics per epoch
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_csi": [],
        "val_csi": [],
        "train_far": [],
        "val_far": [],
        "train_pod": [],
        "val_pod": [],
        "train_bs": [],
        "val_bs": [],
        "train_bss": [],
        "val_bss": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        epoch_stats = {
            "tp": 0,
            "fn": 0,
            "fp": 0,
            "bs_sum": 0.0,
            "prob_count": 0,
            "target_sum": 0.0,
            "target_sq_sum": 0.0,
        }

        print(f"Starting Epoch {epoch}...")

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
            xb = xb.to(device)
            yb = yb.to(device)
            xb, yb = _sanitize_batch(xb, yb)
            yb = (yb > 0).float()

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)
            loss = _safe_loss(loss)
            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

            probs = torch.sigmoid(logits)
            binary_preds = (probs > decision_threshold).int()

            probs_np = probs.detach().cpu().numpy()
            binary_preds_np = binary_preds.detach().cpu().numpy()
            yb_np = yb.detach().cpu().numpy()
            _update_confusion_stats(binary_preds_np, yb_np, epoch_stats)
            epoch_stats["bs_sum"] += float(np.sum((probs_np - yb_np) ** 2))
            epoch_stats["prob_count"] += probs_np.size
            epoch_stats["target_sum"] += float(np.sum(yb_np))
            epoch_stats["target_sq_sum"] += float(np.sum(yb_np**2))

        # Average training metrics over the epoch
        avg_train_loss = (
            total_train_loss / total_train_samples if total_train_samples > 0 else 0.0
        )
        train_metric_values = _finalize_epoch_metrics(epoch_stats)
        avg_train_csi = train_metric_values["csi"]
        avg_train_far = train_metric_values["far"]
        avg_train_pod = train_metric_values["pod"]
        avg_train_bs = train_metric_values["bs"]
        avg_train_bss = train_metric_values["bss"]

        # Validation
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            decision_threshold=decision_threshold,
        )

        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])

        history["train_csi"].append(avg_train_csi)
        history["val_csi"].append(val_metrics["csi"])

        history["train_far"].append(avg_train_far)
        history["val_far"].append(val_metrics["far"])

        history["train_pod"].append(avg_train_pod)
        history["val_pod"].append(val_metrics["pod"])

        history["train_bs"].append(avg_train_bs)
        history["val_bs"].append(val_metrics["bs"])

        history["train_bss"].append(avg_train_bss)
        history["val_bss"].append(val_metrics["bss"])

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}"
        )
        print(
            f"            Train CSI: {avg_train_csi:.4f} | Val CSI: {val_metrics['csi']:.4f}"
        )
        print(
            f"            Train FAR: {avg_train_far:.4f} | Val FAR: {val_metrics['far']:.4f}"
        )
        print(
            f"            Train POD: {avg_train_pod:.4f} | Val POD: {val_metrics['pod']:.4f}"
        )
        print(
            f"            Train BS:  {avg_train_bs:.4f} | Val BS:  {val_metrics['bs']:.4f}"
        )
        print(
            f"            Train BSS: {avg_train_bss:.4f} | Val BSS: {val_metrics['bss']:.4f}"
        )

    return history
