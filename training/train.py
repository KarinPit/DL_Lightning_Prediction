import numpy as np
import torch

from training.contigency_metrics import calc_bs, calc_bss, calc_csi, calc_far, calc_pod


def _safe_mean(values):
    """Mean that safely handles empty lists."""
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


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

    csi_vals = []
    far_vals = []
    pod_vals = []
    bs_vals = []
    bss_vals = []

    with torch.no_grad():
        for xb, yb in data_loader:
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

            # Make sure truth is binary for contingency metrics

            csi_vals.append(calc_csi(binary_preds_np, yb_np))
            far_vals.append(calc_far(binary_preds_np, yb_np))
            pod_vals.append(calc_pod(binary_preds_np, yb_np))
            bs_vals.append(calc_bs(probs_np, yb_np))
            bss_vals.append(calc_bss(probs_np, yb_np))

    avg_loss = total_loss / total_samples if total_samples > 0 else np.nan

    return {
        "loss": avg_loss,
        "csi": _safe_mean(csi_vals),
        "far": _safe_mean(far_vals),
        "pod": _safe_mean(pod_vals),
        "bs": _safe_mean(bs_vals),
        "bss": _safe_mean(bss_vals),
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

        train_csi_vals = []
        train_far_vals = []
        train_pod_vals = []
        train_bs_vals = []
        train_bss_vals = []

        print(f"Starting Epoch {epoch}...")

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb, yb = _sanitize_batch(xb, yb)
            yb = (yb > 0).float()

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)
            # loss = _safe_loss(loss)
            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

            probs = torch.sigmoid(logits)
            print(probs.min(), probs.max())
            binary_preds = (probs > decision_threshold).int()

            probs_np = probs.detach().cpu().numpy()
            binary_preds_np = binary_preds.detach().cpu().numpy()
            yb_np = yb.detach().cpu().numpy()

            train_csi_vals.append(calc_csi(binary_preds_np, yb_np))
            train_far_vals.append(calc_far(binary_preds_np, yb_np))
            train_pod_vals.append(calc_pod(binary_preds_np, yb_np))
            train_bs_vals.append(calc_bs(probs_np, yb_np))
            train_bss_vals.append(calc_bss(probs_np, yb_np))

        # Average training metrics over the epoch
        avg_train_loss = (
            total_train_loss / total_train_samples if total_train_samples > 0 else 0.0
        )
        avg_train_csi = _safe_mean(train_csi_vals)
        avg_train_far = _safe_mean(train_far_vals)
        avg_train_pod = _safe_mean(train_pod_vals)
        avg_train_bs = _safe_mean(train_bs_vals)
        avg_train_bss = _safe_mean(train_bss_vals)

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
