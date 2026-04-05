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


def _build_impossible_mask(
    physics_raw,
    physics_var_names,
    cape_min,
    ki_min,
    tciw_min,
    crr_min,
    w500_max,
    r700_min,
    r850_min,
):
    """
    Build a boolean CPU tensor [B, 1, H, W] where True = lightning physically impossible.

    physics_raw       : [B, K, H, W] tensor on CPU — raw (un-normalised) values of
                        the K physics variables listed in physics_var_names.
    physics_var_names : list[str] of length K — variable name for each channel in physics_raw.

    Rules (each threshold may be None → that constraint is skipped):
      cape  < cape_min   → impossible   (thermodynamic instability required)
      kx    < ki_min     → impossible   (K-Index thunderstorm potential)
      tciw  < tciw_min   → impossible   (ice water for charge separation)
      crr   <= crr_min   → impossible   (active convective core: CRR must be > 0)
      w_500 >= w500_max  → impossible   (ERA5 omega: negative = upward; >= -0.1 = subsidence)
      r_700 < r700_min   → impossible   (mid-level moisture)
      r_850 < r850_min   → impossible   (low-level moisture)
    """
    idx = {name: i for i, name in enumerate(physics_var_names)}
    B, _, H, W = physics_raw.shape
    impossible = torch.zeros(B, 1, H, W, dtype=torch.bool)

    def chan(name):
        """Extract [B, 1, H, W] slice for variable `name` if present."""
        if name in idx:
            return physics_raw[:, idx[name]:idx[name] + 1, :, :]
        return None

    if cape_min is not None:
        c = chan('cape')
        if c is not None:
            impossible |= (c < cape_min)

    if ki_min is not None:
        c = chan('kx')
        if c is not None:
            impossible |= (c < ki_min)

    if tciw_min is not None:
        c = chan('tciw')
        if c is not None:
            impossible |= (c < tciw_min)

    if crr_min is not None:
        c = chan('crr')
        if c is not None:
            # CRR must be strictly greater than crr_min (active convective core)
            impossible |= (c <= crr_min)

    if w500_max is not None:
        c = chan('w_500')
        if c is not None:
            # ERA5 omega: negative = upward motion; >= w500_max means subsidence / no updraft
            impossible |= (c >= w500_max)

    if r700_min is not None:
        c = chan('r_700')
        if c is not None:
            impossible |= (c < r700_min)

    if r850_min is not None:
        c = chan('r_850')
        if c is not None:
            impossible |= (c < r850_min)

    return impossible  # bool [B, 1, H, W] on CPU


def _apply_physics_mask(binary_preds, probs, impossible, device):
    """
    Zero out predictions in cells where lightning is physically impossible.
    impossible : bool [B, 1, H, W] CPU tensor from _build_impossible_mask.
    """
    possible = (~impossible).int().to(device)
    binary_preds = binary_preds * possible
    probs = probs * possible.float()
    return binary_preds, probs


def _physics_penalty(logits, impossible, device):
    """
    Soft physics penalty: penalise any positive predicted probability in cells
    where lightning is physically impossible.  Differentiable — added to main loss.
    impossible : bool [B, 1, H, W] CPU tensor from _build_impossible_mask.
    """
    impossible_float = impossible.float().to(device)
    probs = torch.sigmoid(logits)
    penalty = (probs * impossible_float).mean()
    return penalty


def evaluate_model(
    model,
    data_loader,
    criterion,
    device,
    decision_threshold=0.5,
    use_physics_loss=False,
    physics_weight=1.0,
    physics_var_names=None,
    cape_min=None,
    ki_min=None,
    tciw_min=None,
    crr_min=None,
    w500_max=None,
    r700_min=None,
    r850_min=None,
):
    """
    Evaluate model on a loader and return averaged loss + metrics for one epoch.

    When use_physics_loss=True the DataLoader must yield (xb, yb, physics_raw) 3-tuples,
    where physics_raw is [B, K, H, W] with raw (un-normalised) values for the variables
    listed in physics_var_names.

    Two physics effects are applied:
      - Soft penalty added to loss (differentiable training signal)
      - Hard mask applied to predicted probabilities / binary decisions for metrics
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
        for batch in tqdm(data_loader, desc="Validation", unit="batch", leave=False):
            xb, yb = batch[0], batch[1]
            physics_raw_b = batch[2] if len(batch) > 2 else None

            xb = xb.to(device)
            yb = yb.to(device)
            xb, yb = _sanitize_batch(xb, yb)
            yb = (yb > 0).float()

            logits = model(xb)
            main_loss = criterion(logits, yb)

            impossible = None
            if use_physics_loss and physics_raw_b is not None and physics_var_names:
                impossible = _build_impossible_mask(
                    physics_raw_b, physics_var_names,
                    cape_min, ki_min, tciw_min, crr_min, w500_max, r700_min, r850_min,
                )
                if physics_weight > 0:
                    penalty = _physics_penalty(logits, impossible, device)
                    loss = main_loss + physics_weight * penalty
                else:
                    loss = main_loss
            else:
                loss = main_loss

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

            # Hard physics mask: zero out predictions in physically impossible cells
            if impossible is not None:
                binary_preds, probs = _apply_physics_mask(binary_preds, probs, impossible, device)

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
    scheduler=None,
    use_physics_loss=False,
    physics_weight=1.0,
    physics_var_names=None,
    cape_min=None,
    ki_min=None,
    tciw_min=None,
    crr_min=None,
    w500_max=None,
    r700_min=None,
    r850_min=None,
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

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
            xb, yb = batch[0], batch[1]
            physics_raw_b = batch[2] if len(batch) > 2 else None

            xb = xb.to(device)
            yb = yb.to(device)
            xb, yb = _sanitize_batch(xb, yb)
            yb = (yb > 0).float()

            optimizer.zero_grad()

            logits = model(xb)
            main_loss = criterion(logits, yb)

            impossible = None
            if use_physics_loss and physics_raw_b is not None and physics_var_names:
                impossible = _build_impossible_mask(
                    physics_raw_b, physics_var_names,
                    cape_min, ki_min, tciw_min, crr_min, w500_max, r700_min, r850_min,
                )
                if physics_weight > 0:
                    penalty = _physics_penalty(logits, impossible, device)
                    loss = main_loss + physics_weight * penalty
                else:
                    loss = main_loss
            else:
                loss = main_loss

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

            # Hard physics mask applied to training metrics only (not to gradients)
            if impossible is not None:
                binary_preds, probs = _apply_physics_mask(binary_preds, probs, impossible, device)

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
            use_physics_loss=use_physics_loss,
            physics_weight=physics_weight,
            physics_var_names=physics_var_names,
            cape_min=cape_min,
            ki_min=ki_min,
            tciw_min=tciw_min,
            crr_min=crr_min,
            w500_max=w500_max,
            r700_min=r700_min,
            r850_min=r850_min,
        )

        # Step scheduler based on val CSI
        if scheduler is not None:
            scheduler.step(val_metrics["csi"])

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
