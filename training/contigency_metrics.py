import numpy as np


def calc_csi(binary_preds, true_y):
    """Critical Success Index (CSI)"""
    true_pos = np.sum((binary_preds == 1) & (true_y == 1))
    false_neg = np.sum((binary_preds == 0) & (true_y == 1))
    false_pos = np.sum((binary_preds == 1) & (true_y == 0))

    denominator = true_pos + false_neg + false_pos
    if denominator == 0:
        return 0.0

    return true_pos / denominator


def calc_far(binary_preds, true_y):
    """False Alarm Ratio (FAR)"""
    true_pos = np.sum((binary_preds == 1) & (true_y == 1))
    false_pos = np.sum((binary_preds == 1) & (true_y == 0))

    denominator = true_pos + false_pos
    if denominator == 0:
        return 0.0

    return false_pos / denominator


def calc_pod(binary_preds, true_y):
    """Probability Of Detection (POD)"""
    true_pos = np.sum((binary_preds == 1) & (true_y == 1))
    false_neg = np.sum((binary_preds == 0) & (true_y == 1))

    denominator = true_pos + false_neg
    if denominator == 0:
        return 0.0

    return true_pos / denominator


def calc_bs(pred_probs, true_y):
    """Brier Score (BS)"""
    return np.mean((pred_probs - true_y) ** 2)


def calc_bss(pred_probs, true_y):
    """Brier Skill Score (BSS) relative to climatology"""
    bs = calc_bs(pred_probs, true_y)

    climatology = np.mean(true_y)
    bs_ref = np.mean((climatology - true_y) ** 2)

    if bs_ref == 0:
        return 0.0

    return 1 - (bs / bs_ref)
