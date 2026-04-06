import numpy as np
from scipy.ndimage import uniform_filter


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


# ── FSS ───────────────────────────────────────────────────────────────────────

def calc_fss_from_arrays(pred_probs_all, true_binary_all, threshold=0.5,
                          neighborhood_sizes=None):
    """
    Fractions Skill Score (FSS) at multiple spatial neighborhood sizes.

    Parameters
    ----------
    pred_probs_all    : np.ndarray [N, H, W]  predicted probabilities
    true_binary_all   : np.ndarray [N, H, W]  binary observations (0/1)
    threshold         : float  probability cut-off to binarise predictions
    neighborhood_sizes: list[int]  n×n window sizes, e.g. [1, 3, 5, 7, 9]
                        With ERA5 ~0.25°/cell each step ≈ 25 km.

    Returns
    -------
    dict {n: fss_value}  — FSS pooled across all N samples for each window size.

    Reference: Roberts & Lean (2008), MWR.
    Useful-skill threshold: FSS > 0.5 + f_o/2  (f_o = observed base rate).
    """
    if neighborhood_sizes is None:
        neighborhood_sizes = [1, 3, 5, 7, 9]

    binary_pred = (pred_probs_all >= threshold).astype(np.float32)
    true_bin    = true_binary_all.astype(np.float32)

    results = {}
    for n in neighborhood_sizes:
        fbs_sum     = 0.0   # Σ (frac_pred - frac_obs)²
        fbs_ref_sum = 0.0   # Σ frac_pred² + Σ frac_obs²

        for i in range(binary_pred.shape[0]):
            fp = uniform_filter(binary_pred[i], size=n, mode='constant')
            fo = uniform_filter(true_bin[i],    size=n, mode='constant')

            fbs_sum     += np.sum((fp - fo) ** 2)
            fbs_ref_sum += np.sum(fp ** 2) + np.sum(fo ** 2)

        if fbs_ref_sum == 0.0:
            results[n] = 1.0   # both all-zero → perfect agreement
        else:
            results[n] = float(1.0 - fbs_sum / fbs_ref_sum)

    return results


def fss_useful_threshold(true_binary_all):
    """
    Return the 'useful skill' FSS threshold: 0.5 + f_o / 2
    where f_o is the observed lightning base rate.
    Any FSS above this line is considered skillful (Roberts & Lean 2008).
    """
    f_o = float(np.mean(true_binary_all > 0))
    return 0.5 + f_o / 2.0


# ── Reliability ───────────────────────────────────────────────────────────────

def calc_reliability_data(pred_probs_all, true_binary_all, n_bins=10):
    """
    Compute data for a reliability (attributes) diagram.

    Parameters
    ----------
    pred_probs_all  : np.ndarray, any shape — predicted probabilities
    true_binary_all : np.ndarray, same shape — binary observations
    n_bins          : int  number of probability bins

    Returns
    -------
    dict with:
        bin_centers        : centre of each probability bin  [n_bins]
        obs_frequency      : mean observed frequency per bin [n_bins]
        forecast_frequency : fraction of all forecasts in each bin (sharpness)
        bin_counts         : raw count per bin
        base_rate          : overall observed lightning frequency (climatology line)
    """
    probs_flat = pred_probs_all.flatten().astype(np.float64)
    obs_flat   = true_binary_all.flatten().astype(np.float64)

    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    obs_frequency = np.full(n_bins, np.nan)
    bin_counts    = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include right edge in the last bin
        mask = (probs_flat >= lo) & (probs_flat < hi) if i < n_bins - 1 \
               else (probs_flat >= lo) & (probs_flat <= hi)
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            obs_frequency[i] = obs_flat[mask].mean()

    total = bin_counts.sum()
    forecast_frequency = bin_counts / total if total > 0 else bin_counts

    return {
        'bin_centers':        bin_centers,
        'obs_frequency':      obs_frequency,
        'forecast_frequency': forecast_frequency,
        'bin_counts':         bin_counts,
        'base_rate':          float(obs_flat.mean()),
    }
