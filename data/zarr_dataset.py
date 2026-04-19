"""
zarr_dataset.py — PyTorch Dataset backed by ERA5.zarr + Lightning.zarr
=======================================================================
Replaces the old NC-file + tensor-cache (X_final.pt / Y_final.pt) approach.
Data is loaded lazily from the two Zarr stores in the S3 bucket.

Each item is a 3-tuple  (X, y, physics_raw):
  X            : [lookback*n_vars, lat, lon]  — normalised input
  y            : [1, lat, lon]                 — binary lightning at time T
  physics_raw  : [n_vars, lat, lon]            — raw (un-normalised) values at T
                 passed to the physics constraint checks in train.py

Usage:
    from data.zarr_dataset import ZarrLightningDataset, build_datasets

    train_ds, val_ds = build_datasets(ZARR_CONFIG, MODEL_CONFIG)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2)
"""

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

NORMALIZATION_EPS = 1e-6


class ZarrLightningDataset(Dataset):
    """
    Parameters
    ----------
    era5_zarr_path : str
        Path to ERA5.zarr store (on bucket or local).
    lightning_zarr_path : str
        Path to Lightning.zarr store.
    atm_params : list[str]
        ERA5 variable names to use as input features. Must match variable
        names inside ERA5.zarr exactly.
    start_date : str
        Start of date range, e.g. "2023-01-01".
    end_date : str
        End of date range (inclusive), e.g. "2023-10-31".
    lookback_hours : int
        Number of consecutive hours to stack as input channels.
        lookback=3  → channels are [T-2, T-1, T], shape (3*n_vars, lat, lon).
    hours_of_day : list[int] or None
        Which UTC hours to use as prediction targets (0–23).
        None = all 24 hours.
        Only the TARGET hour is filtered. Lookback context hours are always
        included regardless of this filter.
        Examples:
          list(range(8, 21))  → 08:00–20:00 UTC (daytime only)
          [0, 6, 12, 18]      → synoptic hours only
    norm_stats : dict or None
        {var_name: {'mean': float, 'std': float}}
        Compute on the training split, then pass to val/test via set_norm_stats().
        If None, X is returned un-normalised.
    clip_value : float or None
        After normalisation, clamp X to [-clip_value, +clip_value].
        Mirrors ModelConfig.normalization_clip_value.
    """

    def __init__(
        self,
        era5_zarr_path,
        lightning_zarr_path,
        atm_params,
        start_date,
        end_date,
        lookback_hours=3,
        hours_of_day=None,
        norm_stats=None,
        clip_value=None,
    ):
        self.atm_params     = list(atm_params)
        self.lookback_hours = lookback_hours
        self.hours_of_day   = set(hours_of_day) if hours_of_day is not None else None
        self.norm_stats     = norm_stats
        self.clip_value     = clip_value

        # Open stores lazily — no data read yet
        self.ds_era5      = xr.open_zarr(era5_zarr_path)
        self.ds_lightning = xr.open_zarr(lightning_zarr_path)

        # ── Filter ERA5 time axis to [start_date, end_date] ──────────────────
        all_times = pd.DatetimeIndex(self.ds_era5.time.values)
        mask = (
            (all_times >= pd.Timestamp(start_date)) &
            (all_times <= pd.Timestamp(end_date) + pd.Timedelta(hours=23))
        )
        self._times   = all_times[mask]           # DatetimeIndex (filtered)
        self._indices = np.where(mask)[0]         # integer positions in ERA5 time axis

        # ── Build list of valid target indices ───────────────────────────────
        self._valid = self._build_index()

        n = len(self._valid)
        hour_label = (
            f"{sorted(hours_of_day)}" if hours_of_day is not None else "all"
        )
        if n > 0:
            t0 = self._times[self._valid[0]]
            t1 = self._times[self._valid[-1]]
            print(
                f"  ZarrDataset [{start_date} → {end_date}]  "
                f"{n} windows  hours={hour_label}  "
                f"({t0.strftime('%Y-%m-%d %H')}h → {t1.strftime('%Y-%m-%d %H')}h)"
            )
        else:
            print(f"  WARNING: 0 valid windows for {start_date} → {end_date}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Index building
    # ──────────────────────────────────────────────────────────────────────────

    def _build_index(self):
        """
        Return a list of local indices i (into self._times / self._indices)
        that represent valid prediction targets.

        A valid target at local index i requires:
          1. times[i].hour is in hours_of_day  (if the filter is active)
          2. The lookback window [i-lookback+1 … i] consists of exactly
             consecutive hourly timestamps (no gaps).
        """
        times = self._times
        valid = []
        for i in range(self.lookback_hours - 1, len(times)):
            # Apply hours-of-day filter to the TARGET hour only
            if self.hours_of_day is not None and times[i].hour not in self.hours_of_day:
                continue
            # Verify the lookback window is gap-free
            window = times[i - self.lookback_hours + 1 : i + 1]
            consecutive = all(
                (window[j + 1] - window[j]).total_seconds() == 3600
                for j in range(self.lookback_hours - 1)
            )
            if consecutive:
                valid.append(i)
        return valid

    # ──────────────────────────────────────────────────────────────────────────
    #  Dataset interface
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._valid)

    def __getitem__(self, idx):
        i = self._valid[idx]

        # ERA5 global integer indices for each hour in the lookback window
        window_global = [
            int(self._indices[k])
            for k in range(i - self.lookback_hours + 1, i + 1)
        ]

        # ── Load ERA5 for each hour in the window ─────────────────────────────
        hour_arrays = []
        for gi in window_global:
            # Load all requested variables at this timestep in a single Zarr read
            snap = self.ds_era5[self.atm_params].isel(time=gi).load()
            arr  = np.stack(
                [snap[v].values.astype(np.float32) for v in self.atm_params],
                axis=0,
            )   # (n_vars, lat, lon)
            hour_arrays.append(arr)

        # (lookback * n_vars, lat, lon)  — chronological order T-(L-1), …, T-1, T
        x_np = np.concatenate(hour_arrays, axis=0)
        x    = torch.from_numpy(x_np)

        # Raw (un-normalised) snapshot at time T — used by physics constraints
        physics_raw = torch.from_numpy(hour_arrays[-1].copy())  # (n_vars, lat, lon)

        # ── Normalise ─────────────────────────────────────────────────────────
        if self.norm_stats is not None:
            n_vars = len(self.atm_params)
            for t in range(self.lookback_hours):
                for v, var in enumerate(self.atm_params):
                    c    = t * n_vars + v
                    mean = self.norm_stats[var]['mean']
                    std  = self.norm_stats[var]['std']
                    x[c] = (x[c] - mean) / (std + NORMALIZATION_EPS)
            if self.clip_value is not None:
                x = torch.clamp(x, -self.clip_value, self.clip_value)

        # Replace any remaining NaN/Inf with 0 (edge cases in reanalysis data)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Lightning target at time T (binary) ───────────────────────────────
        gi_target = int(self._indices[i])
        lightning  = self.ds_lightning['entln_count'].values[gi_target]  # (lat, lon)
        y = torch.from_numpy(
            (lightning > 0).astype(np.float32)
        ).unsqueeze(0)   # (1, lat, lon)

        return x, y, physics_raw

    # ──────────────────────────────────────────────────────────────────────────
    #  Normalisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def compute_norm_stats(self):
        """
        Compute per-variable mean and std over the full date range of this dataset.

        Reads each variable from ERA5.zarr in one bulk load (no per-item loop),
        so this is fast even for a full year.

        Call on the TRAINING dataset only, then pass the result to val/test
        datasets via set_norm_stats().

        Returns
        -------
        dict : {var_name: {'mean': float, 'std': float}}
        """
        print("Computing normalisation statistics...")
        stats = {}
        time_slice = self._indices  # integer array of time positions

        for var in self.atm_params:
            # Load entire variable for the training time range at once
            data = (
                self.ds_era5[var]
                .isel(time=time_slice)
                .values
                .astype(np.float64)
            )
            finite = data[np.isfinite(data)]
            mean   = float(np.mean(finite))
            std    = float(np.std(finite))
            std    = max(std, NORMALIZATION_EPS)
            stats[var] = {'mean': mean, 'std': std}
            print(f"    {var:15s}  mean={mean:10.4f}  std={std:10.4f}")

        return stats

    def set_norm_stats(self, stats):
        """Apply normalisation stats (from training set) to this dataset."""
        self.norm_stats = stats


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(zarr_config, model_config):
    """
    Build train and val ZarrLightningDataset from experiment config objects.
    Computes normalisation stats from training data and applies them to both.

    Parameters
    ----------
    zarr_config  : config.schema.ZarrConfig
    model_config : config.schema.ModelConfig

    Returns
    -------
    (train_dataset, val_dataset)
    """
    clip = (
        model_config.normalization_clip_value
        if model_config.clip_after_normalization
        else None
    )

    print("Building training dataset...")
    train_ds = ZarrLightningDataset(
        era5_zarr_path      = zarr_config.era5_zarr,
        lightning_zarr_path = zarr_config.lightning_zarr,
        atm_params          = zarr_config.atm_params,
        start_date          = zarr_config.train_start,
        end_date            = zarr_config.train_end,
        lookback_hours      = zarr_config.lookback_hours,
        hours_of_day        = zarr_config.hours_of_day,
        clip_value          = clip,
    )

    # Stats from training set only
    norm_stats = train_ds.compute_norm_stats()
    train_ds.set_norm_stats(norm_stats)

    print("\nBuilding validation dataset...")
    val_ds = ZarrLightningDataset(
        era5_zarr_path      = zarr_config.era5_zarr,
        lightning_zarr_path = zarr_config.lightning_zarr,
        atm_params          = zarr_config.atm_params,
        start_date          = zarr_config.val_start,
        end_date            = zarr_config.val_end,
        lookback_hours      = zarr_config.lookback_hours,
        hours_of_day        = zarr_config.hours_of_day,
        norm_stats          = norm_stats,
        clip_value          = clip,
    )

    return train_ds, val_ds
