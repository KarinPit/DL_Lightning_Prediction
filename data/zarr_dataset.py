"""
zarr_dataset.py — PyTorch Dataset backed by ERA5.zarr + Lightning.zarr
=======================================================================
Each item is a 3-tuple (X, y, physics_raw):
  X            : [lookback*n_vars, lat, lon]  — normalised input
  y            : [1, lat, lon]                 — binary lightning at time T
  physics_raw  : [n_vars, lat, lon]            — raw (un-normalised) values at T

Preloading (default):
  All ERA5 and lightning data for the date range is loaded into RAM once at
  dataset creation. __getitem__ then does pure numpy slicing — no disk I/O
  during training. Use num_workers=0 in the DataLoader (data already in RAM).
"""

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

NORMALIZATION_EPS = 1e-6


class ZarrLightningDataset(Dataset):

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
        preload=True,          # load all data into RAM at init (fast training)
    ):
        self.atm_params     = list(atm_params)
        self.lookback_hours = lookback_hours
        self.hours_of_day   = set(hours_of_day) if hours_of_day is not None else None
        self.norm_stats     = norm_stats
        self.clip_value     = clip_value

        # Open stores (lazy — used only during preload / norm stats)
        ds_era5      = xr.open_zarr(era5_zarr_path)
        ds_lightning = xr.open_zarr(lightning_zarr_path)

        # Filter time axis to [start_date, end_date]
        all_times = pd.DatetimeIndex(ds_era5.time.values)
        mask = (
            (all_times >= pd.Timestamp(start_date)) &
            (all_times <= pd.Timestamp(end_date) + pd.Timedelta(hours=23))
        )
        self._times   = all_times[mask]
        self._indices = np.where(mask)[0]

        # Build valid window index
        self._valid = self._build_index()

        n          = len(self._valid)
        hour_label = sorted(hours_of_day) if hours_of_day is not None else "all"
        if n > 0:
            t0 = self._times[self._valid[0]]
            t1 = self._times[self._valid[-1]]
            print(f"  ZarrDataset [{start_date} → {end_date}]  "
                  f"{n} windows  hours={hour_label}  "
                  f"({t0.strftime('%Y-%m-%d %H')}h → {t1.strftime('%Y-%m-%d %H')}h)")
        else:
            print(f"  WARNING: 0 valid windows for {start_date} → {end_date}")

        # Preload into RAM
        self._era5_cache      = None
        self._lightning_cache = None
        if preload:
            self._preload(ds_era5, ds_lightning)

        ds_era5.close()
        ds_lightning.close()

    # ──────────────────────────────────────────────────────────────────────────

    def _build_index(self):
        times = self._times
        valid = []
        for i in range(self.lookback_hours - 1, len(times)):
            if self.hours_of_day is not None and times[i].hour not in self.hours_of_day:
                continue
            window = times[i - self.lookback_hours + 1 : i + 1]
            if all((window[j+1] - window[j]).total_seconds() == 3600
                   for j in range(self.lookback_hours - 1)):
                valid.append(i)
        return valid

    def _preload(self, ds_era5, ds_lightning):
        """Load full date range into two numpy arrays in RAM."""
        n_times = len(self._indices)
        n_vars  = len(self.atm_params)

        print(f"  Preloading {n_times} timesteps × {n_vars} vars into RAM...", flush=True)

        # ERA5 — shape (n_times, n_vars, lat, lon)
        ds_sub = ds_era5[self.atm_params].isel(time=self._indices).load()
        self._era5_cache = np.stack(
            [ds_sub[v].values.astype(np.float32) for v in self.atm_params],
            axis=1,
        )  # (n_times, n_vars, lat, lon)

        # Lightning — shape (n_times, lat, lon)
        self._lightning_cache = (
            ds_lightning['entln_count']
            .isel(time=self._indices)
            .values
            .astype(np.float32)
        )

        era5_gb  = self._era5_cache.nbytes / 1e9
        light_mb = self._lightning_cache.nbytes / 1e6
        print(f"  Preloaded: ERA5 {self._era5_cache.shape}  ({era5_gb:.2f} GB)  |  "
              f"Lightning {self._lightning_cache.shape}  ({light_mb:.1f} MB)")

    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._valid)

    def __getitem__(self, idx):
        i            = self._valid[idx]
        window_local = range(i - self.lookback_hours + 1, i + 1)

        if self._era5_cache is not None:
            # Fast path: slice from RAM
            hour_arrays = [self._era5_cache[k] for k in window_local]
            lightning   = self._lightning_cache[i]
        else:
            # Slow fallback: read from Zarr (preload=False)
            ds_era5      = xr.open_zarr.__self__ if False else None  # not used
            raise RuntimeError("preload=False path not supported. Use preload=True.")

        # Stack lookback: (lookback*n_vars, lat, lon)
        x_np        = np.concatenate(hour_arrays, axis=0)
        x           = torch.from_numpy(x_np.copy())
        physics_raw = torch.from_numpy(hour_arrays[-1].copy())  # raw T snapshot

        # Normalise
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

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Lightning target (binary)
        y = torch.from_numpy(
            (lightning > 0).astype(np.float32)
        ).unsqueeze(0)  # (1, lat, lon)

        return x, y, physics_raw

    # ──────────────────────────────────────────────────────────────────────────

    def compute_norm_stats(self):
        """Compute per-variable mean/std. Uses preloaded cache if available."""
        print("Computing normalisation statistics...")
        stats = {}

        if self._era5_cache is not None:
            # Fast: stats from RAM cache — shape (n_times, n_vars, lat, lon)
            for v_idx, var in enumerate(self.atm_params):
                data   = self._era5_cache[:, v_idx, :, :].astype(np.float64)
                finite = data[np.isfinite(data)]
                mean   = float(np.mean(finite))
                std    = float(np.std(finite))
                std    = max(std, NORMALIZATION_EPS)
                stats[var] = {'mean': mean, 'std': std}
                print(f"    {var:15s}  mean={mean:10.4f}  std={std:10.4f}")
        else:
            # Slow: load from Zarr per variable
            ds_era5 = xr.open_zarr(self.era5_zarr_path)
            for var in self.atm_params:
                data   = ds_era5[var].isel(time=self._indices).values.astype(np.float64)
                finite = data[np.isfinite(data)]
                mean   = float(np.mean(finite))
                std    = float(np.std(finite))
                std    = max(std, NORMALIZATION_EPS)
                stats[var] = {'mean': mean, 'std': std}
                print(f"    {var:15s}  mean={mean:10.4f}  std={std:10.4f}")
            ds_era5.close()

        return stats

    def set_norm_stats(self, stats):
        self.norm_stats = stats


# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(zarr_config, model_config):
    clip = (
        model_config.normalization_clip_value
        if model_config.clip_after_normalization else None
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
        preload             = True,
    )

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
        preload             = True,
    )

    return train_ds, val_ds
