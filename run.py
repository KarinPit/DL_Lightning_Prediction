"""
run.py — Main training entry point (Zarr-backed pipeline)
==========================================================
Reads directly from ERA5.zarr + Lightning.zarr. No NC files, no tensor caches.

Usage:
    python run.py                    # train with defaults from experiment.py
    python run.py --hours 10 11 12 13 14 15 16 17 18  # daytime only
    python run.py --train-end 2023-06-30 --val-start 2023-07-01  # custom split
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.experiment import ZARR_CONFIG, MODEL_CONFIG, RUN_CONFIG
from config.schema import ZarrConfig
from data.zarr_dataset import build_datasets
from models.unet import UNet
from training.train import train_model


# ─────────────────────────────────────────────────────────────────────────────
#  CLI overrides
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train UNet lightning predictor on Zarr data')
    p.add_argument('--train-start',  default=None, help='Override train_start (YYYY-MM-DD)')
    p.add_argument('--train-end',    default=None, help='Override train_end   (YYYY-MM-DD)')
    p.add_argument('--val-start',    default=None, help='Override val_start   (YYYY-MM-DD)')
    p.add_argument('--val-end',      default=None, help='Override val_end     (YYYY-MM-DD)')
    p.add_argument(
        '--hours', nargs='+', type=int, default=None,
        metavar='H',
        help='UTC hours to use as prediction targets (0–23). '
             'Default: all 24 hours. E.g. --hours 10 11 12 13 14 15 16 17 18'
    )
    p.add_argument('--lookback',  type=int, default=None, help='Lookback hours (default 3)')
    p.add_argument('--epochs',    type=int, default=None, help='Override num_epochs')
    p.add_argument('--batch',     type=int, default=None, help='Override batch_size')
    p.add_argument('--workers',   type=int, default=4,    help='DataLoader num_workers')
    p.add_argument('--save-dir',  default='/home/ec2-user/thesis-bucket/Models',
                   help='Directory to save trained model')
    return p.parse_args()


def apply_overrides(zarr_cfg, model_cfg, args):
    """Return updated config objects based on CLI arguments."""
    # Build a new ZarrConfig with any CLI overrides applied
    overrides = {}
    if args.train_start: overrides['train_start']  = args.train_start
    if args.train_end:   overrides['train_end']    = args.train_end
    if args.val_start:   overrides['val_start']    = args.val_start
    if args.val_end:     overrides['val_end']      = args.val_end
    if args.hours:       overrides['hours_of_day'] = args.hours
    if args.lookback:    overrides['lookback_hours'] = args.lookback

    if overrides:
        # dataclass is frozen → rebuild with updated fields
        zarr_cfg = ZarrConfig(
            era5_zarr      = zarr_cfg.era5_zarr,
            lightning_zarr = zarr_cfg.lightning_zarr,
            atm_params     = zarr_cfg.atm_params,
            train_start    = overrides.get('train_start',    zarr_cfg.train_start),
            train_end      = overrides.get('train_end',      zarr_cfg.train_end),
            val_start      = overrides.get('val_start',      zarr_cfg.val_start),
            val_end        = overrides.get('val_end',        zarr_cfg.val_end),
            test_start     = zarr_cfg.test_start,
            test_end       = zarr_cfg.test_end,
            lookback_hours = overrides.get('lookback_hours', zarr_cfg.lookback_hours),
            hours_of_day   = overrides.get('hours_of_day',   zarr_cfg.hours_of_day),
        )

    # ModelConfig overrides (epochs, batch)
    if args.epochs or args.batch:
        from dataclasses import replace
        model_cfg = replace(
            model_cfg,
            **(({'num_epochs':  args.epochs} if args.epochs else {})),
            **(({'batch_size':  args.batch}  if args.batch  else {})),
        )

    return zarr_cfg, model_cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    zarr_cfg, model_cfg = apply_overrides(ZARR_CONFIG, MODEL_CONFIG, args)

    # ── Reproducibility ───────────────────────────────────────────────────────
    if RUN_CONFIG.use_seed:
        torch.manual_seed(RUN_CONFIG.seed_value)
        np.random.seed(RUN_CONFIG.seed_value)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'Device        : {device}')
    print(f'Train         : {zarr_cfg.train_start} → {zarr_cfg.train_end}')
    print(f'Val           : {zarr_cfg.val_start}   → {zarr_cfg.val_end}')
    hours_label = sorted(zarr_cfg.hours_of_day) if zarr_cfg.hours_of_day else 'all 24'
    print(f'Hours of day  : {hours_label}')
    print(f'Lookback      : {zarr_cfg.lookback_hours}h')
    print(f'Variables     : {len(zarr_cfg.atm_params)} → {zarr_cfg.atm_params[:4]}...')
    n_channels = zarr_cfg.expected_input_channels
    print(f'Input channels: {n_channels}  ({zarr_cfg.lookback_hours}h × {len(zarr_cfg.atm_params)} vars)')
    print(f'{"="*60}\n')

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(zarr_cfg, model_cfg)

    # num_workers=0: data is preloaded in RAM — no I/O during training,
    # so multiprocessing workers just add overhead from copying large arrays.
    train_loader = DataLoader(
        train_ds,
        batch_size  = model_cfg.batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = (device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = model_cfg.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device.type == 'cuda'),
    )

    print(f'\nTrain batches: {len(train_loader)}  |  Val batches: {len(val_loader)}\n')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(n_channels=n_channels, n_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'UNet  ({total_params:,} trainable parameters)\n')

    # ── Loss / optimizer / scheduler ─────────────────────────────────────────
    pos_weight = torch.tensor([model_cfg.pos_weight]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=model_cfg.learning_rate)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # ── Training ──────────────────────────────────────────────────────────────
    history = train_model(
        model             = model,
        train_loader      = train_loader,
        val_loader        = val_loader,
        optimizer         = optimizer,
        criterion         = criterion,
        num_epochs        = model_cfg.num_epochs,
        device            = device,
        decision_threshold= model_cfg.decision_threshold,
        scheduler         = scheduler,
        use_physics_loss  = model_cfg.use_physics_loss,
        use_physics_mask  = model_cfg.use_physics_mask,
        physics_weight    = model_cfg.physics_weight,
        physics_var_names = list(zarr_cfg.atm_params),
        cape_min          = model_cfg.cape_min,
        ki_min            = model_cfg.ki_min,
        tciw_min          = model_cfg.tciw_min,
        crr_min           = model_cfg.crr_min,
        w500_max          = model_cfg.w500_max,
        r700_min          = model_cfg.r700_min,
        r850_min          = model_cfg.r850_min,
    )

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    hours_tag = (
        f"h{'_'.join(str(h) for h in sorted(zarr_cfg.hours_of_day))}"
        if zarr_cfg.hours_of_day else 'hall'
    )
    model_name = (
        f"unet_lb{zarr_cfg.lookback_hours}"
        f"_{zarr_cfg.train_start}_{zarr_cfg.train_end}"
        f"_{hours_tag}"
        f"_{model_cfg.num_epochs}ep.pth"
    )
    save_path = os.path.join(args.save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f'\nModel saved → {save_path}')

    # Also save norm stats alongside the model (needed for inference)
    import json
    stats_path = save_path.replace('.pth', '_norm_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(train_ds.norm_stats, f, indent=2)
    print(f'Norm stats  → {stats_path}')

    return history


if __name__ == '__main__':
    main()
