from config.schema import ModelConfig, RunConfig, ZarrConfig

# ── Pressure levels — full ERA5 set (all 37 standard levels) ─────────────
_KEY_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 775, 800, 825,
    850, 875, 900, 925, 950, 975, 1000,
]

# Pressure-level variable short names (must match ERA5.zarr variable names)
# clwc removed: all-zero in ERA5 pressure levels
_PRESSURE_VARS = ['u', 'v', 'w', 't', 'q', 'ciwc']

# Single-level variables
_SINGLE_VARS = ["cape"]

# Total: 1 surface + 6 pressure vars × 37 levels = 223 vars × 3 lookback = 669 channels

RUN_CONFIG = RunConfig(
    to_train=True,
    use_seed=True,
    seed_value=42,
    plot_raw_tensors=False,
)

MODEL_CONFIG = ModelConfig(
    learning_rate=0.001,
    pos_weight=60.0,           # matches the good-results run — needed even on storm cases
    num_epochs=30,
    batch_size=8,
    decision_threshold=0.5,
    clip_after_normalization=True,
    normalization_clip_value=4.0,
    train_fraction=0.8,
    visualization_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # Physics-aware constraints
    # NOTE: disabled for baseline — re-enable after baseline converges.
    # ki_min/tciw_min/crr_min set to None: those vars are no longer in atm_params.
    use_physics_loss=False,
    use_physics_mask=False,
    physics_weight=1.0,
    cape_min=100.0,   # J/kg  — cape IS in atm_params; keep this constraint
    ki_min=None,      # kx not in atm_params → disable
    tciw_min=None,    # tciw not in atm_params → disable
    crr_min=None,     # crr not in atm_params → disable
    w500_max=-0.1,    # Pa/s  — w_500 IS in atm_params; keep this constraint
    r700_min=None,    # relative humidity not in ERA5.zarr → disable
    r850_min=None,    # relative humidity not in ERA5.zarr → disable
)

# ── Zarr config (new pipeline) ─────────────────────────────────────────────
#
# Storm cases available in ERA5.zarr (2023 only):
#   Case2: Jan 11–16, 2023
#   Case3: Mar 13–15, 2023
#   Case4: Apr  9–13, 2023
#
# After running data/append_cases_to_zarr.py:
#   Case1: Nov 23–25, 2022  (appended from ARCO)
#   Case5: Jan 26–31, 2024  (appended from ARCO)
#
# Current config: train on Cases 2+3, validate on Case4
# Final config (after append):
#   train on Cases 1+2+3+4, validate on Case5
#
ZARR_CONFIG = ZarrConfig(
    era5_zarr      = '/home/ec2-user/thesis-bucket/Zarr/ERA5.zarr',
    lightning_zarr = '/home/ec2-user/thesis-bucket/Zarr/Lightning.zarr',

    atm_params=(
        _SINGLE_VARS
        + [f"{var}_{lev}" for var in _PRESSURE_VARS for lev in _KEY_LEVELS]
    ),
    # ^ = 1 + 6×7 = 43 vars  ×  3 lookback = 129 input channels

    # ── Training date ranges (Cases 1–4) ─────────────────────────────────────
    # Non-contiguous storm windows — each tuple is (case_start, case_end).
    # Cross-gap lookback windows are automatically rejected by the dataset.
    train_ranges = [
        ('2022-11-23', '2022-11-25'),  # Case1 — Nov 2022 (appended from ARCO)
        ('2023-01-11', '2023-01-16'),  # Case2 — Jan 2023
        ('2023-03-13', '2023-03-15'),  # Case3 — Mar 2023
        ('2023-04-09', '2023-04-13'),  # Case4 — Apr 2023
    ],

    # ── Validation date range (Case 5) ───────────────────────────────────────
    val_start   = '2024-01-26',  # Case5 start (appended from ARCO)
    val_end     = '2024-01-31',  # Case5 end

    # ── Temporal settings ─────────────────────────────────────────────────────
    lookback_hours = 3,    # stack T-2, T-1, T

    # hours_of_day: None = all 24h  |  list(range(10,19)) = peak convection only
    hours_of_day = None,

    # months_of_year: None = use all months in the date range
    # The train range Jan–Mar already spans only storm months → no filter needed
    months_of_year = None,
)
