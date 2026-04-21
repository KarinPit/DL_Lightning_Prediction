from config.schema import ModelConfig, RunConfig, ZarrConfig

# ── Pressure levels — all 37 standard ERA5 levels ─────────────────────────
_KEY_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 775, 800, 825,
    850, 875, 900, 925, 950, 975, 1000,
]

# Pressure-level variable short names (must match ERA5.zarr variable names)
_PRESSURE_VARS = ['u', 'v', 'w', 't', 'q', 'ciwc']  # clwc excluded: all-zero in ERA5 pressure levels

# Single-level surface / column variables — all 17 (matches good-results run)
_SINGLE_VARS = [
    "cape", "cin", "kx", "totot", "d2m", "tcwv", "tciw", "tclw",
    "cp", "crr", "tp", "vimd", "t2m", "msl", "sp", "cbh", "hcc",
]

# Total: 17 + 7×37 = 276 vars × 3 lookback = 828 input channels

RUN_CONFIG = RunConfig(
    to_train=True,
    use_seed=True,
    seed_value=42,
    plot_raw_tensors=False,
)

MODEL_CONFIG = ModelConfig(
    learning_rate=0.001,
    pos_weight=60.0,
    num_epochs=30,
    batch_size=8,
    decision_threshold=0.85,
    clip_after_normalization=True,
    normalization_clip_value=4.0,
    train_fraction=0.8,
    visualization_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    use_physics_loss=False,
    use_physics_mask=False,
    physics_weight=1.0,
    cape_min=100.0,
    ki_min=20.0,
    tciw_min=0.01,
    crr_min=0.0,
    w500_max=-0.1,
    r700_min=None,
    r850_min=None,
)

# ── Zarr config — identical to the good-results run ────────────────────────
ZARR_CONFIG = ZarrConfig(
    era5_zarr      = '/home/ec2-user/thesis-bucket/Zarr/ERA5.zarr',
    lightning_zarr = '/home/ec2-user/thesis-bucket/Zarr/Lightning.zarr',

    atm_params=(
        _SINGLE_VARS
        + [f"{var}_{lev}" for var in _PRESSURE_VARS for lev in _KEY_LEVELS]
    ),
    # ^ = 17 + 7×37 = 276 vars × 3 lookback = 828 input channels

    # Full Jan–Mar 2023 training — same as the run that produced good FSS/reliability
    train_ranges = [('2023-01-01', '2023-03-31')],

    val_start = '2023-04-01',
    val_end   = '2023-04-30',

    lookback_hours = 3,
    hours_of_day   = None,
    months_of_year = None,
)
