from config.constants import CASES
from config.schema import CaseConfig, ModelConfig, RunConfig, ZarrConfig

# ── Available pressure levels in ERA5.zarr (all 37 ARCO levels) ────────────
_ALL_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900, 925,
    950, 975, 1000,
]

# Tropospheric levels only (100–1000 hPa) — skip stratosphere
_TROP_LEVELS = [l for l in _ALL_LEVELS if l >= 100]   # 27 levels

# Pressure-level variable short names (must match ERA5.zarr variable names)
# Naming convention: {var}_{level}  e.g. u_500, ciwc_850
_PRESSURE_VARS = ['u', 'v', 'w', 't', 'q', 'ciwc']  # clwc removed: all-zero in ERA5 pressure levels

# Single-level surface / column variables (17 vars)
_SINGLE_VARS = [
    "cape",  # Convective Available Potential Energy        [J/kg]
    "cin",   # Convective Inhibition                       [J/kg]
    "kx",    # K-Index (thunderstorm potential)             [K]
    "totot", # Total Totals Index
    "d2m",   # 2m Dewpoint Temperature                    [K]
    "tcwv",  # Total Column Water Vapour                   [kg/m²]
    "tciw",  # Total Column Cloud Ice Water                [kg/m²]
    "tclw",  # Total Column Cloud Liquid Water             [kg/m²]
    "cp",    # Convective Precipitation                    [m]
    "crr",   # Convective Rain Rate                        [kg/m²/s]
    "tp",    # Total Precipitation                         [m]
    "vimd",  # Vertically Integrated Moisture Divergence   [kg/m²/s]
    "t2m",   # 2m Temperature                             [K]
    "msl",   # Mean Sea Level Pressure                     [Pa]
    "sp",    # Surface Pressure                            [Pa]
    "cbh",   # Cloud Base Height                           [m]
    "hcc",   # High Cloud Cover                            [0–1]
]

RUN_CONFIG = RunConfig(
    to_train=True,
    use_seed=True,
    seed_value=42,
    plot_raw_tensors=False,
)

MODEL_CONFIG = ModelConfig(
    learning_rate=0.001,
    pos_weight=10.0,
    num_epochs=30,
    batch_size=8,
    decision_threshold=0.5,    # lowered from 0.85 — model needs to reach 85% confidence to predict
    clip_after_normalization=True,
    normalization_clip_value=4.0,
    train_fraction=0.8,
    visualization_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # Physics-aware constraints
    # NOTE: disabled for baseline run — physics penalty in winter months (low CAPE)
    # was penalizing ALL lightning predictions, teaching the model to predict nothing.
    # Re-enable after the baseline model converges.
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

# ── WRF config (original) ──────────────────────────────────────────────────
WRF_CASE_CONFIG = CaseConfig(
    train_cases=CASES[:5],
    val_cases=[],
    test_cases=[],
    tensor_dataset_name="train_cases_1_to_5",
    atm_params=["KI", "CAPE2D", "LPI", "PREC_RATE"],
    with_subparams={},
    space_res="24by24",
    time_res="1_hours",
    min_lat=27.296,
    max_lat=36.598,
    min_lon=27.954,
    max_lon=39.292,
    data_source="wrf",
)

# ── ERA5 config ────────────────────────────────────────────────────────────
ERA5_CASE_CONFIG = CaseConfig(
    train_cases=CASES[:5],
    val_cases=[],
    test_cases=[],
    tensor_dataset_name="ERA5_train_cases_1_to_5_24vars_lookback3",
    atm_params=[
        # Surface / column
        "cape", "kx", "tciw", "d2m", "tcwv", "crr", "msl", "t2m", "hcc",
        # Wind + dynamics at 500, 700, 850 hPa
        "u_500", "u_700", "u_850",
        "v_500", "v_700", "v_850",
        "w_500", "w_700", "w_850",
        "t_500", "t_700", "t_850",
        # NOTE: relative humidity (r_*) does NOT exist in ERA5.zarr.
        # Use specific humidity (q_*) instead.
        "q_500", "q_700", "q_850",
    ],
    with_subparams={},
    space_res="era5",
    time_res="1_hours",
    min_lat=27.296,
    max_lat=36.598,
    min_lon=27.954,
    max_lon=39.292,
    data_source="era5",
    lookback_hours=3,
)

# ── Active config — switch here ────────────────────────────────────────────
CASE_CONFIG = ERA5_CASE_CONFIG   # ← change to WRF_CASE_CONFIG to go back to WRF


# ── Zarr config (new pipeline) ─────────────────────────────────────────────
#
# Variable options:
#
#   _SINGLE_VARS                        17 surface/column vars
#   _SINGLE_VARS + pressure @ 3 levels  17 + 7×3  =  38 vars  ×3 lookback = 114 ch  (fast)
#   _SINGLE_VARS + pressure @ all trop  17 + 7×27 = 206 vars  ×3 lookback = 618 ch  (full)
#   _SINGLE_VARS + pressure @ all 37    17 + 7×37 = 276 vars  ×3 lookback = 828 ch  (everything)
#
# Channels feed directly into the UNet first conv (828→64), so any size works
# architecturally — more channels = more memory and slower training.
#
# Key pressure levels for lightning prediction — covers the full troposphere
# without redundant intermediate levels.
# 200 hPa  → upper troposphere / anvil (ciwc signal, Ehrensperger #1)
# 300 hPa  → upper troposphere (divergence, outflow)
# 400 hPa  → mid-upper troposphere (updraft peak signal)
# 500 hPa  → mid troposphere (classic synoptic level)
# 700 hPa  → mid-lower troposphere (moisture transport)
# 850 hPa  → low troposphere (low-level jet, moisture)
# 925 hPa  → boundary layer top
_KEY_LEVELS = [200, 300, 400, 500, 700, 850, 925]
# → 17 surface + 7 vars × 7 levels = 66 vars × 3 lookback = 198 input channels

ZARR_CONFIG = ZarrConfig(
    era5_zarr      = '/home/ec2-user/thesis-bucket/Zarr/ERA5.zarr',
    lightning_zarr = '/home/ec2-user/thesis-bucket/Zarr/Lightning.zarr',

    atm_params=(
        _SINGLE_VARS
        + [f"{var}_{lev}" for var in _PRESSURE_VARS for lev in _KEY_LEVELS]
    ),

    # ── Date ranges ───────────────────────────────────────────────────────────
    # Train: Jan–Oct 2023  |  Val: Nov–Dec 2023
    train_start = '2023-01-01',
    train_end   = '2023-10-31',
    val_start   = '2023-11-01',
    val_end     = '2023-12-31',

    # ── Temporal settings ─────────────────────────────────────────────────────
    lookback_hours = 3,    # stack T-2, T-1, T

    # hours_of_day: which UTC hours are used as prediction targets.
    # None          → all 24 hours (most training data)
    # list(range(8, 21))  → 08:00–20:00 UTC (daytime storms)
    # list(range(10, 19)) → 10:00–18:00 UTC (peak convection hours)
    # [0, 6, 12, 18]      → synoptic hours only
    hours_of_day = None,
)
