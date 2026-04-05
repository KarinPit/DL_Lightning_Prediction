from config.constants import CASES
from config.schema import CaseConfig, ModelConfig, RunConfig


RUN_CONFIG = RunConfig(
    to_train=False,
    use_seed=True,
    seed_value=42,
    plot_raw_tensors=True,
)

MODEL_CONFIG = ModelConfig(
    learning_rate=0.001,
    pos_weight=60.0,
    num_epochs=15,
    batch_size=8,
    decision_threshold=0.85,
    clip_after_normalization=True,
    normalization_clip_value=4.0,
    train_fraction=0.8,
    visualization_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    # Physics-aware constraints
    use_physics_loss=True,
    physics_weight=0.2,
    cape_min=100.0,    # J/kg  — CAPE must signal some instability
    ki_min=20.0,       # K-Index > 20 required for thunderstorm potential
    tciw_min=0.01,     # kg/m² — ice water needed for charge separation
    crr_min=None,       # mm/h  — active convective rain required (> 0)
    w500_max=-0.1,     # Pa/s  — ERA5 omega at 500 hPa must be negative (upward motion)
    r700_min=50.0,     # %     — 700 hPa relative humidity
    r850_min=60.0,     # %     — 850 hPa relative humidity
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
    tensor_dataset_name="ERA5_train_cases_1_to_5",
    # ERA5 variable short names (must match what's in the NC files)
    atm_params=["cape", "kx", "tciw", "d2m", "tcwv", "crr", "msl", "t2m", "hcc"],
    with_subparams={},
    space_res="era5",       # ERA5 is on ~28km grid
    time_res="1_hours",
    min_lat=27.296,
    max_lat=36.598,
    min_lon=27.954,
    max_lon=39.292,
    data_source="era5",
)

# ── Active config — switch here ────────────────────────────────────────────
CASE_CONFIG = ERA5_CASE_CONFIG   # ← change to WRF_CASE_CONFIG to go back to WRF
