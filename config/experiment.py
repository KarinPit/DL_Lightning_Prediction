from config.constants import CASES
from config.schema import CaseConfig, ModelConfig, RunConfig

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
    # Physics-aware constraints
    use_physics_loss=True,   # soft penalty in loss
    use_physics_mask=False,  # hard mask on predictions — set True to zero impossible cells
    physics_weight=1.0,
    cape_min=100.0,    # J/kg  — CAPE must signal some instability
    ki_min=20.0,       # K-Index > 20 required for thunderstorm potential
    tciw_min=0.01,     # kg/m² — ice water needed for charge separation
    crr_min=0.0,       # mm/h  — active convective rain required (> 0)
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
    tensor_dataset_name="ERA5_train_cases_1_to_5_24vars_lookback3",  # new name → forces tensor rebuild
    # ERA5 variable short names (must match what's in the NC files)
    # 9 surface/column vars + 15 pressure-level vars (500/700/850 hPa: u,v,w,t,r)
    atm_params=[
        # Surface / column
        "cape", "kx", "tciw", "d2m", "tcwv", "crr", "msl", "t2m", "hcc",
        # Wind components (u, v) at 500, 700, 850 hPa
        "u_500", "u_700", "u_850",
        "v_500", "v_700", "v_850",
        # Vertical velocity (omega, Pa/s) at 500, 700, 850 hPa
        "w_500", "w_700", "w_850",
        # Temperature at 500, 700, 850 hPa
        "t_500", "t_700", "t_850",
        # Relative humidity at 500, 700, 850 hPa
        "r_500", "r_700", "r_850",
    ],
    with_subparams={},
    space_res="era5",       # ERA5 is on ~28km grid
    time_res="1_hours",
    min_lat=27.296,
    max_lat=36.598,
    min_lon=27.954,
    max_lon=39.292,
    data_source="era5",
    lookback_hours=3,       # stack T-2, T-1, T → 72 input channels (24 vars × 3 hours)
)

# ── Active config — switch here ────────────────────────────────────────────
CASE_CONFIG = ERA5_CASE_CONFIG   # ← change to WRF_CASE_CONFIG to go back to WRF
