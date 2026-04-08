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
    tensor_dataset_name="ERA5_train_cases_1_to_5_fullpressure_singlelevel_lookback3",  # new name → forces tensor rebuild
    # ERA5 variable short names (must match what's in the NC files after process_era5.py)
    # 9 surface/column vars + 370 pressure-level vars (10 vars × 37 levels)
    # Pressure-level short names: r, ciwc, clwc, q, crwc, cswc, t, u, v, w
    # Levels: 1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,
    #         350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000
    atm_params=[
        # ── Surface / column (single-level ERA5, 17 vars) ─────────────────
        # Instability
        "cape", "cin", "kx", "totot",
        # Moisture
        "d2m", "tcwv", "tciw", "tclw",
        # Precipitation
        "cp", "crr", "tp",
        # Dynamics
        "vimd",
        # Surface
        "t2m", "msl", "sp", "cbh", "hcc",

        # ── Pressure-level variables (10 vars × 37 levels = 370) ──────────
        # Relative humidity (r)
        "r_1","r_2","r_3","r_5","r_7","r_10","r_20","r_30","r_50","r_70",
        "r_100","r_125","r_150","r_175","r_200","r_225","r_250","r_300",
        "r_350","r_400","r_450","r_500","r_550","r_600","r_650","r_700",
        "r_750","r_775","r_800","r_825","r_850","r_875","r_900","r_925",
        "r_950","r_975","r_1000",
        # Cloud ice water content (ciwc) — most important per Ehrensperger
        "ciwc_1","ciwc_2","ciwc_3","ciwc_5","ciwc_7","ciwc_10","ciwc_20","ciwc_30","ciwc_50","ciwc_70",
        "ciwc_100","ciwc_125","ciwc_150","ciwc_175","ciwc_200","ciwc_225","ciwc_250","ciwc_300",
        "ciwc_350","ciwc_400","ciwc_450","ciwc_500","ciwc_550","ciwc_600","ciwc_650","ciwc_700",
        "ciwc_750","ciwc_775","ciwc_800","ciwc_825","ciwc_850","ciwc_875","ciwc_900","ciwc_925",
        "ciwc_950","ciwc_975","ciwc_1000",
        # Cloud liquid water content (clwc)
        "clwc_1","clwc_2","clwc_3","clwc_5","clwc_7","clwc_10","clwc_20","clwc_30","clwc_50","clwc_70",
        "clwc_100","clwc_125","clwc_150","clwc_175","clwc_200","clwc_225","clwc_250","clwc_300",
        "clwc_350","clwc_400","clwc_450","clwc_500","clwc_550","clwc_600","clwc_650","clwc_700",
        "clwc_750","clwc_775","clwc_800","clwc_825","clwc_850","clwc_875","clwc_900","clwc_925",
        "clwc_950","clwc_975","clwc_1000",
        # Specific humidity (q)
        "q_1","q_2","q_3","q_5","q_7","q_10","q_20","q_30","q_50","q_70",
        "q_100","q_125","q_150","q_175","q_200","q_225","q_250","q_300",
        "q_350","q_400","q_450","q_500","q_550","q_600","q_650","q_700",
        "q_750","q_775","q_800","q_825","q_850","q_875","q_900","q_925",
        "q_950","q_975","q_1000",
        # Rain water content (crwc)
        "crwc_1","crwc_2","crwc_3","crwc_5","crwc_7","crwc_10","crwc_20","crwc_30","crwc_50","crwc_70",
        "crwc_100","crwc_125","crwc_150","crwc_175","crwc_200","crwc_225","crwc_250","crwc_300",
        "crwc_350","crwc_400","crwc_450","crwc_500","crwc_550","crwc_600","crwc_650","crwc_700",
        "crwc_750","crwc_775","crwc_800","crwc_825","crwc_850","crwc_875","crwc_900","crwc_925",
        "crwc_950","crwc_975","crwc_1000",
        # Snow water content (cswc) — 2nd most important per Ehrensperger
        "cswc_1","cswc_2","cswc_3","cswc_5","cswc_7","cswc_10","cswc_20","cswc_30","cswc_50","cswc_70",
        "cswc_100","cswc_125","cswc_150","cswc_175","cswc_200","cswc_225","cswc_250","cswc_300",
        "cswc_350","cswc_400","cswc_450","cswc_500","cswc_550","cswc_600","cswc_650","cswc_700",
        "cswc_750","cswc_775","cswc_800","cswc_825","cswc_850","cswc_875","cswc_900","cswc_925",
        "cswc_950","cswc_975","cswc_1000",
        # Temperature (t)
        "t_1","t_2","t_3","t_5","t_7","t_10","t_20","t_30","t_50","t_70",
        "t_100","t_125","t_150","t_175","t_200","t_225","t_250","t_300",
        "t_350","t_400","t_450","t_500","t_550","t_600","t_650","t_700",
        "t_750","t_775","t_800","t_825","t_850","t_875","t_900","t_925",
        "t_950","t_975","t_1000",
        # U wind (u)
        "u_1","u_2","u_3","u_5","u_7","u_10","u_20","u_30","u_50","u_70",
        "u_100","u_125","u_150","u_175","u_200","u_225","u_250","u_300",
        "u_350","u_400","u_450","u_500","u_550","u_600","u_650","u_700",
        "u_750","u_775","u_800","u_825","u_850","u_875","u_900","u_925",
        "u_950","u_975","u_1000",
        # V wind (v)
        "v_1","v_2","v_3","v_5","v_7","v_10","v_20","v_30","v_50","v_70",
        "v_100","v_125","v_150","v_175","v_200","v_225","v_250","v_300",
        "v_350","v_400","v_450","v_500","v_550","v_600","v_650","v_700",
        "v_750","v_775","v_800","v_825","v_850","v_875","v_900","v_925",
        "v_950","v_975","v_1000",
        # Vertical velocity / omega (w)
        "w_1","w_2","w_3","w_5","w_7","w_10","w_20","w_30","w_50","w_70",
        "w_100","w_125","w_150","w_175","w_200","w_225","w_250","w_300",
        "w_350","w_400","w_450","w_500","w_550","w_600","w_650","w_700",
        "w_750","w_775","w_800","w_825","w_850","w_875","w_900","w_925",
        "w_950","w_975","w_1000",
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

# ── Full-year 2023 config ──────────────────────────────────────────────────
FULLYEAR_CASE_CONFIG = CaseConfig(
    train_cases=["Year2023_full"],
    val_cases=[],
    test_cases=[],
    tensor_dataset_name="ERA5_Year2023_full_fullpressure_singlelevel_lookback3",
    atm_params=[
        # ── Surface / column (single-level ERA5, 17 vars) ─────────────────
        "cape", "cin", "kx", "totot",
        "d2m", "tcwv", "tciw", "tclw",
        "cp", "crr", "tp",
        "vimd",
        "t2m", "msl", "sp", "cbh", "hcc",

        # ── Pressure-level variables (10 vars × 37 levels = 370) ──────────
        "r_1","r_2","r_3","r_5","r_7","r_10","r_20","r_30","r_50","r_70",
        "r_100","r_125","r_150","r_175","r_200","r_225","r_250","r_300",
        "r_350","r_400","r_450","r_500","r_550","r_600","r_650","r_700",
        "r_750","r_775","r_800","r_825","r_850","r_875","r_900","r_925",
        "r_950","r_975","r_1000",
        "ciwc_1","ciwc_2","ciwc_3","ciwc_5","ciwc_7","ciwc_10","ciwc_20","ciwc_30","ciwc_50","ciwc_70",
        "ciwc_100","ciwc_125","ciwc_150","ciwc_175","ciwc_200","ciwc_225","ciwc_250","ciwc_300",
        "ciwc_350","ciwc_400","ciwc_450","ciwc_500","ciwc_550","ciwc_600","ciwc_650","ciwc_700",
        "ciwc_750","ciwc_775","ciwc_800","ciwc_825","ciwc_850","ciwc_875","ciwc_900","ciwc_925",
        "ciwc_950","ciwc_975","ciwc_1000",
        "clwc_1","clwc_2","clwc_3","clwc_5","clwc_7","clwc_10","clwc_20","clwc_30","clwc_50","clwc_70",
        "clwc_100","clwc_125","clwc_150","clwc_175","clwc_200","clwc_225","clwc_250","clwc_300",
        "clwc_350","clwc_400","clwc_450","clwc_500","clwc_550","clwc_600","clwc_650","clwc_700",
        "clwc_750","clwc_775","clwc_800","clwc_825","clwc_850","clwc_875","clwc_900","clwc_925",
        "clwc_950","clwc_975","clwc_1000",
        "q_1","q_2","q_3","q_5","q_7","q_10","q_20","q_30","q_50","q_70",
        "q_100","q_125","q_150","q_175","q_200","q_225","q_250","q_300",
        "q_350","q_400","q_450","q_500","q_550","q_600","q_650","q_700",
        "q_750","q_775","q_800","q_825","q_850","q_875","q_900","q_925",
        "q_950","q_975","q_1000",
        "crwc_1","crwc_2","crwc_3","crwc_5","crwc_7","crwc_10","crwc_20","crwc_30","crwc_50","crwc_70",
        "crwc_100","crwc_125","crwc_150","crwc_175","crwc_200","crwc_225","crwc_250","crwc_300",
        "crwc_350","crwc_400","crwc_450","crwc_500","crwc_550","crwc_600","crwc_650","crwc_700",
        "crwc_750","crwc_775","crwc_800","crwc_825","crwc_850","crwc_875","crwc_900","crwc_925",
        "crwc_950","crwc_975","crwc_1000",
        "cswc_1","cswc_2","cswc_3","cswc_5","cswc_7","cswc_10","cswc_20","cswc_30","cswc_50","cswc_70",
        "cswc_100","cswc_125","cswc_150","cswc_175","cswc_200","cswc_225","cswc_250","cswc_300",
        "cswc_350","cswc_400","cswc_450","cswc_500","cswc_550","cswc_600","cswc_650","cswc_700",
        "cswc_750","cswc_775","cswc_800","cswc_825","cswc_850","cswc_875","cswc_900","cswc_925",
        "cswc_950","cswc_975","cswc_1000",
        "t_1","t_2","t_3","t_5","t_7","t_10","t_20","t_30","t_50","t_70",
        "t_100","t_125","t_150","t_175","t_200","t_225","t_250","t_300",
        "t_350","t_400","t_450","t_500","t_550","t_600","t_650","t_700",
        "t_750","t_775","t_800","t_825","t_850","t_875","t_900","t_925",
        "t_950","t_975","t_1000",
        "u_1","u_2","u_3","u_5","u_7","u_10","u_20","u_30","u_50","u_70",
        "u_100","u_125","u_150","u_175","u_200","u_225","u_250","u_300",
        "u_350","u_400","u_450","u_500","u_550","u_600","u_650","u_700",
        "u_750","u_775","u_800","u_825","u_850","u_875","u_900","u_925",
        "u_950","u_975","u_1000",
        "v_1","v_2","v_3","v_5","v_7","v_10","v_20","v_30","v_50","v_70",
        "v_100","v_125","v_150","v_175","v_200","v_225","v_250","v_300",
        "v_350","v_400","v_450","v_500","v_550","v_600","v_650","v_700",
        "v_750","v_775","v_800","v_825","v_850","v_875","v_900","v_925",
        "v_950","v_975","v_1000",
        "w_1","w_2","w_3","w_5","w_7","w_10","w_20","w_30","w_50","w_70",
        "w_100","w_125","w_150","w_175","w_200","w_225","w_250","w_300",
        "w_350","w_400","w_450","w_500","w_550","w_600","w_650","w_700",
        "w_750","w_775","w_800","w_825","w_850","w_875","w_900","w_925",
        "w_950","w_975","w_1000",
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
CASE_CONFIG = ERA5_CASE_CONFIG   # ← ERA5_CASE_CONFIG | FULLYEAR_CASE_CONFIG | WRF_CASE_CONFIG
