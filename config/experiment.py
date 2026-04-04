from config.constants import CASES
from config.schema import CaseConfig, ModelConfig, RunConfig

RUN_CONFIG = RunConfig(
    to_train=True,
    use_seed=True,
    seed_value=42,
    plot_raw_tensors=True,
)

MODEL_CONFIG = ModelConfig(
    learning_rate=0.001,
    pos_weight=70.0,
    num_epochs=20,
    batch_size=8,
    decision_threshold=0.7,
    clip_after_normalization=True,
    normalization_clip_value=4.0,
    train_fraction=0.8,
    visualization_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

CASE_CONFIG = CaseConfig(
    train_cases=[CASES[5]],  # case names must be in a list, even if there is only one item
    # train_cases=CASES[:5],
    val_cases=[],
    test_cases=[],
    tensor_dataset_name=CASES[5],
    # tensor_dataset_name="train_cases_1_to_5",
    # atm_params=["KI", "CAPE2D", "LPI", "PREC_RATE"],
    atm_params=["KI", "CAPE2D", "LPI", "PREC_RATE", "FLUX_PROD", "WMAX_LAYER", "MFLUX_MEAN_LAYER", "WPLUS_MEAN_LAYER"],
    with_subparams={"WDIAG": ["wplus_mean_layer"]},
    # with_subparams={"WDIAG": ["wmax_layer", "mflux_mean_layer", "wplus_mean_layer"]},
    space_res="4by4",
    time_res="1_hours",
    min_lat=27.296,
    max_lat=36.598,
    min_lon=27.954,
    max_lon=39.292,
)
