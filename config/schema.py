from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(frozen=True)
class RunConfig:
    to_train: bool = True
    use_seed: bool = True
    seed_value: int = 42
    plot_raw_tensors: bool = True


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.001
    pos_weight: float = 5000.0
    num_epochs: int = 10
    batch_size: int = 50
    decision_threshold: float = 0.9
    clip_after_normalization: bool = True
    normalization_clip_value: float = 4.0
    train_fraction: float = 0.8
    visualization_thresholds: list[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    )
    # ── Physics-aware training ────────────────────────────────────────────────
    use_physics_loss: bool = False    # soft penalty added to loss during training
    use_physics_mask: bool = False    # hard mask — zero predictions in impossible cells at inference
    physics_weight: float = 1.0      # weight of physics penalty relative to main BCE loss
    # Per-constraint thresholds — set to None to disable an individual constraint.
    # Semantics: value < _min  (or >= _max) → lightning physically impossible in that cell.
    cape_min: Optional[float] = 100.0   # J/kg  — CAPE: thermodynamic instability required
    ki_min:   Optional[float] = 20.0    # dimensionless K-Index: thunderstorm potential
    tciw_min: Optional[float] = 0.01   # kg/m² — ice-water content: charge separation needs ice
    crr_min:  Optional[float] = 0.0    # mm/h  — convective rain > 0: active convective core required
    w500_max: Optional[float] = -0.1   # Pa/s  — ERA5 omega at 500 hPa (negative = upward);
                                        #         >= this threshold means subsidence / no updraft
    r700_min: Optional[float] = 50.0   # %     — relative humidity at 700 hPa: mid-level moisture
    r850_min: Optional[float] = 60.0   # %     — relative humidity at 850 hPa: low-level moisture


@dataclass(frozen=True)
class CaseConfig:
    train_cases: list[str]
    val_cases: list[str]
    test_cases: list[str]
    atm_params: list[str]
    with_subparams: dict[str, list[str]]
    tensor_dataset_name: Optional[str] = None
    space_res: str = "4by4"
    time_res: str = "1_hours"
    min_lat: int = 27.296
    max_lat: int = 36.598
    min_lon: int = 27.954
    max_lon: int = 39.292
    data_source: str = "wrf"   # "wrf" or "era5"

    def _flatten_case_names(self, cases):
        flat_cases = []
        for case in cases:
            if isinstance(case, (list, tuple)):
                flat_cases.extend(self._flatten_case_names(case))
            else:
                flat_cases.append(case)
        return flat_cases

    @property
    def train_case_names(self):
        return self._flatten_case_names(self.train_cases)

    @property
    def val_case_names(self):
        return self._flatten_case_names(self.val_cases)

    @property
    def test_case_names(self):
        return self._flatten_case_names(self.test_cases)

    @property
    def dataset_name(self):
        if self.tensor_dataset_name:
            return self.tensor_dataset_name
        train_cases = self.train_case_names
        if not train_cases:
            raise ValueError("CaseConfig.train_cases must contain at least one case.")
        if len(train_cases) == 1:
            return train_cases[0]
        return "multi__" + "__".join(train_cases)

    @property
    def is_multi_case(self):
        return len(self.train_case_names) > 1

    lookback_hours: int = 1   # how many consecutive hours to stack as input channels
                               # 1 = single snapshot, 3 = T-2, T-1, T stacked


# ── Zarr-backed config (new pipeline — no case concept) ───────────────────────

@dataclass(frozen=True)
class ZarrConfig:
    # ── Required fields (no defaults) ────────────────────────────────────────
    era5_zarr:      str     # path to ERA5.zarr
    lightning_zarr: str     # path to Lightning.zarr
    atm_params:     list    # ERA5 variable names  e.g. ["cape", "u_500", ...]
    train_ranges:   list    # list of ("YYYY-MM-DD", "YYYY-MM-DD") pairs — one per training case
    val_start:      str     # "YYYY-MM-DD"
    val_end:        str     # "YYYY-MM-DD"

    # ── Optional fields (with defaults) ──────────────────────────────────────
    test_start:     Optional[str]  = None
    test_end:       Optional[str]  = None
    lookback_hours: int            = 3     # stack T-(L-1), ..., T-1, T as input
    hours_of_day:   Optional[list] = None  # None=all 24h; e.g. list(range(10,19))
    months_of_year: Optional[list] = None  # None=all months; e.g. [1,3,4] for Jan+Mar+Apr

    @property
    def expected_input_channels(self):
        return self.lookback_hours * len(self.atm_params)

    @property
    def months_label(self):
        if self.months_of_year is None:
            return 'all'
        names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        return '+'.join(names[m] for m in sorted(self.months_of_year))
