from dataclasses import dataclass, field
from typing import Optional


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

    @property
    def input_channel_names(self):
        channel_names = []
        for param in self.atm_params:
            channel_names.extend(self.with_subparams.get(param, [param]))
        return channel_names

    @property
    def expected_input_channels(self):
        return len(self.input_channel_names)
