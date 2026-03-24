from dataclasses import dataclass, field


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
    case: str
    atm_params: list[str]
    space_res: str = "24by24"
    time_res: str = "1_hours"

    @property
    def expected_input_channels(self):
        return len(self.atm_params)
