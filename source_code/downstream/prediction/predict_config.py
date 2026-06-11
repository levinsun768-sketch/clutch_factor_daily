from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PredictConfig:
    run_name_prefix: str = "DailyFingerprintGRU"
    model_run_dir: str = ""
    fingerprint_subdir: str = "fp_dataset"
    fingerprint_file_name: str = ""

    horizon: int = 5
    seq_len: int = 20
    use_gold_features: bool = False
    normalize_window: bool = False
    gold_feature_cols: list[str] = field(default_factory=list)

    train_start_date: str = "20210402"
    train_end_date: str = "20241231"
    valid_start_date: str = "20250101"
    valid_end_date: str = "20260529"

    model_type: str = "ComplexGRUAlpha"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3

    max_epochs: int = 50
    patience: int = 10
    lr_base: float = 1e-4
    lr_start: float = 1e-6
    warmup_epochs: int = 5
    seeds: tuple[int, ...] = (42, 2024, 777)

    device: str = "auto"

    @property
    def target_name(self) -> str:
        return f"ret_{self.horizon}d_t1"


default_predict_config = PredictConfig()
