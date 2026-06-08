from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from training.dataset import load_feature_config, resolve_tensor_dir


@dataclass
class TrainerConfig:
    tensor_dir: str = ""
    train_start: str = ""
    train_end: str = ""
    val_start: str = ""
    val_end: str = ""
    output_root: str = "./artifacts/models"

    model_type: str = "transformer_context"
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    trainable_proj: bool = True

    batch_size: int = 256
    max_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    mask_ratio: float = 0.15
    grad_clip_norm: float = 5.0
    early_stop_patience: int = 5
    num_workers: int = 0
    seed: int = 42

    use_reg_loss: bool = True
    lambda_d: float = 0.3
    lambda_o: float = 0.3
    lambda_u: float = 0.3
    lambda_f: float = 1.0
    lambda_b: float = 1.0

    device: str = "auto"
    price_weights: list[float] | None = None
    feature_config: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.tensor_dir = str(resolve_tensor_dir(self.tensor_dir))
        if not self.feature_config:
            self.feature_config = load_feature_config(self.tensor_dir)
        self.output_root = str(Path(self.output_root).expanduser().resolve())

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by nhead={self.nhead}")

    @property
    def f_dim(self) -> int:
        return int(self.feature_config["F_DIM"])

    @property
    def window_size(self) -> int:
        return int(self.feature_config["WINDOW_SIZE"])

    @property
    def feature_names(self) -> list[str]:
        return list(self.feature_config["FEATURE_NAMES"])

    @property
    def price_idx(self) -> list[int]:
        return list(self.feature_config["PRICE_IDX"])

    @property
    def trade_idx(self) -> list[int]:
        return list(self.feature_config["TRADE_IDX"])

    def to_dict(self) -> dict:
        data = asdict(self)
        data["feature_names"] = self.feature_names
        data["f_dim"] = self.f_dim
        data["window_size"] = self.window_size
        data["price_idx"] = self.price_idx
        data["trade_idx"] = self.trade_idx
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def config_from_env() -> TrainerConfig:
    return TrainerConfig(
        tensor_dir=os.getenv("FPD_TENSOR_DIR", ""),
        train_start=os.getenv("FPD_TRAIN_START", ""),
        train_end=os.getenv("FPD_TRAIN_END", ""),
        val_start=os.getenv("FPD_VAL_START", ""),
        val_end=os.getenv("FPD_VAL_END", ""),
        output_root=os.getenv("FPD_MODEL_ROOT", "./artifacts/models"),
        model_type=os.getenv("FPD_MODEL_TYPE", "transformer_context"),
        d_model=int(os.getenv("FPD_D_MODEL", "64")),
        nhead=int(os.getenv("FPD_NHEAD", "4")),
        num_layers=int(os.getenv("FPD_NUM_LAYERS", "3")),
        dim_feedforward=int(os.getenv("FPD_DIM_FEEDFORWARD", "256")),
        dropout=float(os.getenv("FPD_DROPOUT", "0.1")),
        trainable_proj=os.getenv("FPD_TRAINABLE_PROJ", "true").lower() == "true",
        batch_size=int(os.getenv("FPD_BATCH_SIZE", "256")),
        max_epochs=int(os.getenv("FPD_MAX_EPOCHS", "20")),
        learning_rate=float(os.getenv("FPD_LEARNING_RATE", "1e-4")),
        weight_decay=float(os.getenv("FPD_WEIGHT_DECAY", "1e-4")),
        mask_ratio=float(os.getenv("FPD_MASK_RATIO", "0.15")),
        grad_clip_norm=float(os.getenv("FPD_GRAD_CLIP_NORM", "5.0")),
        early_stop_patience=int(os.getenv("FPD_EARLY_STOP_PATIENCE", "5")),
        num_workers=int(os.getenv("FPD_NUM_WORKERS", "0")),
        seed=int(os.getenv("FPD_SEED", "42")),
        use_reg_loss=os.getenv("FPD_USE_REG_LOSS", "true").lower() == "true",
        lambda_d=float(os.getenv("FPD_LAMBDA_D", "0.3")),
        lambda_o=float(os.getenv("FPD_LAMBDA_O", "0.3")),
        lambda_u=float(os.getenv("FPD_LAMBDA_U", "0.3")),
        lambda_f=float(os.getenv("FPD_LAMBDA_F", "1.0")),
        lambda_b=float(os.getenv("FPD_LAMBDA_B", "1.0")),
        device=os.getenv("FPD_DEVICE", "auto"),
    )
