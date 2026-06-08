from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from downstream.prediction.predict_config import PredictConfig, default_predict_config
from downstream.prediction.train_pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train downstream GRU prediction model from daily fingerprints.")
    parser.add_argument("--model-run-dir", default="", help="Training run dir containing fp_dataset/.")
    parser.add_argument("--fingerprint-file-name", default="", help="Fingerprint parquet file name under fp_dataset/.")
    parser.add_argument("--horizon", type=int, default=0, help="Forward return horizon in trading days.")
    parser.add_argument("--seq-len", type=int, default=0, help="GRU input sequence length.")
    parser.add_argument("--train-start-date", default="", help="YYYYMMDD.")
    parser.add_argument("--train-end-date", default="", help="YYYYMMDD.")
    parser.add_argument("--valid-start-date", default="", help="YYYYMMDD.")
    parser.add_argument("--valid-end-date", default="", help="YYYYMMDD.")
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--normalize-window", action="store_true", help="Apply per-sample time-series standardization to each input window.")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> PredictConfig:
    data = asdict(default_predict_config)
    for key in data:
        value = getattr(args, key, None)
        if value in ("", 0, None):
            continue
        data[key] = value
    return PredictConfig(**data)


def main() -> None:
    config = merge_config(parse_args())
    if not config.model_run_dir:
        raise ValueError("Provide --model-run-dir.")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = Path("downstream/run").resolve()
    exp_dir = exp_root / f"{config.run_name_prefix}_{config.horizon}d_{run_ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "predict_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[prediction] output dir: {exp_dir}")
    run_training(config, exp_dir)


if __name__ == "__main__":
    main()
