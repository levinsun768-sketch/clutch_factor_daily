from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from downstream.prediction.data_process_utils import normalize_feature_window, resolve_fingerprint_path
from downstream.prediction.model_registry import get_model
from downstream.prediction.predict_config import PredictConfig
from downstream.prediction.train_pipeline import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream GRU inference from a trained downstream run.")
    parser.add_argument("--downstream-run-dir", required=True, help="Directory containing predict_config.json and best_model_seed*.pt.")
    parser.add_argument("--model-run-dir", default="", help="Override model_run_dir from predict_config.json.")
    parser.add_argument("--fingerprint-file-name", default="", help="Override fingerprint parquet file name under fp_dataset/.")
    parser.add_argument("--start-date", default="", help="Optional output start end_date in YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="Optional output end end_date in YYYYMMDD.")
    parser.add_argument("--output-name", default="prediction_scores_inference.parquet", help="Output parquet file name under downstream run dir.")
    parser.add_argument("--device", default="", help="Override device from predict_config.json.")
    return parser.parse_args()


def load_config(run_dir: Path, args: argparse.Namespace) -> PredictConfig:
    data = json.loads((run_dir / "predict_config.json").read_text(encoding="utf-8"))
    if args.model_run_dir:
        data["model_run_dir"] = args.model_run_dir
    if args.fingerprint_file_name:
        data["fingerprint_file_name"] = args.fingerprint_file_name
    if args.device:
        data["device"] = args.device
    return PredictConfig(**data)


def load_feature_cols(run_dir: Path) -> list[str]:
    path = run_dir / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature column file: {path}")
    return list(json.loads(path.read_text(encoding="utf-8"))["feature_cols"])


def build_inference_windows(df: pd.DataFrame, feature_cols: list[str], seq_len: int, normalize_window: bool):
    rows: list[dict[str, object]] = []
    windows: list[np.ndarray] = []
    for code, part in df.groupby("ts_code", sort=False):
        part = part.sort_values("end_date")
        vals = part[feature_cols].to_numpy(dtype=np.float32)
        dates = part["end_date"].to_numpy()
        if len(part) < seq_len:
            continue
        for idx in range(seq_len - 1, len(part)):
            raw_window = vals[idx - seq_len + 1 : idx + 1]
            window = normalize_feature_window(raw_window) if normalize_window else raw_window.astype(np.float32, copy=True)
            if not np.isfinite(window).all():
                continue
            rows.append({"ts_code": str(code), "end_date": str(dates[idx])})
            windows.append(window)
    if not windows:
        return pd.DataFrame(columns=["ts_code", "end_date"]), np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
    return pd.DataFrame(rows), np.stack(windows).astype(np.float32, copy=False)


def run_inference(args: argparse.Namespace) -> Path:
    run_dir = Path(args.downstream_run_dir).expanduser().resolve()
    config = load_config(run_dir, args)
    feature_cols = load_feature_cols(run_dir)
    fp_path = resolve_fingerprint_path(config)
    fp = pd.read_parquet(fp_path)
    missing = [col for col in ["ts_code", "end_date", *feature_cols] if col not in fp.columns]
    if missing:
        raise ValueError(f"Fingerprint file missing required columns: {missing}")
    fp = fp[["ts_code", "end_date", *feature_cols]].dropna().sort_values(["ts_code", "end_date"])
    if args.start_date:
        history_dates = sorted(fp["end_date"].astype(str).unique())
        start_pos = max(0, history_dates.index(args.start_date) - config.seq_len + 1) if args.start_date in history_dates else 0
        history_start = history_dates[start_pos]
        fp = fp[fp["end_date"].astype(str) >= history_start]
    if args.end_date:
        fp = fp[fp["end_date"].astype(str) <= args.end_date]

    meta, x_np = build_inference_windows(fp, feature_cols, config.seq_len, config.normalize_window)
    if args.start_date:
        keep = meta["end_date"].astype(str) >= args.start_date
        meta = meta.loc[keep].reset_index(drop=True)
        x_np = x_np[keep.to_numpy()]
    if args.end_date:
        keep = meta["end_date"].astype(str) <= args.end_date
        meta = meta.loc[keep].reset_index(drop=True)
        x_np = x_np[keep.to_numpy()]

    device = resolve_device(config)
    outputs = meta.copy()
    for seed in config.seeds:
        model_path = run_dir / f"best_model_seed{seed}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
        model = get_model(config, input_dim=len(feature_cols)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_np), 8192):
                x = torch.tensor(x_np[start : start + 8192], dtype=torch.float32, device=device)
                preds.append(model(x).detach().cpu().numpy())
        outputs[f"score_seed{seed}"] = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    score_cols = [c for c in outputs.columns if c.startswith("score_seed")]
    outputs["score"] = outputs[score_cols].mean(axis=1)
    outputs = outputs[["ts_code", "end_date", "score", *score_cols]].sort_values(["end_date", "ts_code"])

    out_path = run_dir / args.output_name
    outputs.to_parquet(out_path, index=False)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "downstream_run_dir": str(run_dir),
                "config": asdict(config),
                "fingerprint_path": str(fp_path),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "rows": int(len(outputs)),
                "output_path": str(out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[prediction_inference] rows={len(outputs)} -> {out_path}")
    return out_path


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
