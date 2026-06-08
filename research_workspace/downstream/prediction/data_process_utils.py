from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from data.config import get_settings
from downstream.prediction.predict_config import PredictConfig


def resolve_fingerprint_path(config: PredictConfig) -> Path:
    if not config.model_run_dir:
        raise ValueError("Provide model_run_dir.")
    run_dir = Path(config.model_run_dir).expanduser().resolve()
    fp_dir = run_dir / config.fingerprint_subdir
    if not fp_dir.exists():
        raise FileNotFoundError(f"Fingerprint directory does not exist: {fp_dir}")
    if config.fingerprint_file_name:
        path = fp_dir / config.fingerprint_file_name
        if not path.exists():
            raise FileNotFoundError(f"Fingerprint file does not exist: {path}")
        return path
    candidates = sorted(fp_dir.glob("fingerprints_daily_*.parquet"))
    if not candidates:
        candidates = sorted(fp_dir.glob("fingerprints_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No fingerprint parquet found under: {fp_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def scan_gold_features(end_date: str) -> pl.LazyFrame:
    settings = get_settings()
    pattern = settings.data_root / "gold" / "feature_panel" / "trade_date=*" / "data.parquet"
    return pl.scan_parquet(str(pattern), missing_columns="insert", extra_columns="ignore").filter(
        pl.col("trade_date") <= end_date
    )


def build_target_frame(end_date: str, horizon: int) -> pl.DataFrame:
    lf = (
        scan_gold_features(end_date)
        .select(["ts_code", "trade_date", "close_adj"])
        .sort(["ts_code", "trade_date"])
        .with_columns([
            pl.col("close_adj").shift(-1).over("ts_code").alias("__entry_close"),
            pl.col("close_adj").shift(-horizon).over("ts_code").alias("__exit_close"),
        ])
        .with_columns(
            (pl.col("__exit_close") / pl.col("__entry_close").clip(lower_bound=1e-8) - 1.0).alias("target_return")
        )
        .select(["ts_code", "trade_date", "target_return"])
    )
    return lf.collect()


def load_gold_feature_frame(end_date: str, cols: list[str]) -> pl.DataFrame:
    if not cols:
        return pl.DataFrame(schema={"ts_code": pl.Utf8, "trade_date": pl.Utf8})
    keep = ["ts_code", "trade_date"] + cols
    return scan_gold_features(end_date).select(keep).collect()


def prepare_dataset(config: PredictConfig) -> tuple[pd.DataFrame, list[str]]:
    fp_path = resolve_fingerprint_path(config)
    print(f"[prediction] fingerprint file: {fp_path}")
    fp = pl.read_parquet(fp_path)
    if "end_date" not in fp.columns:
        raise ValueError("Fingerprint parquet must contain end_date.")
    fp = fp.with_columns(pl.col("end_date").cast(pl.Utf8).alias("trade_date"))

    max_date = str(fp.select(pl.col("trade_date").max()).item())
    target = build_target_frame(max_date, config.horizon)
    df = fp.join(target, on=["ts_code", "trade_date"], how="inner")

    if config.use_gold_features:
        gold = load_gold_feature_frame(max_date, config.gold_feature_cols)
        df = df.join(gold, on=["ts_code", "trade_date"], how="left")

    fp_cols = sorted([c for c in df.columns if c.startswith("fp_")])
    feature_cols = fp_cols + [c for c in config.gold_feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    df = (
        df.select(["ts_code", "trade_date", "target_return"] + feature_cols)
        .drop_nulls()
        .sort(["ts_code", "trade_date"])
        .to_pandas()
    )
    df["datetime"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.set_index(["ts_code", "datetime"]).sort_index()
    return df, feature_cols


def normalize_feature_window(window: np.ndarray) -> np.ndarray:
    arr = window.astype(np.float32, copy=True)
    mu = arr.mean(axis=0, keepdims=True)
    sigma = arr.std(axis=0, keepdims=True)
    arr = (arr - mu) / (sigma + 1e-6)
    return arr
