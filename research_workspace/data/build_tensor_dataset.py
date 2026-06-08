from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from data.config import get_settings
from data.feature_spec import (
    FEATURE_NAMES,
    PANEL_PRICE_FEATURES,
    PRICE_IDX,
    TRADE_FEATURES,
    TRADE_IDX,
)


DATE_FMT = "%Y%m%d"
EPS = 1e-8
ANCHOR_CLOSE_CLIP = (-0.95, 4.0)
ANCHOR_HIGH_CLIP = (-0.95, 5.0)
ANCHOR_LOW_CLIP = (-0.95, 4.0)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build rolling-window tensor dataset from gold feature panel.")
    parser.add_argument("--start-date", default="", help="Target start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="Target end date in YYYYMMDD.")
    parser.add_argument("--window-size", type=int, default=settings.window_size, help="Rolling window size.")
    return parser.parse_args()


def scan_gold_features(end_date: str) -> pl.DataFrame:
    settings = get_settings()
    root = settings.data_root / "gold" / "feature_panel"
    paths = sorted(
        p
        for p in root.glob("trade_date=*/data.parquet")
        if p.parent.name.split("=")[1] <= end_date
    )
    if not paths:
        return pl.DataFrame()
    frames = [pl.read_parquet(path) for path in paths]
    return pl.concat(frames, how="diagonal_relaxed").sort(["ts_code", "trade_date"])


def output_dir(start_date: str, end_date: str, window_size: int) -> Path:
    settings = get_settings()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return settings.data_root / "tensors" / f"tensor_dataset_daily_{start_date}_{end_date}_w{window_size}_{ts}"


def write_feature_config(path: Path, window_size: int) -> None:
    lines = [
        f"F_DIM: {len(FEATURE_NAMES)}",
        f"WINDOW_SIZE: {window_size}",
        f"FEATURE_NAMES: {FEATURE_NAMES}",
        f"PRICE_IDX: {PRICE_IDX}",
        f"TRADE_IDX: {TRADE_IDX}",
        "FREQ: day",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(df: pl.DataFrame, start_date: str, end_date: str, window_size: int):
    arrays: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []

    for key, part in df.partition_by(["ts_code", "regime_id"], as_dict=True, maintain_order=True).items():
        if isinstance(key, tuple):
            code = key[0]
            regime_id = key[1]
        else:
            code = key
            regime_id = None
        part = part.sort("trade_date")
        trade_dates = part.get_column("trade_date").to_list()
        price_np = part.select(PANEL_PRICE_FEATURES).to_numpy()
        trade_np = part.select(TRADE_FEATURES).to_numpy()
        close_adj_np = part.get_column("close_adj").to_numpy()
        high_adj_np = part.get_column("high_adj").to_numpy()
        low_adj_np = part.get_column("low_adj").to_numpy()

        for idx in range(window_size - 1, part.height):
            end_dt = trade_dates[idx]
            if end_dt < start_date or end_dt > end_date:
                continue

            start_idx = idx - window_size + 1
            price_window = price_np[start_idx : idx + 1]
            trade_window = trade_np[start_idx : idx + 1]
            anchor_ref = max(float(close_adj_np[start_idx]), EPS)
            anchor_close = np.clip((close_adj_np[start_idx : idx + 1] / anchor_ref) - 1.0, *ANCHOR_CLOSE_CLIP)
            anchor_high = np.clip((high_adj_np[start_idx : idx + 1] / anchor_ref) - 1.0, *ANCHOR_HIGH_CLIP)
            anchor_low = np.clip((low_adj_np[start_idx : idx + 1] / anchor_ref) - 1.0, *ANCHOR_LOW_CLIP)
            anchor_window = np.stack([anchor_close, anchor_high, anchor_low], axis=1)
            window = np.concatenate([price_window, anchor_window, trade_window], axis=1)
            if np.isnan(window).any() or np.isinf(window).any():
                dropped_rows.append({
                    "ts_code": code,
                    "end_date": end_dt,
                    "reason": "nan_or_inf_in_window",
                })
                continue

            arrays.append(window.astype(np.float32, copy=False))
            meta_rows.append({
                "sample_id": len(meta_rows),
                "ts_code": code,
                "regime_id": regime_id,
                "end_date": end_dt,
                "start_date": trade_dates[start_idx],
                "window_size": window_size,
            })

    tensor = np.stack(arrays, axis=0) if arrays else np.empty((0, window_size, len(FEATURE_NAMES)), dtype=np.float32)
    return tensor, meta_rows, dropped_rows


def main() -> None:
    args = parse_args()
    settings = get_settings()
    start_date = args.start_date or settings.start_date
    end_date = args.end_date or settings.end_date or datetime.now().strftime(DATE_FMT)

    datetime.strptime(start_date, DATE_FMT)
    datetime.strptime(end_date, DATE_FMT)

    feature_df = scan_gold_features(end_date)
    tensor, meta_rows, dropped_rows = build_dataset(
        df=feature_df,
        start_date=start_date,
        end_date=end_date,
        window_size=args.window_size,
    )

    out_dir = output_dir(start_date, end_date, args.window_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "clean_tensor.npy", tensor)
    if meta_rows:
        pl.DataFrame(meta_rows).write_csv(out_dir / "tensor_meta.csv")
    else:
        pl.DataFrame(
            schema={
                "sample_id": pl.Int64,
                "ts_code": pl.Utf8,
                "regime_id": pl.Int64,
                "end_date": pl.Utf8,
                "start_date": pl.Utf8,
                "window_size": pl.Int64,
            }
        ).write_csv(out_dir / "tensor_meta.csv")
    write_feature_config(out_dir / "feature_config.yaml", args.window_size)
    (out_dir / "build_config.json").write_text(
        json.dumps(
            {
                "start_date": start_date,
                "end_date": end_date,
                "window_size": args.window_size,
                "feature_count": len(FEATURE_NAMES),
                "sample_count": len(meta_rows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with open(out_dir / "dropped_windows.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts_code", "end_date", "reason"])
        writer.writeheader()
        writer.writerows(dropped_rows)

    print(f"[tensor_dataset] samples={len(meta_rows)} shape={tensor.shape} -> {out_dir}")


if __name__ == "__main__":
    main()
