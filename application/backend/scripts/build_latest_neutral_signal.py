from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

STYLE_FACTORS = [
    "size",
    "value_bp",
    "value_ep",
    "momentum_252_20",
    "reversal_20",
    "beta_120",
    "volatility_60",
    "liquidity_amount_20",
    "liquidity_turnover_20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Barra-neutral signal without forward-return labels.")
    parser.add_argument("--research-workspace", default="../../research_workspace")
    parser.add_argument("--signal-path", required=True)
    parser.add_argument("--signal-col", default="score")
    parser.add_argument("--start-date", default="20250102")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--exposure-root", default="data/barra/exposures")
    parser.add_argument("--output-root", default="artifacts/barra/signal_neutral_ic")
    parser.add_argument("--min-cross-section", type=int, default=500)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--spans", default="3,5,10,20,40")
    return parser.parse_args()


def resolve_research(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (Path(__file__).resolve().parents[3] / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (Path(__file__).resolve().parents[1] / raw).resolve()


def zscore(values: np.ndarray) -> np.ndarray:
    mu = np.nanmean(values)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mu) / sd


def residualize_one_day(pdf: pd.DataFrame, signal_col: str, ridge: float) -> pd.DataFrame:
    pdf = pdf.copy()
    y = zscore(pdf[signal_col].to_numpy(dtype=np.float64))
    x_parts = [np.ones((len(pdf), 1), dtype=np.float64), pdf[STYLE_FACTORS].to_numpy(dtype=np.float64)]
    industry = pd.get_dummies(pdf["sw_l1_code"].fillna("UNKNOWN"), dtype=float)
    if industry.shape[1] > 1:
        industry = industry.iloc[:, 1:]
    x_parts.append(industry.to_numpy(dtype=np.float64))
    x = np.concatenate(x_parts, axis=1)
    xtx = x.T @ x
    xtx.flat[:: xtx.shape[0] + 1] += ridge
    beta = np.linalg.pinv(xtx) @ x.T @ y
    resid = y - x @ beta
    fitted = x @ beta
    denom = float(np.sum((y - y.mean()) ** 2))
    pdf["score"] = zscore(resid)
    pdf["neutral_r2"] = 1.0 - float(np.sum((y - fitted) ** 2)) / denom if denom > 1e-12 else 0.0
    return pdf[["ts_code", "end_date", "score", "neutral_r2"]]


def load_joined(args: argparse.Namespace, research: Path) -> pd.DataFrame:
    signal_path = Path(args.signal_path).expanduser()
    if not signal_path.is_absolute():
        signal_path = research / signal_path
    exposure_root = Path(args.exposure_root).expanduser()
    if not exposure_root.is_absolute():
        exposure_root = research / exposure_root

    signal = (
        pl.scan_parquet(str(signal_path), missing_columns="insert", extra_columns="ignore")
        .select([
            "ts_code",
            pl.col("end_date").cast(pl.Utf8).alias("end_date"),
            pl.col(args.signal_col).cast(pl.Float64).alias("score_raw"),
        ])
        .filter((pl.col("end_date") >= args.start_date) & (pl.col("end_date") <= args.end_date))
    )
    exposures = (
        pl.scan_parquet(str(exposure_root / "trade_date=*" / "data.parquet"), missing_columns="insert", extra_columns="ignore")
        .select(["ts_code", pl.col("trade_date").cast(pl.Utf8).alias("end_date"), "sw_l1_code", *STYLE_FACTORS])
        .filter((pl.col("end_date") >= args.start_date) & (pl.col("end_date") <= args.end_date))
    )
    joined = signal.join(exposures, on=["ts_code", "end_date"], how="inner").drop_nulls(["score_raw", *STYLE_FACTORS]).collect()
    return joined.to_pandas()


def zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("end_date", sort=False)[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) and np.isfinite(s.std(ddof=0)) else 0.0)


def write_ewma(neutral: pd.DataFrame, out_dir: Path, spans: list[int]) -> None:
    ewma_dir = out_dir / "ewma"
    ewma_dir.mkdir(parents=True, exist_ok=True)
    base = neutral[["ts_code", "end_date", "score"]].sort_values(["ts_code", "end_date"]).copy()
    for span in spans:
        smoothed = base.copy()
        smoothed["score"] = smoothed.groupby("ts_code", sort=False)["score"].transform(lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean())
        smoothed = smoothed.sort_values(["end_date", "ts_code"]).copy()
        smoothed["score"] = zscore_by_date(smoothed, "score")
        smoothed[["ts_code", "end_date", "score"]].sort_values(["ts_code", "end_date"]).to_parquet(ewma_dir / f"neutral_signal_ewma{span}.parquet", index=False)


def main() -> None:
    args = parse_args()
    research = resolve_research(args.research_workspace)
    joined = load_joined(args, research)
    if joined.empty:
        raise ValueError("No joined signal/exposure rows.")
    counts = joined.groupby("end_date").size()
    keep_dates = counts[counts >= args.min_cross_section].index
    joined = joined[joined["end_date"].isin(keep_dates)].copy()
    parts = [residualize_one_day(part, "score_raw", args.ridge) for _, part in joined.groupby("end_date", sort=True)]
    neutral = pd.concat(parts, ignore_index=True).sort_values(["end_date", "ts_code"])

    out_root = Path(args.output_root).expanduser()
    if not out_root.is_absolute():
        out_root = research / out_root
    out_dir = out_root / f"{args.start_date}_{args.end_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    neutral[["ts_code", "end_date", "score"]].to_parquet(out_dir / "neutral_signal.parquet", index=False)
    neutral.groupby("end_date", as_index=False)["neutral_r2"].first().to_parquet(out_dir / "neutral_r2.parquet", index=False)
    spans = [int(x) for x in args.spans.split(",") if x.strip()]
    write_ewma(neutral, out_dir, spans)
    config: dict[str, Any] = vars(args) | {
        "method": "signal_only_barra_style_industry_neutralization",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(neutral)),
        "n_dates": int(neutral["end_date"].nunique()),
        "min_date": str(neutral["end_date"].min()),
        "max_date": str(neutral["end_date"].max()),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(neutral), "n_dates": neutral["end_date"].nunique(), "min": neutral["end_date"].min(), "max": neutral["end_date"].max()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
