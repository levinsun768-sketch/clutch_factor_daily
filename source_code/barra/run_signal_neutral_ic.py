from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from barra.build_exposures import STYLE_FACTORS


DATE_FMT = "%Y%m%d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neutralize a signal by Barra style + industry exposures and compute IC.")
    parser.add_argument("--signal-path", required=True, help="Parquet with ts_code, end_date, score.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD.")
    parser.add_argument("--signal-col", default="score")
    parser.add_argument("--exposure-root", default="data/barra/exposures")
    parser.add_argument("--labeled-signal-path", default="", help="Optional backtest labeled_signals parquet with fwd_ret.")
    parser.add_argument("--output-root", default="artifacts/barra/signal_neutral_ic")
    parser.add_argument("--min-cross-section", type=int, default=500)
    parser.add_argument("--no-industry", action="store_true", help="Neutralize by style only.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Small diagonal ridge for numerical stability.")
    parser.add_argument("--save-neutral-signal", action="store_true", help="Write ts_code/end_date/score parquet for backtest.")
    return parser.parse_args()


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def residualize_one_day(pdf: pd.DataFrame, signal_col: str, include_industry: bool, ridge: float) -> pd.DataFrame:
    pdf = pdf.copy()
    y = zscore(pdf[signal_col].to_numpy(dtype=np.float64))
    x_parts = [np.ones((len(pdf), 1), dtype=np.float64), pdf[STYLE_FACTORS].to_numpy(dtype=np.float64)]
    if include_industry:
        industry = pd.get_dummies(pdf["sw_l1_code"].fillna("UNKNOWN"), dtype=float)
        if industry.shape[1] > 1:
            industry = industry.iloc[:, 1:]
        x_parts.append(industry.to_numpy(dtype=np.float64))
    x = np.concatenate(x_parts, axis=1)
    xtx = x.T @ x
    xtx.flat[:: xtx.shape[0] + 1] += ridge
    beta = np.linalg.pinv(xtx) @ x.T @ y
    resid = y - x @ beta
    pdf["score_z"] = y
    pdf["score_neutral"] = zscore(resid)
    fitted = x @ beta
    denom = float(np.sum((y - y.mean()) ** 2))
    pdf["neutral_r2"] = 1.0 - float(np.sum((y - fitted) ** 2)) / denom if denom > 1e-12 else 0.0
    return pdf[["ts_code", "end_date", "score_z", "score_neutral", "neutral_r2"]]


def load_joined(args: argparse.Namespace) -> pd.DataFrame:
    signal = pl.scan_parquet(args.signal_path).select([
        "ts_code",
        pl.col("end_date").cast(pl.Utf8).alias("end_date"),
        pl.col(args.signal_col).cast(pl.Float64).alias("score"),
    ]).filter((pl.col("end_date") >= args.start_date) & (pl.col("end_date") <= args.end_date))

    exposure_pattern = Path(args.exposure_root).expanduser().resolve() / "trade_date=*" / "data.parquet"
    exposures = pl.scan_parquet(str(exposure_pattern), missing_columns="insert", extra_columns="ignore").select(
        ["ts_code", pl.col("trade_date").cast(pl.Utf8).alias("end_date"), "sw_l1_code", *STYLE_FACTORS]
    ).filter((pl.col("end_date") >= args.start_date) & (pl.col("end_date") <= args.end_date))

    if args.labeled_signal_path:
        returns = pl.scan_parquet(args.labeled_signal_path).select([
            "ts_code",
            pl.col("end_date").cast(pl.Utf8).alias("end_date"),
            pl.col("fwd_ret").cast(pl.Float64),
        ])
    else:
        raise ValueError("Provide --labeled-signal-path for fwd_ret in this version.")

    joined = (
        signal.join(exposures, on=["ts_code", "end_date"], how="inner")
        .join(returns, on=["ts_code", "end_date"], how="inner")
        .drop_nulls(["score", "fwd_ret", *STYLE_FACTORS])
        .collect()
    )
    return joined.to_pandas()


def compute_daily_ic(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for dt, part in df.groupby("end_date", sort=True):
        if len(part) < 2:
            continue
        score = part[score_col]
        ret = part["fwd_ret"]
        rows.append({
            "end_date": dt,
            "n": int(len(part)),
            "ic": float(score.corr(ret, method="pearson")),
            "rank_ic": float(score.corr(ret, method="spearman")),
        })
    return pd.DataFrame(rows)


def summarize(ic: pd.DataFrame, label: str, r2_by_date: pd.DataFrame | None = None) -> dict[str, float | int | str]:
    out = {
        "label": label,
        "n_dates": int(len(ic)),
        "mean_n": float(ic["n"].mean()) if len(ic) else 0.0,
        "ic_mean": float(ic["ic"].mean()) if len(ic) else 0.0,
        "ic_ir": float(ic["ic"].mean() / ic["ic"].std()) if len(ic) > 1 and ic["ic"].std() else 0.0,
        "rank_ic_mean": float(ic["rank_ic"].mean()) if len(ic) else 0.0,
        "rank_ic_ir": float(ic["rank_ic"].mean() / ic["rank_ic"].std()) if len(ic) > 1 and ic["rank_ic"].std() else 0.0,
    }
    if r2_by_date is not None and len(r2_by_date):
        out["neutral_r2_mean"] = float(r2_by_date["neutral_r2"].mean())
    return out


def main() -> None:
    args = parse_args()
    datetime.strptime(args.start_date, DATE_FMT)
    datetime.strptime(args.end_date, DATE_FMT)
    joined = load_joined(args)
    if joined.empty:
        raise ValueError("No joined signal/exposure/return rows.")
    counts = joined.groupby("end_date").size()
    joined = joined[joined["end_date"].isin(counts[counts >= args.min_cross_section].index)].copy()

    parts = [residualize_one_day(part, "score", not args.no_industry, args.ridge) for _, part in joined.groupby("end_date", sort=True)]
    neutral = pd.concat(parts, ignore_index=True)
    out_df = joined.merge(neutral, on=["ts_code", "end_date"], how="inner")

    raw_ic = compute_daily_ic(out_df, "score_z")
    neutral_ic = compute_daily_ic(out_df, "score_neutral")
    r2_by_date = out_df.groupby("end_date", as_index=False)["neutral_r2"].first()
    summary = pd.DataFrame([
        summarize(raw_ic, "raw", r2_by_date=None),
        summarize(neutral_ic, "neutral_style_industry" if not args.no_industry else "neutral_style", r2_by_date=r2_by_date),
    ])

    out_dir = Path(args.output_root).expanduser().resolve() / f"{args.start_date}_{args.end_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_ic.to_parquet(out_dir / "raw_ic.parquet", index=False)
    neutral_ic.to_parquet(out_dir / "neutral_ic.parquet", index=False)
    r2_by_date.to_parquet(out_dir / "neutral_r2.parquet", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    if args.save_neutral_signal:
        signal_out = out_df[["ts_code", "end_date", "score_neutral"]].rename(columns={"score_neutral": "score"}).sort_values(["end_date", "ts_code"])
        signal_out.to_parquet(out_dir / "neutral_signal.parquet", index=False)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"[signal_neutral_ic] rows={len(out_df)} -> {out_dir}")


if __name__ == "__main__":
    main()
