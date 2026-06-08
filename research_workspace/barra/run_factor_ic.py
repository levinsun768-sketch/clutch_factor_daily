from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl

from barra.build_exposures import STYLE_FACTORS
from data.config import get_settings


DATE_FMT = "%Y%m%d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple Barra-like factor IC on a test date range.")
    parser.add_argument("--start-date", required=True, help="IC start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="IC end date in YYYYMMDD.")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in trading rows per stock.")
    parser.add_argument("--exposure-root", default="data/barra/exposures", help="Exposure partition root.")
    parser.add_argument("--output-root", default="artifacts/barra", help="Output directory root.")
    return parser.parse_args()


def scan_exposures(root: str, end_date: str) -> pl.LazyFrame:
    pattern = Path(root).expanduser().resolve() / "trade_date=*" / "data.parquet"
    return pl.scan_parquet(str(pattern), missing_columns="insert", extra_columns="ignore").filter(pl.col("trade_date") <= end_date)


def scan_daily_panel(end_date: str) -> pl.LazyFrame:
    settings = get_settings()
    pattern = settings.data_root / "silver" / "daily_panel" / "trade_date=*" / "data.parquet"
    return pl.scan_parquet(str(pattern), missing_columns="insert", extra_columns="ignore").filter(pl.col("trade_date") <= end_date)


def build_forward_return(end_date: str, horizon: int) -> pl.LazyFrame:
    return (
        scan_daily_panel(end_date)
        .filter(pl.col("has_trade").fill_null(True) & ~pl.col("is_st").fill_null(False) & ~pl.col("is_suspend").fill_null(False) & ~pl.col("is_bj").fill_null(False))
        .select(["ts_code", "trade_date", (pl.col("close") * pl.col("adj_factor")).alias("close_adj")])
        .sort(["ts_code", "trade_date"])
        .with_columns([
            pl.col("close_adj").shift(-1).over("ts_code").alias("entry_close"),
            pl.col("close_adj").shift(-horizon).over("ts_code").alias("exit_close"),
        ])
        .with_columns((pl.col("exit_close") / pl.col("entry_close").clip(lower_bound=1e-8) - 1.0).alias("fwd_ret"))
        .select(["ts_code", "trade_date", "fwd_ret"])
    )


def compute_ic(start_date: str, end_date: str, horizon: int, exposure_root: str) -> pl.DataFrame:
    exposures = scan_exposures(exposure_root, end_date).filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))
    returns = build_forward_return(end_date, horizon)
    joined = exposures.join(returns, on=["ts_code", "trade_date"], how="inner").drop_nulls(["fwd_ret", *STYLE_FACTORS])
    rows = []
    for factor in STYLE_FACTORS:
        out = (
            joined.group_by("trade_date")
            .agg([
                pl.corr(factor, "fwd_ret").alias("ic"),
                pl.corr(pl.col(factor).rank(), pl.col("fwd_ret").rank()).alias("rank_ic"),
                pl.len().alias("n"),
            ])
            .with_columns(pl.lit(factor).alias("factor"))
            .select(["trade_date", "factor", "ic", "rank_ic", "n"])
            .collect()
        )
        rows.append(out)
    return pl.concat(rows).sort(["factor", "trade_date"])


def summarize(ic_df: pl.DataFrame) -> pl.DataFrame:
    return (
        ic_df.group_by("factor")
        .agg([
            pl.len().alias("n_dates"),
            pl.col("n").mean().alias("mean_n"),
            pl.col("ic").mean().alias("ic_mean"),
            (pl.col("ic").mean() / pl.col("ic").std()).alias("ic_ir"),
            pl.col("rank_ic").mean().alias("rank_ic_mean"),
            (pl.col("rank_ic").mean() / pl.col("rank_ic").std()).alias("rank_ic_ir"),
        ])
        .sort("rank_ic_mean", descending=True)
    )


def main() -> None:
    args = parse_args()
    datetime.strptime(args.start_date, DATE_FMT)
    datetime.strptime(args.end_date, DATE_FMT)
    ic_df = compute_ic(args.start_date, args.end_date, args.horizon, args.exposure_root)
    summary = summarize(ic_df)
    out_dir = Path(args.output_root).expanduser().resolve() / f"ic_{args.start_date}_{args.end_date}_h{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_df.write_parquet(out_dir / "ic_timeseries.parquet")
    summary.write_csv(out_dir / "summary.csv")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary)
    print(f"[barra_ic] -> {out_dir}")


if __name__ == "__main__":
    main()
