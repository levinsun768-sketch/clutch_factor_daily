from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl

from data.config import get_settings


DATE_FMT = "%Y%m%d"
EPS = 1e-8
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
IDENTITY_COLS = [
    "ts_code",
    "trade_date",
    "name",
    "sw_l1_code",
    "sw_l1_name",
    "is_hs_300",
    "is_csi_500",
    "is_csi_1000",
    "is_csi_2000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simple Barra-like daily style/industry exposures.")
    parser.add_argument("--start-date", required=True, help="Output start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Output end date in YYYYMMDD.")
    parser.add_argument("--output-root", default="data/barra/exposures", help="Partitioned parquet output root.")
    parser.add_argument("--min-history", type=int, default=252, help="Minimum history needed for beta/momentum style factors.")
    parser.add_argument("--market-return", choices=["equal_weight", "csi1000", "hs300"], default="equal_weight")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing output partitions.")
    return parser.parse_args()


def scan_daily_panel(end_date: str) -> pl.LazyFrame:
    settings = get_settings()
    pattern = settings.data_root / "silver" / "daily_panel" / "trade_date=*" / "data.parquet"
    return pl.scan_parquet(str(pattern), missing_columns="insert", extra_columns="ignore").filter(pl.col("trade_date") <= end_date)


def winsor_z(col: str) -> pl.Expr:
    x = pl.col(col)
    med = x.median().over("trade_date")
    mad = (x - med).abs().median().over("trade_date")
    clipped = x.clip(med - 5.0 * mad, med + 5.0 * mad)
    mean = clipped.mean().over("trade_date")
    std = clipped.std().over("trade_date")
    return ((clipped - mean) / (std + EPS)).alias(col)


def build_market_return(lf: pl.LazyFrame, mode: str) -> pl.LazyFrame:
    base = lf.filter(pl.col("is_valid_universe"))
    if mode == "csi1000":
        base = base.filter(pl.col("is_csi_1000").fill_null(False))
    elif mode == "hs300":
        base = base.filter(pl.col("is_hs_300").fill_null(False))
    return base.group_by("trade_date").agg(pl.col("ret_1d").mean().alias("market_ret"))


def build_exposure_frame(start_date: str, end_date: str, min_history: int, market_return: str) -> pl.DataFrame:
    lf = scan_daily_panel(end_date).sort(["ts_code", "trade_date"])
    lf = lf.with_columns([
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("has_trade").fill_null(True).cast(pl.Boolean).alias("has_trade"),
        pl.col("is_st").fill_null(False).cast(pl.Boolean).alias("is_st"),
        pl.col("is_suspend").fill_null(False).cast(pl.Boolean).alias("is_suspend"),
        pl.col("is_bj").fill_null(False).cast(pl.Boolean).alias("is_bj"),
        pl.col("is_hs_300").fill_null(False).cast(pl.Boolean).alias("is_hs_300"),
        pl.col("is_csi_500").fill_null(False).cast(pl.Boolean).alias("is_csi_500"),
        pl.col("is_csi_1000").fill_null(False).cast(pl.Boolean).alias("is_csi_1000"),
        pl.col("is_csi_2000").fill_null(False).cast(pl.Boolean).alias("is_csi_2000"),
    ])
    lf = lf.with_columns(
        (pl.col("has_trade") & ~pl.col("is_st") & ~pl.col("is_suspend") & ~pl.col("is_bj")).alias("is_valid_universe")
    )
    lf = lf.with_columns([
        (pl.col("close") * pl.col("adj_factor")).alias("close_adj"),
        (pl.col("close") * pl.col("adj_factor") / (pl.col("pre_close") * pl.col("adj_factor")).clip(lower_bound=EPS) - 1.0)
        .clip(-0.3, 0.3)
        .alias("ret_1d"),
    ])
    market = build_market_return(lf, market_return)
    lf = lf.join(market, on="trade_date", how="left")
    lf = lf.with_columns([
        pl.int_range(pl.len()).over("ts_code").alias("history_count"),
        pl.col("ret_1d").rolling_mean(120, min_samples=100).over("ts_code").alias("ret_mean_120"),
        pl.col("market_ret").rolling_mean(120, min_samples=100).over("ts_code").alias("mkt_mean_120"),
    ])
    lf = lf.with_columns([
        ((pl.col("ret_1d") - pl.col("ret_mean_120")) * (pl.col("market_ret") - pl.col("mkt_mean_120")))
        .rolling_mean(120, min_samples=100)
        .over("ts_code")
        .alias("cov_mkt_120"),
        ((pl.col("market_ret") - pl.col("mkt_mean_120")) ** 2)
        .rolling_mean(120, min_samples=100)
        .over("ts_code")
        .alias("var_mkt_120"),
    ])
    lf = lf.with_columns([
        pl.col("ret_1d").rolling_std(60, min_samples=40).over("ts_code").alias("volatility_60_raw"),
        (pl.col("close_adj") / pl.col("close_adj").shift(252).over("ts_code").clip(lower_bound=EPS) - 1.0).alias("ret_252"),
        (pl.col("close_adj") / pl.col("close_adj").shift(20).over("ts_code").clip(lower_bound=EPS) - 1.0).alias("ret_20"),
        (pl.col("close_adj").shift(20).over("ts_code") / pl.col("close_adj").shift(252).over("ts_code").clip(lower_bound=EPS) - 1.0)
        .alias("momentum_252_20_raw"),
        pl.col("amount").log1p().rolling_mean(20, min_samples=10).over("ts_code").alias("liquidity_amount_20_raw"),
        pl.col("turnover_rate").rolling_mean(20, min_samples=10).over("ts_code").alias("liquidity_turnover_20_raw"),
    ])
    lf = lf.with_columns([
        pl.col("total_mv").clip(lower_bound=EPS).log().alias("size"),
        pl.when(pl.col("pb") > 0).then(-pl.col("pb").log()).otherwise(None).alias("value_bp"),
        pl.when(pl.col("pe") > 0).then(1.0 / pl.col("pe")).otherwise(None).alias("value_ep"),
        pl.col("momentum_252_20_raw").alias("momentum_252_20"),
        (-pl.col("ret_20")).alias("reversal_20"),
        (pl.col("cov_mkt_120") / pl.col("var_mkt_120").clip(lower_bound=EPS)).clip(-5.0, 5.0).alias("beta_120"),
        pl.col("volatility_60_raw").alias("volatility_60"),
        pl.col("liquidity_amount_20_raw").alias("liquidity_amount_20"),
        pl.col("liquidity_turnover_20_raw").alias("liquidity_turnover_20"),
    ])
    keep_cols = IDENTITY_COLS + ["history_count", "is_valid_universe"] + STYLE_FACTORS
    df = lf.filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date) & pl.col("is_valid_universe"))
    df = df.select([c for c in keep_cols if c in lf.collect_schema().names()]).collect()
    df = df.filter(pl.col("history_count") >= min_history).drop_nulls(STYLE_FACTORS)
    if df.is_empty():
        return df
    return df.with_columns([winsor_z(col) for col in STYLE_FACTORS]).sort(["trade_date", "ts_code"])


def write_partitions(df: pl.DataFrame, output_root: str, skip_existing: bool) -> list[Path]:
    root = Path(output_root).expanduser().resolve()
    written: list[Path] = []
    for key, part in df.partition_by("trade_date", as_dict=True, maintain_order=True).items():
        trade_date = key[0] if isinstance(key, tuple) else key
        out_path = root / f"trade_date={trade_date}" / "data.parquet"
        if skip_existing and out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(out_path)
        written.append(out_path)
        print(f"[barra_exposures] {trade_date} rows={part.height} -> {out_path}")
    return written


def main() -> None:
    args = parse_args()
    datetime.strptime(args.start_date, DATE_FMT)
    datetime.strptime(args.end_date, DATE_FMT)
    df = build_exposure_frame(args.start_date, args.end_date, args.min_history, args.market_return)
    written = write_partitions(df, args.output_root, args.skip_existing)
    meta = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows": df.height,
        "style_factors": STYLE_FACTORS,
        "market_return": args.market_return,
        "min_history": args.min_history,
        "partitions_written": len(written),
    }
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest_build_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
