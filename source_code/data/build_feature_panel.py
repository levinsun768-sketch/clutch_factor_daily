from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from data.config import get_settings
from data.feature_spec import AUXILIARY_COLUMNS, PANEL_FEATURE_NAMES


DATE_FMT = "%Y%m%d"
EPS = 1e-8
VOLUME_LOOKBACK = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold feature panel from silver daily panel.")
    parser.add_argument("--start-date", default="", help="Target start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="Target end date in YYYYMMDD.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing output partitions.")
    return parser.parse_args()


def gold_feature_path(trade_date: str) -> Path:
    settings = get_settings()
    return settings.data_root / "gold" / "feature_panel" / f"trade_date={trade_date}" / "data.parquet"


def scan_silver_panel(end_date: str) -> pl.LazyFrame:
    settings = get_settings()
    pattern = settings.data_root / "silver" / "daily_panel" / "trade_date=*" / "data.parquet"
    return (
        pl.scan_parquet(str(pattern), missing_columns="insert", extra_columns="ignore")
        .filter(pl.col("trade_date") <= end_date)
    )


def build_feature_frame(start_date: str, end_date: str) -> pl.DataFrame:
    lf = scan_silver_panel(end_date)
    lf = lf.sort(["ts_code", "trade_date"])
    schema_names = set(lf.collect_schema().names())

    is_st_expr = pl.col("is_st").fill_null(False) if "is_st" in schema_names else pl.lit(False)
    if "is_suspended" in schema_names:
        is_suspended_expr = pl.col("is_suspended").fill_null(False)
    elif "is_suspend" in schema_names:
        is_suspended_expr = pl.col("is_suspend").fill_null(False)
    else:
        is_suspended_expr = pl.lit(False)

    lf = lf.with_columns([
        is_st_expr.cast(pl.Boolean).alias("__is_st"),
        is_suspended_expr.cast(pl.Boolean).alias("__is_suspended"),
        pl.col("ts_code").str.ends_with(".BJ").alias("__is_bj"),
    ])
    lf = lf.with_columns(
        (
            pl.col("__is_st")
            | pl.col("__is_suspended")
            | pl.col("__is_bj")
        ).alias("__is_invalid")
    )
    lf = lf.with_columns(
        pl.col("__is_invalid").cast(pl.Int32).cum_sum().over("ts_code").alias("regime_id")
    )
    lf = lf.filter(~pl.col("__is_invalid"))

    lf = lf.with_columns([
        (pl.col("open") * pl.col("adj_factor")).alias("open_adj"),
        (pl.col("high") * pl.col("adj_factor")).alias("high_adj"),
        (pl.col("low") * pl.col("adj_factor")).alias("low_adj"),
        (pl.col("close") * pl.col("adj_factor")).alias("close_adj"),
        (pl.col("pre_close") * pl.col("adj_factor")).alias("pre_close_adj"),
        (pl.col("vol") / pl.col("adj_factor").clip(lower_bound=EPS)).alias("vol_adj"),
    ])

    lf = lf.with_columns([
        (pl.col("open_adj") / pl.col("pre_close_adj").clip(lower_bound=EPS) - 1.0).clip(-0.3, 0.3).alias("open_ret_1"),
        (pl.col("high_adj") / pl.col("pre_close_adj").clip(lower_bound=EPS) - 1.0).clip(-0.3, 0.3).alias("high_ret_1"),
        (pl.col("low_adj") / pl.col("pre_close_adj").clip(lower_bound=EPS) - 1.0).clip(-0.3, 0.3).alias("low_ret_1"),
        (pl.col("close_adj") / pl.col("pre_close_adj").clip(lower_bound=EPS) - 1.0).clip(-0.3, 0.3).alias("close_ret_1"),
        (pl.col("close_adj") / pl.col("open_adj").clip(lower_bound=EPS) - 1.0).clip(-0.3, 0.3).alias("intraday_ret"),
        ((pl.col("high_adj") - pl.col("low_adj")) / pl.col("pre_close_adj").clip(lower_bound=EPS)).clip(0.0, 0.5).alias("amp_ratio"),
    ])

    range_expr = (pl.col("high_adj") - pl.col("low_adj")).clip(lower_bound=EPS)
    upper_anchor = pl.max_horizontal("open_adj", "close_adj")
    lower_anchor = pl.min_horizontal("open_adj", "close_adj")
    lf = lf.with_columns([
        ((pl.col("close_adj") - pl.col("open_adj")) / range_expr).clip(-1.0, 1.0).alias("body_ratio"),
        ((pl.col("high_adj") - upper_anchor) / range_expr).clip(0.0, 1.0).alias("upper_shadow_ratio"),
        ((lower_anchor - pl.col("low_adj")) / range_expr).clip(0.0, 1.0).alias("lower_shadow_ratio"),
    ])

    lf = lf.with_columns([
        (
            pl.col("close_adj") / pl.col("close_adj").shift(5).over(["ts_code", "regime_id"]).clip(lower_bound=EPS) - 1.0
        ).clip(-0.8, 2.0).alias("cumret_5"),
        (
            pl.col("close_adj") / pl.col("close_adj").shift(10).over(["ts_code", "regime_id"]).clip(lower_bound=EPS) - 1.0
        ).clip(-0.8, 3.0).alias("cumret_10"),
        (
            pl.col("close_adj") / pl.col("close_adj").shift(20).over(["ts_code", "regime_id"]).clip(lower_bound=EPS) - 1.0
        ).clip(-0.8, 4.0).alias("cumret_20"),
        (
            (pl.col("close_adj") - pl.col("close_adj").rolling_min(VOLUME_LOOKBACK, min_samples=VOLUME_LOOKBACK).over(["ts_code", "regime_id"]))
            /
            (
                pl.col("close_adj").rolling_max(VOLUME_LOOKBACK, min_samples=VOLUME_LOOKBACK).over(["ts_code", "regime_id"])
                - pl.col("close_adj").rolling_min(VOLUME_LOOKBACK, min_samples=VOLUME_LOOKBACK).over(["ts_code", "regime_id"])
                + EPS
            )
        ).clip(0.0, 1.0).alias("close_pos_20"),
        (
            pl.col("close_adj")
            / pl.col("close_adj").rolling_max(VOLUME_LOOKBACK, min_samples=VOLUME_LOOKBACK).over(["ts_code", "regime_id"]).clip(lower_bound=EPS)
            - 1.0
        ).clip(-0.95, 0.0).alias("drawdown_20"),
    ])

    lf = lf.with_columns([
        (pl.col("close") / pl.col("up_limit").clip(lower_bound=EPS) - 1.0).clip(-0.3, 0.0).alias("limit_up_gap"),
        (pl.col("close") / pl.col("down_limit").clip(lower_bound=EPS) - 1.0).clip(0.0, 0.3).alias("limit_down_gap"),
    ])

    lf = lf.with_columns([
        pl.col("vol_adj").log1p().clip(0.0, 30.0).alias("log_vol"),
        pl.col("amount").log1p().clip(0.0, 30.0).alias("log_amount"),
        (
            pl.col("vol_adj")
            /
            pl.col("vol_adj").rolling_mean(VOLUME_LOOKBACK, min_samples=VOLUME_LOOKBACK).shift(1).over(["ts_code", "regime_id"]).clip(lower_bound=EPS)
        ).clip(0.0, 5.0).alias("vol_ma20_ratio"),
        (
            pl.col("amount")
            /
            pl.col("amount").rolling_mean(VOLUME_LOOKBACK, min_samples=VOLUME_LOOKBACK).shift(1).over(["ts_code", "regime_id"]).clip(lower_bound=EPS)
        ).clip(0.0, 5.0).alias("amount_ma20_ratio"),
        pl.col("turnover_rate").clip(0.0, 100.0).alias("turnover_rate"),
        pl.col("turnover_rate_f").clip(0.0, 100.0).alias("turnover_rate_f"),
        pl.col("volume_ratio").clip(0.0, 20.0).alias("volume_ratio"),
    ])

    select_cols = [
        "ts_code",
        "trade_date",
        "days_since_list",
        "regime_id",
    ] + AUXILIARY_COLUMNS + PANEL_FEATURE_NAMES
    df = lf.select(select_cols).collect()
    return df.filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))


def main() -> None:
    args = parse_args()
    settings = get_settings()
    start_date = args.start_date or settings.start_date
    end_date = args.end_date or settings.end_date or datetime.now().strftime(DATE_FMT)

    datetime.strptime(start_date, DATE_FMT)
    datetime.strptime(end_date, DATE_FMT)

    feature_df = build_feature_frame(start_date, end_date)
    if feature_df.is_empty():
        print("No feature rows produced.")
        return

    for trade_date, part in feature_df.partition_by("trade_date", as_dict=True, maintain_order=True).items():
        trade_date_str = trade_date[0] if isinstance(trade_date, tuple) else trade_date
        out_path = gold_feature_path(str(trade_date_str))
        if args.skip_existing and out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(out_path)
        print(f"[feature_panel] {trade_date_str} rows={part.height} -> {out_path}")


if __name__ == "__main__":
    main()
