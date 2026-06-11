from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from data.bronze_io import partition_table_path, static_table_path
from data.config import get_settings


DATE_FMT = "%Y%m%d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build silver daily panel from bronze parquet partitions.")
    parser.add_argument("--start-date", default="", help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="End date in YYYYMMDD.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip output partitions that already exist.")
    return parser.parse_args()


def silver_panel_path(trade_date: str) -> Path:
    settings = get_settings()
    return settings.data_root / "silver" / "daily_panel" / f"trade_date={trade_date}" / "data.parquet"


def universe_flags_path(trade_date: str) -> Path:
    settings = get_settings()
    return settings.data_root / "silver" / "universe_flags" / f"trade_date={trade_date}" / "data.parquet"


def read_bronze_partition(table_name: str, trade_date: str) -> pl.DataFrame:
    settings = get_settings()
    path = partition_table_path(settings.bronze_root, table_name, trade_date)
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def read_universe_flags_partition(trade_date: str) -> pl.DataFrame:
    path = universe_flags_path(trade_date)
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def load_stock_basic() -> pl.DataFrame:
    settings = get_settings()
    path = static_table_path(settings.bronze_root, "stock_basic")
    return pl.read_parquet(path)


def load_open_dates(start_date: str, end_date: str) -> list[str]:
    settings = get_settings()
    path = static_table_path(settings.bronze_root, "trade_cal")
    trade_cal = pl.read_parquet(path)
    return (
        trade_cal
        .filter(
            (pl.col("is_open") == 1)
            & (pl.col("cal_date") >= start_date)
            & (pl.col("cal_date") <= end_date)
        )
        .sort("cal_date")
        .get_column("cal_date")
        .to_list()
    )


def with_listing_features(df: pl.DataFrame, stock_basic: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    sb = stock_basic.with_columns(
        pl.col("list_date").cast(pl.Utf8),
        pl.when(pl.col("delist_date").cast(pl.Utf8) == "")
        .then(None)
        .otherwise(pl.col("delist_date").cast(pl.Utf8))
        .alias("delist_date"),
    )
    out = df.join(sb, on="ts_code", how="left")
    out = out.with_columns([
        pl.col("trade_date").str.strptime(pl.Date, DATE_FMT).alias("__trade_date_dt"),
        pl.col("list_date").str.strptime(pl.Date, DATE_FMT, strict=False).alias("__list_date_dt"),
    ])
    out = out.with_columns(
        (pl.col("__trade_date_dt") - pl.col("__list_date_dt")).dt.total_days().alias("days_since_list")
    )
    return out.drop(["__trade_date_dt", "__list_date_dt"])


def build_panel_for_date(trade_date: str) -> pl.DataFrame:
    daily = read_bronze_partition("daily", trade_date)
    if daily.is_empty():
        return daily

    adj = read_bronze_partition("adj_factor", trade_date)
    basic = read_bronze_partition("daily_basic", trade_date)
    limit_df = read_bronze_partition("stk_limit", trade_date)
    suspend_df = read_bronze_partition("suspend_d", trade_date)
    st_df = read_bronze_partition("stock_st", trade_date)
    universe_flags = read_universe_flags_partition(trade_date)
    stock_basic = load_stock_basic()

    panel = daily.join(adj, on=["ts_code", "trade_date"], how="left")
    panel = panel.join(basic, on=["ts_code", "trade_date"], how="left")

    if not limit_df.is_empty():
        panel = panel.join(limit_df, on=["ts_code", "trade_date"], how="left")

    if not suspend_df.is_empty():
        suspend_flags = (
            suspend_df.select(["ts_code", "trade_date", "suspend_type"])
            .unique(subset=["ts_code", "trade_date"], keep="last")
            .with_columns(pl.lit(True).alias("is_suspended"))
        )
        panel = panel.join(suspend_flags, on=["ts_code", "trade_date"], how="left")

    if not st_df.is_empty():
        st_flags = (
            st_df.select(["ts_code", "trade_date", "type", "type_name"])
            .unique(subset=["ts_code", "trade_date"], keep="last")
            .with_columns(pl.lit(True).alias("is_st"))
        )
        panel = panel.join(st_flags, on=["ts_code", "trade_date"], how="left")

    if not universe_flags.is_empty():
        overlap_cols = [
            name for name in universe_flags.columns
            if name not in {"ts_code", "trade_date"} and name in panel.columns
        ]
        if overlap_cols:
            universe_flags = universe_flags.drop(overlap_cols)
        panel = panel.join(universe_flags, on=["ts_code", "trade_date"], how="left")

    panel = with_listing_features(panel, stock_basic)
    bool_cols = [name for name in panel.columns if name.startswith("is_")]
    if bool_cols:
        panel = panel.with_columns([pl.col(name).fill_null(False) for name in bool_cols])
    panel = panel.sort("ts_code")
    return panel


def write_panel(df: pl.DataFrame, trade_date: str) -> Path:
    path = silver_panel_path(trade_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def main() -> None:
    args = parse_args()
    settings = get_settings()
    start_date = args.start_date or settings.start_date
    end_date = args.end_date or settings.end_date or datetime.now().strftime(DATE_FMT)

    datetime.strptime(start_date, DATE_FMT)
    datetime.strptime(end_date, DATE_FMT)

    for trade_date in load_open_dates(start_date, end_date):
        out_path = silver_panel_path(trade_date)
        if args.skip_existing and out_path.exists():
            continue
        panel = build_panel_for_date(trade_date)
        if panel.is_empty():
            print(f"[daily_panel] {trade_date} skipped: no daily rows")
            continue
        write_panel(panel, trade_date)
        print(f"[daily_panel] {trade_date} rows={panel.height} -> {out_path}")


if __name__ == "__main__":
    main()
