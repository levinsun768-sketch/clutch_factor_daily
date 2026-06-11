from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

import polars as pl

from data.bronze_io import keyed_table_path, named_static_table_path, partition_table_path, static_table_path
from data.config import get_settings
from data.reference_config import REFERENCE_INDICES


DATE_FMT = "%Y%m%d"
LIMIT_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stock universe flags by trade_date.")
    parser.add_argument("--start-date", default="", help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="End date in YYYYMMDD.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip output partitions that already exist.")
    return parser.parse_args()


def universe_flags_path(trade_date: str) -> Path:
    settings = get_settings()
    return settings.data_root / "silver" / "universe_flags" / f"trade_date={trade_date}" / "data.parquet"


def load_open_dates(start_date: str, end_date: str) -> list[str]:
    settings = get_settings()
    trade_cal = pl.read_parquet(static_table_path(settings.bronze_root, "trade_cal"))
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


def load_stock_basic() -> pl.DataFrame:
    settings = get_settings()
    stock_basic = pl.read_parquet(static_table_path(settings.bronze_root, "stock_basic"))
    return stock_basic.with_columns([
        pl.col("list_date").cast(pl.Utf8),
        pl.when(pl.col("delist_date").cast(pl.Utf8) == "")
        .then(None)
        .otherwise(pl.col("delist_date").cast(pl.Utf8))
        .alias("delist_date"),
    ])


def load_trade_universe(trade_date: str) -> pl.DataFrame:
    daily_df = read_trade_partition("daily", trade_date)
    suspend_df = read_trade_partition("suspend_d", trade_date)

    daily_universe = pl.DataFrame(schema={"ts_code": pl.Utf8, "trade_date": pl.Utf8, "close": pl.Float64, "has_trade": pl.Boolean})
    if not daily_df.is_empty():
        daily_universe = (
            daily_df
            .select(["ts_code", "trade_date", "close"])
            .unique(subset=["ts_code"], keep="last")
            .with_columns(pl.lit(True).alias("has_trade"))
        )

    suspend_universe = pl.DataFrame(schema={"ts_code": pl.Utf8, "trade_date": pl.Utf8, "close": pl.Float64, "has_trade": pl.Boolean})
    if not suspend_df.is_empty():
        suspend_universe = (
            suspend_df
            .select(["ts_code", "trade_date"])
            .unique(subset=["ts_code", "trade_date"], keep="last")
            .with_columns([
                pl.lit(None, dtype=pl.Float64).alias("close"),
                pl.lit(False).alias("has_trade"),
            ])
        )

    if daily_universe.is_empty() and suspend_universe.is_empty():
        return daily_universe

    return (
        pl.concat([daily_universe, suspend_universe], how="diagonal")
        .sort(["ts_code", "has_trade"])
        .unique(subset=["ts_code", "trade_date"], keep="last")
        .sort("ts_code")
    )


def load_index_snapshots() -> dict[str, dict[str, object]]:
    settings = get_settings()
    snapshots: dict[str, dict[str, object]] = {}
    for item in REFERENCE_INDICES:
        path = keyed_table_path(settings.bronze_root, "index_weight", "index_code", item["ts_code"])
        weight_df = pl.read_parquet(path).sort(["trade_date", "con_code"])
        trade_dates = weight_df.get_column("trade_date").unique().sort().to_list()
        members = {
            trade_date: set(
                weight_df
                .filter(pl.col("trade_date") == trade_date)
                .get_column("con_code")
                .to_list()
            )
            for trade_date in trade_dates
        }
        snapshots[item["ts_code"]] = {
            "trade_dates": trade_dates,
            "members": members,
            "flag_col": item["flag_col"],
        }
    return snapshots


def memberships_for_date(index_snapshots: dict[str, dict[str, object]], trade_date: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for index_code, meta in index_snapshots.items():
        snapshot_dates = meta["trade_dates"]
        pos = bisect_right(snapshot_dates, trade_date) - 1
        if pos < 0:
            out[meta["flag_col"]] = set()
            continue
        snapshot_date = snapshot_dates[pos]
        out[meta["flag_col"]] = meta["members"][snapshot_date]
    return out


def load_sw_classify() -> pl.DataFrame:
    settings = get_settings()
    path = named_static_table_path(settings.bronze_root, "index_classify", "sw2021_l1.parquet")
    return pl.read_parquet(path).sort("index_code")


def load_sw_member() -> pl.DataFrame:
    settings = get_settings()
    path = named_static_table_path(settings.bronze_root, "sw_member", "sw2021_l1_members.parquet")
    return pl.read_parquet(path).with_columns([
        pl.col("in_date").cast(pl.Utf8),
        pl.when(pl.col("out_date").cast(pl.Utf8) == "")
        .then(None)
        .otherwise(pl.col("out_date").cast(pl.Utf8))
        .alias("out_date"),
    ])


def read_trade_partition(table_name: str, trade_date: str) -> pl.DataFrame:
    settings = get_settings()
    path = partition_table_path(settings.bronze_root, table_name, trade_date)
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def write_stock_st_fallback_audit(trade_date: str, source_date: str, rows: int) -> None:
    settings = get_settings()
    audit_dir = settings.data_root / "silver" / "universe_flags" / "_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trade_date": trade_date,
        "source_date": source_date,
        "rows": rows,
        "reason": "current_stock_st_partition_empty",
    }
    (audit_dir / f"stock_st_fallback_{trade_date}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_stock_st_for_flags(trade_date: str) -> pl.DataFrame:
    current = read_trade_partition("stock_st", trade_date)
    if not current.is_empty():
        return current

    settings = get_settings()
    candidates = sorted(
        p.parent.name.split("=")[1]
        for p in (settings.bronze_root / "stock_st").glob("trade_date=*/data.parquet")
        if p.parent.name.split("=")[1] < trade_date
    )
    for source_date in reversed(candidates):
        fallback = read_trade_partition("stock_st", source_date)
        if fallback.is_empty():
            continue
        fallback = fallback.with_columns(pl.lit(trade_date).alias("trade_date"))
        write_stock_st_fallback_audit(trade_date, source_date, fallback.height)
        print(f"[stock_st_fallback] {trade_date} <- {source_date} rows={fallback.height}")
        return fallback

    return current


def active_sw_members_for_date(sw_member: pl.DataFrame, trade_date: str) -> pl.DataFrame:
    if sw_member.is_empty():
        return pl.DataFrame()

    active = (
        sw_member
        .filter(
            (pl.col("in_date") <= trade_date)
            & (
                pl.col("out_date").is_null()
                | (pl.col("out_date") > trade_date)
            )
        )
        .sort(["con_code", "in_date"])
        .unique(subset=["con_code"], keep="last")
        .select([
            pl.col("con_code").alias("ts_code"),
            pl.col("index_code").alias("sw_l1_code"),
            pl.col("industry_name").alias("sw_l1_name"),
        ])
    )
    return active


def build_flags_for_date(
    trade_date: str,
    stock_basic: pl.DataFrame,
    index_snapshots: dict[str, dict[str, object]],
    sw_member: pl.DataFrame,
    sw_classify: pl.DataFrame,
) -> pl.DataFrame:
    panel = load_trade_universe(trade_date)
    if panel.is_empty():
        return panel

    memberships = memberships_for_date(index_snapshots, trade_date)
    stock_basic_cols = (
        stock_basic
        .select(["ts_code", "name", "list_date", "list_status"])
        .unique(subset=["ts_code"], keep="last")
    )
    panel = panel.join(stock_basic_cols, on="ts_code", how="left")

    for item in REFERENCE_INDICES:
        flag_col = item["flag_col"]
        members = list(memberships[flag_col])
        panel = panel.with_columns(pl.col("ts_code").is_in(members).alias(flag_col))

    st_df = read_stock_st_for_flags(trade_date)
    if not st_df.is_empty():
        st_flags = (
            st_df
            .select(["ts_code", "trade_date"])
            .unique(subset=["ts_code", "trade_date"], keep="last")
            .with_columns(pl.lit(True).alias("is_st"))
        )
        panel = panel.join(st_flags, on=["ts_code", "trade_date"], how="left")

    suspend_df = read_trade_partition("suspend_d", trade_date)
    if not suspend_df.is_empty():
        suspend_flags = (
            suspend_df
            .select(["ts_code", "trade_date"])
            .unique(subset=["ts_code", "trade_date"], keep="last")
            .with_columns(pl.lit(True).alias("is_suspend"))
        )
        panel = panel.join(suspend_flags, on=["ts_code", "trade_date"], how="left")

    limit_df = read_trade_partition("stk_limit", trade_date)
    if not limit_df.is_empty():
        panel = panel.join(
            limit_df.select(["ts_code", "trade_date", "up_limit", "down_limit"]),
            on=["ts_code", "trade_date"],
            how="left",
        )

    sw_active = active_sw_members_for_date(sw_member, trade_date)
    panel = panel.join(sw_active, on="ts_code", how="left")

    for row in sw_classify.iter_rows(named=True):
        panel = panel.with_columns(
            (pl.col("sw_l1_code") == row["index_code"]).fill_null(False).alias(f"is_sw_{row['index_code'][:6]}")
        )

    if "is_st" not in panel.columns:
        panel = panel.with_columns(pl.lit(False).alias("is_st"))
    if "is_suspend" not in panel.columns:
        panel = panel.with_columns(pl.lit(False).alias("is_suspend"))

    panel = panel.with_columns([
        pl.col("has_trade").fill_null(False),
        pl.col("ts_code").str.ends_with(".BJ").alias("is_bj"),
        pl.col("is_st").fill_null(False),
        pl.col("is_suspend").fill_null(False),
        (
            pl.col("up_limit").is_not_null()
            & ((pl.col("close") - pl.col("up_limit")).abs() <= LIMIT_EPS)
        ).fill_null(False).alias("is_up_limit"),
        (
            pl.col("down_limit").is_not_null()
            & ((pl.col("close") - pl.col("down_limit")).abs() <= LIMIT_EPS)
        ).fill_null(False).alias("is_down_limit"),
    ])

    bool_cols = [name for name in panel.columns if name.startswith("is_")]
    panel = panel.with_columns([pl.col(name).fill_null(False) for name in bool_cols])

    base_cols = [
        "ts_code",
        "name",
        "trade_date",
        "has_trade",
        "is_st",
        "is_bj",
        "is_suspend",
        "is_up_limit",
        "is_down_limit",
    ]
    index_cols = [item["flag_col"] for item in REFERENCE_INDICES]
    sw_cols = ["sw_l1_code", "sw_l1_name"]
    extra_sw_cols = [name for name in panel.columns if name.startswith("is_sw_")]
    keep_cols = [name for name in base_cols + index_cols + sw_cols + extra_sw_cols if name in panel.columns]
    return panel.select(keep_cols).sort("ts_code")


def write_flags(df: pl.DataFrame, trade_date: str) -> Path:
    path = universe_flags_path(trade_date)
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

    stock_basic = load_stock_basic()
    index_snapshots = load_index_snapshots()
    sw_classify = load_sw_classify()
    sw_member = load_sw_member()

    for trade_date in load_open_dates(start_date, end_date):
        out_path = universe_flags_path(trade_date)
        if args.skip_existing and out_path.exists():
            continue
        df = build_flags_for_date(
            trade_date=trade_date,
            stock_basic=stock_basic,
            index_snapshots=index_snapshots,
            sw_member=sw_member,
            sw_classify=sw_classify,
        )
        write_flags(df, trade_date)
        print(f"[universe_flags] {trade_date} rows={df.height} -> {out_path}")


if __name__ == "__main__":
    main()
