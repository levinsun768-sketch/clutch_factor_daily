from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import polars as pl

from data.bronze_io import partition_table_path, static_table_path, write_parquet
from data.config import get_settings
from data.tushare_client import build_pro_client


DATE_FMT = "%Y%m%d"
MAX_RETRIES = 5
RETRY_SLEEP_SECONDS = 2.0


@dataclass(frozen=True)
class DailyEndpoint:
    name: str
    fields: str


MARKET_DAILY_ENDPOINTS = {
    "daily": DailyEndpoint(
        name="daily",
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    ),
    "adj_factor": DailyEndpoint(
        name="adj_factor",
        fields="ts_code,trade_date,adj_factor",
    ),
    "daily_basic": DailyEndpoint(
        name="daily_basic",
        fields="ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,total_mv,circ_mv",
    ),
    "stk_limit": DailyEndpoint(
        name="stk_limit",
        fields="trade_date,ts_code,up_limit,down_limit",
    ),
    "suspend_d": DailyEndpoint(
        name="suspend_d",
        fields="ts_code,trade_date,suspend_timing,suspend_type",
    ),
    "stock_st": DailyEndpoint(
        name="stock_st",
        fields="ts_code,name,trade_date,type,type_name",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Tushare bronze tables into partitioned parquet files.")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=[
            "trade_cal",
            "stock_basic",
            "daily",
            "adj_factor",
            "daily_basic",
            "stk_limit",
            "suspend_d",
            "stock_st",
        ],
        help="Tables to fetch.",
    )
    parser.add_argument("--start-date", default="", help="Override FPD_START_DATE, format YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="Override FPD_END_DATE, format YYYYMMDD.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip daily partitions that already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for market-wide daily endpoints.",
    )
    return parser.parse_args()


def iter_open_dates(start_date: str, end_date: str) -> list[str]:
    pro = build_pro_client()
    df = pro.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    open_dates = (
        pl.from_pandas(df)
        .filter(pl.col("is_open") == 1)
        .sort("cal_date")
        .get_column("cal_date")
        .to_list()
    )
    return open_dates


def fetch_trade_cal(start_date: str, end_date: str) -> None:
    settings = get_settings()
    pro = build_pro_client()
    df = pro.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    out_path = static_table_path(settings.bronze_root, "trade_cal")
    write_parquet(df, out_path)
    print(f"[trade_cal] wrote {len(df)} rows -> {out_path}")


def fetch_stock_basic() -> None:
    settings = get_settings()
    pro = build_pro_client()
    df = pro.stock_basic(
        exchange="",
        list_status="L,D,P",
        fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status",
    )
    out_path = static_table_path(settings.bronze_root, "stock_basic")
    write_parquet(df, out_path)
    print(f"[stock_basic] wrote {len(df)} rows -> {out_path}")


def get_partition_path(table_name: str, trade_date: str) -> Path:
    settings = get_settings()
    return partition_table_path(settings.bronze_root, table_name, trade_date)


def call_with_retry(func, kwargs: dict, label: str):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            sleep_s = RETRY_SLEEP_SECONDS * attempt
            print(f"[retry] {label} attempt={attempt}/{MAX_RETRIES} error={exc} sleep={sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise last_exc


def fetch_one_partition(table_name: str, trade_date: str) -> tuple[str, int, str]:
    endpoint = MARKET_DAILY_ENDPOINTS[table_name]
    pro = build_pro_client()
    out_path = get_partition_path(endpoint.name, trade_date)

    func = getattr(pro, endpoint.name)
    kwargs = {"trade_date": trade_date, "fields": endpoint.fields}
    if endpoint.name == "suspend_d":
        kwargs = {"trade_date": trade_date}

    df = call_with_retry(func, kwargs, f"{endpoint.name}:{trade_date}")
    write_parquet(df, out_path)
    return trade_date, len(df), str(out_path)


def fetch_daily_table(table_name: str, start_date: str, end_date: str, skip_existing: bool, workers: int) -> None:
    endpoint = MARKET_DAILY_ENDPOINTS[table_name]
    open_dates = iter_open_dates(start_date, end_date)
    target_dates: list[str] = []

    for trade_date in open_dates:
        out_path = get_partition_path(endpoint.name, trade_date)
        if skip_existing and out_path.exists():
            continue
        target_dates.append(trade_date)

    if not target_dates:
        print(f"[{endpoint.name}] skipped: no partitions to fetch")
        return

    if workers <= 1:
        for trade_date in target_dates:
            _, rows, out_path = fetch_one_partition(table_name, trade_date)
            print(f"[{endpoint.name}] {trade_date} rows={rows} -> {out_path}")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fetch_one_partition, table_name, trade_date): trade_date
            for trade_date in target_dates
        }
        for future in as_completed(future_map):
            trade_date = future_map[future]
            _, rows, out_path = future.result()
            print(f"[{endpoint.name}] {trade_date} rows={rows} -> {out_path}")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    start_date = args.start_date or settings.start_date
    end_date = args.end_date or settings.end_date or datetime.now().strftime(DATE_FMT)

    datetime.strptime(start_date, DATE_FMT)
    datetime.strptime(end_date, DATE_FMT)

    selected = set(args.tables)

    if "trade_cal" in selected:
        fetch_trade_cal(start_date, end_date)
    if "stock_basic" in selected:
        fetch_stock_basic()

    for table_name in ("daily", "adj_factor", "daily_basic", "stk_limit", "suspend_d", "stock_st"):
        if table_name in selected:
            fetch_daily_table(
                table_name=table_name,
                start_date=start_date,
                end_date=end_date,
                skip_existing=args.skip_existing,
                workers=args.workers,
            )


if __name__ == "__main__":
    main()
