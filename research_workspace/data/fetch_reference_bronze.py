from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

from data.bronze_io import keyed_table_path, named_static_table_path, write_parquet
from data.config import get_settings
from data.fetch_bronze import DATE_FMT, call_with_retry
from data.reference_config import REFERENCE_INDEX_CODES, SW_INDEX_LEVEL, SW_INDEX_SRC
from data.tushare_client import build_pro_client


INDEX_DAILY_KEYS = ["ts_code", "trade_date"]
INDEX_WEIGHT_KEYS = ["index_code", "con_code", "trade_date"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch reference index and industry bronze tables.")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=["index_basic", "index_classify", "index_daily", "index_weight", "sw_member"],
        help="Reference tables to fetch.",
    )
    parser.add_argument("--start-date", default="", help="Override FPD_START_DATE, format YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="Override FPD_END_DATE, format YYYYMMDD.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip per-index files only when they already cover --end-date.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for per-index fetches.")
    return parser.parse_args()


def existing_max_trade_date(path) -> str:
    if not path.exists():
        return ""
    try:
        df = pd.read_parquet(path, columns=["trade_date"])
    except Exception:  # noqa: BLE001
        return ""
    if df.empty:
        return ""
    return str(df["trade_date"].max())


def merge_existing(new_df: pd.DataFrame, out_path, key_cols: list[str]) -> pd.DataFrame:
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        frames = [existing_df, new_df]
    else:
        frames = [new_df]

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    return df.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols)


def fetch_index_basic(skip_existing: bool) -> None:
    settings = get_settings()
    out_path = named_static_table_path(settings.bronze_root, "index_basic", "index_basic.parquet")
    if skip_existing and out_path.exists():
        print(f"[index_basic] skipped -> {out_path}")
        return

    pro = build_pro_client()
    df = call_with_retry(
        pro.query,
        {
            "api_name": "index_basic",
            "fields": "ts_code,name,market,publisher,category,list_date",
        },
        "index_basic",
    )
    write_parquet(df, out_path)
    print(f"[index_basic] rows={len(df)} -> {out_path}")


def fetch_index_classify(skip_existing: bool) -> None:
    settings = get_settings()
    out_path = named_static_table_path(settings.bronze_root, "index_classify", "sw2021_l1.parquet")
    if skip_existing and out_path.exists():
        print(f"[index_classify] skipped -> {out_path}")
        return

    pro = build_pro_client()
    df = call_with_retry(
        pro.query,
        {
            "api_name": "index_classify",
            "level": SW_INDEX_LEVEL,
            "src": SW_INDEX_SRC,
            "fields": "index_code,industry_name,level,src",
        },
        "index_classify",
    )
    write_parquet(df, out_path)
    print(f"[index_classify] rows={len(df)} -> {out_path}")


def fetch_index_daily_one(index_code: str, start_date: str, end_date: str) -> tuple[str, int, str]:
    settings = get_settings()
    pro = build_pro_client()
    out_path = keyed_table_path(settings.bronze_root, "index_daily", "index_code", index_code)
    df = call_with_retry(
        pro.query,
        {
            "api_name": "index_daily",
            "ts_code": index_code,
            "start_date": start_date,
            "end_date": end_date,
            "fields": "ts_code,trade_date,close,open,high,low,pre_close,pct_chg,vol,amount",
        },
        f"index_daily:{index_code}",
    )
    df = merge_existing(df, out_path, INDEX_DAILY_KEYS)
    write_parquet(df, out_path)
    return index_code, len(df), str(out_path)


def fetch_index_weight_one(index_code: str, start_date: str, end_date: str) -> tuple[str, int, str]:
    settings = get_settings()
    pro = build_pro_client()
    out_path = keyed_table_path(settings.bronze_root, "index_weight", "index_code", index_code)
    parts: list[pd.DataFrame] = []

    for chunk_start, chunk_end in iter_month_ranges(start_date, end_date):
        df = call_with_retry(
            pro.query,
            {
                "api_name": "index_weight",
                "index_code": index_code,
                "start_date": chunk_start,
                "end_date": chunk_end,
                "fields": "index_code,con_code,trade_date,weight",
            },
            f"index_weight:{index_code}:{chunk_start}:{chunk_end}",
        )
        parts.append(df)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df = merge_existing(df, out_path, INDEX_WEIGHT_KEYS)
    write_parquet(df, out_path)
    return index_code, len(df), str(out_path)


def iter_month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start_dt = datetime.strptime(start_date, DATE_FMT)
    end_dt = datetime.strptime(end_date, DATE_FMT)
    cursor = start_dt.replace(day=1)
    ranges: list[tuple[str, str]] = []

    while cursor <= end_dt:
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            next_month = cursor.replace(month=cursor.month + 1, day=1)
        month_end = next_month - timedelta(days=1)
        chunk_start = max(cursor, start_dt)
        chunk_end = min(month_end, end_dt)
        ranges.append((chunk_start.strftime(DATE_FMT), chunk_end.strftime(DATE_FMT)))
        cursor = next_month

    return ranges


def fetch_sw_member(skip_existing: bool) -> None:
    settings = get_settings()
    out_path = named_static_table_path(settings.bronze_root, "sw_member", "sw2021_l1_members.parquet")
    if skip_existing and out_path.exists():
        print(f"[sw_member] skipped -> {out_path}")
        return

    classify_path = named_static_table_path(settings.bronze_root, "index_classify", "sw2021_l1.parquet")
    if not classify_path.exists():
        fetch_index_classify(skip_existing=False)

    classify_df = pd.read_parquet(classify_path)
    pro = build_pro_client()
    parts: list[pd.DataFrame] = []

    for row in classify_df.itertuples(index=False):
        df = call_with_retry(
            pro.query,
            {
                "api_name": "index_member",
                "index_code": row.index_code,
                "fields": "index_code,con_code,in_date,out_date,is_new",
            },
            f"sw_member:{row.index_code}",
        )
        if not df.empty:
            df["industry_name"] = row.industry_name
            df["level"] = row.level
            df["src"] = row.src
        parts.append(df)
        print(f"[sw_member] {row.index_code} rows={len(df)}")

    out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    write_parquet(out_df, out_path)
    print(f"[sw_member] rows={len(out_df)} -> {out_path}")


def fetch_per_index(
    table_name: str,
    start_date: str,
    end_date: str,
    skip_existing: bool,
    workers: int,
) -> None:
    settings = get_settings()
    fn = fetch_index_daily_one if table_name == "index_daily" else fetch_index_weight_one
    targets: list[str] = []

    for index_code in REFERENCE_INDEX_CODES:
        out_path = keyed_table_path(settings.bronze_root, table_name, "index_code", index_code)
        if skip_existing and existing_max_trade_date(out_path) >= end_date:
            continue
        targets.append(index_code)

    if not targets:
        print(f"[{table_name}] skipped: no files to fetch")
        return

    if workers <= 1:
        for index_code in targets:
            code, rows, out_path = fn(index_code, start_date, end_date)
            print(f"[{table_name}] {code} rows={rows} -> {out_path}")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fn, index_code, start_date, end_date): index_code
            for index_code in targets
        }
        for future in as_completed(future_map):
            code, rows, out_path = future.result()
            print(f"[{table_name}] {code} rows={rows} -> {out_path}")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    start_date = args.start_date or settings.start_date
    end_date = args.end_date or settings.end_date or datetime.now().strftime(DATE_FMT)

    datetime.strptime(start_date, DATE_FMT)
    datetime.strptime(end_date, DATE_FMT)

    selected = set(args.tables)
    if "index_basic" in selected:
        fetch_index_basic(skip_existing=args.skip_existing)
    if "index_classify" in selected:
        fetch_index_classify(skip_existing=args.skip_existing)
    if "index_daily" in selected:
        fetch_per_index(
            table_name="index_daily",
            start_date=start_date,
            end_date=end_date,
            skip_existing=args.skip_existing,
            workers=args.workers,
        )
    if "index_weight" in selected:
        fetch_per_index(
            table_name="index_weight",
            start_date=start_date,
            end_date=end_date,
            skip_existing=args.skip_existing,
            workers=args.workers,
        )
    if "sw_member" in selected:
        fetch_sw_member(skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
