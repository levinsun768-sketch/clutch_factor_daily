from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

from data.config import get_settings


DATE_FMT = "%Y%m%d"
DAILY_TABLES = ["daily", "adj_factor", "daily_basic", "stk_limit", "suspend_d", "stock_st"]
STATIC_TABLES = ["trade_cal", "stock_basic"]
REFERENCE_TABLES = ["index_daily", "index_weight"]


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the after-close daily update pipeline.")
    parser.add_argument("--start-date", default="", help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="End date in YYYYMMDD.")
    parser.add_argument("--refresh-static", action="store_true", help="Refresh trade_cal and stock_basic too.")
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help="Refresh reference index daily/weight tables before building universe flags.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing bronze/silver/gold partitions.")
    parser.add_argument("--build-features", action="store_true", help="Build gold feature_panel after silver daily_panel.")
    parser.add_argument("--build-tensors", action="store_true", help="Build tensor dataset after feature_panel.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=settings.window_size,
        help="Tensor window size when --build-tensors is enabled.",
    )
    return parser.parse_args()


def run_module(module_name: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", module_name] + extra_args
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    start_date = args.start_date or settings.start_date
    end_date = args.end_date or settings.end_date or datetime.now().strftime(DATE_FMT)

    datetime.strptime(start_date, DATE_FMT)
    datetime.strptime(end_date, DATE_FMT)

    date_args = ["--start-date", start_date, "--end-date", end_date]
    partition_args = date_args.copy()
    if args.skip_existing:
        partition_args.append("--skip-existing")

    if args.refresh_static:
        run_module("data.fetch_bronze", ["--tables", *STATIC_TABLES, *date_args])

    run_module("data.fetch_bronze", ["--tables", *DAILY_TABLES, *partition_args])
    if args.refresh_reference:
        run_module("data.fetch_reference_bronze", ["--tables", *REFERENCE_TABLES, *partition_args])
    run_module("data.build_universe_flags", partition_args)
    run_module("data.build_daily_panel", partition_args)
    if args.build_features:
        run_module("data.build_feature_panel", partition_args)
    if args.build_tensors:
        if not args.build_features:
            run_module("data.build_feature_panel", partition_args)
        tensor_args = [*date_args, "--window-size", str(args.window_size)]
        run_module("data.build_tensor_dataset", tensor_args)


if __name__ == "__main__":
    main()
