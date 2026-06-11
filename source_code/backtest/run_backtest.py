from __future__ import annotations

import argparse
from pathlib import Path

from backtest.config import BacktestConfig
from backtest.pipeline import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-sectional daily backtest from a signal parquet.")
    parser.add_argument("--signal-path", required=True, help="Parquet with ts_code, end_date, score.")
    parser.add_argument("--output-root", default="./artifacts/backtests", help="Backtest output root.")
    parser.add_argument("--signal-col", default="score", help="Signal column name.")
    parser.add_argument("--start-date", default="", help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="End date in YYYYMMDD.")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in trade days.")
    parser.add_argument("--cost-bps", type=float, default=12.0, help="One-way transaction cost in bps.")
    parser.add_argument("--groups", type=int, default=10, help="Number of quantile groups.")
    parser.add_argument("--long-group", type=int, default=0, help="Group id used as long leg. 0 means the top group.")
    parser.add_argument("--short-group", type=int, default=0, help="Group id used as short leg. 0 means group 1.")
    parser.add_argument("--min-cross-section", type=int, default=100, help="Minimum valid names per date.")
    parser.add_argument("--benchmark-flag", default="", help="Optional universe flag, e.g. is_hs_300.")
    parser.add_argument("--include-nontradable", action="store_true", help="Keep ST/suspend/BJ/non-trade rows.")
    parser.add_argument("--raw-price", action="store_true", help="Use raw close instead of adjusted close for returns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        signal_path=args.signal_path,
        output_root=args.output_root,
        signal_col=args.signal_col,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        groups=args.groups,
        long_group=args.long_group,
        short_group=args.short_group,
        min_cross_section=args.min_cross_section,
        only_tradable=not args.include_nontradable,
        benchmark_flag=args.benchmark_flag,
        use_adjusted_price=not args.raw_price,
    )
    run_dir = run_backtest(config)
    print(f"Backtest output: {run_dir}")


if __name__ == "__main__":
    main()
