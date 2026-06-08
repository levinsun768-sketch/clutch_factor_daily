from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from backtest.config import BacktestConfig


DATE_FMT = "%Y%m%d"


def _resolve_run_dir(output_root: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root).expanduser().resolve() / f"backtest_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_trade_dates() -> list[str]:
    path = Path("data/bronze/trade_cal/trade_cal.parquet")
    df = pl.read_parquet(path)
    return (
        df.filter(pl.col("is_open") == 1)
        .sort("cal_date")
        .get_column("cal_date")
        .to_list()
    )


def _build_horizon_map(horizon: int) -> pl.DataFrame:
    dates = _load_trade_dates()
    rows: list[dict[str, str | None]] = []
    for idx, end_date in enumerate(dates):
        open_date = dates[idx + 1] if idx + 1 < len(dates) else None
        exit_date = dates[idx + horizon] if idx + horizon < len(dates) else None
        rows.append(
            {
                "end_date": end_date,
                "entry_date": open_date,
                "exit_date": exit_date,
            }
        )
    return pl.DataFrame(rows)


def _scan_daily_panel() -> pl.LazyFrame:
    paths = sorted(Path("data/silver/daily_panel").glob("trade_date=*/data.parquet"))
    return pl.scan_parquet([str(path) for path in paths])


def _scan_universe_flags() -> pl.LazyFrame:
    paths = sorted(Path("data/silver/universe_flags").glob("trade_date=*/data.parquet"))
    return pl.scan_parquet([str(path) for path in paths])


def _load_signals(config: BacktestConfig) -> pl.DataFrame:
    path = Path(config.signal_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")

    df = pl.read_parquet(path)
    required = {"ts_code", "end_date", config.signal_col}
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Signal file missing required columns: {missing}")

    df = (
        df.select(["ts_code", "end_date", config.signal_col])
        .rename({config.signal_col: "score"})
        .with_columns([
            pl.col("end_date").cast(pl.Utf8),
            pl.col("score").cast(pl.Float64),
        ])
        .drop_nulls(["ts_code", "end_date", "score"])
    )

    if config.start_date:
        df = df.filter(pl.col("end_date") >= config.start_date)
    if config.end_date:
        df = df.filter(pl.col("end_date") <= config.end_date)

    return df


def _load_return_panel(config: BacktestConfig) -> pl.DataFrame:
    horizon_map = _build_horizon_map(config.horizon)
    daily = _scan_daily_panel().select(
        [
            "ts_code",
            "trade_date",
            "close",
            "adj_factor",
            "is_st",
            "is_suspend",
            "is_bj",
        ]
    ).with_columns(
        pl.when(pl.lit(config.use_adjusted_price))
        .then(pl.col("close") * pl.col("adj_factor"))
        .otherwise(pl.col("close"))
        .alias("__bt_close")
    )
    flags = _scan_universe_flags().select(
        [
            "ts_code",
            "trade_date",
            "has_trade",
            "is_hs_300",
            "is_csi_500",
            "is_csi_1000",
            "is_csi_2000",
            "is_chinext",
            "is_star_50",
            "sw_l1_code",
            "sw_l1_name",
        ]
    )

    base = (
        horizon_map.lazy()
        .join(
            daily.rename({"trade_date": "entry_date", "__bt_close": "entry_close"}),
            on="entry_date",
            how="left",
        )
        .join(
            daily.rename({"trade_date": "exit_date", "__bt_close": "exit_close"}),
            on=["ts_code", "exit_date"],
            how="left",
        )
        .join(
            flags.rename({"trade_date": "end_date"}),
            on=["ts_code", "end_date"],
            how="left",
        )
        .with_columns(
            (
                pl.col("exit_close") / pl.col("entry_close") - 1.0
            ).alias("fwd_ret")
        )
    )

    if config.benchmark_flag:
        base = base.filter(pl.col(config.benchmark_flag).fill_null(False))

    if config.only_tradable:
        base = base.filter(
            pl.col("has_trade").fill_null(False)
            & ~pl.col("is_st").fill_null(False)
            & ~pl.col("is_suspend").fill_null(False)
            & ~pl.col("is_bj").fill_null(False)
        )

    return base.collect()


def _assign_groups(df: pl.DataFrame, groups: int) -> pl.DataFrame:
    return (
        df.sort(["end_date", "score"])
        .with_columns(
            (pl.int_range(pl.len()).over("end_date") + 1).alias("__rank"),
            pl.len().over("end_date").alias("__n"),
        )
        .with_columns(
            (
                ((pl.col("__rank") - 1) * groups / pl.col("__n"))
                .floor()
                .clip(0, groups - 1)
                .cast(pl.Int64)
                + 1
            ).alias("group_id")
        )
        .drop(["__rank", "__n"])
    )


def _compute_ic(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("end_date")
        .agg(
            pl.corr("score", "fwd_ret").alias("ic"),
            pl.len().alias("n"),
        )
        .sort("end_date")
    )


def _compute_group_returns(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["end_date", "group_id"])
        .agg(
            pl.mean("fwd_ret").alias("group_ret"),
            pl.len().alias("n"),
        )
        .sort(["end_date", "group_id"])
    )


def _compute_long_short(df: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    long_group = config.long_group if config.long_group > 0 else config.groups
    short_group = config.short_group if config.short_group > 0 else 1
    if long_group < 1 or long_group > config.groups:
        raise ValueError(f"long_group must be in [1, {config.groups}], got {long_group}")
    if short_group < 1 or short_group > config.groups:
        raise ValueError(f"short_group must be in [1, {config.groups}], got {short_group}")
    if long_group == short_group:
        raise ValueError("long_group and short_group must be different")

    grouped = (
        df.group_by(["end_date", "group_id"])
        .agg(pl.mean("fwd_ret").alias("group_ret"))
        .sort(["end_date", "group_id"])
    )
    long_leg = grouped.filter(pl.col("group_id") == long_group).select(
        ["end_date", pl.col("group_ret").alias("long_ret")]
    )
    short_leg = grouped.filter(pl.col("group_id") == short_group).select(
        ["end_date", pl.col("group_ret").alias("short_ret")]
    )
    cost = config.cost_bps / 10000.0
    return (
        long_leg.join(short_leg, on="end_date", how="inner")
        .with_columns(
            (pl.col("long_ret") - pl.col("short_ret")).alias("gross_long_short_ret"),
            (pl.col("long_ret") - pl.col("short_ret") - 2.0 * cost).alias("net_long_short_ret"),
        )
        .with_columns(
            (pl.col("gross_long_short_ret") + 1.0).cum_prod().alias("gross_nav"),
            (pl.col("net_long_short_ret") + 1.0).cum_prod().alias("net_nav"),
        )
        .sort("end_date")
    )


def _compute_summary(ic_df: pl.DataFrame, long_short_df: pl.DataFrame) -> dict[str, float | int | str]:
    ic_series = ic_df.get_column("ic").drop_nulls()
    net_ret = long_short_df.get_column("net_long_short_ret")
    gross_ret = long_short_df.get_column("gross_long_short_ret")

    summary = {
        "n_dates": int(ic_df.height),
        "ic_mean": float(ic_series.mean()) if ic_series.len() else 0.0,
        "ic_ir": float(ic_series.mean() / ic_series.std()) if ic_series.len() > 1 and ic_series.std() not in (0, None) else 0.0,
        "gross_cumret": float(long_short_df.get_column("gross_nav").tail(1).item() - 1.0) if long_short_df.height else 0.0,
        "net_cumret": float(long_short_df.get_column("net_nav").tail(1).item() - 1.0) if long_short_df.height else 0.0,
        "gross_avg_ret": float(gross_ret.mean()) if gross_ret.len() else 0.0,
        "net_avg_ret": float(net_ret.mean()) if net_ret.len() else 0.0,
        "gross_win_rate": float((gross_ret > 0).mean()) if gross_ret.len() else 0.0,
        "net_win_rate": float((net_ret > 0).mean()) if net_ret.len() else 0.0,
    }
    return summary


def run_backtest(config: BacktestConfig) -> Path:
    run_dir = _resolve_run_dir(config.output_root)
    signals = _load_signals(config)
    returns = _load_return_panel(config)

    merged = (
        signals.join(returns, on=["ts_code", "end_date"], how="inner")
        .drop_nulls(["score", "fwd_ret", "entry_date", "exit_date"])
        .sort(["end_date", "ts_code"])
    )

    cross_section_size = merged.group_by("end_date").agg(pl.len().alias("__n"))
    merged = (
        merged.join(cross_section_size, on="end_date", how="left")
        .filter(pl.col("__n") >= config.min_cross_section)
        .drop("__n")
    )

    if merged.is_empty():
        raise ValueError("No backtest rows after joining signals with return panel.")

    labeled = _assign_groups(merged, config.groups)
    ic_df = _compute_ic(labeled)
    group_ret_df = _compute_group_returns(labeled)
    long_short_df = _compute_long_short(labeled, config)
    summary = _compute_summary(ic_df, long_short_df)

    labeled.write_parquet(run_dir / "labeled_signals.parquet")
    ic_df.write_parquet(run_dir / "ic_timeseries.parquet")
    group_ret_df.write_parquet(run_dir / "group_returns.parquet")
    long_short_df.write_parquet(run_dir / "long_short_returns.parquet")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_dir
