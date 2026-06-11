from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


DATE_FMT = "%Y%m%d"
DEFAULT_FACTORS = {
    "score": 0.80,
    "reversal_20": 0.15,
    "beta_120": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-only top-N portfolio with top-M sell buffer.")
    parser.add_argument("--signal-path", required=True)
    parser.add_argument("--exposure-root", default="data/barra/exposures")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sell-rank", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--liquidity-floor", type=float, default=-0.5, help="Minimum liquidity_amount_20 exposure z-score.")
    parser.add_argument("--max-volatility", type=float, default=2.5, help="Maximum volatility_60 exposure z-score.")
    parser.add_argument("--max-industry-count", type=int, default=0, help="Max holdings per sw_l1_name. 0 disables the limit.")
    parser.add_argument("--output-root", default="artifacts/portfolio")
    return parser.parse_args()


def scan_exposures(root: str, start_date: str, end_date: str) -> pl.DataFrame:
    pattern = Path(root).expanduser().resolve() / "trade_date=*" / "data.parquet"
    return (
        pl.scan_parquet(str(pattern), missing_columns="insert", extra_columns="ignore")
        .filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))
        .select([
            "ts_code", "trade_date", "name", "sw_l1_name",
            "reversal_20", "beta_120", "liquidity_amount_20", "volatility_60",
        ])
        .collect()
    )


def load_signal(signal_path: str, start_date: str, end_date: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(signal_path)
        .select(["ts_code", pl.col("end_date").cast(pl.Utf8).alias("trade_date"), pl.col("score").cast(pl.Float64)])
        .filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))
        .collect()
    )


def load_trade_dates() -> list[str]:
    cal = pl.read_parquet("data/bronze/trade_cal/trade_cal.parquet")
    return cal.filter(pl.col("is_open") == 1).sort("cal_date").get_column("cal_date").to_list()


def load_next_session_returns(start_date: str, end_date: str) -> pd.DataFrame:
    dates = load_trade_dates()
    prev_map = {dates[i + 1]: dates[i] for i in range(len(dates) - 1)}
    needed_exec_dates = [d for d in dates if d in prev_map and start_date <= prev_map[d] <= end_date]
    if not needed_exec_dates:
        return pd.DataFrame(columns=["ts_code", "trade_date", "exec_date", "ret"])

    paths = [f"data/silver/daily_panel/trade_date={d}/data.parquet" for d in needed_exec_dates]
    daily = (
        pl.scan_parquet(paths, missing_columns="insert", extra_columns="ignore")
        .select(["ts_code", "trade_date", "open", "close", "adj_factor"])
        .with_columns([
            (pl.col("open") * pl.col("adj_factor")).alias("open_adj"),
            (pl.col("close") * pl.col("adj_factor")).alias("close_adj"),
        ])
        .with_columns((pl.col("close_adj") / pl.col("open_adj").clip(lower_bound=1e-8) - 1.0).alias("ret"))
        .select(["ts_code", "trade_date", "ret"])
        .collect()
        .to_pandas()
    )
    daily["exec_date"] = daily["trade_date"]
    daily["trade_date"] = daily["exec_date"].map(prev_map)
    return daily[["ts_code", "trade_date", "exec_date", "ret"]]


def build_rank_table(args: argparse.Namespace) -> pd.DataFrame:
    signal = load_signal(args.signal_path, args.start_date, args.end_date)
    exposures = scan_exposures(args.exposure_root, args.start_date, args.end_date)
    df = signal.join(exposures, on=["ts_code", "trade_date"], how="inner").drop_nulls()
    df = df.filter((pl.col("liquidity_amount_20") >= args.liquidity_floor) & (pl.col("volatility_60") <= args.max_volatility))
    df = df.with_columns(
        (
            pl.col("score") * DEFAULT_FACTORS["score"]
            + pl.col("reversal_20") * DEFAULT_FACTORS["reversal_20"]
            + pl.col("beta_120") * DEFAULT_FACTORS["beta_120"]
        ).alias("composite_score")
    )
    df = df.with_columns(
        pl.col("composite_score").rank(method="ordinal", descending=True).over("trade_date").alias("rank")
    )
    return df.sort(["trade_date", "rank"]).to_pandas()


def compute_turnover(prev_holdings: list[str], new_holdings: list[str]) -> float:
    old_w = {code: 1.0 / len(prev_holdings) for code in prev_holdings} if prev_holdings else {}
    new_w = {code: 1.0 / len(new_holdings) for code in new_holdings} if new_holdings else {}
    names = set(old_w) | set(new_w)
    return 0.5 * sum(abs(new_w.get(code, 0.0) - old_w.get(code, 0.0)) for code in names)


def run_portfolio(args: argparse.Namespace) -> Path:
    rank_df = build_rank_table(args)
    ret_df = load_next_session_returns(args.start_date, args.end_date)
    ret_map = {(r.ts_code, r.trade_date): (r.exec_date, float(r.ret)) for r in ret_df.itertuples(index=False)}

    holdings: list[str] = []
    prev_holdings: list[str] = []
    daily_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    cost_rate = args.cost_bps / 10000.0
    nav = 1.0

    for trade_date, part in rank_df.groupby("trade_date", sort=True):
        part = part.sort_values("rank")
        rank_by_code = dict(zip(part["ts_code"], part["rank"]))
        industry_by_code = dict(zip(part["ts_code"], part["sw_l1_name"].fillna("UNKNOWN")))
        industry_counts: dict[str, int] = {}
        kept: list[str] = []
        for code in holdings:
            industry = str(industry_by_code.get(code, "UNKNOWN"))
            if rank_by_code.get(code, np.inf) > args.sell_rank:
                continue
            if args.max_industry_count > 0 and industry_counts.get(industry, 0) >= args.max_industry_count:
                continue
            kept.append(code)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        holdings = kept

        for code in part["ts_code"]:
            if len(holdings) >= args.top_n:
                break
            if code in holdings:
                continue
            industry = str(industry_by_code.get(code, "UNKNOWN"))
            if args.max_industry_count > 0 and industry_counts.get(industry, 0) >= args.max_industry_count:
                continue
            holdings.append(code)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1

        holdings = holdings[: args.top_n]
        turnover = compute_turnover(prev_holdings, holdings)
        leg_returns = []
        exec_date = None
        for code in holdings:
            item = ret_map.get((code, trade_date))
            code_exec_date = item[0] if item is not None else None
            if item is not None:
                exec_date = item[0]
                leg_returns.append(item[1])
            row = part[part["ts_code"] == code].iloc[0]
            holding_rows.append({
                "trade_date": trade_date,
                "exec_date": code_exec_date,
                "ts_code": code,
                "name": row.get("name"),
                "sw_l1_name": row.get("sw_l1_name"),
                "rank": int(row["rank"]),
                "composite_score": float(row["composite_score"]),
            })

        gross_ret = float(np.mean(leg_returns)) if leg_returns else 0.0
        cost = turnover * cost_rate
        net_ret = gross_ret - cost
        nav *= 1.0 + net_ret
        daily_rows.append({
            "trade_date": trade_date,
            "exec_date": exec_date,
            "holding_count": len(holdings),
            "gross_ret": gross_ret,
            "turnover": turnover,
            "cost": cost,
            "net_ret": net_ret,
            "net_nav": nav,
        })
        prev_holdings = list(holdings)

    daily = pd.DataFrame(daily_rows)
    holding = pd.DataFrame(holding_rows)
    industry_tag = f"_indmax{args.max_industry_count}" if args.max_industry_count > 0 else ""
    out_dir = Path(args.output_root).expanduser().resolve() / f"top{args.top_n}_buffer{args.sell_rank}{industry_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_dir / "daily_returns.parquet", index=False)
    holding.to_parquet(out_dir / "holdings.parquet", index=False)
    rank_df.to_parquet(out_dir / "rank_table.parquet", index=False)

    dd = daily["net_nav"] / daily["net_nav"].cummax() - 1.0
    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "top_n": args.top_n,
        "sell_rank": args.sell_rank,
        "cost_bps": args.cost_bps,
        "factor_weights": DEFAULT_FACTORS,
        "liquidity_floor": args.liquidity_floor,
        "max_volatility": args.max_volatility,
        "max_industry_count": args.max_industry_count,
        "n_dates": int(len(daily)),
        "final_nav": float(daily["net_nav"].iloc[-1]) if len(daily) else 1.0,
        "cumret": float(daily["net_nav"].iloc[-1] - 1.0) if len(daily) else 0.0,
        "avg_daily_net_ret": float(daily["net_ret"].mean()) if len(daily) else 0.0,
        "daily_net_vol": float(daily["net_ret"].std()) if len(daily) > 1 else 0.0,
        "sharpe_daily_sqrt252": float(daily["net_ret"].mean() / daily["net_ret"].std() * np.sqrt(252)) if len(daily) > 1 and daily["net_ret"].std() else 0.0,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "avg_turnover": float(daily["turnover"].mean()) if len(daily) else 0.0,
        "median_turnover": float(daily["turnover"].median()) if len(daily) else 0.0,
        "p90_turnover": float(daily["turnover"].quantile(0.9)) if len(daily) else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plot_results(daily, out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[topn_portfolio] -> {out_dir}")
    return out_dir


def plot_results(daily: pd.DataFrame, out_dir: Path) -> None:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["exec_date"].fillna(daily["trade_date"]))
    dd = daily["net_nav"] / daily["net_nav"].cummax() - 1.0

    plt.figure(figsize=(12, 6))
    plt.plot(daily["date"], daily["net_nav"], label="net NAV", linewidth=1.8)
    plt.title("Top10 Buffer20 Portfolio NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "nav.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.fill_between(daily["date"], dd.values, 0, color="tab:red", alpha=0.35)
    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(daily["date"], daily["turnover"], linewidth=1.0)
    plt.title("Daily One-way Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "turnover.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    datetime.strptime(args.start_date, DATE_FMT)
    datetime.strptime(args.end_date, DATE_FMT)
    if args.sell_rank < args.top_n:
        raise ValueError("--sell-rank must be >= --top-n")
    run_portfolio(args)


if __name__ == "__main__":
    main()
