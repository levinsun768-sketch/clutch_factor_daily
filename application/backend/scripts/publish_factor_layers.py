from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

FACTOR_COLS = [f"fp_{i:03d}" for i in range(64)]
UNIVERSE_FLAGS = {
    "all": None,
    "hs300": "is_hs_300",
    "csi500": "is_csi_500",
    "csi1000": "is_csi_1000",
    "csi2000": "is_csi_2000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish layered backtests for all 64 fingerprint factors.")
    parser.add_argument("--research-workspace", default="../../research_workspace")
    parser.add_argument("--product-root", default="", help="Default: <research>/artifacts/product/current")
    parser.add_argument("--fingerprint-file", default="")
    parser.add_argument("--factor-metric-file", default="")
    parser.add_argument("--start-date", default="20250102")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--horizon", type=int, default=1, help="Close-to-close holding horizon in trade days. Default 1 = signal t, enter at close(t+1), exit at close(t+2).")
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--universes", default="all,hs300,csi500,csi1000")
    return parser.parse_args()


def resolve_research(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (Path(__file__).resolve().parents[3] / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (Path(__file__).resolve().parents[1] / raw).resolve()


def latest_file(patterns: list[Path]) -> Path | None:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(str(pattern)))
    files = [p for p in files if p.exists()]
    if not files:
        return None
    return max(files, key=lambda p: (p.stat().st_mtime, p.name))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}" if len(value) == 8 else value


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def resolve_fingerprint(research: Path, explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    path = latest_file([research / "artifacts" / "models" / "*" / "fp_dataset" / "fingerprints_daily_*.parquet"])
    if not path:
        path = latest_file([research / "artifacts" / "models" / "*" / "fp_incremental" / "fingerprints_daily_*.parquet"])
    if not path:
        raise FileNotFoundError("No fingerprint parquet found.")
    return path


def resolve_metric_file(research: Path, explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.exists() else None
    return latest_file([research / "artifacts" / "backtests" / "*" / "fingerprint_dim_ic" / "fingerprint_dim_ic_summary.parquet"])


def alpha_signs(metric_file: Path | None) -> dict[str, float]:
    signs = {factor: 1.0 for factor in FACTOR_COLS}
    if not metric_file or not metric_file.exists():
        return signs
    metrics = pl.read_parquet(metric_file).select(["factor", "rankic_mean"])
    for row in metrics.to_dicts():
        factor = str(row["factor"]).lower()
        if factor in signs:
            signs[factor] = 1.0 if float(row.get("rankic_mean") or 0.0) >= 0 else -1.0
    return signs


def trade_dates(research: Path) -> list[str]:
    cal = pl.read_parquet(research / "data" / "bronze" / "trade_cal" / "trade_cal.parquet")
    return cal.filter(pl.col("is_open") == 1).sort("cal_date").get_column("cal_date").to_list()


def build_horizon_frame(research: Path, start_date: str, end_date: str, horizon: int) -> pl.DataFrame:
    dates = trade_dates(research)
    rows = []
    for idx, date in enumerate(dates):
        if date < start_date or date > end_date:
            continue
        if idx + horizon + 1 >= len(dates):
            continue
        # Fingerprints for signal date t are known only after close(t).
        # Use the next close for execution, then hold for `horizon` close-to-close sessions.
        rows.append({"end_date": date, "entry_date": dates[idx + 1], "exit_date": dates[idx + horizon + 1]})
    return pl.DataFrame(rows)


def scan_daily(research: Path, dates: list[str]) -> pl.LazyFrame:
    paths = [research / "data" / "silver" / "daily_panel" / f"trade_date={date}" / "data.parquet" for date in dates]
    paths = [str(path) for path in paths if Path(path).exists()]
    return (
        pl.scan_parquet(paths, missing_columns="insert", extra_columns="ignore")
        .select(["ts_code", "trade_date", "close", "adj_factor"])
        .with_columns((pl.col("close") * pl.col("adj_factor")).alias("close_adj"))
        .select(["ts_code", "trade_date", "close_adj"])
    )


def scan_flags(research: Path, dates: list[str]) -> pl.LazyFrame:
    paths = [research / "data" / "silver" / "universe_flags" / f"trade_date={date}" / "data.parquet" for date in dates]
    paths = [str(path) for path in paths if Path(path).exists()]
    return pl.scan_parquet(paths, missing_columns="insert", extra_columns="ignore")


def return_panel(research: Path, start_date: str, end_date: str, horizon: int, universe: str) -> pl.DataFrame:
    hmap = build_horizon_frame(research, start_date, end_date, horizon)
    if hmap.is_empty():
        return pl.DataFrame()
    entry_dates = hmap.get_column("entry_date").unique().to_list()
    exit_dates = hmap.get_column("exit_date").unique().to_list()
    end_dates = hmap.get_column("end_date").unique().to_list()
    daily = scan_daily(research, sorted(set(entry_dates + exit_dates)))
    flags = scan_flags(research, end_dates).select([
        "ts_code", pl.col("trade_date").cast(pl.Utf8).alias("end_date"), "has_trade", "is_st", "is_suspend", "is_bj",
        "is_hs_300", "is_csi_500", "is_csi_1000", "is_csi_2000",
    ])
    base = (
        hmap.lazy()
        .join(daily.rename({"trade_date": "entry_date", "close_adj": "entry_close"}), on="entry_date", how="left")
        .join(daily.rename({"trade_date": "exit_date", "close_adj": "exit_close"}), on=["ts_code", "exit_date"], how="left")
        .join(flags, on=["ts_code", "end_date"], how="left")
        .filter(
            pl.col("has_trade").fill_null(False)
            & ~pl.col("is_st").fill_null(False)
            & ~pl.col("is_suspend").fill_null(False)
            & ~pl.col("is_bj").fill_null(False)
        )
    )
    flag = UNIVERSE_FLAGS.get(universe)
    if flag:
        base = base.filter(pl.col(flag).fill_null(False))
    return (
        base.with_columns((pl.col("exit_close") / pl.col("entry_close") - 1.0).alias("fwd_ret"))
        .select(["ts_code", "end_date", "fwd_ret"])
        .drop_nulls()
        .collect()
    )


def assign_groups(df: pl.DataFrame, groups: int) -> pl.DataFrame:
    return (
        df.sort(["end_date", "score"])
        .with_columns([
            (pl.int_range(pl.len()).over("end_date") + 1).alias("__rank"),
            pl.len().over("end_date").alias("__n"),
        ])
        .with_columns((((pl.col("__rank") - 1) * groups / pl.col("__n")).floor().clip(0, groups - 1).cast(pl.Int64) + 1).alias("group_id"))
        .drop(["__rank", "__n"])
    )


def turnover(prev: set[str], curr: set[str]) -> float:
    if not prev and not curr:
        return 0.0
    old_w = {code: 1.0 / len(prev) for code in prev} if prev else {}
    new_w = {code: 1.0 / len(curr) for code in curr} if curr else {}
    names = set(old_w) | set(new_w)
    return 0.5 * sum(abs(new_w.get(code, 0.0) - old_w.get(code, 0.0)) for code in names)


def build_layer_payload(grouped: pl.DataFrame, factor: str, factor_id: str, universe: str, alpha_sign: float, groups: int, cost_bps: float, horizon: int) -> dict[str, Any]:
    group_ret = grouped.group_by(["end_date", "group_id"]).agg(pl.mean("fwd_ret").alias("group_ret"), pl.len().alias("n")).sort(["end_date", "group_id"])
    pdf = group_ret.to_pandas()
    pivot = pdf.pivot(index="end_date", columns="group_id", values="group_ret").sort_index()
    for group_id in range(1, groups + 1):
        if group_id not in pivot.columns:
            pivot[group_id] = 0.0
    pivot = pivot[[group_id for group_id in range(1, groups + 1)]].fillna(0.0)
    nav = (1.0 + pivot).cumprod()

    hold_pdf = grouped.filter(pl.col("group_id").is_in([1, groups])).select(["end_date", "group_id", "ts_code"]).to_pandas()
    top_prev: set[str] = set()
    bot_prev: set[str] = set()
    gross_nav = 1.0
    net_nav = 1.0
    ls_rows = []
    top_turnovers = []
    bottom_turnovers = []
    for date in pivot.index:
        top = set(hold_pdf[(hold_pdf["end_date"] == date) & (hold_pdf["group_id"] == groups)]["ts_code"])
        bot = set(hold_pdf[(hold_pdf["end_date"] == date) & (hold_pdf["group_id"] == 1)]["ts_code"])
        top_to = turnover(top_prev, top)
        bot_to = turnover(bot_prev, bot)
        ls_to = top_to + bot_to
        gross_ret = float(pivot.loc[date, groups] - pivot.loc[date, 1])
        net_ret = gross_ret - ls_to * cost_bps / 10000.0
        gross_nav *= 1.0 + gross_ret
        net_nav *= 1.0 + net_ret
        top_turnovers.append(top_to)
        bottom_turnovers.append(bot_to)
        ls_rows.append({
            "date": display_date(str(date)),
            "gross_ret": safe_float(gross_ret),
            "net_ret": safe_float(net_ret),
            "gross_nav": safe_float(gross_nav),
            "net_nav": safe_float(net_nav),
            "top_turnover": safe_float(top_to),
            "bottom_turnover": safe_float(bot_to),
            "ls_turnover": safe_float(ls_to),
        })
        top_prev = top
        bot_prev = bot

    group_nav = []
    for date, row in nav.iterrows():
        item = {"date": display_date(str(date))}
        for group_id in range(1, groups + 1):
            item[f"group_{group_id}"] = safe_float(row[group_id])
        group_nav.append(item)

    net_rets = [row["net_ret"] for row in ls_rows if row["net_ret"] is not None]
    summary = {
        "factor_id": factor_id,
        "factor": factor,
        "universe": universe,
        "alpha_sign": alpha_sign,
        "horizon": horizon,
        "return_mode": "lagged_close_to_close",
        "rebalance": "daily",
        "signal_available": "after_signal_date_close",
        "entry_timing": "next_trade_date_close",
        "exit_timing": "entry_plus_horizon_close",
        "groups": groups,
        "n_dates": len(ls_rows),
        "gross_cumret": safe_float(gross_nav - 1.0),
        "net_cumret": safe_float(net_nav - 1.0),
        "net_win_rate": safe_float(float(np.mean([x > 0 for x in net_rets])) if net_rets else None),
        "top_turnover_mean": safe_float(float(np.mean(top_turnovers)) if top_turnovers else None),
        "bottom_turnover_mean": safe_float(float(np.mean(bottom_turnovers)) if bottom_turnovers else None),
        "ls_turnover_mean": safe_float(float(np.mean([a + b for a, b in zip(top_turnovers, bottom_turnovers)])) if top_turnovers else None),
        "final_net_nav": safe_float(net_nav),
    }
    return {"summary": summary, "group_nav": group_nav, "long_short": ls_rows}


def update_factor_metrics(product_root: Path, universe: str, summaries: list[dict[str, Any]]) -> None:
    metrics_path = product_root / "factors" / "metrics" / f"universe={universe}" / "metrics.json"
    if not metrics_path.exists():
        return
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_factor = {item["factor"]: item for item in summaries}
    for item in payload.get("items", []):
        factor = str(item.get("factor", "")).lower()
        summary = by_factor.get(factor)
        if not summary:
            item["backtest_available"] = False
            continue
        item["backtest_available"] = True
        item["net_cumret"] = summary.get("net_cumret")
        item["gross_cumret"] = summary.get("gross_cumret")
        item["turnover"] = summary.get("ls_turnover_mean")
        item["top_turnover"] = summary.get("top_turnover_mean")
        item["bottom_turnover"] = summary.get("bottom_turnover_mean")
        item["final_net_nav"] = summary.get("final_net_nav")
        item["layer_path"] = summary.get("path")
    write_json(metrics_path, payload)


def update_manifest(product_root: Path) -> None:
    manifest_path = product_root / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("paths", {})["factor_layers"] = "factors/layers/universe={universe}/factor={factor_id}/data.json"
    manifest.setdefault("paths", {})["factor_layers_summary"] = "factors/layers/universe={universe}/summary.json"
    write_json(manifest_path, manifest)
    latest = product_root.parent / "latest_manifest.json"
    if latest.exists():
        write_json(latest, manifest)


def main() -> None:
    args = parse_args()
    research = resolve_research(args.research_workspace)
    product_root = Path(args.product_root).expanduser().resolve() if args.product_root else research / "artifacts" / "product" / "current"
    fp_file = resolve_fingerprint(research, args.fingerprint_file)
    metric_file = resolve_metric_file(research, args.factor_metric_file)
    signs = alpha_signs(metric_file)
    universes = [item.strip().lower() for item in args.universes.split(",") if item.strip()]

    meta = {
        "schema_version": 1,
        "published_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint_file": str(fp_file),
        "metric_file": str(metric_file) if metric_file else None,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizon": args.horizon,
        "return_mode": "lagged_close_to_close",
        "rebalance": "daily",
        "signal_available": "after_signal_date_close",
        "entry_timing": "next_trade_date_close",
        "exit_timing": "entry_plus_horizon_close",
        "groups": args.groups,
        "cost_bps": args.cost_bps,
        "universes": universes,
    }

    for universe in universes:
        print(f"[factor_layers] universe={universe} return panel")
        ret = return_panel(research, args.start_date, args.end_date, args.horizon, universe)
        if ret.is_empty():
            print(f"[factor_layers] universe={universe} empty return panel")
            continue
        summary_rows = []
        ret_lf = ret.lazy()
        for factor in FACTOR_COLS:
            factor_id = factor.upper()
            sign = signs.get(factor, 1.0)
            print(f"[factor_layers] {universe} {factor_id}")
            sig = (
                pl.scan_parquet(str(fp_file), missing_columns="insert", extra_columns="ignore")
                .select(["ts_code", pl.col("end_date").cast(pl.Utf8).alias("end_date"), pl.col(factor).cast(pl.Float64).alias("raw_score")])
                .filter((pl.col("end_date") >= args.start_date) & (pl.col("end_date") <= args.end_date))
            )
            joined = sig.join(ret_lf, on=["ts_code", "end_date"], how="inner").drop_nulls(["raw_score", "fwd_ret"]).with_columns((pl.col("raw_score") * sign).alias("score")).collect()
            if joined.is_empty():
                continue
            grouped = assign_groups(joined, args.groups)
            payload = build_layer_payload(grouped, factor, factor_id, universe, sign, args.groups, args.cost_bps, args.horizon)
            out_path = product_root / "factors" / "layers" / f"universe={universe}" / f"factor={factor_id}" / "data.json"
            write_json(out_path, payload)
            summary = dict(payload["summary"])
            summary["path"] = str(out_path.relative_to(product_root))
            summary_rows.append(summary)
        write_json(product_root / "factors" / "layers" / f"universe={universe}" / "summary.json", {"meta": meta, "items": summary_rows})
        update_factor_metrics(product_root, universe, summary_rows)

    write_json(product_root / "factors" / "layers" / "meta.json", meta)
    update_manifest(product_root)
    print(json.dumps({"product_root": str(product_root), "universes": universes}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
