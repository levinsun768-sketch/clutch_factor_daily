from __future__ import annotations

import argparse
import glob
import json
import math
import shutil

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

STYLE_FACTORS = [
    "size",
    "value_bp",
    "value_ep",
    "momentum_252_20",
    "reversal_20",
    "beta_120",
    "volatility_60",
    "liquidity_amount_20",
    "liquidity_turnover_20",
]
UNIVERSE_FLAGS = {
    "all": None,
    "hs300": "is_hs_300",
    "csi500": "is_csi_500",
    "csi1000": "is_csi_1000",
    "csi2000": "is_csi_2000",
}
STYLE_NAMES = {
    "size": "Size",
    "value_bp": "Value",
    "value_ep": "Value",
    "momentum_252_20": "Momentum",
    "reversal_20": "Reversal",
    "beta_120": "Beta",
    "volatility_60": "Volatility",
    "liquidity_amount_20": "Liquidity",
    "liquidity_turnover_20": "Liquidity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish research artifacts into a stable product layer for backend APIs.")
    parser.add_argument("--research-workspace", default="../../source_code")
    parser.add_argument("--output-root", default="", help="Default: <research>/artifacts/product")
    parser.add_argument("--date", default="", help="Signal/product date YYYYMMDD. Default: latest portfolio date, then latest signal date.")
    parser.add_argument("--portfolio-run", default="", help="Optional explicit portfolio run directory.")
    parser.add_argument("--fingerprint-file", default="", help="Optional explicit fingerprint parquet.")
    parser.add_argument("--signal-file", default="", help="Optional explicit composite/neutral signal parquet.")
    parser.add_argument("--factor-metric-file", default="", help="Optional explicit fingerprint factor metric parquet.")
    parser.add_argument("--rankic-timeseries-file", default="", help="Optional explicit rankIC timeseries parquet.")
    parser.add_argument("--single-dim-summary-file", default="", help="Optional explicit single dim summary csv.")
    parser.add_argument("--universes", default="all,hs300,csi500,csi1000")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--copy-heavy", action="store_true", help="Copy heavy full fingerprint/signal files instead of referencing them.")
    parser.add_argument("--skip-style-exposure", action="store_true", help="Skip factor style exposure correlations.")
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
    script_candidate = (Path(__file__).resolve().parents[1] / raw).resolve()
    return script_candidate


def latest_file(patterns: list[Path]) -> Path | None:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(str(pattern)))
    files = [p for p in files if p.exists()]
    if not files:
        return None
    return max(files, key=lambda p: (p.stat().st_mtime, p.name))


def latest_portfolio_run(research: Path) -> Path | None:
    candidates = [p.parent for p in (research / "artifacts" / "portfolio").glob("*/summary.json")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "summary.json").stat().st_mtime)


def latest_date_from_parquet(path: Path, col: str) -> str | None:
    if not path or not path.exists():
        return None
    try:
        value = pl.scan_parquet(str(path)).select(pl.max(pl.col(col).cast(pl.Utf8))).collect().item()
    except Exception:
        return None
    return str(value) if value else None


def latest_product_date(portfolio_run: Path | None, signal_file: Path | None, explicit: str) -> str:
    if explicit:
        return explicit.replace("-", "")
    if portfolio_run and (portfolio_run / "holdings.parquet").exists():
        date = latest_date_from_parquet(portfolio_run / "holdings.parquet", "trade_date")
        if date:
            return date
    if signal_file:
        date = latest_date_from_parquet(signal_file, "end_date")
        if date:
            return date
    raise FileNotFoundError("Cannot infer product date from portfolio or signal artifacts.")


def rel_to(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def resolve_inputs(args: argparse.Namespace, research: Path) -> dict[str, Path | None]:
    portfolio_run = Path(args.portfolio_run).expanduser().resolve() if args.portfolio_run else latest_portfolio_run(research)
    fingerprint_file = Path(args.fingerprint_file).expanduser().resolve() if args.fingerprint_file else latest_file([
        research / "artifacts" / "models" / "*" / "fp_dataset" / "fingerprints_daily_*.parquet",
        research / "artifacts" / "models" / "*" / "fp_incremental" / "fingerprints_daily_*.parquet",
    ])
    signal_file = Path(args.signal_file).expanduser().resolve() if args.signal_file else latest_file([
        research / "artifacts" / "barra" / "signal_neutral_ic" / "*" / "ewma" / "neutral_signal_ewma20.parquet",
        research / "artifacts" / "barra" / "signal_neutral_ic" / "*" / "neutral_signal.parquet",
    ])
    factor_metric_file = Path(args.factor_metric_file).expanduser().resolve() if args.factor_metric_file else latest_file([
        research / "artifacts" / "backtests" / "*" / "fingerprint_dim_ic" / "fingerprint_dim_ic_summary.parquet",
    ])
    rankic_file = Path(args.rankic_timeseries_file).expanduser().resolve() if args.rankic_timeseries_file else latest_file([
        research / "artifacts" / "backtests" / "*" / "fingerprint_dim_ic" / "fingerprint_dim_rankic_timeseries.parquet",
    ])
    single_summary = Path(args.single_dim_summary_file).expanduser().resolve() if args.single_dim_summary_file else latest_file([
        research / "artifacts" / "backtests" / "*" / "single_dim_signals" / "single_dim_backtest_summary.csv",
    ])
    return {
        "portfolio_run": portfolio_run,
        "fingerprint_file": fingerprint_file,
        "signal_file": signal_file,
        "factor_metric_file": factor_metric_file,
        "rankic_timeseries_file": rankic_file,
        "single_dim_summary_file": single_summary,
    }


def read_exposure(research: Path, date: str, universe: str) -> pl.DataFrame:
    path = research / "data" / "barra" / "exposures" / f"trade_date={date}" / "data.parquet"
    if not path.exists():
        return pl.DataFrame()
    lf = pl.scan_parquet(str(path), missing_columns="insert", extra_columns="ignore")
    flag = UNIVERSE_FLAGS.get(universe)
    if flag and flag in lf.collect_schema().names():
        lf = lf.filter(pl.col(flag).fill_null(False))
    return lf.collect()


def publish_daily_scores(out: Path, date: str, universe: str, signal_file: Path, exposures: pl.DataFrame, top_n: int) -> pl.DataFrame:
    scores = (
        pl.scan_parquet(str(signal_file), missing_columns="insert", extra_columns="ignore")
        .select(["ts_code", pl.col("end_date").cast(pl.Utf8).alias("trade_date"), pl.col("score").cast(pl.Float64)])
        .filter(pl.col("trade_date") == date)
        .collect()
    )
    if scores.is_empty() or exposures.is_empty():
        result = pl.DataFrame(schema={"ts_code": pl.Utf8, "trade_date": pl.Utf8, "score": pl.Float64})
    else:
        result = scores.join(exposures.select(["ts_code", "name", "sw_l1_code", "sw_l1_name"]), on="ts_code", how="left")
        result = result.sort("score", descending=True).with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
    write_parquet(out / "daily_scores" / f"trade_date={date}" / f"universe={universe}" / "data.parquet", result)
    recommendations = result.head(top_n) if not result.is_empty() else result
    write_parquet(out / "recommendations" / f"trade_date={date}" / f"universe={universe}" / "data.parquet", recommendations)
    return result


def publish_market_snapshot(research: Path, out: Path, date: str, universe: str) -> dict[str, Any]:
    path = research / "data" / "silver" / "daily_panel" / f"trade_date={date}" / "data.parquet"
    if not path.exists():
        payload = {"available": False, "date": date, "universe": universe}
        write_json(out / "market" / f"trade_date={date}" / f"universe={universe}" / "overview.json", payload)
        return payload
    lf = pl.scan_parquet(str(path), missing_columns="insert", extra_columns="ignore")
    flag = UNIVERSE_FLAGS.get(universe)
    if flag and flag in lf.collect_schema().names():
        lf = lf.filter(pl.col(flag).fill_null(False))
    cols = lf.collect_schema().names()
    exprs = [pl.len().alias("n")]
    if "pct_chg" in cols:
        exprs += [
            (pl.col("pct_chg") > 0).sum().alias("up"),
            (pl.col("pct_chg") < 0).sum().alias("down"),
            (pl.col("pct_chg") == 0).sum().alias("flat"),
            pl.col("pct_chg").mean().alias("avg_pct_chg"),
        ]
    if "amount" in cols:
        exprs.append(pl.col("amount").sum().alias("amount"))
    if "is_up_limit" in cols:
        exprs.append(pl.col("is_up_limit").fill_null(False).sum().alias("limit_up"))
    if "is_down_limit" in cols:
        exprs.append(pl.col("is_down_limit").fill_null(False).sum().alias("limit_down"))
    row = lf.select(exprs).collect().to_dicts()[0]
    payload = {"available": True, "date": date, "universe": universe, **{k: safe_float(v) if isinstance(v, float) else v for k, v in row.items()}}
    write_json(out / "market" / f"trade_date={date}" / f"universe={universe}" / "overview.json", payload)
    return payload


def publish_style_monitor(research: Path, out: Path, date: str, universe: str, exposures: pl.DataFrame) -> dict[str, Any]:
    daily_path = research / "data" / "silver" / "daily_panel" / f"trade_date={date}" / "data.parquet"
    if exposures.is_empty() or not daily_path.exists():
        payload = {
            "date": date,
            "universe": universe,
            "n": 0,
            "unit": "bps",
            "method": "same_day_cross_section_regression",
            "styles": {},
            "style_spread": {},
        }
        write_json(out / "style_monitor" / f"trade_date={date}" / f"universe={universe}" / "data.json", payload)
        return payload

    daily = pl.scan_parquet(str(daily_path), missing_columns="insert", extra_columns="ignore")
    cols = daily.collect_schema().names()
    if "pct_chg" in cols:
        returns = daily.select(["ts_code", (pl.col("pct_chg").cast(pl.Float64) / 100.0).alias("ret_1d")]).collect()
    else:
        returns = (
            daily.select(["ts_code", "close", "pre_close"])
            .with_columns((pl.col("close") / pl.col("pre_close").clip(lower_bound=1e-8) - 1.0).alias("ret_1d"))
            .select(["ts_code", "ret_1d"])
            .collect()
        )

    keep = ["ts_code", "sw_l1_code", "sw_l1_name", *STYLE_FACTORS]
    joined = exposures.select([col for col in keep if col in exposures.columns]).join(returns, on="ts_code", how="inner").drop_nulls(["ret_1d", *STYLE_FACTORS])
    if joined.height < 100:
        payload = {
            "date": date,
            "universe": universe,
            "n": joined.height,
            "unit": "bps",
            "method": "same_day_cross_section_regression",
            "styles": {},
            "style_spread": {},
        }
        write_json(out / "style_monitor" / f"trade_date={date}" / f"universe={universe}" / "data.json", payload)
        return payload

    pdf = joined.to_pandas()
    y = pdf["ret_1d"].to_numpy(dtype=np.float64)
    x_parts = [np.ones((len(pdf), 1), dtype=np.float64), pdf[STYLE_FACTORS].to_numpy(dtype=np.float64)]
    industry_names: list[str] = []
    industry_base: str | None = None
    industry_name_by_code: dict[str, str] = {}
    industry_count_by_code: dict[str, int] = {}
    if "sw_l1_code" in pdf.columns:
        industry_codes = pdf["sw_l1_code"].fillna("UNKNOWN").astype(str)
        if "sw_l1_name" in pdf.columns:
            industry_display = pdf["sw_l1_name"].fillna(industry_codes).astype(str)
        else:
            industry_display = industry_codes
        industry_name_by_code = dict(zip(industry_codes, industry_display))
        industry_count_by_code = industry_codes.value_counts().to_dict()
        industry = pd.get_dummies(industry_codes, dtype=float)
        industry_names = list(industry.columns)
        if industry.shape[1] > 1:
            industry_base = industry_names[0]
            x_parts.append(industry.iloc[:, 1:].to_numpy(dtype=np.float64))
            industry_names = industry_names[1:]
    x = np.concatenate(x_parts, axis=1)
    beta = np.linalg.pinv(x.T @ x + np.eye(x.shape[1]) * 1e-8) @ x.T @ y
    style_beta_bps = {style: safe_float(beta[idx + 1] * 10000.0) for idx, style in enumerate(STYLE_FACTORS)}

    industry_start = 1 + len(STYLE_FACTORS)
    industry_beta_bps = {name: safe_float(beta[industry_start + idx] * 10000.0) for idx, name in enumerate(industry_names)}
    if industry_base is not None:
        industry_beta_bps[industry_base] = 0.0
    industry_payload = [
        {
            "industry_code": code,
            "industry": industry_name_by_code.get(code, code),
            "premium_bps": value,
            "n": int(industry_count_by_code.get(code, 0)),
        }
        for code, value in industry_beta_bps.items()
        if value is not None
    ]
    industry_payload = sorted(industry_payload, key=lambda item: abs(item["premium_bps"] or 0.0), reverse=True)

    spreads = {}
    for style in STYLE_FACTORS:
        lo = pdf[style].quantile(0.3)
        hi = pdf[style].quantile(0.7)
        low_ret = pdf.loc[pdf[style] <= lo, "ret_1d"].mean()
        high_ret = pdf.loc[pdf[style] >= hi, "ret_1d"].mean()
        spreads[style] = safe_float((high_ret - low_ret) * 10000.0)

    fitted = x @ beta
    denom = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - fitted) ** 2).sum()) / denom if denom > 1e-12 else 0.0
    payload = {
        "date": date,
        "universe": universe,
        "n": int(joined.height),
        "unit": "bps",
        "method": "same_day_cross_section_regression",
        "description": "Same-day stock returns regressed on Barra-style exposures and industry dummies. Values are daily style premia in basis points per 1 z-score exposure.",
        "styles": style_beta_bps,
        "style_spread": spreads,
        "industry_premium": industry_payload,
        "industry_base": industry_base,
        "r2": safe_float(r2),
    }
    write_json(out / "style_monitor" / f"trade_date={date}" / f"universe={universe}" / "data.json", payload)
    return payload


def main_style(exposure: dict[str, float | None]) -> str:
    clean = {k: abs(v) for k, v in exposure.items() if v is not None}
    if not clean:
        return "Unknown"
    key, value = max(clean.items(), key=lambda x: x[1])
    if value < 0.25:
        return "Neutral Alpha"
    return STYLE_NAMES.get(key, "Unknown")


def publish_factor_metrics(out: Path, date: str, universe: str, inputs: dict[str, Path | None], exposures: pl.DataFrame, fingerprint_file: Path | None, skip_style: bool) -> list[dict[str, Any]]:
    metric_file = inputs["factor_metric_file"]
    if not metric_file or not metric_file.exists():
        return []
    metrics = pl.read_parquet(metric_file)
    single = inputs.get("single_dim_summary_file")
    if single and single.exists():
        extra = pl.read_csv(single).rename({"signal": "factor"})
        metrics = metrics.join(extra, on="factor", how="left")

    style_by_factor: dict[str, dict[str, float | None]] = {}
    if not skip_style and fingerprint_file and fingerprint_file.exists() and not exposures.is_empty():
        factor_cols = [f"fp_{i:03d}" for i in range(64)]
        fp = (
            pl.scan_parquet(str(fingerprint_file), missing_columns="insert", extra_columns="ignore")
            .select(["ts_code", pl.col("end_date").cast(pl.Utf8).alias("end_date"), *factor_cols])
            .filter(pl.col("end_date") == date)
            .drop("end_date")
            .collect()
        )
        joined = fp.join(exposures.select(["ts_code", *STYLE_FACTORS]), on="ts_code", how="inner").drop_nulls()
        if joined.height >= 100:
            pdf = joined.to_pandas()
            for factor in factor_cols:
                style_by_factor[factor] = {style: safe_float(pdf[factor].corr(pdf[style])) for style in STYLE_FACTORS}

    rows = []
    for item in metrics.to_dicts():
        factor = str(item["factor"]).lower()
        if factor.startswith("fp_"):
            factor_id = f"fp_{int(factor.split('_', 1)[1]):03d}"
        else:
            continue
        exposure = style_by_factor.get(factor_id, {})
        rows.append({
            "factor_id": factor_id.upper(),
            "factor": factor_id,
            "rank_ic": safe_float(item.get("rankic_mean")),
            "rank_ic_abs": abs(safe_float(item.get("rankic_mean")) or 0.0),
            "icir": safe_float(item.get("rankic_ir")),
            "ic_mean": safe_float(item.get("ic_mean")),
            "ic_ir": safe_float(item.get("ic_ir")),
            "net_cumret": safe_float(item.get("net_cumret")),
            "turnover": safe_float(item.get("ls_turnover")),
            "main_style": main_style(exposure),
            "style_exposure": exposure,
        })
    rows = sorted(rows, key=lambda r: r["rank_ic_abs"], reverse=True)
    write_json(out / "factors" / "metrics" / f"universe={universe}" / "metrics.json", {"date": date, "universe": universe, "items": rows})

    rankic = inputs.get("rankic_timeseries_file")
    if rankic and rankic.exists():
        target = out / "factors" / "rankic_timeseries.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rankic, target)
    return rows


def publish_portfolio(out: Path, portfolio_run: Path | None) -> dict[str, Any] | None:
    if not portfolio_run or not portfolio_run.exists():
        return None
    target = out / "portfolio" / "main"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ["summary.json", "config.json", "daily_returns.parquet", "holdings.parquet", "rank_table.parquet"]:
        src = portfolio_run / name
        if src.exists():
            shutil.copy2(src, target / name)
            copied.append(name)
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8")) if (target / "summary.json").exists() else {}
    return {"source": str(portfolio_run), "path": "portfolio/main", "files": copied, "summary": summary}


def publish_references(out: Path, inputs: dict[str, Path | None], copy_heavy: bool) -> dict[str, Any]:
    refs = {}
    heavy_dir = out / "references"
    heavy_dir.mkdir(parents=True, exist_ok=True)
    for key in ["fingerprint_file", "signal_file", "factor_metric_file", "single_dim_summary_file"]:
        src = inputs.get(key)
        if not src:
            refs[key] = None
            continue
        if copy_heavy and key in {"fingerprint_file", "signal_file"}:
            dst = heavy_dir / src.name
            shutil.copy2(src, dst)
            refs[key] = str(dst.relative_to(out))
        else:
            refs[key] = str(src)
    return refs


def main() -> None:
    args = parse_args()
    research = resolve_research(args.source_code)
    out_root = Path(args.output_root).expanduser().resolve() if args.output_root else research / "artifacts" / "product"
    current = out_root / "current"
    current.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(args, research)
    date = latest_product_date(inputs["portfolio_run"], inputs["signal_file"], args.date)
    universes = [u.strip().lower() for u in args.universes.split(",") if u.strip()]

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "published_at": datetime.now().isoformat(timespec="seconds"),
        "source_code": str(research),
        "product_root": str(current),
        "trade_date": date,
        "universes": universes,
        "references": publish_references(current, inputs, args.copy_heavy),
        "sources": {key: str(value) if value else None for key, value in inputs.items()},
        "paths": {
            "daily_scores": "daily_scores/trade_date={date}/universe={universe}/data.parquet",
            "recommendations": "recommendations/trade_date={date}/universe={universe}/data.parquet",
            "factor_metrics": "factors/metrics/universe={universe}/metrics.json",
            "rankic_timeseries": "factors/rankic_timeseries.parquet",
            "factor_layers": "factors/layers/universe={universe}/factor={factor_id}/data.json",
            "factor_layers_summary": "factors/layers/universe={universe}/summary.json",
            "market_overview": "market/trade_date={date}/universe={universe}/overview.json",
            "style_monitor": "style_monitor/trade_date={date}/universe={universe}/data.json",
            "portfolio": "portfolio/main",
        },
    }

    portfolio = publish_portfolio(current, inputs["portfolio_run"])
    if portfolio:
        manifest["portfolio"] = portfolio

    for universe in universes:
        exposures = read_exposure(research, date, universe)
        publish_market_snapshot(research, current, date, universe)
        publish_style_monitor(research, current, date, universe, exposures)
        if inputs["signal_file"]:
            publish_daily_scores(current, date, universe, inputs["signal_file"], exposures, args.top_n)
        publish_factor_metrics(current, date, universe, inputs, exposures, inputs["fingerprint_file"], args.skip_style_exposure)

    write_json(current / "manifest.json", manifest)
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "latest_manifest.json", manifest)
    print(json.dumps({"trade_date": date, "product_root": str(current), "manifest": str(current / "manifest.json")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
