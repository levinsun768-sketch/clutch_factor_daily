from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from publish_artifacts import (
    latest_date_from_parquet,
    publish_daily_scores,
    publish_market_snapshot,
    publish_style_monitor,
    read_exposure,
    resolve_inputs,
    resolve_research,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish date-partitioned product artifacts for a historical date range.")
    parser.add_argument("--research-workspace", default="../../research_workspace")
    parser.add_argument("--output-root", default="", help="Default: <research>/artifacts/product")
    parser.add_argument("--start-date", default="20250101")
    parser.add_argument("--end-date", default="", help="Default: latest available daily_panel date")
    parser.add_argument("--universes", default="all,hs300,csi500,csi1000")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--portfolio-run", default="")
    parser.add_argument("--fingerprint-file", default="")
    parser.add_argument("--signal-file", default="")
    parser.add_argument("--factor-metric-file", default="")
    parser.add_argument("--rankic-timeseries-file", default="")
    parser.add_argument("--single-dim-summary-file", default="")
    parser.add_argument("--copy-heavy", action="store_true")
    parser.add_argument("--skip-market", action="store_true")
    parser.add_argument("--skip-style-monitor", action="store_true")
    parser.add_argument("--skip-scores", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def partition_dates(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name.split("=", 1)[1] for path in root.glob("trade_date=*") if "=" in path.name)


def trade_dates(research: Path, start_date: str, end_date: str) -> list[str]:
    cal_path = research / "data" / "bronze" / "trade_cal" / "trade_cal.parquet"
    cal = pl.read_parquet(cal_path).with_columns(pl.col("cal_date").cast(pl.Utf8))
    return (
        cal.filter((pl.col("is_open") == 1) & (pl.col("cal_date") >= start_date) & (pl.col("cal_date") <= end_date))
        .sort("cal_date")
        .get_column("cal_date")
        .to_list()
    )


def parquet_date_set(path: Path | None, col: str, start_date: str, end_date: str) -> set[str]:
    if not path or not path.exists():
        return set()
    try:
        values = (
            pl.scan_parquet(str(path))
            .select(pl.col(col).cast(pl.Utf8).alias("date"))
            .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
            .unique()
            .collect()
            .get_column("date")
            .to_list()
        )
    except Exception:
        return set()
    return set(values)


def load_manifest(out_root: Path, current: Path, research: Path, universes: list[str]) -> dict[str, Any]:
    manifest_path = current / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "research_workspace": str(research),
            "product_root": str(current),
            "universes": universes,
            "paths": {},
        }
    manifest["published_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["research_workspace"] = str(research)
    manifest["product_root"] = str(current)
    manifest["universes"] = universes
    manifest.setdefault("paths", {}).update({
        "daily_scores": "daily_scores/trade_date={date}/universe={universe}/data.parquet",
        "recommendations": "recommendations/trade_date={date}/universe={universe}/data.parquet",
        "factor_metrics": "factors/metrics/universe={universe}/metrics.json",
        "rankic_timeseries": "factors/rankic_timeseries.parquet",
        "factor_layers": "factors/layers/universe={universe}/factor={factor_id}/data.json",
        "factor_layers_summary": "factors/layers/universe={universe}/summary.json",
        "market_overview": "market/trade_date={date}/universe={universe}/overview.json",
        "style_monitor": "style_monitor/trade_date={date}/universe={universe}/data.json",
        "portfolio": "portfolio/main",
    })
    return manifest


def main() -> None:
    args = parse_args()
    research = resolve_research(args.research_workspace)
    out_root = Path(args.output_root).expanduser().resolve() if args.output_root else research / "artifacts" / "product"
    current = out_root / "current"
    current.mkdir(parents=True, exist_ok=True)
    universes = [u.strip().lower() for u in args.universes.split(",") if u.strip()]

    inputs = resolve_inputs(args, research)
    daily_dates = partition_dates(research / "data" / "silver" / "daily_panel")
    if not daily_dates:
        raise FileNotFoundError("No daily_panel trade_date partitions found.")
    start_date = args.start_date.replace("-", "")
    end_date = (args.end_date or max(daily_dates)).replace("-", "")
    dates = trade_dates(research, start_date, end_date)
    daily_available = set(daily_dates)
    exposure_available = set(partition_dates(research / "data" / "barra" / "exposures"))
    score_dates = parquet_date_set(inputs.get("signal_file"), "end_date", start_date, end_date)

    counts: dict[str, dict[str, int]] = {
        "market": {u: 0 for u in universes},
        "style_monitor": {u: 0 for u in universes},
        "daily_scores": {u: 0 for u in universes},
        "recommendations": {u: 0 for u in universes},
    }

    for date in dates:
        for universe in universes:
            exposures = None
            if date in daily_available and not args.skip_market:
                counts["market"][universe] += 1
                if not args.dry_run:
                    publish_market_snapshot(research, current, date, universe)

            if date in daily_available and date in exposure_available and not args.skip_style_monitor:
                exposures = read_exposure(research, date, universe)
                counts["style_monitor"][universe] += 1
                if not args.dry_run:
                    publish_style_monitor(research, current, date, universe, exposures)

            if date in score_dates and not args.skip_scores and inputs.get("signal_file"):
                if exposures is None:
                    exposures = read_exposure(research, date, universe)
                counts["daily_scores"][universe] += 1
                counts["recommendations"][universe] += 1
                if not args.dry_run:
                    publish_daily_scores(current, date, universe, inputs["signal_file"], exposures, args.top_n)

    if not args.dry_run:
        manifest = load_manifest(out_root, current, research, universes)
        manifest.setdefault("history", {})["date_partitioned"] = {
            "published_at": datetime.now().isoformat(timespec="seconds"),
            "start_date": start_date,
            "end_date": end_date,
            "market_start_date": min(d for d in dates if d in daily_available) if any(d in daily_available for d in dates) else None,
            "market_end_date": max(d for d in dates if d in daily_available) if any(d in daily_available for d in dates) else None,
            "style_start_date": min(d for d in dates if d in daily_available and d in exposure_available) if any(d in daily_available and d in exposure_available for d in dates) else None,
            "style_end_date": max(d for d in dates if d in daily_available and d in exposure_available) if any(d in daily_available and d in exposure_available for d in dates) else None,
            "score_start_date": min(score_dates) if score_dates else None,
            "score_end_date": max(score_dates) if score_dates else None,
        }
        write_json(current / "manifest.json", manifest)
        write_json(out_root / "latest_manifest.json", manifest)

    print(json.dumps({
        "product_root": str(current),
        "start_date": start_date,
        "end_date": end_date,
        "universes": universes,
        "counts": counts,
        "signal_file": str(inputs.get("signal_file")) if inputs.get("signal_file") else None,
        "signal_date_count": len(score_dates),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
