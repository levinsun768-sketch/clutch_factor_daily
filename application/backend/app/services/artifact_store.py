from __future__ import annotations

import json
import math
import glob
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from app.core.config import Settings


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

NICKNAMES = {
    "Momentum": "Momentum Pulse",
    "Reversal": "Reversal Drift",
    "Size": "Size Tilt",
    "Value": "Value Signal",
    "Volatility": "Volatility Break",
    "Liquidity": "Liquidity Flow",
    "Beta": "Beta Link",
    "Neutral Alpha": "Neutral Alpha",
    "Unknown": "Fingerprint Signal",
}


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    return value.replace("-", "")


def display_date(value: str | None) -> str | None:
    if not value or len(value) != 8:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def normalize_factor_id(value: str) -> str:
    raw = value.lower()
    if raw.startswith("fp_"):
        suffix = raw.split("_", 1)[1]
    elif raw.startswith("fp"):
        suffix = raw[2:]
    else:
        suffix = raw
    return f"fp_{int(suffix):03d}"


def display_factor_id(value: str) -> str:
    return normalize_factor_id(value).upper()


def status_lights(rank_ic: float | None, icir: float | None, max_drawdown: float | None, turnover: float | None) -> dict[str, str]:
    rank_ic_value = abs(rank_ic or 0.0)
    predictive = "green" if rank_ic_value >= 0.04 else "yellow" if rank_ic_value >= 0.02 else "red"
    stable = "green" if (icir or 0.0) >= 1.0 else "yellow" if (icir or 0.0) >= 0.5 else "red"
    drawdown_value = abs(max_drawdown or 0.0)
    turnover_value = turnover if turnover is not None else 0.0
    risk = "green" if drawdown_value <= 0.08 and turnover_value <= 0.40 else "yellow" if drawdown_value <= 0.15 and turnover_value <= 0.65 else "red"
    return {"predictive": predictive, "stable": stable, "risk": risk}


class ArtifactStore:
    def __init__(self, settings: Settings):
        self.root = settings.research_workspace
        self.product_root = self.root / "artifacts" / "product"

    def product_manifest_path(self) -> Path:
        return self.product_root / "latest_manifest.json"

    @lru_cache(maxsize=1)
    def product_manifest(self) -> dict[str, Any] | None:
        path = self.product_manifest_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def product_current_root(self) -> Path | None:
        manifest = self.product_manifest()
        if not manifest:
            return None
        product_root = manifest.get("product_root")
        if product_root:
            return Path(product_root).expanduser().resolve()
        return self.product_root / "current"

    def product_path(self, key: str, date: str | None = None, universe: str = "all") -> Path | None:
        manifest = self.product_manifest()
        current = self.product_current_root()
        if not manifest or current is None:
            return None
        template = manifest.get("paths", {}).get(key)
        if not template:
            return None
        dt = date or manifest.get("trade_date")
        return current / template.format(date=dt, universe=universe)

    def latest_manifest(self) -> dict[str, Any]:
        product = self.product_manifest()
        portfolio_run = self.latest_portfolio_run()
        return {
            "research_workspace": str(self.root),
            "exists": self.root.exists(),
            "product_manifest": str(self.product_manifest_path()) if self.product_manifest_path().exists() else None,
            "product_trade_date": display_date(product.get("trade_date")) if product else None,
            "latest_date": display_date(self.latest_business_date()),
            "fingerprint_file": self._rel(self.fingerprint_file()),
            "prediction_file": self._rel(self.prediction_file()),
            "neutral_signal_file": self._rel(self.neutral_signal_file()),
            "barra_exposure_root": self._rel(self.exposure_root()),
            "portfolio_run": self._rel(portfolio_run),
            "factor_metric_file": self._rel(self.factor_metric_file()),
        }

    def health(self) -> dict[str, Any]:
        manifest = self.latest_manifest()
        required = {
            "fingerprint_file": bool(manifest["fingerprint_file"]),
            "neutral_signal_file": bool(manifest["neutral_signal_file"]),
            "barra_exposure_root": self.exposure_root().exists(),
            "portfolio_run": bool(manifest["portfolio_run"]),
        }
        return {"ok": self.root.exists(), "required": required, "manifest": manifest}

    def latest_business_date(self) -> str | None:
        product = self.product_manifest()
        if product and product.get("trade_date"):
            return str(product["trade_date"])

        portfolio_run = self.latest_portfolio_run()
        if portfolio_run:
            holdings = portfolio_run / "holdings.parquet"
            if holdings.exists():
                try:
                    date = pl.scan_parquet(str(holdings)).select(pl.max("trade_date")).collect().item()
                    if date:
                        return str(date)
                except Exception:
                    pass

        signal = self.neutral_signal_file()
        if signal:
            try:
                date = pl.scan_parquet(str(signal)).select(pl.max(pl.col("end_date").cast(pl.Utf8))).collect().item()
                if date:
                    return str(date)
            except Exception:
                pass

        dates = self._partition_dates(self.exposure_root())
        if not dates:
            return None
        return max(dates)

    def overview(self, date: str | None, universe: str) -> dict[str, Any]:
        dt = normalize_date(date) or self.latest_business_date()
        if not dt:
            return {"date": None, "universe": universe, "warnings": ["No available date found."]}
        return {
            "date": display_date(dt),
            "universe": universe,
            "market": self.market_overview(dt, universe),
            "style_monitor": self.style_monitor(dt, universe),
            "top_recommendations": self.composite_recommendations(dt, universe, limit=8),
            "portfolio": self.portfolio_today(dt, universe),
        }

    def market_overview(self, date: str, universe: str) -> dict[str, Any]:
        product_path = self.product_path("market_overview", date, universe)
        if product_path and product_path.exists():
            payload = json.loads(product_path.read_text(encoding="utf-8"))
            payload["date"] = display_date(str(payload.get("date")))
            return payload

        path = self.root / "data" / "silver" / "daily_panel" / f"trade_date={date}" / "data.parquet"
        if not path.exists():
            return {"date": display_date(date), "available": False}
        lf = pl.scan_parquet(str(path), missing_columns="insert", extra_columns="ignore")
        lf = self._apply_universe(lf, universe)
        columns = lf.collect_schema().names()
        exprs = [pl.len().alias("n")]
        if "pct_chg" in columns:
            exprs += [
                (pl.col("pct_chg") > 0).sum().alias("up"),
                (pl.col("pct_chg") < 0).sum().alias("down"),
                (pl.col("pct_chg") == 0).sum().alias("flat"),
                pl.col("pct_chg").mean().alias("avg_pct_chg"),
            ]
        if "amount" in columns:
            exprs.append(pl.col("amount").sum().alias("amount"))
        if "is_up_limit" in columns:
            exprs.append(pl.col("is_up_limit").fill_null(False).sum().alias("limit_up"))
        if "is_down_limit" in columns:
            exprs.append(pl.col("is_down_limit").fill_null(False).sum().alias("limit_down"))
        row = lf.select(exprs).collect().to_dicts()[0]
        return {key: clean_float(value) if isinstance(value, float) else value for key, value in row.items()} | {"date": display_date(date), "available": True}

    def style_monitor(self, date: str, universe: str) -> dict[str, Any]:
        product_path = self.product_path("style_monitor", date, universe)
        if product_path and product_path.exists():
            payload = json.loads(product_path.read_text(encoding="utf-8"))
            payload["date"] = display_date(str(payload.get("date")))
            return payload

        exposures = self.exposure_frame(date, universe)
        if exposures.is_empty():
            return {"date": display_date(date), "styles": {}, "n": 0}
        summary = exposures.select([pl.col(col).mean().alias(col) for col in STYLE_FACTORS]).to_dicts()[0]
        return {"date": display_date(date), "n": exposures.height, "styles": {k: clean_float(v) for k, v in summary.items()}}

    def factor_list(self, date: str | None, universe: str, sort: str, style: str | None, limit: int) -> list[dict[str, Any]]:
        layer_summaries = self.factor_layer_summaries(universe)
        product_path = self.product_path("factor_metrics", normalize_date(date) or self.latest_business_date(), universe)
        if product_path and product_path.exists():
            payload = json.loads(product_path.read_text(encoding="utf-8"))
            rows = []
            for item in payload.get("items", []):
                main_style = item.get("main_style") or "Unknown"
                if style and style.lower() not in {"all", main_style.lower()}:
                    continue
                factor_key = normalize_factor_id(str(item.get("factor", item.get("factor_id"))))
                layer = layer_summaries.get(factor_key, {})
                rank_ic = clean_float(item.get("rank_ic"))
                icir = clean_float(item.get("icir"))
                turnover = clean_float(item.get("turnover"))
                if turnover is None:
                    turnover = clean_float(layer.get("ls_turnover_mean"))
                net_cumret = clean_float(item.get("net_cumret"))
                if net_cumret is None:
                    net_cumret = clean_float(layer.get("net_cumret"))
                max_drawdown = clean_float(item.get("max_drawdown"))
                rows.append({
                    "factor_id": item.get("factor_id"),
                    "nickname": NICKNAMES.get(main_style, NICKNAMES["Unknown"]),
                    "rank_ic": rank_ic,
                    "icir": icir,
                    "sharpe": clean_float(item.get("sharpe")),
                    "turnover": turnover,
                    "max_drawdown": max_drawdown,
                    "net_cumret": net_cumret,
                    "main_style": main_style,
                    "style_exposure": item.get("style_exposure") or {},
                    "backtest_available": bool(item.get("backtest_available")) or bool(layer),
                    "layer_path": item.get("layer_path") or layer.get("path"),
                    "final_net_nav": clean_float(item.get("final_net_nav")) or clean_float(layer.get("final_net_nav")),
                    "status": status_lights(rank_ic, icir, max_drawdown, turnover),
                    "sparkline": self.factor_sparkline(factor_key),
                })
            sort_map = {
                "rankic": lambda x: abs(x.get("rank_ic") or 0.0),
                "icir": lambda x: x.get("icir") or 0.0,
                "turnover": lambda x: -(x.get("turnover") or 0.0),
                "return": lambda x: x.get("net_cumret") or 0.0,
                "style": lambda x: x.get("main_style") or "",
            }
            key_fn = sort_map.get(sort.lower(), sort_map["rankic"])
            return sorted(rows, key=key_fn, reverse=True)[:limit]

        metric_file = self.factor_metric_file()
        if not metric_file:
            return []
        metrics = pl.read_parquet(metric_file)
        single_summary = self.single_dim_summary_file()
        if single_summary:
            extra = pl.read_csv(single_summary).rename({"signal": "factor"})
            metrics = metrics.join(extra, on="factor", how="left")

        style_snapshot = self.factor_style_snapshot(normalize_date(date) or self.latest_business_date(), universe)
        rows = []
        for item in metrics.to_dicts():
            factor = normalize_factor_id(str(item["factor"]))
            layer = layer_summaries.get(factor, {})
            rank_ic = clean_float(item.get("rankic_mean"))
            icir = clean_float(item.get("rankic_ir"))
            turnover = clean_float(item.get("ls_turnover"))
            if turnover is None:
                turnover = clean_float(layer.get("ls_turnover_mean"))
            net_cumret = clean_float(item.get("net_cumret"))
            if net_cumret is None:
                net_cumret = clean_float(layer.get("net_cumret"))
            exposure = style_snapshot.get(factor, {})
            main_style = self._main_style(exposure)
            if style and style.lower() not in {"all", main_style.lower()}:
                continue
            rows.append({
                "factor_id": display_factor_id(factor),
                "nickname": NICKNAMES.get(main_style, NICKNAMES["Unknown"]),
                "rank_ic": rank_ic,
                "icir": icir,
                "sharpe": None,
                "turnover": turnover,
                "max_drawdown": None,
                "net_cumret": net_cumret,
                "main_style": main_style,
                "style_exposure": exposure,
                "backtest_available": bool(layer),
                "layer_path": layer.get("path"),
                "final_net_nav": clean_float(layer.get("final_net_nav")),
                "status": status_lights(rank_ic, icir, None, turnover),
                "sparkline": self.factor_sparkline(factor),
            })

        sort_map = {
            "rankic": lambda x: abs(x.get("rank_ic") or 0.0),
            "icir": lambda x: x.get("icir") or 0.0,
            "turnover": lambda x: -(x.get("turnover") or 0.0),
            "return": lambda x: x.get("net_cumret") or 0.0,
            "style": lambda x: x.get("main_style") or "",
        }
        key_fn = sort_map.get(sort.lower(), sort_map["rankic"])
        return sorted(rows, key=key_fn, reverse=True)[:limit]

    def factor_summary(self, factor_id: str, date: str | None, universe: str) -> dict[str, Any]:
        factor = normalize_factor_id(factor_id)
        items = self.factor_list(date, universe, sort="rankic", style=None, limit=128)
        item = next((x for x in items if normalize_factor_id(x["factor_id"]) == factor), None)
        if item is None:
            item = {"factor_id": display_factor_id(factor), "nickname": NICKNAMES["Unknown"]}
        item["recommendations"] = self.factor_recommendations(factor, normalize_date(date) or self.latest_business_date(), universe, 20)
        item["ic_timeseries"] = self.factor_ic_timeseries(factor)
        item["exposure"] = self.factor_style_snapshot(normalize_date(date) or self.latest_business_date(), universe).get(factor, {})
        item["layer_backtest"] = self.factor_layer_backtest(factor, universe)
        return item

    def factor_layer_path(self, factor_id: str, universe: str) -> Path | None:
        factor = display_factor_id(factor_id)
        current = self.product_current_root()
        if current is None:
            return None

        manifest = self.product_manifest() or {}
        template = manifest.get("paths", {}).get("factor_layers")
        if template:
            path = current / template.format(universe=universe, factor_id=factor, factor=factor.lower())
            if path.exists():
                return path

        path = current / "factors" / "layers" / f"universe={universe}" / f"factor={factor}" / "data.json"
        return path if path.exists() else None

    def factor_layer_backtest(self, factor_id: str, universe: str) -> dict[str, Any]:
        path = self.factor_layer_path(factor_id, universe)
        if not path or not path.exists():
            return {"available": False, "factor_id": display_factor_id(factor_id), "universe": universe}
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["available"] = True
        payload["path"] = self._rel(path)
        return payload

    def factor_layer_summaries(self, universe: str) -> dict[str, dict[str, Any]]:
        current = self.product_current_root()
        if current is None:
            return {}
        summary_path = current / "factors" / "layers" / f"universe={universe}" / "summary.json"
        rows: list[dict[str, Any]] = []
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = payload.get("items", [])
        else:
            root = current / "factors" / "layers" / f"universe={universe}"
            for path in sorted(root.glob("factor=FP_*/data.json")):
                try:
                    summary = json.loads(path.read_text(encoding="utf-8")).get("summary", {})
                except json.JSONDecodeError:
                    continue
                if summary:
                    summary = dict(summary)
                    summary["path"] = str(path.relative_to(current))
                    rows.append(summary)
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            factor = row.get("factor") or row.get("factor_id")
            if not factor:
                continue
            out[normalize_factor_id(str(factor))] = row
        return out

    def factor_ic_timeseries(self, factor_id: str) -> list[dict[str, Any]]:
        factor = normalize_factor_id(factor_id)
        product_path = self.product_path("rankic_timeseries")
        path = product_path if product_path and product_path.exists() else self.root / "artifacts" / "backtests" / "backtest_20260531_202830" / "fingerprint_dim_ic" / "fingerprint_dim_rankic_timeseries.parquet"
        if not path.exists():
            return []
        return (
            pl.scan_parquet(str(path))
            .select(["end_date", factor])
            .drop_nulls()
            .collect()
            .rename({factor: "value"})
            .with_columns(pl.col("end_date").map_elements(display_date, return_dtype=pl.Utf8).alias("date"))
            .select(["date", "value"])
            .to_dicts()
        )

    def factor_sparkline(self, factor_id: str, points: int = 60) -> list[dict[str, Any]]:
        rows = self.factor_ic_timeseries(factor_id)
        return rows[-points:]

    def factor_recommendations(self, factor_id: str, date: str | None, universe: str, limit: int) -> list[dict[str, Any]]:
        dt = normalize_date(date) or self.latest_business_date()
        if not dt:
            return []
        factor = normalize_factor_id(factor_id)
        fp = self.fingerprint_values(dt, [factor])
        exposures = self.exposure_frame(dt, universe).select(["ts_code", "name", "sw_l1_name"])
        if fp.is_empty() or exposures.is_empty():
            return []
        sign = self.factor_alpha_sign(factor)
        joined = fp.join(exposures, on="ts_code", how="inner").with_columns((pl.col(factor) * sign).alias("score"))
        return (
            joined.sort("score", descending=True)
            .head(limit)
            .with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
            .rename({"sw_l1_name": "industry"})
            .select(["rank", "ts_code", "name", "industry", "score"])
            .to_dicts()
        )

    def composite_recommendations(self, date: str, universe: str, limit: int = 20) -> list[dict[str, Any]]:
        product_path = self.product_path("recommendations", date, universe)
        if product_path and product_path.exists():
            df = pl.read_parquet(product_path)
            if df.is_empty():
                return []
            rename = {"sw_l1_name": "industry"} if "sw_l1_name" in df.columns else {}
            return df.head(limit).rename(rename).to_dicts()

        signal = self.neutral_signal_file()
        if not signal:
            return []
        scores = (
            pl.scan_parquet(str(signal))
            .select(["ts_code", pl.col("end_date").cast(pl.Utf8).alias("trade_date"), pl.col("score").cast(pl.Float64)])
            .filter(pl.col("trade_date") == date)
            .collect()
        )
        exposures = self.exposure_frame(date, universe)
        if scores.is_empty() or exposures.is_empty():
            return []
        joined = scores.join(exposures.select(["ts_code", "name", "sw_l1_name", "liquidity_amount_20", "volatility_60"]), on="ts_code", how="inner")
        return (
            joined.sort("score", descending=True)
            .head(limit)
            .with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
            .rename({"sw_l1_name": "industry"})
            .select(["rank", "ts_code", "name", "industry", "score", "liquidity_amount_20", "volatility_60"])
            .to_dicts()
        )

    def stock_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        latest = self.latest_business_date()
        if not latest:
            return []
        exposure = self.exposure_frame(latest, "all")
        if exposure.is_empty():
            return []
        q = query.lower()
        return (
            exposure
            .filter(pl.col("ts_code").str.to_lowercase().str.contains(q, literal=True) | pl.col("name").str.contains(query, literal=True))
            .select(["ts_code", "name", "sw_l1_name"])
            .head(limit)
            .rename({"sw_l1_name": "industry"})
            .to_dicts()
        )

    def stock_fingerprint(self, ts_code: str, date: str | None) -> dict[str, Any]:
        dt = normalize_date(date) or self.latest_business_date()
        cols = [f"fp_{i:03d}" for i in range(64)]
        fp = self.fingerprint_values(dt, cols).filter(pl.col("ts_code") == ts_code)
        if fp.is_empty():
            return {"ts_code": ts_code, "date": display_date(dt), "embedding": []}
        row = fp.to_dicts()[0]
        return {"ts_code": ts_code, "date": display_date(dt), "embedding": [clean_float(row[col]) for col in cols]}

    def stock_profile(self, ts_code: str, date: str | None, universe: str) -> dict[str, Any]:
        dt = normalize_date(date) or self.latest_business_date()
        exposure = self.exposure_frame(dt, universe).filter(pl.col("ts_code") == ts_code)
        base = {"ts_code": ts_code, "date": display_date(dt)}
        if not exposure.is_empty():
            row = exposure.to_dicts()[0]
            base.update({
                "name": row.get("name"),
                "industry": row.get("sw_l1_name"),
                "style_exposure": {col: clean_float(row.get(col)) for col in STYLE_FACTORS},
            })
        signal = self.neutral_signal_file()
        if signal:
            score = (
                pl.scan_parquet(str(signal))
                .filter((pl.col("ts_code") == ts_code) & (pl.col("end_date").cast(pl.Utf8) == dt))
                .select(pl.col("score").cast(pl.Float64))
                .collect()
            )
            if not score.is_empty():
                base["composite_score"] = clean_float(score.item())
        return base

    def similar_stocks(self, ts_code: str, date: str | None, universe: str, top_n: int) -> list[dict[str, Any]]:
        dt = normalize_date(date) or self.latest_business_date()
        cols = [f"fp_{i:03d}" for i in range(64)]
        fp = self.fingerprint_values(dt, cols)
        exposures = self.exposure_frame(dt, universe).select(["ts_code", "name", "sw_l1_name"])
        joined = fp.join(exposures, on="ts_code", how="inner")
        if joined.is_empty() or ts_code not in set(joined.get_column("ts_code").to_list()):
            return []
        query = joined.filter(pl.col("ts_code") == ts_code).select(cols).to_numpy()[0]
        q_norm = math.sqrt(float((query * query).sum()))
        if q_norm <= 1e-12:
            return []
        arr = joined.select(cols).to_numpy()
        denom = (arr * arr).sum(axis=1) ** 0.5 * q_norm
        sims = (arr @ query) / (denom + 1e-12)
        out = joined.select(["ts_code", "name", "sw_l1_name"]).with_columns(pl.Series("similarity", sims)).filter(pl.col("ts_code") != ts_code)
        return (
            out.sort("similarity", descending=True)
            .head(top_n)
            .with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
            .rename({"sw_l1_name": "industry"})
            .select(["rank", "ts_code", "name", "industry", "similarity"])
            .to_dicts()
        )

    def portfolio_today(self, date: str | None, universe: str) -> dict[str, Any]:
        product_portfolio = self.product_path("portfolio")
        run = product_portfolio if product_portfolio and product_portfolio.exists() else self.latest_portfolio_run()
        if not run:
            return {"available": False}
        dt = normalize_date(date)
        holdings_path = run / "holdings.parquet"
        returns_path = run / "daily_returns.parquet"
        if not holdings_path.exists():
            return {"available": False, "portfolio_run": self._rel(run)}
        holdings = pl.read_parquet(holdings_path)
        if dt is None or dt not in set(holdings.get_column("trade_date").to_list()):
            dt = str(holdings.get_column("trade_date").max())
        current = holdings.filter(pl.col("trade_date") == dt)
        previous_dates = sorted([x for x in holdings.get_column("trade_date").unique().to_list() if x < dt])
        previous = holdings.filter(pl.col("trade_date") == previous_dates[-1]) if previous_dates else pl.DataFrame()
        current_codes = set(current.get_column("ts_code").to_list()) if not current.is_empty() else set()
        previous_codes = set(previous.get_column("ts_code").to_list()) if not previous.is_empty() else set()
        returns = pl.read_parquet(returns_path).filter(pl.col("trade_date") == dt) if returns_path.exists() else pl.DataFrame()
        exposure = self.holdings_style_exposure(dt, list(current_codes))
        payload = {
            "available": True,
            "portfolio_run": self._rel(run),
            "trade_date": display_date(dt),
            "universe": universe,
            "holdings": current.rename({"sw_l1_name": "industry"}).with_columns(pl.lit(1.0 / max(current.height, 1)).alias("weight")).to_dicts(),
            "buys": current.filter(pl.col("ts_code").is_in(list(current_codes - previous_codes))).rename({"sw_l1_name": "industry"}).to_dicts(),
            "sells": previous.filter(pl.col("ts_code").is_in(list(previous_codes - current_codes))).rename({"sw_l1_name": "industry"}).to_dicts() if not previous.is_empty() else [],
            "style_exposure": exposure,
        }
        if not returns.is_empty():
            payload["return_row"] = returns.to_dicts()[0]
        return payload

    def portfolio_backtest(self, portfolio_id: str = "main") -> dict[str, Any]:
        product_portfolio = self.product_path("portfolio")
        run = product_portfolio if product_portfolio and product_portfolio.exists() else self.latest_portfolio_run()
        if not run:
            return {"available": False}
        summary_path = run / "summary.json"
        returns_path = run / "daily_returns.parquet"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        daily = []
        if returns_path.exists():
            daily = (
                pl.scan_parquet(str(returns_path))
                .with_columns(pl.col("exec_date").map_elements(display_date, return_dtype=pl.Utf8).alias("date"))
                .select(["date", "net_nav", "net_ret", "turnover"])
                .collect()
                .to_dicts()
            )
        return {"available": True, "portfolio_id": portfolio_id, "portfolio_run": self._rel(run), "summary": summary, "daily": daily}

    def holdings_style_exposure(self, date: str, codes: list[str]) -> dict[str, float | None]:
        if not codes:
            return {}
        exposure = self.exposure_frame(date, "all").filter(pl.col("ts_code").is_in(codes))
        if exposure.is_empty():
            return {}
        row = exposure.select([pl.col(col).mean().alias(col) for col in STYLE_FACTORS]).to_dicts()[0]
        return {k: clean_float(v) for k, v in row.items()}

    @lru_cache(maxsize=64)
    def factor_style_snapshot(self, date: str | None, universe: str) -> dict[str, dict[str, float | None]]:
        if not date:
            return {}
        factor_cols = [f"fp_{i:03d}" for i in range(64)]
        fp = self.fingerprint_values(date, factor_cols)
        exposures = self.exposure_frame(date, universe).select(["ts_code", *STYLE_FACTORS])
        if fp.is_empty() or exposures.is_empty():
            return {}
        joined = fp.join(exposures, on="ts_code", how="inner").drop_nulls()
        if joined.height < 100:
            return {}
        rows = joined.to_pandas()
        out: dict[str, dict[str, float | None]] = {}
        for factor in factor_cols:
            exposure = {}
            for style in STYLE_FACTORS:
                exposure[style] = clean_float(rows[factor].corr(rows[style]))
            out[factor] = exposure
        return out

    def factor_alpha_sign(self, factor: str) -> float:
        metric_file = self.factor_metric_file()
        if not metric_file:
            return 1.0
        row = pl.read_parquet(metric_file).filter(pl.col("factor") == factor)
        if row.is_empty():
            return 1.0
        value = clean_float(row.select("rankic_mean").item())
        return 1.0 if (value or 0.0) >= 0 else -1.0

    def fingerprint_values(self, date: str, columns_key: tuple[str, ...] | list[str]) -> pl.DataFrame:
        return self._fingerprint_values_cached(date, tuple(columns_key))

    @lru_cache(maxsize=128)
    def _fingerprint_values_cached(self, date: str, columns_key: tuple[str, ...]) -> pl.DataFrame:
        columns = list(columns_key)
        fp = self.fingerprint_file()
        if not fp:
            return pl.DataFrame({"ts_code": []})
        return (
            pl.scan_parquet(str(fp), missing_columns="insert", extra_columns="ignore")
            .select(["ts_code", pl.col("end_date").cast(pl.Utf8).alias("end_date"), *columns])
            .filter(pl.col("end_date") == date)
            .drop("end_date")
            .collect()
        )

    @lru_cache(maxsize=128)
    def exposure_frame(self, date: str, universe: str) -> pl.DataFrame:
        path = self.exposure_root() / f"trade_date={date}" / "data.parquet"
        if not path.exists():
            return pl.DataFrame()
        lf = pl.scan_parquet(str(path), missing_columns="insert", extra_columns="ignore")
        lf = self._apply_universe(lf, universe)
        return lf.collect()

    def fingerprint_file(self) -> Path | None:
        product = self.product_manifest()
        if product:
            ref = product.get("references", {}).get("fingerprint_file")
            if ref:
                path = Path(ref)
                if not path.is_absolute():
                    current = self.product_current_root()
                    path = (current / path).resolve() if current else (self.root / ref).resolve()
                if path.exists():
                    return path

        full_dataset = self._latest_file([self.root / "artifacts" / "models" / "*" / "fp_dataset" / "fingerprints_daily_*.parquet"])
        if full_dataset:
            return full_dataset
        return self._latest_file([self.root / "artifacts" / "models" / "*" / "fp_incremental" / "fingerprints_daily_*.parquet"])

    def prediction_file(self) -> Path | None:
        return self._latest_file([self.root / "downstream" / "run" / "*" / "prediction_scores*_inference*.parquet"])

    def neutral_signal_file(self) -> Path | None:
        product = self.product_manifest()
        if product:
            ref = product.get("references", {}).get("signal_file")
            if ref:
                path = Path(ref)
                if not path.is_absolute():
                    current = self.product_current_root()
                    path = (current / path).resolve() if current else (self.root / ref).resolve()
                if path.exists():
                    return path
        return self._latest_file([
            self.root / "artifacts" / "barra" / "signal_neutral_ic" / "*" / "ewma" / "neutral_signal_ewma20.parquet",
            self.root / "artifacts" / "barra" / "signal_neutral_ic" / "*" / "neutral_signal.parquet",
        ])

    def exposure_root(self) -> Path:
        return self.root / "data" / "barra" / "exposures"

    def factor_metric_file(self) -> Path | None:
        product = self.product_manifest()
        if product:
            ref = product.get("references", {}).get("factor_metric_file")
            if ref:
                path = Path(ref)
                if path.exists():
                    return path
        return self._latest_file([self.root / "artifacts" / "backtests" / "*" / "fingerprint_dim_ic" / "fingerprint_dim_ic_summary.parquet"])

    def single_dim_summary_file(self) -> Path | None:
        product = self.product_manifest()
        if product:
            ref = product.get("references", {}).get("single_dim_summary_file")
            if ref:
                path = Path(ref)
                if path.exists():
                    return path
        return self._latest_file([self.root / "artifacts" / "backtests" / "*" / "single_dim_signals" / "single_dim_backtest_summary.csv"])

    def latest_portfolio_run(self) -> Path | None:
        candidates = [path.parent for path in (self.root / "artifacts" / "portfolio").glob("*/summary.json")]
        if not candidates:
            return None
        return max(candidates, key=lambda path: (path / "summary.json").stat().st_mtime)

    def _apply_universe(self, lf: pl.LazyFrame, universe: str) -> pl.LazyFrame:
        flag = UNIVERSE_FLAGS.get(universe.lower())
        if not flag:
            return lf
        names = lf.collect_schema().names()
        if flag not in names:
            return lf
        return lf.filter(pl.col(flag).fill_null(False))

    def _main_style(self, exposure: dict[str, float | None]) -> str:
        clean = {k: abs(v) for k, v in exposure.items() if v is not None}
        if not clean:
            return "Unknown"
        style_key, value = max(clean.items(), key=lambda item: item[1])
        if value < 0.25:
            return "Neutral Alpha"
        return STYLE_NAMES.get(style_key, "Unknown")

    def _partition_dates(self, root: Path) -> list[str]:
        if not root.exists():
            return []
        return sorted(path.name.split("=", 1)[1] for path in root.glob("trade_date=*") if "=" in path.name)

    def _latest_file(self, patterns: list[Path]) -> Path | None:
        files: list[Path] = []
        for pattern in patterns:
            files.extend(Path(path) for path in glob.glob(str(pattern)))
        files = [path for path in files if path.exists()]
        if not files:
            return None
        return max(files, key=lambda path: path.stat().st_mtime)

    def _rel(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)
