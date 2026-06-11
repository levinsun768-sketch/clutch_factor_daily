from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import load_settings
from app.services.artifact_store import ArtifactStore, normalize_date, normalize_factor_id
from app.services.agent_service import AgentContext, AgentService


settings = load_settings()
store = ArtifactStore(settings)
agent_service = AgentService(store, settings)

app = FastAPI(title="Clutch Factor Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentRequest(BaseModel):
    message: str
    context: dict[str, Any] = {}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return store.health()


@app.get("/api/meta/latest")
def latest_manifest() -> dict[str, Any]:
    return store.latest_manifest()


@app.get("/api/overview")
def overview(date: str | None = None, universe: str = "all") -> dict[str, Any]:
    return store.overview(date, universe)


@app.get("/api/factors")
def factors(
    date: str | None = None,
    universe: str = "all",
    sort: str = "rankic",
    style: str | None = None,
    limit: int = Query(default=64, ge=1, le=128),
) -> dict[str, Any]:
    items = store.factor_list(date, universe, sort, style, limit)
    return {"date": date, "universe": universe, "items": items}


@app.get("/api/factors/{factor_id}/summary")
def factor_summary(factor_id: str, date: str | None = None, universe: str = "all") -> dict[str, Any]:
    try:
        normalize_factor_id(factor_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid factor id: {factor_id}") from exc
    return store.factor_summary(factor_id, date, universe)


@app.get("/api/factors/{factor_id}/layers")
def factor_layers(factor_id: str, universe: str = "all") -> dict[str, Any]:
    try:
        normalize_factor_id(factor_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid factor id: {factor_id}") from exc
    return store.factor_layer_backtest(factor_id, universe)


@app.get("/api/factors/{factor_id}/recommendations")
def factor_recommendations(
    factor_id: str,
    date: str | None = None,
    universe: str = "all",
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    dt = normalize_date(date) or store.latest_business_date()
    return {
        "factor_id": normalize_factor_id(factor_id).upper(),
        "date": dt,
        "universe": universe,
        "items": store.factor_recommendations(factor_id, dt, universe, limit),
    }


@app.get("/api/stocks/search")
def stock_search(q: str, limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    return {"query": q, "items": store.stock_search(q, limit)}


@app.get("/api/stocks/{ts_code}/profile")
def stock_profile(ts_code: str, date: str | None = None, universe: str = "all") -> dict[str, Any]:
    return store.stock_profile(ts_code, date, universe)


@app.get("/api/stocks/{ts_code}/fingerprint")
def stock_fingerprint(ts_code: str, date: str | None = None) -> dict[str, Any]:
    return store.stock_fingerprint(ts_code, date)


@app.get("/api/stocks/{ts_code}/similar")
def similar_stocks(
    ts_code: str,
    date: str | None = None,
    universe: str = "all",
    top_n: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    return {"ts_code": ts_code, "date": date, "universe": universe, "items": store.similar_stocks(ts_code, date, universe, top_n)}


@app.get("/api/portfolio/today")
def portfolio_today(date: str | None = None, universe: str = "all", portfolio_id: str = "main") -> dict[str, Any]:
    payload = store.portfolio_today(date, universe)
    payload["portfolio_id"] = portfolio_id
    return payload


@app.get("/api/portfolio/backtest")
def portfolio_backtest(universe: str = "all", portfolio_id: str = "main") -> dict[str, Any]:
    payload = store.portfolio_backtest(portfolio_id)
    payload["universe"] = universe
    return payload


@app.post("/api/agent/chat")
def agent_chat(req: AgentRequest) -> dict[str, Any]:
    context = req.context or {}
    agent_context = AgentContext(
        message=req.message,
        route=str(context.get("route") or "/"),
        date=context.get("date"),
        universe=str(context.get("universe") or "all"),
        benchmark=context.get("benchmark"),
        locale=context.get("locale"),
        selected_factor=context.get("selected_factor"),
        selected_stock=context.get("selected_stock"),
        selected_portfolio=context.get("selected_portfolio"),
    )
    return agent_service.respond(agent_context)
