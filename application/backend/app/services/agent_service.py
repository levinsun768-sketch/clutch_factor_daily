from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.services.artifact_store import ArtifactStore, normalize_date, normalize_factor_id
from app.services.tushare_mcp import TushareMCPClient
from app.services.agent_tools import AGENT_TOOLS, execute_tool


@dataclass(frozen=True)
class AgentContext:
    message: str
    route: str
    date: str | None
    universe: str
    benchmark: str | None
    locale: str | None
    selected_factor: str | None = None
    selected_stock: str | None = None
    selected_portfolio: str | None = None


class AgentService:
    def __init__(self, store: ArtifactStore, settings: Settings):
        self.store = store
        self.settings = settings

    def dashboard_snapshot(self, context: AgentContext) -> dict[str, Any]:
        date = normalize_date(context.date) or self.store.latest_business_date()
        universe = context.universe or "all"
        route = context.route or "/"
        snapshot: dict[str, Any] = {
            "route": route,
            "date": date,
            "universe": universe,
            "benchmark": context.benchmark,
            "locale": context.locale,
            "market": self.store.market_overview(date, universe) if date else {},
            "style_monitor": self.store.style_monitor(date, universe) if date else {},
            "portfolio": self.store.portfolio_today(date, universe) if date else {},
        }

        factor_id = self._current_factor(context)
        if factor_id and date:
            snapshot["factor"] = self.store.factor_summary(factor_id, date, universe)

        ts_code = self._current_stock(context)
        if ts_code and date:
            snapshot["stock"] = self.store.stock_profile(ts_code, date, universe)
            snapshot["similar_stocks"] = self.store.similar_stocks(ts_code, date, universe, 10)

        if route.startswith("/portfolio"):
            snapshot["backtest"] = self.store.portfolio_backtest(context.selected_portfolio or "main")

        snapshot["tushare"] = self.tushare_snapshot()

        return snapshot

    def respond(self, context: AgentContext) -> dict[str, Any]:
        snapshot = self.dashboard_snapshot(context)
        if not self.settings.openrouter_api_key:
            return self._fallback_response(context, snapshot)

        prompt = self._build_prompt(context, snapshot)
        try:
            content = self._call_openrouter(prompt)
            return {
                "markdown": content,
                "artifacts": self._build_artifacts(snapshot),
                "context": snapshot,
                "provider": "openrouter",
            }
        except Exception as exc:
            return {
                "markdown": self._fallback_markdown(context, snapshot, error=str(exc)),
                "artifacts": self._build_artifacts(snapshot),
                "context": snapshot,
                "provider": "fallback",
            }

    def tushare_snapshot(self) -> dict[str, Any]:
        if not self.settings.tushare_mcp_url:
            return {"available": False, "reason": "TUSHARE_MCP_URL not configured"}
        try:
            client = TushareMCPClient(self.settings.tushare_mcp_url)
            result = client.list_tools()
            if result.ok:
                tools = result.data.get("tools") or []
                return {"available": True, "tools": tools, "session_id": result.data.get("session_id") }
            return {"available": False, "error": result.error}
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _fallback_response(self, context: AgentContext, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "markdown": self._fallback_markdown(context, snapshot),
            "artifacts": self._build_artifacts(snapshot),
            "context": snapshot,
            "provider": "fallback",
        }

    def _build_prompt(self, context: AgentContext, snapshot: dict[str, Any]) -> str:
        locale = context.locale or "zh"
        if locale == "zh":
            system = (
                "你是一个A股量化研究助手。你可以使用提供的工具查询因子、股票、市场数据。"
                "回答时要简洁、准确，引用具体的数据字段。使用中文回答。"
            )
        else:
            system = (
                "You are an equity research assistant for A-share quantitative analysis. "
                "Use the provided tools to query factors, stocks, and market data. "
                "Be concise, factual, and cite specific data fields."
            )

        payload = json.dumps(snapshot, ensure_ascii=False, default=str, indent=2)
        return (
            f"{system}\n\n"
            f"User question: {context.message}\n\n"
            f"Current context:\n{payload}"
        )

    def _call_openrouter(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.settings.openrouter_model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "tools": AGENT_TOOLS,
                "tool_choice": "auto",
                "temperature": 0.3,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "X-Title": self.settings.openrouter_app_name,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenRouter HTTP {exc.code}") from exc
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")
        message = choices[0].get("message") or {}

        # Handle tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            tool_results = []
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                tool_name = func.get("name")
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                result = execute_tool(tool_name, arguments, self.store, self.settings)
                tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })

            # Format tool results for display
            content_parts = []
            if message.get("content"):
                content_parts.append(str(message["content"]))

            for tr in tool_results:
                content_parts.append(f"\n**工具调用: {tr['tool']}**\n")
                content_parts.append(f"```json\n{json.dumps(tr['result'], ensure_ascii=False, indent=2)}\n```\n")

            return "\n".join(content_parts)

        content = message.get("content")
        if not content:
            raise RuntimeError("OpenRouter returned empty content")
        return str(content)

    def _build_artifacts(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        factor = snapshot.get("factor") or {}
        if factor.get("factor_id"):
            artifacts.append(
                {
                    "type": "link",
                    "title": f"Open {factor.get('factor_id')}",
                    "route": f"/factors/{factor.get('factor_id')}",
                }
            )
        stock = snapshot.get("stock") or {}
        if stock.get("ts_code"):
            artifacts.append(
                {
                    "type": "link",
                    "title": f"Open {stock.get('ts_code')}",
                    "route": f"/stocks/{stock.get('ts_code')}",
                }
            )
        artifacts.append({"type": "link", "title": "Open Portfolio Board", "route": "/portfolio"})
        return artifacts

    def _fallback_markdown(self, context: AgentContext, snapshot: dict[str, Any], error: str | None = None) -> str:
        lines = [
            f"当前路由：`{context.route}`",
            f"日期：`{snapshot.get('date') or 'latest'}`，股票池：`{context.universe}`",
        ]
        market = snapshot.get("market") or {}
        if market:
            lines.append(f"市场样本数：`{market.get('n', '-')}`，平均涨跌幅：`{market.get('avg_pct_chg', '-')}`")
        style_monitor = snapshot.get("style_monitor") or {}
        styles = style_monitor.get("styles") or {}
        top_style = next(iter(styles.items()), None)
        if top_style:
            lines.append(f"风格快照：`{top_style[0]}` = `{top_style[1]}`")
        portfolio = snapshot.get("portfolio") or {}
        return_row = portfolio.get("return_row") or {}
        if return_row:
            lines.append(
                f"组合当日净收益：`{return_row.get('net_ret', '-')}`，换手：`{return_row.get('turnover', '-')}`"
            )
        factor = snapshot.get("factor") or {}
        if factor:
            lines.append(
                f"当前因子：`{factor.get('factor_id')}`，RankIC：`{factor.get('rank_ic', '-')}`，ICIR：`{factor.get('icir', '-')}`"
            )
        stock = snapshot.get("stock") or {}
        if stock:
            lines.append(f"当前股票：`{stock.get('ts_code')}`，名称：`{stock.get('name', '-')}`")
        if error:
            lines.append(f"外部模型暂不可用，已回退到本地摘要：`{error}`")
        return "\n\n".join(lines)

    def _current_factor(self, context: AgentContext) -> str | None:
        if context.selected_factor:
            return normalize_factor_id(context.selected_factor)
        parts = [part for part in context.route.split("/") if part]
        if len(parts) >= 2 and parts[0] == "factors":
            return normalize_factor_id(parts[1])
        return None

    def _current_stock(self, context: AgentContext) -> str | None:
        if context.selected_stock:
            return context.selected_stock
        parts = [part for part in context.route.split("/") if part]
        if len(parts) >= 2 and parts[0] == "stocks":
            return parts[1]
        return None
