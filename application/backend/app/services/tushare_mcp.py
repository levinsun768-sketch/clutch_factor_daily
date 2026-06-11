from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class McpResult:
    ok: bool
    data: dict[str, Any]
    error: str | None = None


class TushareMCPClient:
    def __init__(self, url: str):
        self.url = url
        self._session_id: str | None = None
        self._id = 0

    def list_tools(self) -> McpResult:
        try:
            self._ensure_initialized()
            payload = self._request({"method": "tools/list", "params": {}})
            result = payload.get("result") or {}
            tools = result.get("tools") or []
            return McpResult(ok=True, data={"tools": tools, "raw": result, "session_id": self._session_id})
        except Exception as exc:
            return McpResult(ok=False, data={}, error=str(exc))

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> McpResult:
        try:
            self._ensure_initialized()
            payload = self._request({
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            })
            result = payload.get("result") or {}
            content = result.get("content") or []
            error = payload.get("error")
            if error:
                return McpResult(ok=False, data={"result": result}, error=str(error))
            return McpResult(ok=True, data={"content": content, "result": result})
        except Exception as exc:
            return McpResult(ok=False, data={}, error=str(exc))

    def _ensure_initialized(self) -> None:
        if self._session_id:
            return
        payload = self._request(
            {
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "clutch-factor-backend", "version": "0.1.0"},
                },
            }
        )
        result = payload.get("result") or {}
        if not result:
            raise RuntimeError(f"Tushare MCP initialize returned empty result: {payload!r}")
        self._session_id = self._session_header_from_response(payload.get("_headers") or {}) or self._session_id
        self._send_notification("notifications/initialized")

    def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            response = client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            self._session_id = response.headers.get("mcp-session-id") or response.headers.get("Mcp-Session-Id") or self._session_id

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        self._id += 1
        is_notification = bool(message.pop("notification", False))
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": message["method"]}
        if not is_notification:
            payload["id"] = self._id
        if message.get("params") is not None:
            payload["params"] = message["params"]

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            response = client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            self._session_id = response.headers.get("mcp-session-id") or response.headers.get("Mcp-Session-Id") or self._session_id
            content_type = response.headers.get("content-type", "")
            if content_type.startswith("application/json"):
                data = response.json()
                if isinstance(data, dict):
                    data["_headers"] = dict(response.headers)
                return data
            if "text/event-stream" in content_type:
                return self._parse_sse(response.text, payload.get("id"))
            text = response.text.strip()
            if text.startswith("{"):
                data = json.loads(text)
                if isinstance(data, dict):
                    data["_headers"] = dict(response.headers)
                return data
            raise RuntimeError(f"Unexpected Tushare MCP content-type: {content_type or 'unknown'}")

    def _parse_sse(self, text: str, request_id: int | None) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        current_data: list[str] = []
        current_event: str | None = None
        for line in text.splitlines():
            line = line.rstrip("\r")
            if not line.strip():
                if current_data:
                    events.append({"event": current_event, "data": "\n".join(current_data)})
                    current_data = []
                    current_event = None
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
                continue
            if line.startswith("data:"):
                current_data.append(line.split(":", 1)[1].lstrip())
        if current_data:
            events.append({"event": current_event, "data": "\n".join(current_data)})
        for item in reversed(events):
            raw = item.get("data") or ""
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and (request_id is None or obj.get("id") == request_id):
                return obj
        raise RuntimeError(f"Unable to parse MCP SSE response: {events!r}")

    def _session_header_from_response(self, headers: dict[str, Any]) -> str | None:
        return headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
