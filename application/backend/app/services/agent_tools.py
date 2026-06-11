"""Agent tool definitions for LLM function calling."""
from __future__ import annotations

from typing import Any

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_factor_summary",
            "description": "Get detailed summary of a specific fingerprint factor including metrics, recommendations, IC timeseries, and style exposure",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor_id": {
                        "type": "string",
                        "description": "Factor ID like FP_00, FP_12, fp_005, etc."
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYYMMDD format, e.g. 20260610"
                    },
                    "universe": {
                        "type": "string",
                        "enum": ["all", "hs300", "csi500", "csi1000"],
                        "description": "Stock universe"
                    }
                },
                "required": ["factor_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_factor_list",
            "description": "Get list of all factors with metrics, sorted and filtered by criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYYMMDD format"
                    },
                    "universe": {
                        "type": "string",
                        "enum": ["all", "hs300", "csi500", "csi1000"]
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["rankic", "icir", "return", "turnover", "style"],
                        "description": "Sort criterion"
                    },
                    "style": {
                        "type": "string",
                        "description": "Filter by style: Momentum, Reversal, Size, Value, Volatility, Liquidity, Beta, Neutral Alpha"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of factors to return",
                        "default": 20
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_profile",
            "description": "Get stock profile including name, industry, style exposure, and composite score",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "Stock code like 600519.SH, 000858.SZ"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYYMMDD format"
                    },
                    "universe": {
                        "type": "string",
                        "enum": ["all", "hs300", "csi500", "csi1000"]
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_stocks",
            "description": "Find stocks with similar fingerprint embeddings to a given stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "Stock code"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYYMMDD format"
                    },
                    "universe": {
                        "type": "string",
                        "enum": ["all", "hs300", "csi500", "csi1000"]
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of similar stocks to return",
                        "default": 10
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_tushare",
            "description": "Query Tushare data via MCP server. Use this to get real-time market data, fundamentals, or other Tushare API data",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Tushare MCP tool name, e.g. 'query_stock_basic', 'query_daily'"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the Tushare tool",
                        "additionalProperties": True
                    }
                },
                "required": ["tool_name", "arguments"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_overview",
            "description": "Get market overview statistics for a given date and universe",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYYMMDD format"
                    },
                    "universe": {
                        "type": "string",
                        "enum": ["all", "hs300", "csi500", "csi1000"]
                    }
                }
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: dict[str, Any], store: Any, settings: Any) -> dict[str, Any]:
    """Execute an agent tool and return results."""
    from app.services.artifact_store import normalize_date, normalize_factor_id
    from app.services.tushare_mcp import TushareMCPClient

    if tool_name == "get_factor_summary":
        factor_id = normalize_factor_id(arguments["factor_id"])
        date = normalize_date(arguments.get("date")) or store.latest_business_date()
        universe = arguments.get("universe", "all")
        return store.factor_summary(factor_id, date, universe)

    elif tool_name == "get_factor_list":
        date = normalize_date(arguments.get("date"))
        universe = arguments.get("universe", "all")
        sort = arguments.get("sort", "rankic")
        style = arguments.get("style")
        limit = arguments.get("limit", 20)
        items = store.factor_list(date, universe, sort, style, limit)
        return {"items": items, "count": len(items)}

    elif tool_name == "get_stock_profile":
        ts_code = arguments["ts_code"]
        date = normalize_date(arguments.get("date")) or store.latest_business_date()
        universe = arguments.get("universe", "all")
        return store.stock_profile(ts_code, date, universe)

    elif tool_name == "get_similar_stocks":
        ts_code = arguments["ts_code"]
        date = normalize_date(arguments.get("date")) or store.latest_business_date()
        universe = arguments.get("universe", "all")
        top_n = arguments.get("top_n", 10)
        items = store.similar_stocks(ts_code, date, universe, top_n)
        return {"query_ts_code": ts_code, "items": items}

    elif tool_name == "query_tushare":
        if not settings.tushare_mcp_url:
            return {"error": "Tushare MCP not configured"}
        client = TushareMCPClient(settings.tushare_mcp_url)
        mcp_tool_name = arguments["tool_name"]
        mcp_args = arguments.get("arguments", {})
        result = client.call_tool(mcp_tool_name, mcp_args)
        if result.ok:
            return {"content": result.data.get("content", []), "raw": result.data}
        return {"error": result.error}

    elif tool_name == "get_market_overview":
        date = normalize_date(arguments.get("date")) or store.latest_business_date()
        universe = arguments.get("universe", "all")
        return store.market_overview(date, universe)

    else:
        return {"error": f"Unknown tool: {tool_name}"}
