from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    source_code: Path
    cors_origins: list[str]
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_app_name: str
    tushare_mcp_url: str | None


def load_settings() -> Settings:
    backend_root = Path(__file__).resolve().parents[2]
    load_dotenv(backend_root / ".env", override=False)
    default_workspace = backend_root.parent.parent / "source_code"
    source_code = Path(os.getenv("SOURCE_CODE_DIR", str(default_workspace))).expanduser()
    if not source_code.is_absolute():
        source_code = (backend_root / source_code).resolve()
    else:
        source_code = source_code.resolve()

    cors = os.getenv("API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return Settings(
        source_code=source_code,
        cors_origins=[item.strip() for item in cors.split(",") if item.strip()],
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat"),
        openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", "Clutch Factor Research Terminal"),
        tushare_mcp_url=os.getenv("TUSHARE_MCP_URL") or None,
    )
