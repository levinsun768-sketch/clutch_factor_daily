from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    research_workspace: Path
    cors_origins: list[str]


def load_settings() -> Settings:
    backend_root = Path(__file__).resolve().parents[2]
    default_workspace = backend_root.parent.parent / "research_workspace"
    research_workspace = Path(os.getenv("RESEARCH_WORKSPACE", str(default_workspace))).expanduser()
    if not research_workspace.is_absolute():
        research_workspace = (backend_root / research_workspace).resolve()
    else:
        research_workspace = research_workspace.resolve()

    cors = os.getenv("API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return Settings(
        research_workspace=research_workspace,
        cors_origins=[item.strip() for item in cors.split(",") if item.strip()],
    )
