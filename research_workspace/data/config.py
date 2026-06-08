from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    tushare_token: str
    tushare_http_url: str
    start_date: str
    end_date: str
    data_root: Path
    window_size: int

    @property
    def bronze_root(self) -> Path:
        return self.data_root / "bronze"


def get_settings() -> Settings:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise ValueError("Missing TUSHARE_TOKEN in environment or .env")

    http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    start_date = os.getenv("FPD_START_DATE", "20160101").strip()
    end_date = os.getenv("FPD_END_DATE", "").strip()
    data_root = Path(os.getenv("FPD_DATA_ROOT", "./data")).expanduser().resolve()
    window_size = int(os.getenv("FPD_WINDOW_SIZE", "30").strip())

    return Settings(
        tushare_token=token,
        tushare_http_url=http_url,
        start_date=start_date,
        end_date=end_date,
        data_root=data_root,
        window_size=window_size,
    )
