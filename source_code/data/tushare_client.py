from __future__ import annotations

import tushare as ts

from data.config import get_settings


def build_pro_client():
    settings = get_settings()
    pro = ts.pro_api(settings.tushare_token)
    pro._DataApi__token = settings.tushare_token
    pro._DataApi__http_url = settings.tushare_http_url
    return pro
