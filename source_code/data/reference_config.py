from __future__ import annotations

REFERENCE_INDICES = [
    {"ts_code": "000001.SH", "name": "上证指数", "flag_col": "is_sse_composite"},
    {"ts_code": "000300.SH", "name": "沪深300", "flag_col": "is_hs_300"},
    {"ts_code": "000905.SH", "name": "中证500", "flag_col": "is_csi_500"},
    {"ts_code": "000852.SH", "name": "中证1000", "flag_col": "is_csi_1000"},
    {"ts_code": "932000.CSI", "name": "中证2000", "flag_col": "is_csi_2000"},
    {"ts_code": "399006.SZ", "name": "创业板指", "flag_col": "is_chinext"},
    {"ts_code": "000688.SH", "name": "科创50", "flag_col": "is_star_50"},
]

REFERENCE_INDEX_CODES = [item["ts_code"] for item in REFERENCE_INDICES]
REFERENCE_INDEX_FLAG_MAP = {item["ts_code"]: item["flag_col"] for item in REFERENCE_INDICES}

SW_INDEX_LEVEL = "L1"
SW_INDEX_SRC = "SW2021"
