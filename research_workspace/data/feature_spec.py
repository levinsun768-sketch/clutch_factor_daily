from __future__ import annotations

DAILY_PRICE_FEATURES = [
    "open_ret_1",
    "high_ret_1",
    "low_ret_1",
    "close_ret_1",
    "intraday_ret",
    "amp_ratio",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "limit_up_gap",
    "limit_down_gap",
]

ROLLING_PRICE_FEATURES = [
    "cumret_5",
    "cumret_10",
    "cumret_20",
    "close_pos_20",
    "drawdown_20",
]

ANCHOR_PRICE_FEATURES = [
    "anchor_close_ret",
    "anchor_high_ret",
    "anchor_low_ret",
]

TRADE_FEATURES = [
    "log_vol",
    "log_amount",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "vol_ma20_ratio",
    "amount_ma20_ratio",
]

PANEL_PRICE_FEATURES = DAILY_PRICE_FEATURES + ROLLING_PRICE_FEATURES
PANEL_FEATURE_NAMES = PANEL_PRICE_FEATURES + TRADE_FEATURES
PRICE_FEATURES = PANEL_PRICE_FEATURES + ANCHOR_PRICE_FEATURES
FEATURE_NAMES = PRICE_FEATURES + TRADE_FEATURES

PRICE_IDX = list(range(len(PRICE_FEATURES)))
TRADE_IDX = list(range(len(PRICE_FEATURES), len(FEATURE_NAMES)))

AUXILIARY_COLUMNS = [
    "close_adj",
    "high_adj",
    "low_adj",
]
