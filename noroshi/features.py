from __future__ import annotations

import numpy as np
import pandas as pd

from .data import normalize_history_frame

FEATURE_COLUMNS = (
    "RETURN_1D",
    "RETURN_2D",
    "RETURN_5D",
    "RETURN_10D",
    "RETURN_20D",
    "PRICE_TO_MA5",
    "PRICE_TO_MA10",
    "PRICE_TO_MA20",
    "PRICE_TO_MA50",
    "MA5_MA20_RATIO",
    "MA10_MA50_RATIO",
    "EMA5_EMA20_RATIO",
    "VOLATILITY_5D",
    "VOLATILITY_10D",
    "VOLATILITY_20D",
    "RSI7",
    "RSI14",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "BB_WIDTH_PCT",
    "BB_POSITION",
    "ATR14_PCT",
    "VOLUME_RATIO_5D",
    "VOLUME_RATIO_20D",
    "INTRADAY_RETURN",
    "OVERNIGHT_GAP",
    "RANGE_PCT",
    "CLOSE_TO_20D_HIGH",
    "CLOSE_TO_20D_LOW",
    "DAY_OF_WEEK_SIN",
    "DAY_OF_WEEK_COS",
)


def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((gain > 0) & (loss == 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
    return rsi.fillna(50.0)


def create_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Build causal features and next-session targets.

    All feature columns use only the current or earlier rows. The final target
    remains NaN, preventing the former bug that silently labelled it as DOWN.
    """
    df = normalize_history_frame(history)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_price = df["Open"].astype(float)
    volume = df["Volume"].astype(float)

    for period in (1, 2, 5, 10, 20):
        df[f"RETURN_{period}D"] = close.pct_change(period)

    for period in (5, 10, 20, 50):
        df[f"MA{period}"] = close.rolling(period, min_periods=period).mean()
        df[f"EMA{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()

    df["PRICE_TO_MA5"] = close / df["MA5"] - 1
    df["PRICE_TO_MA10"] = close / df["MA10"] - 1
    df["PRICE_TO_MA20"] = close / df["MA20"] - 1
    df["PRICE_TO_MA50"] = close / df["MA50"] - 1
    df["MA5_MA20_RATIO"] = df["MA5"] / df["MA20"] - 1
    df["MA10_MA50_RATIO"] = df["MA10"] / df["MA50"] - 1
    df["EMA5_EMA20_RATIO"] = df["EMA5"] / df["EMA20"] - 1

    one_day_return = close.pct_change()
    for period in (5, 10, 20):
        df[f"VOLATILITY_{period}D"] = one_day_return.rolling(
            period, min_periods=period
        ).std()

    df["RSI7"] = calculate_rsi(close, 7) / 100.0
    df["RSI14"] = calculate_rsi(close, 14) / 100.0

    ema12 = close.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_raw = ema12 - ema26
    macd_signal_raw = macd_raw.ewm(span=9, adjust=False, min_periods=9).mean()
    price_scale = close.replace(0, np.nan)
    df["MACD"] = macd_raw / price_scale
    df["MACD_SIGNAL"] = macd_signal_raw / price_scale
    df["MACD_HIST"] = (macd_raw - macd_signal_raw) / price_scale

    ma20 = df["MA20"]
    std20 = close.rolling(20, min_periods=20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    band_width = (upper - lower).replace(0, np.nan)
    df["BB_WIDTH_PCT"] = band_width / price_scale
    df["BB_POSITION"] = (close - lower) / band_width

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR14_PCT"] = true_range.rolling(14, min_periods=14).mean() / price_scale

    df["VOLUME_RATIO_5D"] = volume / volume.rolling(5, min_periods=5).mean() - 1
    df["VOLUME_RATIO_20D"] = volume / volume.rolling(20, min_periods=20).mean() - 1
    df["INTRADAY_RETURN"] = close / open_price.replace(0, np.nan) - 1
    df["OVERNIGHT_GAP"] = open_price / previous_close.replace(0, np.nan) - 1
    df["RANGE_PCT"] = (high - low) / price_scale

    rolling_high = close.rolling(20, min_periods=20).max()
    rolling_low = close.rolling(20, min_periods=20).min()
    df["CLOSE_TO_20D_HIGH"] = close / rolling_high.replace(0, np.nan) - 1
    df["CLOSE_TO_20D_LOW"] = close / rolling_low.replace(0, np.nan) - 1

    day_of_week = df["Date"].dt.dayofweek.astype(float)
    df["DAY_OF_WEEK_SIN"] = np.sin(2 * np.pi * day_of_week / 5.0)
    df["DAY_OF_WEEK_COS"] = np.cos(2 * np.pi * day_of_week / 5.0)

    next_return = close.shift(-1) / close - 1
    df["TARGET_RETURN"] = next_return
    df["TARGET_UP"] = np.where(
        next_return.notna(),
        (next_return > 0).astype(float),
        np.nan,
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df
