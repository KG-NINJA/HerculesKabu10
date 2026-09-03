from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
MODEL_VERSION = "noroshi-3.0.0"


@dataclass(frozen=True)
class MarketConfig:
    code: str
    timezone: str
    close_time: str


@dataclass(frozen=True)
class TickerConfig:
    ticker: str
    name: str
    market: str
    currency: str


@dataclass(frozen=True)
class PipelineConfig:
    history_period: str
    max_data_age_days: int
    minimum_history_rows: int
    min_valid_tickers: int
    backtest_splits: int
    high_confidence_threshold: float
    buy_probability_threshold: float
    sell_probability_threshold: float
    markets: dict[str, MarketConfig]
    tickers: tuple[TickerConfig, ...]


def _require(mapping: dict[str, Any], key: str, expected: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected):
        raise ValueError(f"config.{key} must be {expected.__name__}")
    return value


def load_config(path: Path | str) -> PipelineConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")

    raw_markets = _require(raw, "markets", dict)
    markets: dict[str, MarketConfig] = {}
    for code, value in raw_markets.items():
        if not isinstance(code, str) or not isinstance(value, dict):
            raise ValueError("markets entries must be objects")
        markets[code] = MarketConfig(
            code=code,
            timezone=str(value.get("timezone", "UTC")),
            close_time=str(value.get("close_time", "23:59")),
        )

    raw_tickers = _require(raw, "tickers", list)
    tickers: list[TickerConfig] = []
    seen: set[str] = set()
    for item in raw_tickers:
        if not isinstance(item, dict):
            raise ValueError("ticker entries must be objects")
        ticker = str(item.get("ticker", "")).strip().upper()
        market = str(item.get("market", "")).strip().upper()
        if not ticker:
            raise ValueError("ticker must not be empty")
        if ticker in seen:
            raise ValueError(f"duplicate ticker: {ticker}")
        if market not in markets:
            raise ValueError(f"unknown market {market} for {ticker}")
        seen.add(ticker)
        tickers.append(
            TickerConfig(
                ticker=ticker,
                name=str(item.get("name", ticker)).strip() or ticker,
                market=market,
                currency=str(item.get("currency", "")).strip(),
            )
        )

    if not tickers:
        raise ValueError("at least one ticker is required")

    buy_threshold = float(raw.get("buy_probability_threshold", 0.60))
    sell_threshold = float(raw.get("sell_probability_threshold", 0.40))
    if not (0.5 < buy_threshold < 1.0):
        raise ValueError("buy_probability_threshold must be between 0.5 and 1.0")
    if not (0.0 < sell_threshold < 0.5):
        raise ValueError("sell_probability_threshold must be between 0.0 and 0.5")

    return PipelineConfig(
        history_period=str(raw.get("history_period", "5y")),
        max_data_age_days=int(raw.get("max_data_age_days", 7)),
        minimum_history_rows=int(raw.get("minimum_history_rows", 260)),
        min_valid_tickers=int(raw.get("min_valid_tickers", max(1, len(tickers) - 2))),
        backtest_splits=int(raw.get("backtest_splits", 5)),
        high_confidence_threshold=float(raw.get("high_confidence_threshold", 0.65)),
        buy_probability_threshold=buy_threshold,
        sell_probability_threshold=sell_threshold,
        markets=markets,
        tickers=tuple(tickers),
    )
