from __future__ import annotations

import json
import os
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from datetime import datetime, time as time_of_day, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .settings import MarketConfig, PipelineConfig, TickerConfig

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
OPTIONAL_COLUMNS = ("Adj Close",)


def _flatten_column(column: object) -> str:
    if isinstance(column, tuple):
        candidates = [str(part) for part in column if str(part)]
        for candidate in candidates:
            if candidate.lower() in {
                "open",
                "high",
                "low",
                "close",
                "adj close",
                "adjclose",
                "volume",
                "date",
            }:
                return candidate
        return candidates[0] if candidates else ""
    return str(column)


def normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical Date/OHLCV frame without mutating the caller's object."""
    if frame is None or frame.empty:
        raise ValueError("market history is empty")

    df = frame.copy()
    df.columns = [_flatten_column(column).strip() for column in df.columns]

    if "Date" not in df.columns:
        index_name = str(df.index.name or "Date")
        df = df.reset_index()
        first = str(df.columns[0])
        df = df.rename(columns={first: "Date", index_name: "Date"})

    aliases = {
        "date": "Date",
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }
    rename: dict[str, str] = {}
    for column in df.columns:
        key = str(column).strip().lower().replace("_", " ")
        if key in aliases:
            rename[column] = aliases[key]
        else:
            # Legacy cache columns such as CLOSE_NVDA.
            for alias_key, target in aliases.items():
                if key.startswith(f"{alias_key} ") or key.startswith(f"{alias_key}-"):
                    rename[column] = target
                    break
    df = df.rename(columns=rename)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"market history missing columns: {', '.join(missing)}")

    def normalize_session_date(value: object) -> pd.Timestamp:
        if value is None or pd.isna(value):
            return pd.NaT
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        # Daily bars represent an exchange-local calendar session. Converting a
        # Tokyo midnight to UTC would incorrectly move it to the previous date.
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp.normalize()

    df["Date"] = df["Date"].map(normalize_session_date)
    for column in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    selected = ["Date", *REQUIRED_COLUMNS]
    if "Adj Close" in df.columns:
        selected.insert(5, "Adj Close")
    df = df[selected]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Date", *REQUIRED_COLUMNS])
    df = df[df["Close"] > 0]
    df = df[df["Volume"] >= 0]
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    if df.empty:
        raise ValueError("market history has no valid rows")
    return df


def _parse_close_time(value: str) -> time_of_day:
    try:
        hour, minute = value.split(":", 1)
        return time_of_day(hour=int(hour), minute=int(minute))
    except Exception as exc:
        raise ValueError(f"invalid market close_time: {value}") from exc


def drop_incomplete_session(
    frame: pd.DataFrame,
    market: MarketConfig,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    """Exclude a same-day bar when the exchange has not completed its session."""
    df = normalize_history_frame(frame)
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(ZoneInfo(market.timezone))
    last_date = pd.Timestamp(df.iloc[-1]["Date"]).date()
    if last_date == local_now.date() and local_now.time() < _parse_close_time(market.close_time):
        df = df.iloc[:-1].copy()
    if df.empty:
        raise ValueError("no completed market sessions are available")
    return df.reset_index(drop=True)


def load_cache(path: Path) -> pd.DataFrame:
    return normalize_history_frame(pd.read_csv(path))


def save_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = normalize_history_frame(frame)
    temporary = path.with_suffix(path.suffix + ".tmp")
    canonical.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    os.replace(temporary, path)


def _frame_from_yahoo_chart(payload: dict[str, object]) -> pd.DataFrame:
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError("Yahoo chart payload is missing chart")
    if chart.get("error"):
        raise ValueError(f"Yahoo chart error: {chart.get('error')}")
    results = chart.get("result")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise ValueError("Yahoo chart payload has no result")
    result = results[0]
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise ValueError("Yahoo chart payload is incomplete")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or not quotes or not isinstance(quotes[0], dict):
        raise ValueError("Yahoo chart payload has no OHLCV quote")
    quote_data = quotes[0]
    size = len(timestamps)
    dates = pd.to_datetime(timestamps, unit="s", errors="coerce", utc=True)
    metadata = result.get("meta")
    exchange_timezone = metadata.get("exchangeTimezoneName") if isinstance(metadata, dict) else None
    if isinstance(exchange_timezone, str) and exchange_timezone:
        try:
            dates = dates.tz_convert(ZoneInfo(exchange_timezone)).tz_localize(None).normalize()
        except Exception:
            dates = dates.tz_convert(None).normalize()
    else:
        dates = dates.tz_convert(None).normalize()

    def values(key: str) -> list[object]:
        raw = quote_data.get(key)
        if not isinstance(raw, list) or len(raw) != size:
            return [np.nan] * size
        return raw

    data: dict[str, object] = {
        "Date": dates,
        "Open": values("open"),
        "High": values("high"),
        "Low": values("low"),
        "Close": values("close"),
        "Volume": values("volume"),
    }
    adjusted = indicators.get("adjclose")
    if isinstance(adjusted, list) and adjusted and isinstance(adjusted[0], dict):
        adjusted_values = adjusted[0].get("adjclose")
        if isinstance(adjusted_values, list) and len(adjusted_values) == size:
            data["Adj Close"] = adjusted_values
    return normalize_history_frame(pd.DataFrame(data))


def _download_yahoo_chart(ticker: str, period: str, timeout: int = 30) -> pd.DataFrame:
    supported = {"1y", "2y", "5y", "10y", "max", "ytd", "6mo", "3mo", "1mo"}
    requested_range = period if period in supported else "5y"
    query = urlencode(
        {
            "range": requested_range,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    errors: list[str] = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{quote(ticker, safe='')}?{query}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; NOROSHI/3.0; +https://github.com/KG-NINJA/HerculesKabu10)",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _frame_from_yahoo_chart(payload)
        except Exception as exc:  # provider/network behavior is external
            errors.append(f"{host}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def download_history(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    """Fetch current daily history with yfinance and a direct Yahoo chart fallback."""
    errors: list[str] = []
    try:
        import yfinance as yf

        try:
            yf.config.network.retries = 2
        except Exception:
            pass

        for attempt in range(1, retries + 1):
            try:
                frame = yf.Ticker(ticker).history(
                    period=period,
                    interval="1d",
                    actions=False,
                    auto_adjust=False,
                    repair=True,
                    keepna=False,
                    timeout=30,
                    raise_errors=True,
                )
                return normalize_history_frame(frame)
            except Exception as exc:  # network/provider behavior is external
                errors.append(f"yfinance attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < retries:
                    time.sleep(2 ** (attempt - 1))
    except ImportError as exc:  # direct chart fallback remains available
        errors.append(f"yfinance unavailable: {exc}")

    try:
        return _download_yahoo_chart(ticker, period)
    except Exception as exc:
        errors.append(f"direct chart fallback: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"failed to download {ticker}: " + " | ".join(errors)[-1800:])


def data_age_days(frame: pd.DataFrame, market: MarketConfig, now_utc: datetime | None = None) -> int:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_today = now.astimezone(ZoneInfo(market.timezone)).date()
    last_date = pd.Timestamp(frame.iloc[-1]["Date"]).date()
    return max(0, (local_today - last_date).days)


def refresh_market_data(
    root: Path,
    config: PipelineConfig,
    now_utc: datetime | None = None,
    downloader: Callable[[str, str], pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    """Refresh all configured tickers, using a recent cache only as a bounded fallback."""
    now = now_utc or datetime.now(timezone.utc)
    fetch = downloader or download_history
    cache_dir = root / "data" / "cache"
    frames: dict[str, pd.DataFrame] = {}
    statuses: list[dict[str, object]] = []

    for ticker_cfg in config.tickers:
        cache_path = cache_dir / f"{ticker_cfg.ticker}.csv"
        source = "download"
        error: str | None = None
        frame: pd.DataFrame | None = None
        try:
            downloaded = fetch(ticker_cfg.ticker, config.history_period)
            frame = drop_incomplete_session(
                downloaded,
                config.markets[ticker_cfg.market],
                now,
            )
            if len(frame) < config.minimum_history_rows:
                raise ValueError(
                    f"only {len(frame)} rows; need {config.minimum_history_rows}"
                )
            live_age = data_age_days(frame, config.markets[ticker_cfg.market], now)
            if live_age > config.max_data_age_days:
                raise ValueError(
                    f"downloaded data is stale: {live_age} days old "
                    f"(last session {pd.Timestamp(frame.iloc[-1]['Date']).date().isoformat()})"
                )
            # Do not overwrite a potentially fresher cache until the live frame passes
            # both row-count and freshness gates.
            save_cache(cache_path, frame)
        except Exception as exc:
            source = "cache_fallback"
            error = str(exc)[:500]
            try:
                frame = drop_incomplete_session(
                    load_cache(cache_path),
                    config.markets[ticker_cfg.market],
                    now,
                )
            except Exception as cache_exc:
                error = f"{error}; cache unavailable: {str(cache_exc)[:300]}"
                frame = None

        if frame is None:
            statuses.append(
                {
                    "ticker": ticker_cfg.ticker,
                    "market": ticker_cfg.market,
                    "status": "unavailable",
                    "source": source,
                    "rows": 0,
                    "as_of": None,
                    "age_days": None,
                    "error": error,
                }
            )
            continue

        age = data_age_days(frame, config.markets[ticker_cfg.market], now)
        fresh = age <= config.max_data_age_days and len(frame) >= config.minimum_history_rows
        status = "fresh" if fresh and source == "download" else "fallback" if fresh else "stale"
        statuses.append(
            {
                "ticker": ticker_cfg.ticker,
                "market": ticker_cfg.market,
                "status": status,
                "source": source,
                "rows": int(len(frame)),
                "as_of": pd.Timestamp(frame.iloc[-1]["Date"]).date().isoformat(),
                "age_days": int(age),
                "error": error,
            }
        )
        if fresh:
            frames[ticker_cfg.ticker] = frame

    return frames, statuses
