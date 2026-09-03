#!/usr/bin/env python3
"""Verify current US and Japanese market ingestion without writing repository data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noroshi.data import data_age_days, download_history, drop_incomplete_session  # noqa: E402
from noroshi.settings import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "markets.json")
    parser.add_argument("--tickers", nargs="+", default=["NVDA", "7203.T"])
    args = parser.parse_args()

    config = load_config(args.config)
    lookup = {item.ticker: item for item in config.tickers}
    failures: list[str] = []
    now = datetime.now(timezone.utc)
    for ticker in args.tickers:
        ticker_config = lookup.get(ticker)
        if ticker_config is None:
            failures.append(f"unknown ticker: {ticker}")
            continue
        try:
            frame = download_history(ticker, "2y", retries=3)
            frame = drop_incomplete_session(frame, config.markets[ticker_config.market], now)
            age = data_age_days(frame, config.markets[ticker_config.market], now)
            if age > config.max_data_age_days:
                raise RuntimeError(f"stale last bar: {frame.iloc[-1]['Date'].date()}, age={age}d")
            print(f"{ticker}: rows={len(frame)}, last={frame.iloc[-1]['Date'].date()}, age={age}d")
        except Exception as error:
            failures.append(f"{ticker}: {error}")

    if failures:
        print("Live market-data smoke failed: " + " | ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
