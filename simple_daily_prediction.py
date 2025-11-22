#!/usr/bin/env python3
# Daily prediction (GitHub Actions 用) #KGNINJA

import json
from datetime import datetime
from pathlib import Path

import yfinance as yf

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "data" / "daily_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_prediction(ticker: str) -> dict:
    data = yf.download(ticker, period="60d")
    close = data["Close"]

    current = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    pred = current + (current - ma20) * 0.3
    pct = (pred - current) / current * 100

    trend = "強気" if pct > 0.5 else "弱気" if pct < -0.5 else "横ばい"

    return {
        "ticker": ticker,
        "current_price": current,
        "predicted_price": pred,
        "predicted_change_pct": pct,
        "trend": trend,
        "ma20": ma20,
        "prediction_method": "technical_analysis",
    }


def main():
    tickers_us = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    tickers_jp = ["7203.T", "6758.T", "9984.T", "6861.T", "8035.T"]

    preds_us = [get_prediction(t) for t in tickers_us]
    preds_jp = [get_prediction(t) for t in tickers_jp]

    now = datetime.now()
    result = {
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "markets": {
            "米国市場": preds_us,
            "日本市場": preds_jp,
        },
        "market_context": {
            "retrieved_at": now.isoformat(),
            "spy_close": float(yf.download("SPY", period="5d")["Close"].iloc[-1]),
            "vix_close": float(yf.download("^VIX", period="5d")["Close"].iloc[-1]),
        },
    }

    output_path = OUT_DIR / "latest_predictions.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Prediction saved to {output_path.as_posix()}. #KGNINJA")


if __name__ == "__main__":
    main()
