#!/usr/bin/env python3
# Daily prediction (GitHub Actions 用) #KGNINJA

import json
from datetime import datetime
from pathlib import Path

from typing import Dict, Optional

import pandas as pd
import yfinance as yf
from lightgbm import LGBMRegressor


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "data" / "daily_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"] = df["Close"].rolling(window=5).mean()
    df["ma20"] = df["Close"].rolling(window=20).mean()
    df["ma50"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    return df


def fit_lightgbm(features: pd.DataFrame, target: pd.Series) -> Optional[LGBMRegressor]:
    if len(features) < 60:
        return None

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    model.fit(features, target)
    return model


def determine_trend(pct: float) -> str:
    return "強気" if pct > 0.5 else "弱気" if pct < -0.5 else "横ばい"


def technical_fallback(close: pd.Series) -> Dict[str, float]:
    if close.empty:
        return {
            "predicted_price": 0.0,
            "predicted_change_pct": 0.0,
            "ma20": 0.0,
            "prediction_method": "technical_analysis",
            "features": {},
        }


    current = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    pred = current + (current - ma20) * 0.3
    pct = (pred - current) / current * 100
    return {
        "predicted_price": pred,
        "predicted_change_pct": pct,
        "ma20": ma20,
        "prediction_method": "technical_analysis",
        "features": {},
    }


def get_prediction(ticker: str) -> dict:
    data = yf.download(ticker, period="180d")
    if data.empty or len(data) < 60:
        close = data["Close"] if not data.empty else pd.Series(dtype=float)
        fallback = technical_fallback(close)
        current_price = float(close.iloc[-1]) if not close.empty else 0.0
        trend = determine_trend(fallback.get("predicted_change_pct", 0.0))
        fallback.update({"ticker": ticker, "current_price": current_price, "trend": trend})
        return fallback

    df = compute_indicators(data)
    df = df.dropna()

    if df.empty or len(df) < 60:
        close = data["Close"]
        current = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        fallback = technical_fallback(close)
        trend = determine_trend(fallback.get("predicted_change_pct", 0.0))
        fallback.update({"ticker": ticker, "current_price": current, "trend": trend})
        fallback["ma20"] = ma20
        return fallback

    df["target_change_pct"] = df["Close"].pct_change().shift(-1) * 100
    df = df.dropna()

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ma5",
        "ma20",
        "ma50",
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
    ]

    features = df[feature_cols]
    target = df["target_change_pct"]

    model = fit_lightgbm(features.iloc[:-1], target.iloc[:-1])
    close = df["Close"]
    current = float(close.iloc[-1])
    latest_features = features.iloc[[-1]]

    if model is None:
        fallback = technical_fallback(close)
        trend = determine_trend(fallback.get("predicted_change_pct", 0.0))
        fallback.update({
            "ticker": ticker,
            "current_price": current,
            "trend": trend,
        })
        return fallback

    predicted_change_pct = float(model.predict(latest_features)[0])
    predicted_price = current * (1 + predicted_change_pct / 100)
    trend = "強気" if predicted_change_pct > 0.5 else "弱気" if predicted_change_pct < -0.5 else "横ばい"

    latest_feature_values = {col: float(latest_features.iloc[0][col]) for col in feature_cols}

    return {
        "ticker": ticker,
        "current_price": current,
        "predicted_price": predicted_price,
        "predicted_change_pct": predicted_change_pct,
        "trend": trend,
        "ma20": float(latest_feature_values["ma20"]),
        "prediction_method": "lightgbm",
        "features": latest_feature_values,
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
