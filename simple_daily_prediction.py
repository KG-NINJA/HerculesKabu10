#!/usr/bin/env python3
# LightGBM NOROSHI Daily Prediction #KGNINJA

import json
from datetime import datetime
from pathlib import Path
import yfinance as yf
import numpy as np
import pandas as pd
import lightgbm as lgb


BASE = Path(__file__).resolve().parents[0]
OUT_DIR = BASE / "data" / "daily_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# Feature Engineering
# ======================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()

    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()

    df["price_change_1d"] = df["Close"].diff()
    df["price_change_pct_1d"] = df["Close"].pct_change() * 100

    df = df.dropna()
    df.columns = [c.replace("(", "_").replace(")", "_").replace("%","pct") for c in df.columns]
    return df


# ======================================================
# LightGBM training
# ======================================================

def train_lgbm(features, target):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_pre_filter": False,
    }
    train_data = lgb.Dataset(features, label=target)
    model = lgb.train(params, train_data, num_boost_round=300)
    return model


# ======================================================
# Prediction Pipeline
# ======================================================

def get_prediction(ticker: str):
    df = yf.download(ticker, period="180d", progress=False)
    df = create_features(df)

    target = df["Close"]
    feat = df.drop(columns=["Close"])

    model = train_lgbm(feat.iloc[:-1], target.iloc[:-1])
    pred = model.predict(feat.iloc[-1:], num_iteration=model.best_iteration)[0]

    current = float(target.iloc[-1])
    pct = (pred - current) / current * 100
    trend = "強気" if pct > 0.5 else "弱気" if pct < -0.5 else "横ばい"

    return {
        "ticker": ticker,
        "current_price": current,
        "predicted_price": float(pred),
        "predicted_change": float(pred - current),
        "predicted_change_pct": pct,
        "trend": trend,
        "features": feat.iloc[-1].to_dict(),
        "prediction_method": "lightgbm"
    }


# ======================================================
# Main
# ======================================================

def main():
    tickers_us = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    tickers_jp = ["7203.T", "6758.T", "9984.T", "6861.T", "8035.T"]

    preds_us = [get_prediction(t) for t in tickers_us]
    preds_jp = [get_prediction(t) for t in tickers_jp]

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "markets": {
            "米国市場": preds_us,
            "日本市場": preds_jp
        }
    }

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    (OUT_DIR / "latest_predictions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("LightGBM prediction done #KGNINJA")


if __name__ == "__main__":
    main()
