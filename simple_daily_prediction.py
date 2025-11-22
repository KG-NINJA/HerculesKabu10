#!/usr/bin/env python3
# NOROSHI LightGBM Daily Prediction Engine #KGNINJA

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import numpy as np
from prediction_data_manager import PredictionDataManager

BASE = Path(__file__).resolve().parents[0]
DATA_DIR = BASE / "data" / "daily_predictions"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# MultiIndex-safe feature builder #KGNINJA
# ======================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- MultiIndex カラム名を完全フラット化 ---
    df.columns = [
        "_".join([str(c) for c in col]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]

    # --- 主要テクニカル指標 ---
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df["price_change_1d"] = df["Close"].diff()
    df["price_change_pct_1d"] = df["Close"].pct_change() * 100

    df = df.dropna()

    # LightGBM が嫌う記号を排除
    df.columns = [
        c.replace("(", "_").replace(")", "_").replace("%", "pct")
        for c in df.columns
    ]

    return df


# ======================================================
# LightGBM モデル #KGNINJA
# ======================================================
def fit_lightgbm(train_X, train_y):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(train_X, train_y)
    return model


# ======================================================
# 単一ティッカー予測 #KGNINJA
# ======================================================
def predict_ticker(ticker: str):
    df = yf.download(ticker, period="180d", progress=False)

    if df is None or len(df) < 60:
        return {"ticker": ticker, "error": "NO_DATA"}

    df = create_features(df)

    # 入力特徴量
    feature_cols = [c for c in df.columns if c not in ["Close"]]
    X = df[feature_cols]
    y = df["Close"]

    # LightGBM モデル学習
    model = fit_lightgbm(X.iloc[:-1], y.iloc[:-1])

    # 直近値の予測
    latest_features = X.iloc[-1:].values
    pred = float(model.predict(latest_features)[0])
    current = float(y.iloc[-1])
    pct = (pred - current) / current * 100

    trend = "強気" if pct > 0.5 else "弱気" if pct < -0.5 else "横ばい"

    return {
        "ticker": ticker,
        "current_price": current,
        "predicted_price": pred,
        "predicted_change": pred - current,
        "predicted_change_pct": pct,
        "trend": trend,
        "prediction_method": "lightgbm",
        "generated_at": datetime.now().isoformat(),
    }


# ======================================================
# メイン処理 #KGNINJA
# ======================================================
def main():
    tickers_us = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    tickers_jp = ["7203.T", "6758.T", "9984.T", "6861.T", "8035.T"]

    preds_us = [predict_ticker(t) for t in tickers_us]
    preds_jp = [predict_ticker(t) for t in tickers_jp]

    spy = yf.download("SPY", period="5d")["Close"].iloc[-1]
    vix = yf.download("^VIX", period="5d")["Close"].iloc[-1]

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "markets": {
            "US": preds_us,
            "JP": preds_jp,
        },
        "market_context": {
            "retrieved_at": datetime.now().isoformat(),
            "spy_close": float(spy),
            "vix_close": float(vix),
        },
        "engine": "NOROSHI LightGBM v1 #KGNINJA"
    }

    PredictionDataManager.save_latest(result)
    print("LightGBM Prediction Done #KGNINJA")


if __name__ == "__main__":
    main()
