#!/usr/bin/env python3
# NOROSHI Chart Generator #KGNINJA

import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "analytics"
OUT.mkdir(exist_ok=True)


def load_latest_predictions(data_dir=DATA):
    prediction_files = sorted(data_dir.glob("predictions_*.json"))
    if not prediction_files:
        raise FileNotFoundError("No prediction files found")

    payload = json.loads(prediction_files[-1].read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Latest prediction payload must be a list")

    us = [x for x in payload if not x["ticker"].endswith(".T")]
    jp = [x for x in payload if x["ticker"].endswith(".T")]
    return us, jp


def generate_chart():
    us, jp = load_latest_predictions()

    # US チャート
    tickers = [x["ticker"] for x in us]
    pct = [x["prediction_details"]["predicted_change_pct"] for x in us]

    plt.figure(figsize=(10,5))
    plt.bar(tickers, pct)
    plt.title("US Market Prediction (%) #KGNINJA")
    plt.savefig(OUT / "us_predictions.png")
    plt.close()

    # Japan チャート
    tickers = [x["ticker"] for x in jp]
    pct = [x["prediction_details"]["predicted_change_pct"] for x in jp]
    plt.figure(figsize=(10,5))
    plt.bar(tickers, pct, color="orange")
    plt.title("Japan Market Prediction (%) #KGNINJA")
    plt.savefig(OUT / "jp_predictions.png")
    plt.close()

    print("Charts generated #KGNINJA")


if __name__ == "__main__":
    generate_chart()
