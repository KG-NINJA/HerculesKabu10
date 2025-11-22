#!/usr/bin/env python3

import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
PRED = BASE / "data" / "daily_predictions" / "latest_predictions.json"
OUT = BASE / "analytics"
OUT.mkdir(exist_ok=True, parents=True)

def generate():
    data = json.loads(PRED.read_text(encoding="utf-8"))

    for market, items in data["markets"].items():
        for item in items:
            ticker = item["ticker"]
            preds = item["predicted_price"]
            current = item["current_price"]

            plt.figure(figsize=(6,4))
            plt.title(f"{ticker} Prediction #KGNINJA")
            plt.bar(["Current","Predicted"], [current, preds])
            plt.ylabel("Price")

            plt.savefig(OUT / f"{ticker}.png", bbox_inches="tight")
            plt.close()

    print("Charts generated #KGNINJA")

if __name__ == "__main__":
    generate()
