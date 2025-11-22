#!/usr/bin/env python3

import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]

# どちらか存在するパスを使う
PRED_FILE = None
CANDIDATES = [
    BASE / "data" / "daily_predictions" / "latest_predictions.json",
    BASE / "daily_predictions" / "latest_predictions.json",
]

for c in CANDIDATES:
    if c.exists():
        PRED_FILE = c
        break

if PRED_FILE is None:
    raise FileNotFoundError("No latest_predictions.json found in any known directory")

IMG_DIR = BASE / "analytics"
IMG_DIR.mkdir(exist_ok=True)

def generate_nvda_chart():
    with open(PRED_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    markets = payload["markets"].get("米国市場", [])
    nvda = next((x for x in markets if x["ticker"] == "NVDA"), None)

    if nvda is None:
        raise RuntimeError("NVDA prediction data not found")

    plt.figure(figsize=(6,4))
    plt.title("NVDA Predicted Change (%) #KGNINJA")
    plt.bar(["Predicted Change"], [nvda["predicted_change_pct"]])
    plt.ylabel("%")
    plt.savefig(IMG_DIR / "nvda_prediction.png", dpi=150)
    plt.close()

def main():
    generate_nvda_chart()
    print("Charts generated. #KGNINJA")

if __name__ == "__main__":
    main()
