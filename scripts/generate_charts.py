#!/usr/bin/env python3
# Generate charts for NOROSHI Dashboard #KGNINJA

import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]

# 修正ポイント（新パスに変更）
PRED_FILE = BASE / "data" / "daily_predictions" / "latest_predictions.json"
OUT_DIR = BASE / "analytics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_nvda_chart():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_FILE}")

    payload = json.loads(PRED_FILE.read_text(encoding="utf-8"))

    nvda = None
    for item in payload["markets"]["米国市場"]:
        if item["ticker"] == "NVDA":
            nvda = item
            break

    if not nvda:
        raise ValueError("NVDA prediction not found in JSON")

    current = nvda["current_price"]
    predicted = nvda["predicted_price"]

    plt.figure(figsize=(6, 4))
    plt.bar(["Current", "Predicted"], [current, predicted])
    plt.title("NVDA Prediction #KGNINJA")
    plt.ylabel("Price (USD)")

    plt.savefig(OUT_DIR / "nvda_prediction.png")
    plt.close()


def main():
    generate_nvda_chart()
    print("Charts generated. #KGNINJA")

if __name__ == "__main__":
    main()
