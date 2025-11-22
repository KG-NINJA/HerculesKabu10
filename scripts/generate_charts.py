#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
PRED_CANDIDATES = [
    BASE / "data" / "daily_predictions" / "latest_predictions.json",
    BASE / "daily_predictions" / "latest_predictions.json",
]

PRED_FILE = next((p for p in PRED_CANDIDATES if p.exists()), None)
if PRED_FILE is None:
    raise FileNotFoundError("No latest_predictions.json found in any known directory")

ANALYTICS_DIR = BASE / "analytics"
DOCS_ASSETS = BASE / "docs" / "assets"
for folder in (ANALYTICS_DIR, DOCS_ASSETS):
    folder.mkdir(parents=True, exist_ok=True)


def _save_chart(fig_path: Path):
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


def load_nvda(data: dict) -> dict:
    markets = data["markets"].get("米国市場", [])
    nvda = next((x for x in markets if x["ticker"] == "NVDA"), None)
    if nvda is None:
        raise RuntimeError("NVDA prediction data not found")
    return nvda


def generate_nvda_chart(data: dict):
    nvda = load_nvda(data)
    chart_path = ANALYTICS_DIR / "nvda_prediction.png"
    plt.figure(figsize=(6, 4))
    plt.title("NVDA Predicted vs Current #KGNINJA")
    plt.bar(["Current", "Predicted"], [nvda["current_price"], nvda["predicted_price"]])
    plt.ylabel("USD")
    _save_chart(chart_path)
    mirror = DOCS_ASSETS / chart_path.name
    mirror.write_bytes(chart_path.read_bytes())
    return chart_path


def generate_market_context_chart(data: dict):
    ctx = data.get("market_context", {})
    chart_path = ANALYTICS_DIR / "market_context.png"
    plt.figure(figsize=(6, 4))
    plt.title("SPY / VIX Snapshot #KGNINJA")
    plt.bar(["SPY Close", "VIX Close"], [ctx.get("spy_close", 0), ctx.get("vix_close", 0)])
    plt.ylabel("Value")
    _save_chart(chart_path)
    mirror = DOCS_ASSETS / chart_path.name
    mirror.write_bytes(chart_path.read_bytes())
    return chart_path


def main():
    with open(PRED_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    generate_nvda_chart(payload)
    generate_market_context_chart(payload)

    print("Charts generated. #KGNINJA")


if __name__ == "__main__":
    main()
