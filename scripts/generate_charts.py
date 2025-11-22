# scripts/generate_charts.py
# 自動チャート生成スクリプト #KGNINJA

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED_FILE = DATA / "daily_predictions" / "latest_predictions.json"
CHART_DIR = DATA / "charts"
CHART_DIR.mkdir(exist_ok=True)

def add_tag(fig):
    fig.text(0.95, 0.02, "#KGNINJA", ha="right", fontsize=8, alpha=0.4)

def generate_nvda_chart():
    payload = json.loads(PRED_FILE.read_text(encoding="utf-8"))
    nvda = [
        x for x in payload["markets"]["米国市場"]
        if x["ticker"] == "NVDA"
    ][0]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["Current", "Predicted"], [nvda["current_price"], nvda["predicted_price"]])
    ax.set_title(f"NVDA Prediction ({nvda['trend']})")
    ax.set_ylabel("Price ($)")

    add_tag(fig)
    fig.savefig(CHART_DIR / "nvda_signal.png", dpi=150)
    plt.close()

def generate_context_chart():
    payload = json.loads(PRED_FILE.read_text(encoding="utf-8"))
    ctx = payload["market_context"]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["SPY %", "VIX %"], [ctx["spy_change_pct"], ctx["vix_change_pct"]])
    ax.set_title("Market Context (SPY / VIX)")
    ax.set_ylabel("% change")

    add_tag(fig)
    fig.savefig(CHART_DIR / "spy_vix_context.png", dpi=150)
    plt.close()

def main():
    generate_nvda_chart()
    generate_context_chart()
    print("[charts] Generated charts #KGNINJA")

if __name__ == "__main__":
    main()
