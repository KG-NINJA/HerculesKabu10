#!/usr/bin/env python3
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PRED = BASE / "data" / "daily_predictions" / "latest_predictions.json"
README = BASE / "README.md"

def generate():
    data = json.loads(PRED.read_text(encoding="utf-8"))

    md = "# NOROSHI Prediction Dashboard #KGNINJA\n\n"
    md += f"**更新日時：{data['timestamp']}**\n\n"

    for market, items in data["markets"].items():
        md += f"## {market}\n\n"
        for d in items:
            md += f"- **{d['ticker']}** → 予測: {d['predicted_price']:.2f} ({d['trend']})\n"

    README.write_text(md, encoding="utf-8")
    print("README updated #KGNINJA")

if __name__ == "__main__":
    generate()
