# scripts/generate_readme.py
# README 自動生成 #KGNINJA

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TEMPLATE = BASE / "README_template.md"
READ = BASE / "README.md"
DATA = BASE / "data/daily_predictions/latest_predictions.json"

def main():
    pred = json.loads(DATA.read_text(encoding="utf-8"))

    nvda = [
        x for x in pred["markets"]["米国市場"]
        if x["ticker"] == "NVDA"
    ][0]

    with TEMPLATE.open("r", encoding="utf-8") as f:
        text = f.read()

    rendered = text.format(
        date=pred["date"],
        nvda_price=nvda["current_price"],
        nvda_pred=nvda["predicted_price"],
        nvda_pct=nvda["predicted_change_pct"],
        trend=nvda["trend"]
    )

    rendered += "\n\n---\nGenerated automatically by NOROSHI Prediction System. #KGNINJA\n"

    READ.write_text(rendered, encoding="utf-8")
    print("[readme] Updated README.md #KGNINJA")

if __name__ == "__main__":
    main()
