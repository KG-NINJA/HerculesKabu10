#!/usr/bin/env python3
# README 自動生成 #KGNINJA

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TEMPLATE = BASE / "README_template.md"
README = BASE / "README.md"
PRED_CANDIDATES = [
    BASE / "data" / "daily_predictions" / "latest_predictions.json",
    BASE / "daily_predictions" / "latest_predictions.json",
]
PRED_FILE = next((p for p in PRED_CANDIDATES if p.exists()), None)
DOCS_INDEX = BASE / "docs" / "index.html"

if PRED_FILE is None:
    raise FileNotFoundError("latest_predictions.json not found")


def load_prediction():
    return json.loads(PRED_FILE.read_text(encoding="utf-8"))


def render_readme(pred: dict):
    nvda = [x for x in pred["markets"]["米国市場"] if x["ticker"] == "NVDA"][0]
    with TEMPLATE.open("r", encoding="utf-8") as f:
        text = f.read()

    rendered = text.format(
        date=pred["date"],
        nvda_price=nvda["current_price"],
        nvda_pred=nvda["predicted_price"],
        nvda_pct=round(nvda["predicted_change_pct"], 2),
        trend=f"{nvda['trend']} ({nvda.get('prediction_method', 'n/a')})",
    )

    rendered += "\n\n---\nGenerated automatically by NOROSHI Prediction System. #KGNINJA\n"
    README.write_text(rendered, encoding="utf-8")
    print("[readme] Updated README.md #KGNINJA")


def render_docs(pred: dict):
    nvda = [x for x in pred["markets"]["米国市場"] if x["ticker"] == "NVDA"][0]
    market_ctx = pred.get("market_context", {})
    method = nvda.get("prediction_method", "n/a")
    features = nvda.get("features", {})

    html = f"""<!DOCTYPE html>
<html lang=\"ja\">
<head>
<meta charset=\"UTF-8\" />
<title>NOROSHI Dashboard</title>
<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 24px; }}
section {{ margin-bottom: 32px; }}
.card {{ border: 1px solid #ccc; padding: 16px; border-radius: 8px; max-width: 720px; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>📈 NOROSHI Prediction Dashboard</h1>
<p>最終更新: {pred['timestamp']} (自動生成 #KGNINJA)</p>
<section class=\"card\">
  <h2>NVDA 予測スナップショット</h2>
  <ul>
    <li>モデル: {method}</li>
    <li>現在価格: ${nvda['current_price']}</li>
    <li>予測価格: ${nvda['predicted_price']}</li>
    <li>予測変動率: {round(nvda['predicted_change_pct'], 2)}%</li>
    <li>トレンド: {nvda['trend']}</li>
  </ul>
</section>
<section>
  <h2>チャート</h2>
  <h3>NVDA 予測</h3>
  <img src=\"assets/nvda_prediction.png\" alt=\"NVDA Prediction\" />
  <h3>特徴量スナップショット</h3>
  <img src=\"assets/nvda_feature_snapshot.png\" alt=\"NVDA Feature Snapshot\" />
  <h3>市場スナップショット (SPY / VIX)</h3>
  <img src=\"assets/market_context.png\" alt=\"Market Context\" />
</section>
<section class=\"card\">
  <h2>市場コンテキスト</h2>
  <ul>
    <li>SPY 終値: {market_ctx.get('spy_close')}</li>
    <li>VIX 終値: {market_ctx.get('vix_close')}</li>
    <li>NVDA ma20: {features.get('ma20')}</li>
    <li>NVDA RSI14: {features.get('rsi14')}</li>
  </ul>
</section>
</body>
</html>
"""

    DOCS_INDEX.write_text(html, encoding="utf-8")
    print("[docs] Updated docs/index.html #KGNINJA")


def main():
    pred = load_prediction()
    render_readme(pred)
    render_docs(pred)


if __name__ == "__main__":
    main()
