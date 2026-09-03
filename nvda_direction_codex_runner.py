#!/usr/bin/env python3
"""Display NVDA prediction, walk-forward validation, and measured live results."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "蓄積待ち" if value is None else f"{float(value) * 100:.2f}%"


def main() -> None:
    latest = read_json(ROOT / "data" / "latest_predictions.json")
    metrics = read_json(ROOT / "data" / "metrics.json")
    prediction = next(item for item in latest["predictions"] if item["ticker"] == "NVDA")
    validation = prediction["model"]["validation"]
    live = metrics.get("live", {}).get("by_ticker", {}).get("NVDA", {})

    print("=== NVDA Prediction ===")
    print(f"Data as-of: {prediction['as_of']}")
    print(f"Direction: {prediction['direction']} ({prediction['direction_ja']})")
    print(f"Estimated return: {prediction['predicted_return_pct']:+.2f}%")
    print(f"Model confidence: {prediction['confidence'] * 100:.2f}% (not historical accuracy)")
    print(f"Research signal: {prediction['signal']}")
    print("\n=== Expanding walk-forward validation ===")
    print(f"Direction accuracy: {pct(validation['direction_accuracy'])}")
    print(f"Persistence baseline: {pct(validation['persistence_baseline_accuracy'])}")
    print(f"Skill vs baseline: {float(validation['skill_vs_persistence']) * 100:+.2f} points")
    print(f"Evaluated rows: {validation['samples']}")
    print("\n=== Live performance ===")
    print(f"Resolved predictions: {int(live.get('direction_evaluable', 0))}")
    print(f"Direction accuracy: {pct(live.get('direction_accuracy'))}")
    print(f"Baseline accuracy: {pct(live.get('baseline_direction_accuracy'))}")


if __name__ == "__main__":
    main()
