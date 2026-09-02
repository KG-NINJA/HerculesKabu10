#!/usr/bin/env python3
# NOROSHI README Auto Generator #KGNINJA

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
README = BASE / "README.md"


def load_latest_predictions(data_dir=DATA):
    prediction_files = sorted(data_dir.glob("predictions_*.json"))
    if not prediction_files:
        raise FileNotFoundError("No prediction files found")

    payload = json.loads(prediction_files[-1].read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Latest prediction payload must be a list")

    return payload


def update_readme():
    payload = load_latest_predictions()
    us = [x for x in payload if not x["ticker"].endswith(".T")]
    jp = [x for x in payload if x["ticker"].endswith(".T")]
    updated = max((x.get("timestamp", "") for x in payload), default="")

    md = []
    md.append("# NOROSHI Auto Stock Prediction #KGNINJA\n")
    md.append(f"Updated: **{updated}**\n")

    md.append("## US Market\n")
    for x in us:
        details = x["prediction_details"]
        md.append(
            f"- **{x['ticker']}** → {details['predicted_change_pct']:.2f}% "
            f"({details['direction']})"
        )

    md.append("\n## Japan Market\n")
    for x in jp:
        details = x["prediction_details"]
        md.append(
            f"- **{x['ticker']}** → {details['predicted_change_pct']:.2f}% "
            f"({details['direction']})"
        )

    README.write_text("\n".join(md), encoding="utf-8")
    print("README Updated #KGNINJA")


if __name__ == "__main__":
    update_readme()
