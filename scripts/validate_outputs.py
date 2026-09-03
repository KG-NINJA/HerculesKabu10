#!/usr/bin/env python3
"""Validate NOROSHI output structure, freshness, and operational health."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    require(path.exists(), f"missing output: {path}")
    require(path.stat().st_size > 0, f"empty output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path, allow_unhealthy: bool) -> None:
    latest = read_json(root / "data" / "latest_predictions.json")
    metrics = read_json(root / "data" / "metrics.json")
    status = read_json(root / "data" / "status.json")
    docs_index = root / "docs" / "index.html"
    require(docs_index.exists() and docs_index.stat().st_size > 0, "dashboard is missing")

    require(latest.get("schema_version") == 2, "latest schema_version must be 2")
    require(metrics.get("schema_version") == 2, "metrics schema_version must be 2")
    require(status.get("schema_version") == 2, "status schema_version must be 2")
    require(status.get("health") in {"healthy", "degraded", "unhealthy"}, "invalid health")

    predictions = latest.get("predictions", [])
    require(isinstance(predictions, list), "predictions must be an array")
    if status.get("health") != "unhealthy":
        require(len(predictions) >= 8, "healthy/degraded output requires at least 8 predictions")
        require(len({item.get("prediction_id") for item in predictions}) == len(predictions), "duplicate prediction IDs")
        today = date.today()
        for item in predictions:
            as_of = date.fromisoformat(str(item["as_of"]))
            require(as_of <= today, f"{item['ticker']}: future as_of date")
            require(item.get("direction") in {"UP", "DOWN"}, f"{item['ticker']}: invalid direction")
            confidence = float(item.get("confidence", -1))
            require(0.5 <= confidence <= 1.0, f"{item['ticker']}: invalid confidence")
            validation = item.get("model", {}).get("validation", {})
            for key in (
                "direction_accuracy",
                "persistence_baseline_accuracy",
                "balanced_accuracy",
                "brier_score",
            ):
                value = float(validation[key])
                require(0.0 <= value <= 1.0, f"{item['ticker']}: invalid {key}")

    html = docs_index.read_text(encoding="utf-8")
    require("Confidenceは過去の的中率ではありません" in html, "dashboard confidence warning missing")
    require("データ鮮度" in html, "dashboard freshness table missing")
    require("Live実績" in html, "dashboard live metrics missing")

    if status.get("health") == "unhealthy" and not allow_unhealthy:
        raise ValueError(f"pipeline health is unhealthy: {status.get('message')}")
    print(f"NOROSHI output validation passed (health={status.get('health')})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-unhealthy", action="store_true")
    args = parser.parse_args()
    try:
        validate(args.root.resolve(), args.allow_unhealthy)
    except Exception as error:
        print(f"Output validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
