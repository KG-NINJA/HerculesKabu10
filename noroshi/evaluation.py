from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .settings import PipelineConfig, SCHEMA_VERSION


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _archive_payloads(prediction_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(prediction_dir.glob("*.json")):
        payload = _read_json(path, None)
        if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
            payloads.append(payload)
    payloads.sort(key=lambda item: str(item.get("generated_at", "")))
    return payloads


def evaluate_prediction_archives(
    root: Path,
    cache_frames: dict[str, pd.DataFrame],
    config: PipelineConfig,
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    """Score immutable v2 predictions once the next completed session exists."""
    now = updated_at or datetime.now(timezone.utc)
    path = root / "data" / "evaluations" / "live_results.json"
    existing = _read_json(path, {"results": []})
    results = existing.get("results", []) if isinstance(existing, dict) else []
    if not isinstance(results, list):
        results = []

    by_id = {
        str(item.get("prediction_id")): item
        for item in results
        if isinstance(item, dict) and item.get("prediction_id")
    }

    for archive in _archive_payloads(root / "data" / "predictions_v2"):
        for prediction in archive.get("predictions", []):
            if not isinstance(prediction, dict):
                continue
            prediction_id = str(prediction.get("prediction_id", ""))
            ticker = str(prediction.get("ticker", ""))
            if not prediction_id or prediction_id in by_id or ticker not in cache_frames:
                continue

            as_of = pd.to_datetime(prediction.get("as_of"), errors="coerce")
            if pd.isna(as_of):
                continue
            frame = cache_frames[ticker]
            future = frame[frame["Date"] > as_of]
            if future.empty:
                continue

            actual_row = future.iloc[0]
            current_price = float(prediction.get("current_price", 0.0))
            actual_close = float(actual_row["Close"])
            if current_price <= 0 or actual_close <= 0:
                continue

            actual_return = actual_close / current_price - 1.0
            if actual_return > 0:
                actual_direction = "UP"
            elif actual_return < 0:
                actual_direction = "DOWN"
            else:
                actual_direction = "FLAT"

            predicted_direction = str(prediction.get("direction", ""))
            direction_evaluable = actual_direction != "FLAT" and predicted_direction in {"UP", "DOWN"}
            direction_correct = (
                predicted_direction == actual_direction if direction_evaluable else None
            )
            probability_up = float(prediction.get("probability_up", 0.5))
            predicted_return_pct = float(prediction.get("predicted_return_pct", 0.0))
            predicted_price = float(prediction.get("predicted_price", current_price))
            confidence = float(prediction.get("confidence", max(probability_up, 1 - probability_up)))
            baseline_direction = str(prediction.get("baseline_direction", ""))
            baseline_evaluable = actual_direction != "FLAT" and baseline_direction in {"UP", "DOWN"}
            baseline_correct = (
                baseline_direction == actual_direction if baseline_evaluable else None
            )

            record = {
                "prediction_id": prediction_id,
                "ticker": ticker,
                "market": prediction.get("market"),
                "model_version": prediction.get("model", {}).get("version"),
                "generated_at": prediction.get("generated_at"),
                "as_of": pd.Timestamp(as_of).date().isoformat(),
                "target_date": pd.Timestamp(actual_row["Date"]).date().isoformat(),
                "current_price": current_price,
                "predicted_price": predicted_price,
                "actual_close": actual_close,
                "predicted_return_pct": predicted_return_pct,
                "actual_return_pct": float(actual_return * 100),
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "direction_evaluable": direction_evaluable,
                "direction_correct": direction_correct,
                "baseline_direction": baseline_direction,
                "baseline_evaluable": baseline_evaluable,
                "baseline_correct": baseline_correct,
                "probability_up": probability_up,
                "confidence": confidence,
                "high_confidence": confidence >= config.high_confidence_threshold,
                "return_absolute_error_pct": abs(predicted_return_pct - actual_return * 100),
                "price_absolute_error_pct": abs(predicted_price - actual_close) / actual_close * 100,
                "brier_score": (
                    (probability_up - (1.0 if actual_direction == "UP" else 0.0)) ** 2
                    if direction_evaluable
                    else None
                ),
                "evaluated_at": now.isoformat().replace("+00:00", "Z"),
            }
            by_id[prediction_id] = record

    ordered = sorted(
        by_id.values(),
        key=lambda item: (str(item.get("target_date", "")), str(item.get("ticker", ""))),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now.isoformat().replace("+00:00", "Z"),
        "results": ordered,
    }
    _write_json_atomic(path, payload)
    return payload


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "evaluated_predictions": 0,
            "direction_evaluable": 0,
            "direction_accuracy": None,
            "baseline_direction_accuracy": None,
            "skill_vs_baseline": None,
            "return_mae_pct": None,
            "price_mae_pct": None,
            "brier_score": None,
            "high_confidence_predictions": 0,
            "high_confidence_accuracy": None,
        }

    evaluable = [item for item in records if item.get("direction_evaluable")]
    baseline_evaluable = [item for item in records if item.get("baseline_evaluable")]
    high = [item for item in evaluable if item.get("high_confidence")]
    direction_accuracy = (
        float(np.mean([bool(item["direction_correct"]) for item in evaluable]))
        if evaluable
        else None
    )
    baseline_accuracy = (
        float(np.mean([bool(item["baseline_correct"]) for item in baseline_evaluable]))
        if baseline_evaluable
        else None
    )

    return {
        "evaluated_predictions": len(records),
        "direction_evaluable": len(evaluable),
        "direction_accuracy": direction_accuracy,
        "baseline_direction_accuracy": baseline_accuracy,
        "skill_vs_baseline": (
            direction_accuracy - baseline_accuracy
            if direction_accuracy is not None and baseline_accuracy is not None
            else None
        ),
        "return_mae_pct": float(
            np.mean([float(item["return_absolute_error_pct"]) for item in records])
        ),
        "price_mae_pct": float(
            np.mean([float(item["price_absolute_error_pct"]) for item in records])
        ),
        "brier_score": (
            float(np.mean([float(item["brier_score"]) for item in evaluable]))
            if evaluable
            else None
        ),
        "high_confidence_predictions": len(high),
        "high_confidence_accuracy": (
            float(np.mean([bool(item["direction_correct"]) for item in high]))
            if high
            else None
        ),
    }


def summarize_live_metrics(
    evaluation_payload: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    records = [
        item
        for item in evaluation_payload.get("results", [])
        if isinstance(item, dict) and item.get("target_date")
    ]

    def within(days: int) -> list[dict[str, Any]]:
        cutoff = current.date() - timedelta(days=days)
        selected: list[dict[str, Any]] = []
        for item in records:
            target = pd.to_datetime(item.get("target_date"), errors="coerce")
            if not pd.isna(target) and pd.Timestamp(target).date() >= cutoff:
                selected.append(item)
        return selected

    tickers = sorted({str(item.get("ticker")) for item in records if item.get("ticker")})
    by_ticker = {
        ticker: _summary([item for item in records if item.get("ticker") == ticker])
        for ticker in tickers
    }

    return {
        "all_time": _summary(records),
        "last_90_days": _summary(within(90)),
        "last_30_days": _summary(within(30)),
        "by_ticker": by_ticker,
        "metric_note": (
            "Live metrics use only predictions saved before the next completed trading session. "
            "Legacy v1 files are intentionally excluded."
        ),
    }
