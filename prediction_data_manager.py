#!/usr/bin/env python3
"""Compatibility helpers for canonical NOROSHI JSON output files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noroshi.reporting import write_json_atomic

ROOT = Path(__file__).resolve().parent


class PredictionDataManager:
    @staticmethod
    def save_latest(payload: dict[str, Any], root: Path = ROOT) -> Path:
        path = root / "data" / "latest_predictions.json"
        write_json_atomic(path, payload)
        return path

    @staticmethod
    def save_metrics(payload: dict[str, Any], root: Path = ROOT) -> Path:
        path = root / "data" / "metrics.json"
        write_json_atomic(path, payload)
        return path
