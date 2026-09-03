#!/usr/bin/env python3
"""Regenerate the canonical dashboard from existing pipeline outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noroshi.reporting import generate_reports  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        latest = read_json(root / "data" / "latest_predictions.json")
        metrics = read_json(root / "data" / "metrics.json")
        status = read_json(root / "data" / "status.json")
        generate_reports(root, latest, metrics, status)
    except Exception as error:
        print(f"Report generation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print("NOROSHI reports generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
