#!/usr/bin/env python3
"""Regenerate only README.md from canonical outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noroshi.reporting import generate_readme  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        latest = json.loads((root / "data" / "latest_predictions.json").read_text(encoding="utf-8"))
        metrics = json.loads((root / "data" / "metrics.json").read_text(encoding="utf-8"))
        status = json.loads((root / "data" / "status.json").read_text(encoding="utf-8"))
        generate_readme(root, latest, metrics, status)
    except Exception as error:
        print(f"README generation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
