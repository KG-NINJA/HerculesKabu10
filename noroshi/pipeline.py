from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .data import refresh_market_data
from .evaluation import evaluate_prediction_archives, summarize_live_metrics
from .modeling import aggregate_backtest_metrics, predict_ticker
from .reporting import generate_reports, write_json_atomic
from .settings import MODEL_VERSION, SCHEMA_VERSION, PipelineConfig, load_config


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _data_signature(predictions: list[dict[str, Any]]) -> str:
    identity = [
        {
            "ticker": item["ticker"],
            "as_of": item["as_of"],
            "current_price": round(float(item["current_price"]), 8),
            "model_version": item["model"]["version"],
        }
        for item in sorted(predictions, key=lambda value: value["ticker"])
    ]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _empty_latest(now: datetime) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "data_signature": None,
        "predictions": [],
    }


def _write_failure_state(
    root: Path,
    now: datetime,
    statuses: list[dict[str, object]],
    warnings: list[str],
    message: str,
) -> None:
    latest = _read_json(root / "data" / "latest_predictions.json", _empty_latest(now))
    metrics = _read_json(
        root / "data" / "metrics.json",
        {"schema_version": SCHEMA_VERSION, "backtest": {}, "live": {}},
    )
    status = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now.isoformat().replace("+00:00", "Z"),
        "health": "unhealthy",
        "message": message,
        "valid_tickers": 0,
        "warnings": warnings,
        "tickers": statuses,
    }
    write_json_atomic(root / "data" / "latest_predictions.json", latest)
    write_json_atomic(root / "data" / "metrics.json", metrics)
    write_json_atomic(root / "data" / "status.json", status)
    generate_reports(root, latest, metrics, status)


def run_pipeline(
    root: Path,
    config: PipelineConfig,
    now_utc: datetime | None = None,
    downloader: Callable[[str, str], pd.DataFrame] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    frames, statuses = refresh_market_data(
        root=root,
        config=config,
        now_utc=now,
        downloader=downloader,
    )
    warnings: list[str] = []
    for item in statuses:
        if item.get("status") != "fresh":
            detail = f"{item.get('ticker')}: {item.get('status')} via {item.get('source')}"
            if item.get("error"):
                detail += f" ({item.get('error')})"
            warnings.append(detail)

    markets_present = {
        ticker.market for ticker in config.tickers if ticker.ticker in frames
    }
    if len(frames) < config.min_valid_tickers or not {"US", "JP"}.issubset(markets_present):
        message = (
            f"insufficient fresh market data: {len(frames)}/{len(config.tickers)} tickers; "
            f"markets={sorted(markets_present)}"
        )
        _write_failure_state(root, now, statuses, warnings, message)
        if strict:
            raise RuntimeError(message)
        return {"health": "unhealthy", "message": message}

    ticker_lookup = {item.ticker: item for item in config.tickers}
    predictions: list[dict[str, Any]] = []
    prediction_errors: list[str] = []
    for ticker in sorted(frames):
        try:
            predictions.append(
                predict_ticker(
                    ticker_config=ticker_lookup[ticker],
                    history=frames[ticker],
                    config=config,
                    generated_at=now,
                )
            )
        except Exception as exc:
            prediction_errors.append(f"{ticker}: {str(exc)[:500]}")

    warnings.extend(prediction_errors)
    if len(predictions) < config.min_valid_tickers:
        message = (
            f"insufficient valid predictions: {len(predictions)}/{len(config.tickers)}; "
            + "; ".join(prediction_errors)
        )
        _write_failure_state(root, now, statuses, warnings, message)
        if strict:
            raise RuntimeError(message)
        return {"health": "unhealthy", "message": message}

    signature = _data_signature(predictions)
    archive_dir = root / "data" / "predictions_v2"
    archive_dir.mkdir(parents=True, exist_ok=True)
    max_as_of = max(item["as_of"] for item in predictions)
    archive_path = archive_dir / f"{max_as_of}_{signature[:12]}.json"

    new_payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "data_signature": signature,
        "horizon": "next_trading_close",
        "prediction_count": len(predictions),
        "predictions": sorted(predictions, key=lambda item: (item["market"], item["ticker"])),
    }
    if archive_path.exists():
        official_payload = _read_json(archive_path, new_payload)
    else:
        write_json_atomic(archive_path, new_payload)
        official_payload = new_payload
    write_json_atomic(root / "data" / "latest_predictions.json", official_payload)

    evaluation = evaluate_prediction_archives(
        root=root,
        cache_frames=frames,
        config=config,
        updated_at=now,
    )
    backtest = aggregate_backtest_metrics(official_payload.get("predictions", []))
    live = summarize_live_metrics(evaluation, now)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "updated_at": now.isoformat().replace("+00:00", "Z"),
        "backtest": backtest,
        "live": live,
        "definitions": {
            "backtest": "Expanding time-series validation with a one-session gap.",
            "live": "Immutable saved predictions scored against the next completed session.",
            "confidence": "Current model class probability; not historical accuracy.",
            "baseline": "Previous-session direction persistence, evaluated on the same observations.",
        },
    }
    write_json_atomic(root / "data" / "metrics.json", metrics)

    live_count = int(live.get("all_time", {}).get("direction_evaluable", 0))
    if live_count < 30:
        warnings.append(
            f"live accuracy is provisional: only {live_count} direction-evaluable v2 predictions"
        )
    health = "healthy" if not prediction_errors and all(
        item.get("status") == "fresh" for item in statuses
    ) else "degraded"
    status = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "updated_at": now.isoformat().replace("+00:00", "Z"),
        "health": health,
        "message": "pipeline completed",
        "valid_tickers": len(predictions),
        "configured_tickers": len(config.tickers),
        "archive": str(archive_path.relative_to(root)),
        "data_signature": signature,
        "warnings": warnings,
        "tickers": statuses,
    }
    write_json_atomic(root / "data" / "status.json", status)
    generate_reports(root, official_payload, metrics, status)

    return {
        "health": health,
        "latest": official_payload,
        "metrics": metrics,
        "status": status,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NOROSHI auditable prediction pipeline")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--config", default="config/markets.json", help="Configuration JSON")
    parser.add_argument(
        "--allow-degraded-failure",
        action="store_true",
        help="Write an unhealthy dashboard instead of exiting non-zero when minimum data is unavailable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    result = run_pipeline(
        root=root,
        config=config,
        strict=not args.allow_degraded_failure,
    )
    print(
        json.dumps(
            {
                "health": result.get("health"),
                "prediction_count": len(result.get("latest", {}).get("predictions", [])),
                "generated_at": result.get("latest", {}).get("generated_at"),
            },
            ensure_ascii=False,
        )
    )
    return 0
