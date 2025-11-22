#!/usr/bin/env python3
"""NVDA向けCONFIG評価ランナー + 自動JSONログ保存"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# === CONFIG ===
CONFIG: Dict[str, Any] = {
    "target_ticker": "NVDA",
    "cv_sources": [
        {
            "name": "continuous_registry",
            "type": "registry",
            "path": "continuous_models/model_registry.json",
            "weight": 0.35,
        },
        {
            "name": "stable_registry",
            "type": "registry",
            "path": "stable_continuous_models/model_registry.json",
            "weight": 0.35,
        },
        {
            "name": "recent_validation",
            "type": "validation",
            "directory": "validation_results",
            "weight": 0.30,
            "max_files": 4,
        },
    ],
    "confidence_weights": {
        "cv": 0.5,
        "mape": 0.2,
        "signal": 0.3,
    },
    "signal_rule": {
        "predictions_file": "daily_predictions/latest_predictions.json",
        "change_scale_pct": 5.0,
        "buy_threshold_pct": 1.0,
        "sell_threshold_pct": -1.0,
    },
    "required": {
        "cv_average": 60.0,
        "high_confidence": 65.0,
    },
}

# === データ構造 ===
@dataclass
class CVComponent:
    name: str
    score: float
    weight: float


# === ユーティリティ ===
def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gather_registry_score(project_root: Path, source: Dict[str, Any], ticker: str) -> Optional[CVComponent]:
    data = load_json(project_root / source["path"])
    if not data or ticker not in data:
        return None
    accuracy = data[ticker].get("last_accuracy", {}).get("direction_accuracy")
    if accuracy is None:
        return None
    return CVComponent(name=source["name"], score=float(accuracy), weight=float(source["weight"]))


def gather_validation_component(project_root: Path, source: Dict[str, Any], ticker: str) -> Tuple[Optional[CVComponent], Optional[float]]:
    directory = project_root / source.get("directory", "validation_results")
    pattern = f"validation_{ticker}_"
    files = sorted(
        [p for p in directory.glob("*.json") if p.name.startswith(pattern)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    selected = files[: int(source.get("max_files", 3))]
    if not selected:
        return (None, None)

    cv_scores, mapes = [], []
    for path in selected:
        payload = load_json(path)
        if not payload:
            continue
        metrics = payload.get("metrics", {})
        mape = metrics.get("mape")
        if mape is None:
            continue
        mapes.append(float(mape))
        cv_scores.append(max(0.0, 100.0 - float(mape)))

    if not cv_scores:
        return (None, None)

    avg_score = mean(cv_scores)
    component = CVComponent(source["name"], avg_score, float(source["weight"]))
    return (component, mean(mapes))


def normalize_weights(components: List[CVComponent]) -> List[CVComponent]:
    total = sum(c.weight for c in components)
    if total == 0:
        return components
    return [CVComponent(c.name, c.score, c.weight / total) for c in components]


def calculate_signal(project_root: Path, cfg: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    predictions_file = project_root / cfg["predictions_file"]
    payload = load_json(predictions_file) or {}
    markets = payload.get("markets", {})
    entries = []
    for market_stocks in markets.values():
        entries.extend(market_stocks)

    entry = next((item for item in entries if item.get("ticker") == ticker), None)
    if not entry:
        return {"status": "missing"}

    change_pct = float(entry.get("predicted_change_pct", 0.0))
    trend = entry.get("trend", "N/A")
    scale = max(0.1, float(cfg.get("change_scale_pct", 5.0)))
    normalized = max(-1.0, min(1.0, change_pct / scale))
    strength_pct = (normalized + 1.0) / 2.0 * 100.0

    if change_pct >= cfg.get("buy_threshold_pct", 1.0):
        signal = "BUY"
    elif change_pct <= cfg.get("sell_threshold_pct", -1.0):
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "status": "ok",
        "predicted_change_pct": change_pct,
        "trend": trend,
        "signal": signal,
        "signal_strength": strength_pct,
    }


def print_section(title: str):
    print(f"\n=== {title} ===")


# === メイン ===
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ticker = CONFIG["target_ticker"]

    cv_components: List[CVComponent] = []
    validation_mape: Optional[float] = None

    for source in CONFIG["cv_sources"]:
        if source["type"] == "registry":
            c = gather_registry_score(project_root, source, ticker)
            if c:
                cv_components.append(c)
        elif source["type"] == "validation":
            c, m = gather_validation_component(project_root, source, ticker)
            if c:
                cv_components.append(c)
            if m is not None:
                validation_mape = m if validation_mape is None else (validation_mape + m) / 2

    if not cv_components:
        print("CV情報が見つかりません。CONFIGを確認。")
        return

    cv_components = normalize_weights(cv_components)
    cv_average = sum(c.score * c.weight for c in cv_components)

    print_section("CV Components")
    for c in cv_components:
        print(f"- {c.name}: {c.score:.2f}% (weight {c.weight:.2f})")
    print(f"\n=> 加重CV平均: {cv_average:.2f}%")

    signal_info = calculate_signal(project_root, CONFIG["signal_rule"], ticker)
    signal_strength = signal_info.get("signal_strength", 50.0)
    mape_component = 100.0 - validation_mape if validation_mape is not None else 50.0

    conf_cfg = CONFIG["confidence_weights"]
    conf_total = sum(conf_cfg.values()) or 1.0
    confidence = (
        cv_average * conf_cfg["cv"]
        + mape_component * conf_cfg["mape"]
        + signal_strength * conf_cfg["signal"]
    ) / conf_total

    print_section("Confidence & Signal")
    print(f"- 平均MAPE: {validation_mape:.3f}% -> スコア {mape_component:.2f}%" if validation_mape else "- 検証MAPE: データなし (50%)")

    if signal_info["status"] == "ok":
        print(f"- 最新予測: {signal_info['predicted_change_pct']:+.2f}% / Trend {signal_info['trend']}")
        print(f"- シグナル: {signal_info['signal']} (強度 {signal_strength:.1f}%)")
    else:
        print("- シグナル: データなし")

    print(f"\n=> 信頼スコア: {confidence:.2f}%")

    req = CONFIG["required"]
    print_section("Goal Check")
    print(f"- CV平均 >= {req['cv_average']}% : {'PASS' if cv_average >= req['cv_average'] else 'FAIL'}")
    print(f"- 高確信 >= {req['high_confidence']}% : {'PASS' if confidence >= req['high_confidence'] else 'FAIL'}")

    # --- 自動ログ保存 ---
    save_run_log(cv_average, confidence, signal_info, signal_strength)


# === JSONログ保存機能 ===
def save_run_log(cv_average=None, confidence=None, signal_info=None, signal_strength=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "cv_average": round(cv_average or 0, 2),
        "confidence": round(confidence or 0, 2),
        "signal": signal_info.get("signal") if signal_info else None,
        "signal_strength": round(signal_strength or 0, 2),
    }

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"cv_run_{datetime.now():%Y%m%d_%H%M%S}.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    print(f"\n[ログ保存完了] {log_path}")


if __name__ == "__main__":
    main()
