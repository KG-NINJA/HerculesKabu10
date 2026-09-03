from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLUMNS, create_feature_frame
from .settings import MODEL_VERSION, PipelineConfig, TickerConfig


def _classifier(seed: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=140,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=0.35,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def _regressor(seed: int = 42) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=0.35,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def _prediction_id(ticker: str, as_of: str) -> str:
    source = f"{MODEL_VERSION}|{ticker}|{as_of}|next_trading_close"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    return f"{ticker}:{as_of}:{digest}"


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(accuracy_score(y_true, y_pred))
    return float(balanced_accuracy_score(y_true, y_pred))


def _time_series_validation(
    labelled: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    X = labelled.loc[:, FEATURE_COLUMNS].astype(float)
    y_class = labelled["TARGET_UP"].astype(int)
    y_return = labelled["TARGET_RETURN"].astype(float)

    n_samples = len(labelled)
    n_splits = min(config.backtest_splits, max(2, (n_samples - 100) // 25))
    test_size = max(20, min(40, (n_samples - 100) // n_splits))
    while n_splits * test_size + 2 >= n_samples - 60 and test_size > 20:
        test_size -= 1
    if n_splits < 2 or n_splits * test_size + 2 >= n_samples:
        raise ValueError(f"insufficient labelled rows for time-series validation: {n_samples}")

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=1)
    true_class: list[int] = []
    predicted_class: list[int] = []
    probability_up: list[float] = []
    true_return: list[float] = []
    predicted_return: list[float] = []
    persistence_class: list[int] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_class, y_test_class = y_class.iloc[train_index], y_class.iloc[test_index]
        y_train_return, y_test_return = y_return.iloc[train_index], y_return.iloc[test_index]

        if y_train_class.nunique() < 2:
            continue

        classifier = _classifier(42 + fold)
        regressor = _regressor(42 + fold)
        classifier.fit(X_train, y_train_class)
        regressor.fit(X_train, y_train_return)

        fold_probability = classifier.predict_proba(X_test)[:, 1]
        fold_class = (fold_probability >= 0.5).astype(int)
        fold_return = regressor.predict(X_test)

        true_class.extend(y_test_class.astype(int).tolist())
        predicted_class.extend(fold_class.astype(int).tolist())
        probability_up.extend(fold_probability.astype(float).tolist())
        true_return.extend(y_test_return.astype(float).tolist())
        predicted_return.extend(np.asarray(fold_return, dtype=float).tolist())
        persistence_class.extend((X_test["RETURN_1D"] > 0).astype(int).tolist())

    if not true_class:
        raise ValueError("time-series validation produced no usable folds")

    y_true = np.asarray(true_class, dtype=int)
    y_pred = np.asarray(predicted_class, dtype=int)
    y_prob = np.asarray(probability_up, dtype=float)
    y_return_true = np.asarray(true_return, dtype=float)
    y_return_pred = np.asarray(predicted_return, dtype=float)
    persistence = np.asarray(persistence_class, dtype=int)
    confidence = np.maximum(y_prob, 1 - y_prob)
    high_mask = confidence >= config.high_confidence_threshold

    direction_accuracy = float(accuracy_score(y_true, y_pred))
    persistence_accuracy = float(accuracy_score(y_true, persistence))
    return {
        "method": "expanding_time_series_split_with_gap",
        "splits": int(n_splits),
        "test_size_per_split": int(test_size),
        "samples": int(len(y_true)),
        "direction_accuracy": direction_accuracy,
        "balanced_accuracy": _safe_balanced_accuracy(y_true, y_pred),
        "persistence_baseline_accuracy": persistence_accuracy,
        "skill_vs_persistence": direction_accuracy - persistence_accuracy,
        "always_up_baseline_accuracy": float(np.mean(y_true == 1)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "return_mae_pct": float(mean_absolute_error(y_return_true, y_return_pred) * 100),
        "high_confidence_threshold": float(config.high_confidence_threshold),
        "high_confidence_coverage": float(high_mask.mean()),
        "high_confidence_accuracy": (
            float(accuracy_score(y_true[high_mask], y_pred[high_mask]))
            if high_mask.any()
            else None
        ),
    }


def predict_ticker(
    ticker_config: TickerConfig,
    history: pd.DataFrame,
    config: PipelineConfig,
    generated_at: datetime,
) -> dict[str, Any]:
    feature_frame = create_feature_frame(history)
    usable = feature_frame.dropna(subset=list(FEATURE_COLUMNS)).copy()
    labelled = usable.dropna(subset=["TARGET_UP", "TARGET_RETURN"]).copy()

    if len(labelled) < config.minimum_history_rows - 60:
        raise ValueError(
            f"{ticker_config.ticker} has only {len(labelled)} labelled feature rows"
        )
    if usable.empty:
        raise ValueError(f"{ticker_config.ticker} has no current feature row")

    validation = _time_series_validation(labelled, config)
    X_train = labelled.loc[:, FEATURE_COLUMNS].astype(float)
    y_class = labelled["TARGET_UP"].astype(int)
    y_return = labelled["TARGET_RETURN"].astype(float)
    if y_class.nunique() < 2:
        raise ValueError(f"{ticker_config.ticker} training target has one class")

    classifier = _classifier()
    regressor = _regressor()
    classifier.fit(X_train, y_class)
    regressor.fit(X_train, y_return)

    latest = usable.iloc[-1]
    X_latest = latest.loc[list(FEATURE_COLUMNS)].astype(float).to_frame().T
    probability_up = float(classifier.predict_proba(X_latest)[0, 1])
    raw_return = float(regressor.predict(X_latest)[0])
    volatility = float(latest.get("VOLATILITY_20D", 0.01) or 0.01)
    volatility = max(0.002, min(abs(volatility), 0.08))
    probability_edge = (probability_up - 0.5) * 2.0
    blended_return = 0.55 * raw_return + 0.45 * probability_edge * volatility
    cap = min(0.10, max(0.005, 3.0 * volatility))
    magnitude = min(abs(blended_return), cap)
    magnitude = max(magnitude, min(cap, max(0.0005, abs(probability_edge) * volatility * 0.25)))
    predicted_return = magnitude if probability_up >= 0.5 else -magnitude

    current_price = float(latest["Close"])
    predicted_price = current_price * (1 + predicted_return)
    direction = "UP" if probability_up >= 0.5 else "DOWN"
    confidence = max(probability_up, 1 - probability_up)

    as_of = pd.Timestamp(latest["Date"]).date().isoformat()
    training_start = pd.Timestamp(labelled.iloc[0]["Date"]).date().isoformat()
    training_end = pd.Timestamp(labelled.iloc[-1]["Date"]).date().isoformat()
    baseline_direction = "UP" if float(latest["RETURN_1D"]) > 0 else "DOWN"
    quality_gate_passed = (
        validation["samples"] >= 100
        and validation["direction_accuracy"] >= 0.50
        and validation["skill_vs_persistence"] >= 0.0
    )

    if not quality_gate_passed:
        signal = "HOLD"
    elif probability_up >= config.buy_probability_threshold and predicted_return > 0:
        signal = "BUY"
    elif probability_up <= config.sell_probability_threshold and predicted_return < 0:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "prediction_id": _prediction_id(ticker_config.ticker, as_of),
        "ticker": ticker_config.ticker,
        "name": ticker_config.name,
        "market": ticker_config.market,
        "currency": ticker_config.currency,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "as_of": as_of,
        "horizon": "next_trading_close",
        "current_price": current_price,
        "predicted_price": float(predicted_price),
        "predicted_return_pct": float(predicted_return * 100),
        "direction": direction,
        "direction_ja": "強気" if direction == "UP" else "弱気",
        "probability_up": probability_up,
        "confidence": confidence,
        "baseline_direction": baseline_direction,
        "quality_gate_passed": quality_gate_passed,
        "signal": signal,
        "model": {
            "version": MODEL_VERSION,
            "type": "LightGBM classifier + LightGBM return regressor",
            "training_rows": int(len(labelled)),
            "training_start": training_start,
            "training_end": training_end,
            "feature_count": len(FEATURE_COLUMNS),
            "validation": validation,
        },
        "features_snapshot": {
            "return_1d_pct": float(latest["RETURN_1D"] * 100),
            "volatility_20d_pct": float(volatility * 100),
            "rsi14": float(latest["RSI14"] * 100),
            "macd_pct": float(latest["MACD"] * 100),
            "volume_ratio_20d": float(latest["VOLUME_RATIO_20D"]),
        },
        "interpretation": {
            "confidence_is_model_probability": True,
            "confidence_is_not_historical_accuracy": True,
            "research_only": True,
        },
    }


def aggregate_backtest_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[tuple[int, dict[str, Any]]] = []
    by_ticker: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        validation = prediction["model"]["validation"]
        samples = int(validation.get("samples", 0))
        if samples <= 0:
            continue
        by_ticker[prediction["ticker"]] = validation
        rows.append((samples, validation))

    total = sum(samples for samples, _ in rows)
    if total == 0:
        return {"samples": 0, "by_ticker": by_ticker}

    weighted_fields = (
        "direction_accuracy",
        "balanced_accuracy",
        "persistence_baseline_accuracy",
        "skill_vs_persistence",
        "always_up_baseline_accuracy",
        "brier_score",
        "return_mae_pct",
        "high_confidence_coverage",
    )
    overall: dict[str, Any] = {"samples": total}
    for field in weighted_fields:
        overall[field] = sum(samples * float(values[field]) for samples, values in rows) / total

    hc_weight = 0.0
    hc_correct_weight = 0.0
    for samples, values in rows:
        coverage = float(values.get("high_confidence_coverage") or 0.0)
        high_samples = samples * coverage
        accuracy = values.get("high_confidence_accuracy")
        if accuracy is not None and high_samples > 0:
            hc_weight += high_samples
            hc_correct_weight += high_samples * float(accuracy)
    overall["high_confidence_accuracy"] = hc_correct_weight / hc_weight if hc_weight else None
    overall["by_ticker"] = by_ticker
    overall["method"] = "weighted expanding time-series validation"
    return overall
