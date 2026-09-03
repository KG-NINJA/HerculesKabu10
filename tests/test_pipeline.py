from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from noroshi.data import (
    _frame_from_yahoo_chart,
    drop_incomplete_session,
    load_cache,
    normalize_history_frame,
    refresh_market_data,
    save_cache,
)
from noroshi.evaluation import evaluate_prediction_archives, summarize_live_metrics
from noroshi.features import FEATURE_COLUMNS, calculate_rsi, create_feature_frame
from noroshi.modeling import predict_ticker
from noroshi.pipeline import run_pipeline
from noroshi.settings import MarketConfig, PipelineConfig, TickerConfig


def synthetic_history(end: str = "2026-09-02", rows: int = 380, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=rows)
    returns = rng.normal(0.0004, 0.012, size=rows)
    close = 100 * np.exp(np.cumsum(returns))
    open_price = close * (1 + rng.normal(0, 0.003, size=rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.012, size=rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.012, size=rows))
    volume = rng.integers(1_000_000, 5_000_000, size=rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }
    )


def small_config() -> PipelineConfig:
    markets = {
        "US": MarketConfig("US", "America/New_York", "16:15"),
        "JP": MarketConfig("JP", "Asia/Tokyo", "15:45"),
    }
    return PipelineConfig(
        history_period="5y",
        max_data_age_days=7,
        minimum_history_rows=180,
        min_valid_tickers=2,
        backtest_splits=3,
        high_confidence_threshold=0.65,
        buy_probability_threshold=0.60,
        sell_probability_threshold=0.40,
        markets=markets,
        tickers=(
            TickerConfig("TEST", "Test US", "US", "USD"),
            TickerConfig("TEST.T", "Test JP", "JP", "JPY"),
        ),
    )


class DataAndFeatureTests(unittest.TestCase):
    def test_normalize_history_frame(self) -> None:
        frame = synthetic_history(rows=80).set_index("Date")
        normalized = normalize_history_frame(frame)
        self.assertEqual(list(normalized.columns[:6]), ["Date", "Open", "High", "Low", "Close", "Adj Close"])
        self.assertTrue(normalized["Date"].is_monotonic_increasing)

    def test_incomplete_same_day_bar_is_removed(self) -> None:
        frame = synthetic_history(end="2026-09-03", rows=80)
        market = MarketConfig("US", "America/New_York", "16:15")
        now = datetime(2026, 9, 3, 17, 0, tzinfo=timezone.utc)  # 13:00 New York
        completed = drop_incomplete_session(frame, market, now)
        self.assertEqual(completed.iloc[-1]["Date"].date().isoformat(), "2026-09-02")

    def test_timezone_aware_daily_index_preserves_exchange_session_date(self) -> None:
        frame = synthetic_history(end="2026-09-03", rows=80).set_index("Date")
        frame.index = frame.index.tz_localize("Asia/Tokyo")
        normalized = normalize_history_frame(frame)
        self.assertEqual(normalized.iloc[-1]["Date"].date().isoformat(), "2026-09-03")

    def test_direct_yahoo_chart_fallback_preserves_market_date(self) -> None:
        timestamp = int(pd.Timestamp("2026-09-03 00:00:00", tz="Asia/Tokyo").timestamp())
        payload = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {"exchangeTimezoneName": "Asia/Tokyo"},
                        "timestamp": [timestamp],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0],
                                    "high": [102.0],
                                    "low": [99.0],
                                    "close": [101.0],
                                    "volume": [1000],
                                }
                            ],
                            "adjclose": [{"adjclose": [101.0]}],
                        },
                    }
                ],
            }
        }
        frame = _frame_from_yahoo_chart(payload)
        self.assertEqual(frame.iloc[-1]["Date"].date().isoformat(), "2026-09-03")
        self.assertEqual(float(frame.iloc[-1]["Close"]), 101.0)

    def test_rsi_handles_uninterrupted_gains_and_losses(self) -> None:
        rising = calculate_rsi(pd.Series(np.arange(1.0, 25.0)), 14)
        falling = calculate_rsi(pd.Series(np.arange(25.0, 1.0, -1.0)), 14)
        self.assertEqual(float(rising.iloc[-1]), 100.0)
        self.assertEqual(float(falling.iloc[-1]), 0.0)

    def test_stale_download_does_not_replace_fresher_cache(self) -> None:
        cfg = small_config()
        now = datetime(2026, 9, 3, 23, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh = synthetic_history(end="2026-09-02", rows=220)
            for ticker in cfg.tickers:
                save_cache(root / "data" / "cache" / f"{ticker.ticker}.csv", fresh)

            def stale_downloader(ticker: str, period: str) -> pd.DataFrame:
                return synthetic_history(end="2025-11-21", rows=220)

            frames, statuses = refresh_market_data(root, cfg, now, stale_downloader)
            self.assertEqual(set(frames), {"TEST", "TEST.T"})
            self.assertTrue(all(item["status"] == "fallback" for item in statuses))
            cached = load_cache(root / "data" / "cache" / "TEST.csv")
            self.assertEqual(cached.iloc[-1]["Date"].date().isoformat(), "2026-09-02")

    def test_final_target_is_nan_not_down(self) -> None:
        features = create_feature_frame(synthetic_history())
        self.assertTrue(pd.isna(features.iloc[-1]["TARGET_UP"]))
        self.assertTrue(pd.isna(features.iloc[-1]["TARGET_RETURN"]))
        self.assertFalse(features.dropna(subset=list(FEATURE_COLUMNS)).empty)


class ModelingTests(unittest.TestCase):
    def test_time_series_model_produces_auditable_prediction(self) -> None:
        cfg = small_config()
        generated = datetime(2026, 9, 3, 23, 0, tzinfo=timezone.utc)
        prediction = predict_ticker(cfg.tickers[0], synthetic_history(), cfg, generated)
        self.assertEqual(prediction["as_of"], "2026-09-02")
        self.assertIn(prediction["direction"], {"UP", "DOWN"})
        self.assertIn(prediction["signal"], {"BUY", "SELL", "HOLD"})
        self.assertGreater(prediction["model"]["validation"]["samples"], 0)
        self.assertEqual(
            prediction["model"]["validation"]["method"],
            "expanding_time_series_split_with_gap",
        )
        self.assertTrue(prediction["interpretation"]["confidence_is_not_historical_accuracy"])

    def test_failed_audit_gate_forces_hold_signal(self) -> None:
        cfg = small_config()
        generated = datetime(2026, 9, 3, 23, 0, tzinfo=timezone.utc)
        validation = {
            "method": "expanding_time_series_split_with_gap",
            "splits": 3,
            "test_size_per_split": 20,
            "samples": 120,
            "direction_accuracy": 0.51,
            "balanced_accuracy": 0.51,
            "persistence_baseline_accuracy": 0.55,
            "skill_vs_persistence": -0.04,
            "always_up_baseline_accuracy": 0.50,
            "brier_score": 0.25,
            "return_mae_pct": 1.0,
            "high_confidence_threshold": 0.65,
            "high_confidence_coverage": 0.25,
            "high_confidence_accuracy": 0.50,
        }
        with patch("noroshi.modeling._time_series_validation", return_value=validation):
            prediction = predict_ticker(cfg.tickers[0], synthetic_history(), cfg, generated)
        self.assertFalse(prediction["quality_gate_passed"])
        self.assertEqual(prediction["signal"], "HOLD")


class EvaluationTests(unittest.TestCase):
    def test_saved_prediction_is_scored_on_next_session(self) -> None:
        cfg = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_dir = root / "data" / "predictions_v2"
            prediction_dir.mkdir(parents=True)
            archive = {
                "generated_at": "2026-09-02T22:00:00Z",
                "predictions": [
                    {
                        "prediction_id": "TEST:2026-09-02:abc",
                        "ticker": "TEST",
                        "market": "US",
                        "generated_at": "2026-09-02T22:00:00Z",
                        "as_of": "2026-09-02",
                        "current_price": 100.0,
                        "predicted_price": 101.0,
                        "predicted_return_pct": 1.0,
                        "direction": "UP",
                        "probability_up": 0.7,
                        "confidence": 0.7,
                        "baseline_direction": "DOWN",
                        "model": {"version": "test"},
                    }
                ],
            }
            (prediction_dir / "sample.json").write_text(json.dumps(archive), encoding="utf-8")
            frame = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-09-02", "2026-09-03"]),
                    "Open": [100, 101],
                    "High": [101, 103],
                    "Low": [99, 100],
                    "Close": [100, 102],
                    "Volume": [1000, 1100],
                }
            )
            evaluated = evaluate_prediction_archives(
                root,
                {"TEST": frame},
                cfg,
                datetime(2026, 9, 3, 23, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(len(evaluated["results"]), 1)
            self.assertTrue(evaluated["results"][0]["direction_correct"])
            metrics = summarize_live_metrics(evaluated, datetime(2026, 9, 3, tzinfo=timezone.utc))
            self.assertEqual(metrics["all_time"]["direction_accuracy"], 1.0)
            self.assertEqual(metrics["all_time"]["baseline_direction_accuracy"], 0.0)
            self.assertEqual(metrics["all_time"]["skill_vs_baseline"], 1.0)


class EndToEndTests(unittest.TestCase):
    def test_pipeline_writes_fresh_outputs_and_dashboard(self) -> None:
        cfg = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def downloader(ticker: str, period: str) -> pd.DataFrame:
                self.assertEqual(period, "5y")
                return synthetic_history(seed=11 if ticker == "TEST" else 17)

            result = run_pipeline(
                root=root,
                config=cfg,
                now_utc=datetime(2026, 9, 3, 23, 0, tzinfo=timezone.utc),
                downloader=downloader,
                strict=True,
            )
            self.assertEqual(result["health"], "healthy")
            self.assertEqual(len(result["latest"]["predictions"]), 2)
            self.assertTrue((root / "data" / "latest_predictions.json").exists())
            self.assertTrue((root / "data" / "metrics.json").exists())
            self.assertTrue((root / "docs" / "index.html").exists())
            self.assertTrue((root / "docs" / "assets" / "accuracy.png").exists())
            html = (root / "docs" / "index.html").read_text(encoding="utf-8")
            self.assertIn("Confidenceは過去の的中率ではありません", html)

    def test_unavailable_data_publishes_explicit_unhealthy_state(self) -> None:
        cfg = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def failing_downloader(ticker: str, period: str) -> pd.DataFrame:
                raise RuntimeError("provider unavailable")

            result = run_pipeline(
                root=root,
                config=cfg,
                now_utc=datetime(2026, 9, 3, 23, 0, tzinfo=timezone.utc),
                downloader=failing_downloader,
                strict=False,
            )
            self.assertEqual(result["health"], "unhealthy")
            for relative in (
                "data/latest_predictions.json",
                "data/metrics.json",
                "data/status.json",
                "docs/index.html",
            ):
                self.assertTrue((root / relative).exists(), relative)
            status = json.loads((root / "data" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["health"], "unhealthy")
            html = (root / "docs" / "index.html").read_text(encoding="utf-8")
            self.assertIn("停止", html)


if __name__ == "__main__":
    unittest.main()
