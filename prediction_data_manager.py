#!/usr/bin/env python3
"""
継続学習用のデータ管理クラス
日次予測ログ・実績・学習データセットを一元管理
"""

import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd


class PredictionDataManager:
    """予測結果と実績データを保存・整形するためのユーティリティ"""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw" / "daily_predictions"
        self.processed_dir = self.base_dir / "processed"
        self.feedback_dir = self.base_dir / "feedback"
        self.logs_dir = self.base_dir / "logs"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """必要なディレクトリ群を作成"""
        for path in [self.raw_dir, self.processed_dir, self.feedback_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _sanitize_for_json(self, obj: Any):
        """JSONに保存できるように NaN や numpy 型を再帰的に整理"""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        if isinstance(obj, (float, int)):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj
        if hasattr(obj, "item"):
            value = obj.item()
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return value
        return obj

    def save_daily_prediction(self, prediction_dict: Dict[str, Any]) -> None:
        """日次予測データを raw ディレクトリに保存"""
        sanitized = self._sanitize_for_json(prediction_dict)
        date_str = sanitized.get("date", datetime.now().strftime("%Y-%m-%d"))

        # 日付ごとのファイル
        daily_path = self.raw_dir / f"{date_str}.json"
        with open(daily_path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, ensure_ascii=False, indent=2)

        # 月次の JSONL に追記してストリーム処理を可能にする
        month_key = date_str[:7]
        jsonl_path = self.raw_dir / f"{month_key}.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")

    def update_actuals(self, prediction_date: str, ticker: str, actual_data: Dict[str, Any]) -> None:
        """実績値をフィードバックとして保存"""
        sanitized = self._sanitize_for_json(actual_data)
        sanitized.update({
            "prediction_date": prediction_date,
            "ticker": ticker,
            "recorded_at": datetime.now().isoformat()
        })
        month_key = prediction_date[:7]
        feedback_path = self.feedback_dir / f"actuals_{month_key}.jsonl"
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")

    def build_training_dataset(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        保存済みの予測ログと実績フィードバックを結合して学習用データセットを構築
        戻り値として DataFrame を返し、同時に Parquet にも保存する
        """
        records: List[Dict[str, Any]] = []
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        for json_file in sorted(self.raw_dir.glob("*.json")):
            if json_file.name == "latest_predictions.json":
                continue
            with open(json_file, "r", encoding="utf-8") as f:
                prediction = json.load(f)

            date_str = prediction.get("date")
            if not date_str:
                continue
            pred_dt = datetime.fromisoformat(date_str)
            if start_dt and pred_dt < start_dt:
                continue
            if end_dt and pred_dt > end_dt:
                continue

            for market, stocks in prediction.get("markets", {}).items():
                for stock in stocks:
                    actual = self._lookup_actual_record(stock["ticker"], date_str)
                    if actual is None:
                        continue  # 実績が未登録の場合はスキップ

                    feature_snapshot = stock.get("features", {})
                    record = {
                        "record_id": f"{stock['ticker']}_{date_str}",
                        "date": date_str,
                        "market": market,
                        "ticker": stock["ticker"],
                        "target_price": actual.get("actual_price"),
                        "target_change_pct": actual.get("actual_change_pct"),
                        "predicted_price": stock.get("predicted_price"),
                        "predicted_change_pct": stock.get("predicted_change_pct"),
                        "prediction_method": stock.get("prediction_method"),
                        "model_version": stock.get("metadata", {}).get("model_version"),
                    }

                    # 特徴量をフラット化して追加
                    for feat_key, feat_value in feature_snapshot.items():
                        record[f"feat_{feat_key}"] = feat_value

                    # 誤差情報
                    if actual.get("actual_price") and stock.get("predicted_price"):
                        try:
                            error_pct = abs(stock["predicted_price"] - actual["actual_price"]) / actual["actual_price"] * 100
                        except ZeroDivisionError:
                            error_pct = None
                        record["prediction_error_pct"] = error_pct

                    records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        dataset_path = self.processed_dir / "training_dataset.parquet"
        df.to_parquet(dataset_path, index=False)
        self._save_dataset_metadata(df, dataset_path)
        return df

    def _lookup_actual_record(self, ticker: str, prediction_date: str) -> Optional[Dict[str, Any]]:
        """feedback ディレクトリから対応する実績レコードを取得"""
        month_key = prediction_date[:7]
        feedback_path = self.feedback_dir / f"actuals_{month_key}.jsonl"
        if not feedback_path.exists():
            return None
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("ticker") == ticker and record.get("prediction_date") == prediction_date:
                    return record
        return None

    def _save_dataset_metadata(self, df: pd.DataFrame, dataset_path: Path) -> None:
        """学習データセットのメタ情報を保存"""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "num_records": int(len(df)),
            "date_min": df["date"].min(),
            "date_max": df["date"].max(),
            "tickers": sorted(df["ticker"].unique()),
            "dataset_path": str(dataset_path),
            "columns": list(df.columns),
        }
        meta_file = self.processed_dir / "training_dataset_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
