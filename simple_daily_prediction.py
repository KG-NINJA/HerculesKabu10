#!/usr/bin/env python3
"""
日米主要銘柄 毎日予測アプリ（シンプル版）
二日比較分析をスキップして安定性を向上
"""

import schedule
import time
import json
from datetime import datetime, timedelta
import os
import math
import yfinance as yf
import pandas as pd
from prediction_data_manager import PredictionDataManager
try:
    from colorama import init as colorama_init, Fore, Style, Back

    # Windows環境でもANSIカラーを有効化するための初期化
    colorama_init(autoreset=True)
    COLOR_OUTPUT_ENABLED = True
except ImportError:
    # coloramaが無い環境でも動作するようにダミー定義を用意
    class _ColorFallback:
        def __getattr__(self, _name):
            return ""

    Fore = Style = Back = _ColorFallback()
    COLOR_OUTPUT_ENABLED = False

class SimpleDailyPredictionApp:
    """シンプルで安定した毎日予測アプリ"""
    
    def __init__(self):
        self.results_dir = "daily_predictions"
        self.create_results_directory()
        
        # 主要銘柄リスト
        self.major_stocks = {
            "米国市場": {
                "AAPL": "Apple",
                "GOOGL": "Google",
                "MSFT": "Microsoft",
                "NVDA": "NVIDIA",
                "TSLA": "Tesla"
            },
            "日本市場": {
                "7203.T": "トヨタ自動車",
                "6758.T": "ソニーグループ",
                "9984.T": "ソフトバンク",
                "6861.T": "キーエンス",
                "8035.T": "東京エレクトロン"
            }
        }
        # 見やすい配色を適用するための設定（ライト背景でも視認性を確保）
        self.default_text = Fore.LIGHTBLUE_EX if COLOR_OUTPUT_ENABLED else ""
        self.market_color = {
            "米国市場": Back.BLUE + Fore.WHITE + Style.BRIGHT if COLOR_OUTPUT_ENABLED else "",
            "日本市場": Back.MAGENTA + Fore.WHITE + Style.BRIGHT if COLOR_OUTPUT_ENABLED else ""
        }
        self.trend_colors = {
            "強気": Fore.GREEN + Style.BRIGHT if COLOR_OUTPUT_ENABLED else "",
            "弱気": Fore.RED + Style.BRIGHT if COLOR_OUTPUT_ENABLED else "",
            "横ばい": Fore.YELLOW + Style.BRIGHT if COLOR_OUTPUT_ENABLED else ""
        }
        self.warning_color = Fore.YELLOW + Style.BRIGHT if COLOR_OUTPUT_ENABLED else ""
        self.alert_color = Fore.RED + Style.BRIGHT if COLOR_OUTPUT_ENABLED else ""
        self.info_color = Fore.CYAN + Style.BRIGHT if COLOR_OUTPUT_ENABLED else ""
        # 市場コンテキストを先に読み込み（欠損時は None を許容）
        self.global_context = self._load_market_context()
        # データ管理クラス（継続学習用）の初期化
        self.data_manager = PredictionDataManager()

    def _print_line(self, message: str = "", color: str = None, end: str = "\n"):
        """ライト背景でも読めるように、デフォルトで青系の文字色を適用"""
        if COLOR_OUTPUT_ENABLED:
            color_code = color if color is not None else self.default_text
            print(f"{color_code}{message}{Style.RESET_ALL}", end=end)
        else:
            print(message, end=end)

    def _load_market_context(self) -> dict:
        """市場全体の状況をJSONに保存できる形で取得"""
        try:
            spy = yf.Ticker("SPY").history(period="10d")
            vix = yf.Ticker("^VIX").history(period="10d")
            context = {
                "retrieved_at": datetime.now().isoformat(),
                "spy_close": float(spy["Close"].iloc[-1]) if len(spy) else None,
                "spy_change_pct": float(((spy["Close"].iloc[-1] - spy["Close"].iloc[-2]) / spy["Close"].iloc[-2] * 100)) if len(spy) > 1 else None,
                "vix_close": float(vix["Close"].iloc[-1]) if len(vix) else None,
                "vix_change_pct": float(((vix["Close"].iloc[-1] - vix["Close"].iloc[-2]) / vix["Close"].iloc[-2] * 100)) if len(vix) > 1 else None
            }
            return context
        except Exception:
            # 取得失敗時は None を保持してアプリの安定性を優先
            return {
                "retrieved_at": datetime.now().isoformat(),
                "spy_close": None,
                "spy_change_pct": None,
                "vix_close": None,
                "vix_change_pct": None
            }

    def _calculate_rsi(self, close_series: pd.Series, period: int = 14) -> float:
        """RSI(14)を計算し、十分なデータが無い場合は NaN を返す"""
        if len(close_series) < period + 1:
            return float("nan")
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def _calculate_macd(self, close_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD とシグナル・ヒストグラムを計算"""
        if len(close_series) < slow + signal:
            return float("nan"), float("nan"), float("nan")
        ema_fast = close_series.ewm(span=fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])

    def _build_feature_snapshot(self, data: pd.DataFrame) -> dict:
        """学習ログに保存する特徴量スナップショットを構築"""
        latest = data.iloc[-1]
        features = {
            "feature_timestamp": latest.name.isoformat() if hasattr(latest.name, "isoformat") else datetime.now().isoformat(),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),
            "volume": float(latest["Volume"]),
            "ma5": float(data["Close"].rolling(window=5).mean().iloc[-1]) if len(data) >= 5 else float("nan"),
            "ma20": float(data["Close"].rolling(window=20).mean().iloc[-1]) if len(data) >= 20 else float("nan"),
            "ma50": float(data["Close"].rolling(window=50).mean().iloc[-1]) if len(data) >= 50 else float("nan"),
            "rsi14": self._calculate_rsi(data["Close"], 14),
        }
        macd, macd_signal, macd_hist = self._calculate_macd(data["Close"])
        features.update({
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        })
        if len(data) >= 2:
            prev_close = data["Close"].iloc[-2]
            features.update({
                "price_change_1d": float(latest["Close"] - prev_close),
                "price_change_pct_1d": float((latest["Close"] - prev_close) / prev_close * 100),
            })
        else:
            features.update({
                "price_change_1d": float("nan"),
                "price_change_pct_1d": float("nan"),
            })
        return features

    def _sanitize_data(self, obj):
        """JSON保存用にNaNやnumpy型を整理"""
        if isinstance(obj, dict):
            return {k: self._sanitize_data(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_data(v) for v in obj]
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

    def create_results_directory(self):
        """結果保存用ディレクトリを作成"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def get_basic_predictions(self):
        """基本的な予測取得（二日比較分析なし）"""
        predictions_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S"),
            "markets": {},
            "market_context": self.global_context
        }
        self._print_line("🌏 日米主要銘柄 毎日予測レポート（シンプル版）", color=self.info_color)
        self._print_line("=" * 80)
        self._print_line(f"実行日時: {predictions_data['timestamp']}")
        self._print_line("=" * 80)
        
        for market, stocks in self.major_stocks.items():
            # 市場ごとに視認性の高いカラーでセクション見出しを表示
            market_header = self.market_color.get(market, None)
            self._print_line(f"\n📊 {market}", color=market_header)
            self._print_line("-" * 50)

            market_predictions = []
            success_count = 0

            for ticker, name in stocks.items():
                try:
                    # 現在の株価取得
                    stock = yf.Ticker(ticker)
                    data = stock.history(period='120d')
                    
                    if len(data) < 2:
                        self._print_line(f"❌ {name} ({ticker}): データ不足", color=self.alert_color)
                        continue
                    
                    # 特徴量を計算（継続学習のために保存）
                    feature_snapshot = self._build_feature_snapshot(data)
                    
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    daily_change = current_price - prev_price
                    daily_change_pct = (daily_change / prev_price) * 100
                    
                    # 移動平均を計算
                    ma20 = feature_snapshot.get("ma20")
                    ma50 = feature_snapshot.get("ma50")
                    if pd.isna(ma20):
                        ma20 = current_price
                    if pd.isna(ma50):
                        ma50 = current_price
                    
                    # シンプルな予測ロジック（移動平均ベース）
                    if current_price > ma20 and current_price > ma50:
                        # 強気シグナル
                        pred_change_pct = 1.5 + (daily_change_pct * 0.3)
                        trend_text = "強気"
                        trend_icon = "📈"
                    elif current_price < ma20 and current_price < ma50:
                        # 弱気シグナル
                        pred_change_pct = -1.5 + (daily_change_pct * 0.3)
                        trend_text = "弱気"
                        trend_icon = "📉"
                    else:
                        # 中立
                        pred_change_pct = daily_change_pct * 0.5
                        trend_text = "横ばい"
                        trend_icon = "➡️"
                    
                    # 予測価格を計算
                    pred_price = current_price * (1 + pred_change_pct / 100)
                    pred_change = pred_price - current_price
                    
                    # 通貨表示
                    currency = "$" if market == "米国市場" else "¥"
                    if market == "米国市場":
                        current_str = f"{currency}{current_price:.2f}"
                        pred_str = f"{currency}{pred_price:.2f}"
                    else:
                        current_str = f"{currency}{current_price:.0f}"
                        pred_str = f"{currency}{pred_price:.0f}"
                    
                    # 技術的指標
                    ma20_signal = "上" if current_price > ma20 else "下"
                    ma50_signal = "上" if current_price > ma50 else "下"
                    
                    # コンソール表示（ライト背景向けに視認性を高めた配色）
                    trend_color = self.trend_colors.get(trend_text, self.default_text)
                    self._print_line(f"{trend_icon} {name} ({ticker})", color=trend_color)
                    self._print_line(f"   現在価格: {current_str}")
                    self._print_line(f"   明日予測: {pred_str} ({trend_text})")
                    self._print_line(f"   予測変動: {pred_change:+.2f} ({pred_change_pct:+.2f}%)")
                    self._print_line(f"   前日比: {daily_change:+.2f} ({daily_change_pct:+.2f}%)")
                    self._print_line(f"   20日線: {ma20_signal} | 50日線: {ma50_signal}")
                    
                    # 警告表示
                    if abs(pred_change_pct) > 5:
                        self._print_line("   ⚠️ 大きな変動予測", color=self.alert_color)
                    elif abs(pred_change_pct) > 2:
                        self._print_line("   🔸 中程度の変動", color=self.warning_color)
                    else:
                        self._print_line("   ✅ 安定した予測")
                    
                    self._print_line()
                    
                    # データ保存用
                    stock_data = {
                        "ticker": ticker,
                        "name": name,
                        "current_price": current_price,
                        "predicted_price": pred_price,
                        "predicted_change": pred_change,
                        "predicted_change_pct": pred_change_pct,
                        "daily_change": daily_change,
                        "daily_change_pct": daily_change_pct,
                        "trend": trend_text,
                        "ma20": ma20,
                        "ma50": ma50,
                        "currency": currency,
                        "prediction_method": "technical_analysis",
                        "features": feature_snapshot,
                        "metadata": {
                            "data_quality": "complete" if all(not pd.isna(v) for v in feature_snapshot.values()) else "partial",
                            "generated_at": datetime.now().isoformat()
                        }
                    }
                    
                    market_predictions.append(stock_data)
                    success_count += 1
                        
                except Exception as e:
                    self._print_line(f"❌ {name} ({ticker}): エラー - {str(e)[:50]}...", color=self.alert_color)
                    self._print_line()
            
            predictions_data["markets"][market] = market_predictions
            self._print_line(f"✅ {market}: {success_count}/{len(stocks)} 銘柄の予測成功", color=self.info_color)
        
        # 統計サマリー
        self.print_summary(predictions_data)
        
        # 結果を保存
        self.save_results(predictions_data)
        
        return predictions_data
    
    def print_summary(self, data):
        """サマリー表示"""
        self._print_line("\n" + "=" * 80)
        self._print_line("📈 予測サマリー", color=self.info_color)
        self._print_line("=" * 80)
        
        total_stocks = 0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for market, stocks in data["markets"].items():
            market_stocks = len(stocks)
            total_stocks += market_stocks
            
            if market_stocks > 0:
                market_bullish = len([s for s in stocks if s["predicted_change_pct"] > 1])
                market_bearish = len([s for s in stocks if s["predicted_change_pct"] < -1])
                market_neutral = len([s for s in stocks if abs(s["predicted_change_pct"]) <= 1])
                
                bullish_count += market_bullish
                bearish_count += market_bearish
                neutral_count += market_neutral
                
                avg_change = sum(s["predicted_change_pct"] for s in stocks) / market_stocks
                
                self._print_line(f"\n{market}:")
                self._print_line(f"  分析銘柄数: {market_stocks}")
                self._print_line(f"  強気予測: {market_bullish} | 弱気予測: {market_bearish} | 横ばい: {market_neutral}")
                self._print_line(f"  平均予測変動: {avg_change:+.2f}%")
        
        if total_stocks > 0:
            self._print_line("\n全体市場:")
            self._print_line(f"  総銘柄数: {total_stocks}")
            self._print_line(f"  強気: {bullish_count} ({bullish_count/total_stocks*100:.1f}%)")
            self._print_line(f"  弱気: {bearish_count} ({bearish_count/total_stocks*100:.1f}%)")
            self._print_line(f"  横ばい: {neutral_count} ({neutral_count/total_stocks*100:.1f}%)")
            
            # 市場センチメント
            if bearish_count > bullish_count * 1.5:
                sentiment = "🐻 弱気市場"
            elif bullish_count > bearish_count * 1.5:
                sentiment = "🐂 強気市場"
            else:
                sentiment = "⚖️ 中立市場"
            
            self._print_line(f"  市場センチメント: {sentiment}", color=self.info_color)
            
            self._print_line("\n💡 投資アドバイス:", color=self.info_color)
            if bearish_count > total_stocks * 0.6:
                self._print_line("  • 市場全体に弱気サイン。リスク管理を重視しましょう")
                self._print_line("  • 利確や損切りを検討する良いタイミングかもしれません")
            elif bullish_count > total_stocks * 0.6:
                self._print_line("  • 市場全体に強気サイン。新規投資のチャンスかもしれません")
                self._print_line("  • ただし、過熱しすぎていないか注意も必要です")
            else:
                self._print_line("  • 市場は混迷。銘柄選別が重要です")
                self._print_line("  • 業績の良い銘柄の割安なタイミングを狙いましょう")
            
            self._print_line("\n📊 予測方法:", color=self.info_color)
            self._print_line("  • 移動平均線（20日・50日）ベースの技術分析")
            self._print_line("  • 前日の値動きを考慮した予測")
            self._print_line("  • AI学習モデルより安定した動作")
        
        self._print_line("=" * 80)
    
    def save_results(self, data):
        """予測結果をファイルに保存"""
        sanitized = self._sanitize_data(data)
        filename = f"{self.results_dir}/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sanitized, f, ensure_ascii=False, indent=2)
        
        self._print_line(f"\n💾 結果を保存しました: {filename}", color=self.info_color)
        
        # 最新結果のコピーも保存
        latest_file = f"{self.results_dir}/latest_predictions.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(sanitized, f, ensure_ascii=False, indent=2)

        # 継続学習用のデータ管理にも保存
        try:
            self.data_manager.save_daily_prediction(sanitized)
        except Exception as e:
            self._print_line(f"⚠️ データマネージャへの保存に失敗: {e}", color=self.alert_color)
    
    def run_scheduled_predictions(self):
        """スケジュール実行"""
        self._print_line("🤖 毎日予測アプリ スケジューラー起動", color=self.info_color)
        self._print_line("=" * 50)
        
        # 毎朝8時に実行
        schedule.every().day.at("08:00").do(self.get_basic_predictions)
        
        # 毎夕17時に実行（市場終了後）
        schedule.every().day.at("17:00").do(self.get_basic_predictions)
        
        self._print_line("⏰ スケジュール設定:", color=self.info_color)
        self._print_line("  • 毎朝 08:00 (市場開始前)")
        self._print_line("  • 毎夕 17:00 (市場終了後)")
        self._print_line("\nCtrl+Cで停止できます")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1分ごとにチェック
    
    def run_once(self):
        """一度だけ実行"""
        return self.get_basic_predictions()

def main():
    """メイン実行"""
    app = SimpleDailyPredictionApp()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":
        # スケジュール実行モード
        app.run_scheduled_predictions()
    else:
        # 一度だけ実行モード
        app.run_once()

if __name__ == "__main__":
    main()
