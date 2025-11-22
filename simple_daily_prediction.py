"""
NOROSHI Prediction System - Polygon.io API Edition
Production-grade stock price prediction using LightGBM
Fully GitHub Actions compatible
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime, timedelta
import warnings
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

warnings.filterwarnings('ignore')

# 設定
TICKERS_US = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
TICKERS_JP = ["7203.T", "6758.T", "9984.T", "6861.T", "8035.T"]
OUTPUT_DIR = "./data"
ANALYTICS_DIR = "./analytics"
MODEL_DIR = "./models"
CACHE_DIR = "./data/cache"

# Polygon.io API Key（GitHub Secretsから取得）
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '').strip()

# JPFUNDS API Key（日本株用 - 無料API）
JPFUNDS_API_KEY = os.environ.get('JPFUNDS_API_KEY', '').strip()

# ディレクトリ作成
for d in [OUTPUT_DIR, ANALYTICS_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# デバッグ用
if not POLYGON_API_KEY:
    print("WARNING: POLYGON_API_KEY not set")
if not JPFUNDS_API_KEY:
    print("WARNING: JPFUNDS_API_KEY not set (Japanese stocks will use cache only)")


def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def create_features(df):
    """Create comprehensive technical features"""
    try:
        # 列名を統一（大文字に）
        df.columns = df.columns.str.upper()
        
        # 必須列の確認
        required_cols = ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        close = df['CLOSE'].astype(float)
        high = df['HIGH'].astype(float)
        low = df['LOW'].astype(float)
        volume = df['VOLUME'].astype(float)
        
        # ===== 価格系指標 =====
        df['RETURN_1D'] = close.pct_change()
        df['RETURN_2D'] = close.pct_change(2)
        df['RETURN_5D'] = close.pct_change(5)
        df['PRICE_CHANGE'] = close - close.shift(1)
        
        # ===== 移動平均系 =====
        for period in [5, 10, 20, 50]:
            df[f'MA{period}'] = close.rolling(period).mean()
            df[f'EMA{period}'] = close.ewm(span=period).mean()
        
        # 移動平均からの乖離
        df['PRICE_TO_MA5'] = close / df['MA5']
        df['PRICE_TO_MA20'] = close / df['MA20']
        df['MA5_MA20_RATIO'] = df['MA5'] / df['MA20']
        
        # ===== ボラティリティ系 =====
        df['VOLATILITY_5D'] = df['RETURN_1D'].rolling(5).std()
        df['VOLATILITY_20D'] = df['RETURN_1D'].rolling(20).std()
        df['LOG_RETURN'] = np.log(close / close.shift(1))
        
        # ===== 強度指標 =====
        df['RSI14'] = calculate_rsi(close, 14)
        df['RSI7'] = calculate_rsi(close, 7)
        
        # ===== MACD =====
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = calculate_macd(close)
        
        # ===== ボリンジャーバンド =====
        upper, sma, lower = calculate_bollinger_bands(close, 20, 2)
        df['BB_UPPER'] = upper
        df['BB_MIDDLE'] = sma
        df['BB_LOWER'] = lower
        df['BB_WIDTH'] = upper - lower
        df['BB_POSITION'] = (close - lower) / (upper - lower)
        
        # ===== 出来高系 =====
        df['VOLUME_MA20'] = volume.rolling(20).mean()
        df['VOLUME_RATIO'] = volume / df['VOLUME_MA20']
        df['ADL'] = ((close - low) - (high - close)) / (high - low) * volume
        df['ADL_MA'] = df['ADL'].rolling(20).mean()
        
        # ===== トレンド系 =====
        df['HIGH_LOW_RATIO'] = high / low
        df['CLOSE_OPEN_RATIO'] = close / df['OPEN'].astype(float)
        
        # ===== 高値安値からの距離 =====
        rolling_high = close.rolling(20).max()
        rolling_low = close.rolling(20).min()
        df['CLOSE_TO_HIGH'] = (close - rolling_low) / (rolling_high - rolling_low)
        
        # ===== 前日との比較 =====
        df['HIGHER_THAN_YESTERDAY'] = (close > close.shift(1)).astype(int)
        df['CLOSE_HIGHER_THAN_OPEN'] = (close > df['OPEN'].astype(float)).astype(int)
        
        return df
    
    except Exception as e:
        print(f"Error in create_features: {e}")
        return None


def prepare_training_data(df, target_periods=1):
    """Prepare training data"""
    try:
        df = df.dropna()
        
        if len(df) < 50:
            return None, None, None
        
        # ターゲット変数を作成
        df['TARGET'] = (df['CLOSE'].shift(-target_periods) > df['CLOSE']).astype(int)
        df = df[df['TARGET'].notna()].copy()
        
        # 特徴量を選択
        feature_cols = [col for col in df.columns 
                       if col not in ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME', 
                                     'DATE', 'TARGET', 'RETURN_1D', 'RETURN_2D', 
                                     'RETURN_5D', 'LOG_RETURN', 'PRICE_CHANGE']]
        
        X = df[feature_cols].fillna(0)
        y = df['TARGET']
        
        return X, y, feature_cols
    
    except Exception as e:
        print(f"Error preparing training data: {e}")
        return None, None, None


def train_or_load_model(ticker, df):
    """Train LightGBM model or load existing"""
    try:
        model_path = f"{MODEL_DIR}/lgb_model_{ticker}.pkl"
        
        X, y, feature_cols = prepare_training_data(df)
        
        if X is None or len(X) < 30:
            return None, None, None
        
        # モデルが存在する場合は読み込み
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)
            model = model_info['model']
            feature_cols = model_info['features']
            scaler = model_info['scaler']
            return model, feature_cols, scaler
        
        # 新規モデル訓練
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        model_info = {
            'model': model,
            'features': feature_cols,
            'scaler': scaler,
            'accuracy': model.score(X_test, y_test)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"  Model trained for {ticker} (Accuracy: {model_info['accuracy']:.4f})")
        
        return model, feature_cols, scaler
    
    except Exception as e:
        print(f"Error in model training for {ticker}: {e}")
        return None, None, None


def fetch_polygon_io(symbol, api_key):
    """Fetch data from Polygon.io API (US stocks)"""
    if not api_key or len(api_key) < 10:
        return None
    
    try:
        # 180日前の日付を計算
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "apikey": api_key,
            "limit": 50000,
            "sort": "asc"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # ステータスチェック
        if not data.get('status') or data['status'] != 'OK':
            print(f"    API Error: {data.get('status')}")
            return None
        
        results = data.get('results', [])
        if not results:
            print(f"    No data returned")
            return None
        
        # DataFrameに変換
        records = []
        for bar in results:
            try:
                # タイムスタンプはミリ秒で返される
                date = pd.to_datetime(bar['t'], unit='ms')
                records.append({
                    'Date': date,
                    'Open': float(bar.get('o', 0)),
                    'High': float(bar.get('h', 0)),
                    'Low': float(bar.get('l', 0)),
                    'Close': float(bar.get('c', 0)),
                    'Volume': float(bar.get('v', 0))
                })
            except (KeyError, ValueError) as e:
                continue
        
        if not records:
            print(f"    No valid records found")
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"    ✓ Got {len(df)} days of data")
        return df
    
    except requests.exceptions.RequestException as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            print(f"    API Key invalid or expired")
        elif "429" in str(e):
            print(f"    Rate limited")
        else:
            print(f"    Network error: {str(e)[:40]}")
        return None
    except Exception as e:
        print(f"    Error: {str(e)[:50]}")
        return None


def fetch_japan_stocks_api(symbol, api_key):
    """
    Fetch Japanese stock data from JPFUNDS API
    Symbol format: "7203.T" -> use "7203" for API
    """
    if not api_key or len(api_key) < 10:
        return None
    
    try:
        # シンボルを処理 (例: "7203.T" -> "7203")
        symbol_code = symbol.split('.')[0] if '.' in symbol else symbol
        
        # JPFUNDS APIエンドポイント
        url = f"https://jpfunds.com/api/json/stocks/{symbol_code}/daily"
        params = {
            "apikey": api_key,
            "days": 180
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # レスポンス形式チェック
        if not isinstance(data, list) or len(data) == 0:
            print(f"    No data returned")
            return None
        
        # DataFrameに変換
        records = []
        for item in data:
            try:
                records.append({
                    'Date': pd.to_datetime(item['date']),
                    'Open': float(item.get('open', 0)),
                    'High': float(item.get('high', 0)),
                    'Low': float(item.get('low', 0)),
                    'Close': float(item.get('close', 0)),
                    'Volume': float(item.get('volume', 0))
                })
            except (KeyError, ValueError):
                continue
        
        if not records:
            print(f"    No valid records found")
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"    ✓ Got {len(df)} days of data")
        return df
    
    except requests.exceptions.RequestException as e:
        if "401" in str(e):
            print(f"    API Key invalid")
        elif "429" in str(e):
            print(f"    Rate limited")
        else:
            print(f"    Network error: {str(e)[:40]}")
        return None
    except Exception as e:
        print(f"    Error: {str(e)[:50]}")
        return None


def fetch_data_with_fallback(ticker, market_type='us', polygon_key='', japan_key=''):
    """
    Fetch data based on market type
    US: Polygon.io API
    JP: JPFUNDS API
    Both fallback to cache
    """
    # Step 1: API から取得を試みる
    if market_type == 'us' and polygon_key and len(polygon_key) > 10:
        df_api = fetch_polygon_io(ticker, polygon_key)
        if df_api is not None and not df_api.empty:
            # キャッシュに保存
            cache_file = f"{CACHE_DIR}/{ticker}.csv"
            df_api.to_csv(cache_file, index=False)
            return df_api
    
    elif market_type == 'jp' and japan_key and len(japan_key) > 10:
        df_api = fetch_japan_stocks_api(ticker, japan_key)
        if df_api is not None and not df_api.empty:
            # キャッシュに保存
            cache_file = f"{CACHE_DIR}/{ticker}.csv"
            df_api.to_csv(cache_file, index=False)
            return df_api
    
    # Step 2: キャッシュから読み込み
    cache_file = f"{CACHE_DIR}/{ticker}.csv"
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['Date'])
            print(f"    ✓ Using cache")
            return df
        except:
            return None
    
    return None


def predict_ticker(ticker, market_type='us', polygon_key='', japan_key=''):
    """Predict for a single ticker using LightGBM"""
    try:
        # データ取得（APIまたはキャッシュ）
        df = fetch_data_with_fallback(ticker, market_type, polygon_key, japan_key)
        
        if df is None or df.empty:
            return None
        
        # 特徴量作成
        df_features = create_features(df.copy())
        if df_features is None or df_features.empty:
            return None
        
        # モデル訓練/読み込み
        model, feature_cols, scaler = train_or_load_model(ticker, df_features)
        if model is None:
            return None
        
        # 最新データ取得
        latest_row = df_features.iloc[-1].to_dict()
        
        # 特徴量抽出
        X_latest = pd.DataFrame([latest_row])[feature_cols].fillna(0)
        X_latest_scaled = scaler.transform(X_latest)
        
        # 予測
        pred_proba = model.predict_proba(X_latest_scaled)[0]
        pred_class = model.predict(X_latest_scaled)[0]
        
        close_price = float(latest_row.get('CLOSE', 0))
        trend_direction = '強気' if pred_class == 1 else '弱気'
        confidence = float(max(pred_proba))
        
        volatility = float(latest_row.get('VOLATILITY_5D', 0.01))
        predicted_change = (volatility * 0.5) if pred_class == 1 else -(volatility * 0.5)
        predicted_price = close_price * (1 + predicted_change)
        
        result = {
            'ticker': ticker,
            'timestamp': datetime.utcnow().isoformat(),
            'predicted_price': float(predicted_price),
            'features': {
                'close': close_price,
                'ma_5': float(latest_row.get('MA5', 0)),
                'ma_20': float(latest_row.get('MA20', 0)),
                'return_1d': float(latest_row.get('RETURN_1D', 0)),
                'volatility_5d': volatility,
                'rsi14': float(latest_row.get('RSI14', 50)),
                'macd': float(latest_row.get('MACD', 0)),
                'macd_hist': float(latest_row.get('MACD_HIST', 0)),
                'as_of': str(latest_row.get('Date', datetime.now())),
                'volume': float(latest_row.get('VOLUME', 0))
            },
            'prediction_details': {
                'direction': trend_direction,
                'confidence': confidence,
                'predicted_change_pct': predicted_change * 100
            },
            'status': 'completed'
        }
        
        return result
    
    except Exception as e:
        print(f"Error predicting {ticker}: {e}")
        return None


def fetch_market_context(api_key):
    """Fetch market context (SPY)"""
    try:
        df = fetch_polygon_io("SPY", api_key)
        
        if df is None or df.empty or len(df) < 2:
            return None
        
        df = df.sort_values('Date')
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        spy_close = float(latest['Close'])
        spy_change = float((latest['Close'] - prev['Close']) / prev['Close'] * 100)
        
        return {
            'spy_close': spy_close,
            'spy_change_pct': spy_change
        }
    
    except Exception as e:
        print(f"Error fetching market context: {e}")
        return None


def main():
    """Main execution"""
    print("=" * 60)
    print("NOROSHI Prediction System - Multi-API Edition")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    predictions = []
    
    # US市場予測（Polygon.io）
    print("\n[US Market - Polygon.io]")
    for ticker in TICKERS_US:
        print(f"Processing {ticker}...", end=" ")
        pred = predict_ticker(ticker, 'us', POLYGON_API_KEY, '')
        if pred:
            predictions.append(pred)
            print(f"✓ ({pred['prediction_details']['direction']})")
        else:
            print("✗")
        
        time.sleep(0.2)
    
    # 日本市場予測（JPFUNDS API）
    print("\n[Japanese Market - JPFUNDS API]")
    for ticker in TICKERS_JP:
        print(f"Processing {ticker}...", end=" ")
        pred = predict_ticker(ticker, 'jp', '', JPFUNDS_API_KEY)
        if pred:
            predictions.append(pred)
            print(f"✓ ({pred['prediction_details']['direction']})")
        else:
            print("✗")
        
        time.sleep(0.2)
    
    # 市場コンテキスト（SPY - US株）
    print("\n[Market Context]")
    market_ctx = fetch_market_context(POLYGON_API_KEY)
    if market_ctx:
        print(f"SPY: {market_ctx['spy_close']:.2f} ({market_ctx['spy_change_pct']:+.2f}%)")
    else:
        print("✗ Market context unavailable")
    
    # 結果保存
    if predictions:
        output_file = f"{OUTPUT_DIR}/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\n✓ Saved {len(predictions)} predictions to {output_file}")
    else:
        print("\n✗ No predictions generated")
    
    print("\n" + "=" * 60)
    print("Daily run complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
