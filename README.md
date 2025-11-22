# HerculesKabu10
# 📈 Stock Auto Predictor  
AI-driven Daily Market Forecast System  
Powered by GitHub Actions ＋ Python  
#KGNINJA

---

## 🚀 Overview  
This repository automatically executes two prediction systems every day:

1. **Daily Stock Prediction (US + JP markets)**  
   - SMA-based technical analysis  
   - RSI / MACD / Trend extraction  
   - Market context (SPY, VIX)  
   - Results saved to /daily_predictions/

2. **NVDA Direction CV Evaluator**  
   - Multi-registry CV aggregation  
   - Validation-based MAPE scoring  
   - Trend-based Signal calculation  
   - Confidence score generation  
   - Logs stored in /logs/

All predictions are executed on:
- **08:00 JST**
- **17:00 JST**
via GitHub Actions.

---

## 📊 GitHub Pages: Prediction Dashboard  
The dashboard visualizes:

- Daily prediction results  
- NVDA confidence trends  
- BUY / SELL signal history  
- Market heatmaps  
- AI accuracy & backtesting logs  

GitHub Pages URL:  
(Your URL will appear here after enabling Pages)

---

## 📁 Repository Structure
```
stock-autopredictor/
│
├── simple_daily_prediction.py
├── nvda_direction_codex_runner.py
│
├── daily_predictions/    # Daily market forecasts
├── logs/                 # CV runner logs
│
└── .github/
       └── workflows/
             └── daily-stock.yml
```

---

## ⚙️ Automation Workflow
GitHub Actions automatically:

1. Runs daily predictions  
2. Saves output JSON  
3. Generates trend graphs  
4. Updates dashboard  
5. Commits & pushes results  

---

## 📘 Dashboard Preview

- NVDA Confidence Score Trend  
- Daily Market Sentiment Graph  
- BUY/SELL Heatmap  
- Price Prediction Trendline  
- SPY/VIX Macro Context Panel  

---

## 🧪 Backtesting & Research  
All historical data is preserved for:

- Feature extraction  
- Weight optimization  
- Market behavior analysis  
- AIEO (AI Existence Observation) research

---

## 🧱 Tech Stack  
- Python 3.10  
- yfinance  
- pandas  
- matplotlib  
- GitHub Actions  
- GitHub Pages  
- JSON structured logs  

---

## © KGNINJA  
Autonomous AI trading research.  
Always evolving via continuous logs and signals.

