# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-25T19:11:48.321664Z**
- Expanding time-series validation accuracy: **49.5%**
- Persistence baseline: **49.2%**
- Live 90-day direction accuracy: **48.4%** (548 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-25 | DOWN | -1.27% | 75.1% | 54.5% | 51.5% | SELL |
| 6861.T | 2026-11-25 | DOWN | -0.37% | 56.5% | 53.5% | 48.0% | HOLD |
| 7203.T | 2026-11-25 | DOWN | -0.08% | 50.0% | 46.5% | 52.5% | HOLD |
| 8035.T | 2026-11-25 | UP | +1.61% | 71.5% | 48.0% | 46.5% | HOLD |
| 9984.T | 2026-11-25 | UP | +2.28% | 67.1% | 46.5% | 48.5% | HOLD |
| AAPL | 2026-11-24 | UP | +0.30% | 59.3% | 49.5% | 43.5% | HOLD |
| GOOGL | 2026-11-24 | DOWN | -0.82% | 66.8% | 47.0% | 47.5% | HOLD |
| MSFT | 2026-11-24 | UP | +0.72% | 70.3% | 47.0% | 49.0% | HOLD |
| NVDA | 2026-11-24 | UP | +0.07% | 50.2% | 47.5% | 53.5% | HOLD |
| TSLA | 2026-11-24 | DOWN | -0.76% | 63.2% | 55.0% | 51.5% | SELL |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
