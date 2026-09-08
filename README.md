# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-08T13:34:21.182721Z**
- Expanding time-series validation accuracy: **50.6%**
- Persistence baseline: **49.9%**
- Live 90-day direction accuracy: **52.0%** (25 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-08 | DOWN | -0.90% | 74.1% | 53.5% | 53.5% | SELL |
| 6861.T | 2026-09-08 | UP | +0.53% | 64.4% | 49.0% | 49.5% | HOLD |
| 7203.T | 2026-09-08 | UP | +0.36% | 62.0% | 45.5% | 53.5% | HOLD |
| 8035.T | 2026-09-08 | UP | +0.75% | 61.6% | 55.5% | 46.0% | BUY |
| 9984.T | 2026-09-08 | UP | +1.62% | 63.2% | 51.0% | 49.5% | BUY |
| AAPL | 2026-09-04 | DOWN | -0.14% | 58.1% | 52.5% | 47.0% | HOLD |
| GOOGL | 2026-09-04 | UP | +0.17% | 52.7% | 50.5% | 47.5% | HOLD |
| MSFT | 2026-09-04 | UP | +0.12% | 53.4% | 45.5% | 52.0% | HOLD |
| NVDA | 2026-09-04 | UP | +1.01% | 69.5% | 49.0% | 49.5% | HOLD |
| TSLA | 2026-09-04 | UP | +0.72% | 67.5% | 54.0% | 50.5% | BUY |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
