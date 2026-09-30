# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-30T15:53:57.556262Z**
- Expanding time-series validation accuracy: **49.0%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **43.2%** (169 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-30 | UP | +0.09% | 54.2% | 52.5% | 53.0% | HOLD |
| 6861.T | 2026-09-30 | DOWN | -0.07% | 52.0% | 51.0% | 48.5% | HOLD |
| 7203.T | 2026-09-30 | DOWN | -0.25% | 59.4% | 43.5% | 52.5% | HOLD |
| 8035.T | 2026-09-30 | UP | +0.29% | 55.9% | 51.5% | 47.5% | HOLD |
| 9984.T | 2026-09-30 | DOWN | -0.50% | 58.1% | 47.5% | 48.5% | HOLD |
| AAPL | 2026-09-29 | DOWN | -0.05% | 52.7% | 52.0% | 45.5% | HOLD |
| GOOGL | 2026-09-29 | UP | +0.19% | 62.4% | 48.0% | 50.5% | HOLD |
| MSFT | 2026-09-29 | DOWN | -0.11% | 57.8% | 45.5% | 50.0% | HOLD |
| NVDA | 2026-09-29 | UP | +0.63% | 67.8% | 47.5% | 53.0% | HOLD |
| TSLA | 2026-09-29 | DOWN | -1.05% | 68.0% | 51.5% | 50.5% | SELL |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
