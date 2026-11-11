# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-11T17:04:20.432955Z**
- Expanding time-series validation accuracy: **50.3%**
- Persistence baseline: **49.5%**
- Live 90-day direction accuracy: **48.9%** (458 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-11 | UP | +0.05% | 52.7% | 52.0% | 53.5% | HOLD |
| 6861.T | 2026-11-11 | UP | +0.55% | 66.8% | 54.5% | 48.0% | BUY |
| 7203.T | 2026-11-11 | UP | +0.09% | 52.0% | 48.0% | 53.0% | HOLD |
| 8035.T | 2026-11-11 | DOWN | -0.16% | 57.0% | 50.0% | 46.0% | HOLD |
| 9984.T | 2026-11-11 | DOWN | -0.33% | 51.1% | 46.5% | 49.0% | HOLD |
| AAPL | 2026-11-10 | UP | +0.41% | 63.0% | 54.0% | 43.5% | BUY |
| GOOGL | 2026-11-10 | UP | +0.73% | 60.6% | 46.0% | 48.5% | HOLD |
| MSFT | 2026-11-10 | DOWN | -0.45% | 59.1% | 44.5% | 49.5% | HOLD |
| NVDA | 2026-11-10 | UP | +0.68% | 55.6% | 53.0% | 54.0% | HOLD |
| TSLA | 2026-11-10 | DOWN | -0.52% | 58.2% | 55.0% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
