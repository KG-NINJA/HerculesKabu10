# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-26T16:05:21.992777Z**
- Expanding time-series validation accuracy: **49.0%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **48.1%** (343 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-26 | UP | +0.42% | 64.3% | 50.5% | 54.5% | HOLD |
| 6861.T | 2026-10-26 | UP | +0.07% | 51.9% | 51.5% | 49.0% | HOLD |
| 7203.T | 2026-10-26 | UP | +0.05% | 52.0% | 47.0% | 54.0% | HOLD |
| 8035.T | 2026-10-26 | UP | +1.40% | 72.9% | 51.5% | 46.5% | BUY |
| 9984.T | 2026-10-26 | DOWN | -0.13% | 54.5% | 45.5% | 48.0% | HOLD |
| AAPL | 2026-10-23 | DOWN | -0.07% | 52.2% | 52.0% | 45.0% | HOLD |
| GOOGL | 2026-10-23 | DOWN | -0.16% | 59.9% | 50.0% | 49.5% | HOLD |
| MSFT | 2026-10-23 | DOWN | -0.37% | 70.2% | 45.0% | 49.0% | HOLD |
| NVDA | 2026-10-23 | DOWN | -0.13% | 53.5% | 48.5% | 53.5% | HOLD |
| TSLA | 2026-10-23 | UP | +0.09% | 51.4% | 49.0% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
