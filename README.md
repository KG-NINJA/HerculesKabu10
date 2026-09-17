# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-17T14:16:36.041596Z**
- Expanding time-series validation accuracy: **49.0%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **47.9%** (94 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-17 | UP | +0.17% | 60.2% | 53.0% | 53.0% | BUY |
| 6861.T | 2026-09-17 | DOWN | -0.16% | 55.0% | 51.0% | 49.0% | HOLD |
| 7203.T | 2026-09-17 | DOWN | -0.12% | 56.5% | 46.0% | 53.0% | HOLD |
| 8035.T | 2026-09-17 | UP | +0.51% | 68.8% | 49.5% | 48.0% | HOLD |
| 9984.T | 2026-09-17 | UP | +0.53% | 59.3% | 47.0% | 49.5% | HOLD |
| AAPL | 2026-09-16 | DOWN | -0.05% | 53.1% | 51.0% | 47.0% | HOLD |
| GOOGL | 2026-09-16 | DOWN | -0.05% | 51.1% | 49.0% | 49.0% | HOLD |
| MSFT | 2026-09-16 | DOWN | -0.43% | 70.9% | 45.0% | 51.5% | HOLD |
| NVDA | 2026-09-16 | UP | +0.05% | 50.9% | 48.5% | 51.0% | HOLD |
| TSLA | 2026-09-16 | DOWN | -0.24% | 55.4% | 50.5% | 50.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
