# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-13T16:55:35.460223Z**
- Expanding time-series validation accuracy: **49.5%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **45.5%** (253 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-13 | UP | +0.50% | 66.6% | 53.5% | 54.0% | HOLD |
| 6861.T | 2026-10-13 | UP | +0.05% | 55.0% | 52.5% | 48.5% | HOLD |
| 7203.T | 2026-10-13 | DOWN | -0.20% | 57.0% | 45.0% | 53.0% | HOLD |
| 8035.T | 2026-10-13 | UP | +0.51% | 53.3% | 49.0% | 47.0% | HOLD |
| 9984.T | 2026-10-13 | UP | +1.67% | 67.1% | 47.0% | 48.5% | HOLD |
| AAPL | 2026-10-12 | UP | +0.10% | 58.5% | 52.0% | 46.0% | HOLD |
| GOOGL | 2026-10-12 | DOWN | -0.15% | 50.2% | 49.0% | 49.0% | HOLD |
| MSFT | 2026-10-12 | DOWN | -0.40% | 67.8% | 42.5% | 51.0% | HOLD |
| NVDA | 2026-10-12 | DOWN | -0.44% | 68.5% | 52.5% | 53.5% | HOLD |
| TSLA | 2026-10-12 | DOWN | -0.83% | 74.6% | 52.5% | 51.0% | SELL |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
