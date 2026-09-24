# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-24T14:11:48.835673Z**
- Expanding time-series validation accuracy: **49.6%**
- Persistence baseline: **50.4%**
- Live 90-day direction accuracy: **45.0%** (129 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-24 | UP | +0.05% | 50.3% | 53.5% | 53.5% | HOLD |
| 6861.T | 2026-09-24 | DOWN | -0.05% | 51.4% | 52.0% | 49.5% | HOLD |
| 7203.T | 2026-09-24 | UP | +0.15% | 53.4% | 49.0% | 53.5% | HOLD |
| 8035.T | 2026-09-24 | DOWN | -0.25% | 58.6% | 49.0% | 48.0% | HOLD |
| 9984.T | 2026-09-24 | UP | +0.56% | 55.9% | 47.5% | 50.0% | HOLD |
| AAPL | 2026-09-23 | UP | +0.27% | 57.4% | 55.5% | 46.0% | HOLD |
| GOOGL | 2026-09-23 | DOWN | -0.53% | 75.8% | 49.5% | 50.0% | HOLD |
| MSFT | 2026-09-23 | UP | +0.46% | 73.6% | 48.5% | 51.0% | HOLD |
| NVDA | 2026-09-23 | DOWN | -0.42% | 58.1% | 47.0% | 52.5% | HOLD |
| TSLA | 2026-09-23 | DOWN | -1.31% | 74.4% | 45.0% | 50.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
