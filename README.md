# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-28T17:37:44.461201Z**
- Expanding time-series validation accuracy: **50.0%**
- Persistence baseline: **50.2%**
- Live 90-day direction accuracy: **45.6%** (149 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-28 | UP | +0.31% | 68.8% | 50.0% | 53.0% | HOLD |
| 6861.T | 2026-09-28 | DOWN | -0.05% | 52.5% | 52.0% | 49.0% | HOLD |
| 7203.T | 2026-09-28 | UP | +0.25% | 68.7% | 45.0% | 53.5% | HOLD |
| 8035.T | 2026-09-28 | UP | +0.43% | 61.6% | 48.5% | 48.5% | HOLD |
| 9984.T | 2026-09-28 | UP | +1.38% | 70.0% | 47.5% | 49.5% | HOLD |
| AAPL | 2026-09-25 | UP | +0.29% | 60.3% | 56.5% | 45.5% | BUY |
| GOOGL | 2026-09-25 | UP | +0.44% | 67.0% | 53.5% | 50.5% | BUY |
| MSFT | 2026-09-25 | UP | +0.26% | 57.2% | 49.0% | 50.0% | HOLD |
| NVDA | 2026-09-25 | UP | +0.66% | 63.3% | 47.0% | 53.0% | HOLD |
| TSLA | 2026-09-25 | UP | +0.86% | 61.6% | 50.5% | 50.0% | BUY |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
