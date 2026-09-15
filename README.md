# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-15T14:18:54.250001Z**
- Expanding time-series validation accuracy: **49.4%**
- Persistence baseline: **50.4%**
- Live 90-day direction accuracy: **50.0%** (74 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-15 | UP | +0.05% | 52.4% | 48.5% | 53.0% | HOLD |
| 6861.T | 2026-09-15 | DOWN | -0.26% | 58.5% | 53.5% | 49.5% | HOLD |
| 7203.T | 2026-09-15 | UP | +0.05% | 52.6% | 49.5% | 53.5% | HOLD |
| 8035.T | 2026-09-15 | DOWN | -0.13% | 58.0% | 47.5% | 48.0% | HOLD |
| 9984.T | 2026-09-15 | DOWN | -0.29% | 52.6% | 47.0% | 50.0% | HOLD |
| AAPL | 2026-09-14 | DOWN | -0.05% | 55.0% | 52.5% | 48.0% | HOLD |
| GOOGL | 2026-09-14 | UP | +0.28% | 54.9% | 50.5% | 49.0% | HOLD |
| MSFT | 2026-09-14 | UP | +0.07% | 50.2% | 47.0% | 52.0% | HOLD |
| NVDA | 2026-09-14 | DOWN | -0.22% | 54.4% | 46.5% | 50.5% | HOLD |
| TSLA | 2026-09-14 | UP | +0.05% | 52.2% | 51.5% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
