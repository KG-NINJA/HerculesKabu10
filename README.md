# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-28T15:58:40.952569Z**
- Expanding time-series validation accuracy: **50.5%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **48.8%** (363 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-28 | DOWN | -0.29% | 60.2% | 52.0% | 55.0% | HOLD |
| 6861.T | 2026-10-28 | DOWN | -0.11% | 51.6% | 52.0% | 49.0% | HOLD |
| 7203.T | 2026-10-28 | DOWN | -0.05% | 52.5% | 46.5% | 54.0% | HOLD |
| 8035.T | 2026-10-28 | DOWN | -0.74% | 64.5% | 49.5% | 46.5% | HOLD |
| 9984.T | 2026-10-28 | UP | +0.75% | 59.1% | 49.0% | 48.0% | HOLD |
| AAPL | 2026-10-27 | UP | +0.16% | 54.0% | 51.5% | 45.0% | HOLD |
| GOOGL | 2026-10-27 | DOWN | -0.23% | 62.6% | 49.0% | 49.0% | HOLD |
| MSFT | 2026-10-27 | DOWN | -0.25% | 65.4% | 47.0% | 49.5% | HOLD |
| NVDA | 2026-10-27 | UP | +0.36% | 53.6% | 51.5% | 53.5% | HOLD |
| TSLA | 2026-10-27 | UP | +0.16% | 57.5% | 57.5% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
