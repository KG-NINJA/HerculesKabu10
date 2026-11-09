# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-09T19:45:18.251400Z**
- Expanding time-series validation accuracy: **48.5%**
- Persistence baseline: **49.6%**
- Live 90-day direction accuracy: **48.2%** (438 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-09 | DOWN | -0.90% | 74.4% | 54.5% | 53.5% | SELL |
| 6861.T | 2026-11-09 | UP | +0.05% | 51.5% | 51.5% | 48.0% | HOLD |
| 7203.T | 2026-11-09 | DOWN | -0.98% | 69.8% | 44.0% | 53.0% | HOLD |
| 8035.T | 2026-11-09 | UP | +1.32% | 66.8% | 52.0% | 46.0% | BUY |
| 9984.T | 2026-11-09 | UP | +0.60% | 56.0% | 43.0% | 49.5% | HOLD |
| AAPL | 2026-11-06 | UP | +0.13% | 54.6% | 48.0% | 43.5% | HOLD |
| GOOGL | 2026-11-06 | UP | +0.62% | 64.1% | 48.0% | 48.0% | HOLD |
| MSFT | 2026-11-06 | DOWN | -0.54% | 65.0% | 45.0% | 50.5% | HOLD |
| NVDA | 2026-11-06 | DOWN | -0.28% | 52.0% | 48.0% | 54.0% | HOLD |
| TSLA | 2026-11-06 | UP | +0.65% | 60.9% | 51.0% | 50.0% | BUY |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
