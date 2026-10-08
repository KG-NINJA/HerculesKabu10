# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-08T16:26:07.630647Z**
- Expanding time-series validation accuracy: **50.0%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **44.5%** (229 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-08 | UP | +0.17% | 62.8% | 51.0% | 54.0% | HOLD |
| 6861.T | 2026-10-08 | UP | +0.18% | 54.9% | 47.0% | 48.5% | HOLD |
| 7203.T | 2026-10-08 | DOWN | -0.22% | 60.8% | 49.0% | 52.0% | HOLD |
| 8035.T | 2026-10-08 | DOWN | -0.67% | 69.1% | 49.5% | 48.0% | HOLD |
| 9984.T | 2026-10-08 | DOWN | -0.44% | 56.1% | 49.5% | 48.0% | HOLD |
| AAPL | 2026-10-07 | DOWN | -0.07% | 50.0% | 55.0% | 46.0% | HOLD |
| GOOGL | 2026-10-07 | DOWN | -0.07% | 53.1% | 50.5% | 50.0% | HOLD |
| MSFT | 2026-10-07 | UP | +0.11% | 57.6% | 46.5% | 51.0% | HOLD |
| NVDA | 2026-10-07 | DOWN | -0.55% | 68.6% | 51.5% | 53.5% | HOLD |
| TSLA | 2026-10-07 | DOWN | -0.55% | 63.1% | 51.0% | 50.5% | SELL |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
