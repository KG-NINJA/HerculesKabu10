# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-07T16:24:30.183352Z**
- Expanding time-series validation accuracy: **49.2%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **43.8%** (219 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-07 | UP | +0.53% | 70.1% | 50.5% | 53.5% | HOLD |
| 6861.T | 2026-10-07 | UP | +0.16% | 56.3% | 45.5% | 48.5% | HOLD |
| 7203.T | 2026-10-07 | UP | +0.05% | 53.0% | 50.0% | 52.5% | HOLD |
| 8035.T | 2026-10-07 | DOWN | -0.10% | 56.2% | 51.0% | 47.5% | HOLD |
| 9984.T | 2026-10-07 | DOWN | -0.09% | 52.4% | 48.0% | 48.0% | HOLD |
| AAPL | 2026-10-06 | UP | +0.11% | 50.5% | 49.0% | 45.5% | HOLD |
| GOOGL | 2026-10-06 | UP | +0.48% | 64.4% | 50.5% | 49.5% | BUY |
| MSFT | 2026-10-06 | DOWN | -0.28% | 66.2% | 47.0% | 50.5% | HOLD |
| NVDA | 2026-10-06 | UP | +0.12% | 53.9% | 51.0% | 53.5% | HOLD |
| TSLA | 2026-10-06 | DOWN | -0.34% | 60.8% | 49.5% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
