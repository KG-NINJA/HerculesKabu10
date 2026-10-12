# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-12T18:58:37.556577Z**
- Expanding time-series validation accuracy: **49.5%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **45.5%** (244 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-09 | UP | +0.09% | 55.8% | 52.0% | 53.5% | HOLD |
| 6861.T | 2026-10-09 | UP | +0.12% | 53.8% | 53.5% | 48.5% | HOLD |
| 7203.T | 2026-10-09 | UP | +0.05% | 51.1% | 47.5% | 52.5% | HOLD |
| 8035.T | 2026-10-09 | DOWN | -0.30% | 57.5% | 49.5% | 47.5% | HOLD |
| 9984.T | 2026-10-09 | DOWN | -0.36% | 64.5% | 47.5% | 48.0% | HOLD |
| AAPL | 2026-10-09 | UP | +0.22% | 59.9% | 50.0% | 46.0% | HOLD |
| GOOGL | 2026-10-09 | UP | +0.25% | 61.8% | 49.0% | 49.0% | HOLD |
| MSFT | 2026-10-09 | DOWN | -0.14% | 52.2% | 44.5% | 50.5% | HOLD |
| NVDA | 2026-10-09 | UP | +0.12% | 50.9% | 48.5% | 53.5% | HOLD |
| TSLA | 2026-10-09 | UP | +0.08% | 50.1% | 52.5% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
