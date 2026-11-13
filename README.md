# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-13T17:09:50.746122Z**
- Expanding time-series validation accuracy: **48.9%**
- Persistence baseline: **49.5%**
- Live 90-day direction accuracy: **49.0%** (478 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-13 | DOWN | -0.21% | 57.7% | 50.0% | 53.5% | HOLD |
| 6861.T | 2026-11-13 | UP | +0.60% | 64.2% | 55.0% | 48.0% | BUY |
| 7203.T | 2026-11-13 | UP | +0.42% | 60.3% | 48.0% | 52.5% | HOLD |
| 8035.T | 2026-11-13 | UP | +0.18% | 53.8% | 49.5% | 47.0% | HOLD |
| 9984.T | 2026-11-13 | DOWN | -0.09% | 53.3% | 44.5% | 49.0% | HOLD |
| AAPL | 2026-11-12 | UP | +0.18% | 59.5% | 53.5% | 43.0% | HOLD |
| GOOGL | 2026-11-12 | DOWN | -1.00% | 72.9% | 49.0% | 48.0% | HOLD |
| MSFT | 2026-11-12 | DOWN | -0.36% | 62.4% | 43.0% | 49.0% | HOLD |
| NVDA | 2026-11-12 | UP | +0.50% | 53.0% | 44.5% | 53.5% | HOLD |
| TSLA | 2026-11-12 | DOWN | -0.29% | 55.6% | 52.5% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
