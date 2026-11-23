# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-23T21:15:50.123061Z**
- Expanding time-series validation accuracy: **49.6%**
- Persistence baseline: **49.5%**
- Live 90-day direction accuracy: **48.0%** (533 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-20 | DOWN | -0.13% | 50.8% | 53.0% | 52.0% | HOLD |
| 6861.T | 2026-11-20 | DOWN | -0.11% | 50.9% | 55.5% | 48.0% | HOLD |
| 7203.T | 2026-11-20 | DOWN | -0.34% | 57.8% | 45.0% | 53.0% | HOLD |
| 8035.T | 2026-11-20 | UP | +0.55% | 56.5% | 48.0% | 47.0% | HOLD |
| 9984.T | 2026-11-20 | UP | +2.29% | 66.3% | 46.0% | 49.0% | HOLD |
| AAPL | 2026-11-23 | UP | +0.24% | 61.6% | 54.0% | 43.0% | BUY |
| GOOGL | 2026-11-23 | UP | +0.26% | 52.3% | 47.5% | 47.5% | HOLD |
| MSFT | 2026-11-23 | UP | +0.26% | 58.7% | 45.5% | 49.0% | HOLD |
| NVDA | 2026-11-23 | DOWN | -1.45% | 62.6% | 48.5% | 54.0% | HOLD |
| TSLA | 2026-11-23 | UP | +0.50% | 50.3% | 53.0% | 52.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
