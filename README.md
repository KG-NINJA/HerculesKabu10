# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-24T20:07:15.023736Z**
- Expanding time-series validation accuracy: **50.0%**
- Persistence baseline: **49.4%**
- Live 90-day direction accuracy: **48.1%** (538 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-24 | UP | +0.05% | 52.2% | 55.5% | 51.5% | HOLD |
| 6861.T | 2026-11-24 | DOWN | -0.05% | 51.9% | 53.5% | 48.0% | HOLD |
| 7203.T | 2026-11-24 | DOWN | -1.17% | 73.9% | 46.0% | 53.0% | HOLD |
| 8035.T | 2026-11-24 | DOWN | -0.58% | 64.5% | 50.0% | 47.0% | SELL |
| 9984.T | 2026-11-24 | UP | +1.14% | 56.0% | 46.5% | 48.5% | HOLD |
| AAPL | 2026-11-23 | UP | +0.18% | 57.6% | 54.0% | 43.0% | HOLD |
| GOOGL | 2026-11-23 | UP | +0.26% | 52.2% | 47.5% | 47.5% | HOLD |
| MSFT | 2026-11-23 | UP | +0.26% | 58.7% | 45.5% | 49.0% | HOLD |
| NVDA | 2026-11-23 | DOWN | -1.45% | 62.6% | 48.5% | 54.0% | HOLD |
| TSLA | 2026-11-23 | UP | +0.68% | 52.3% | 53.0% | 52.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
