# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-04T17:10:54.131569Z**
- Expanding time-series validation accuracy: **48.5%**
- Persistence baseline: **49.9%**
- Live 90-day direction accuracy: **48.8%** (408 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-04 | UP | +0.05% | 57.0% | 47.5% | 55.0% | HOLD |
| 6861.T | 2026-11-04 | DOWN | -0.29% | 61.7% | 51.5% | 48.5% | SELL |
| 7203.T | 2026-11-04 | DOWN | -0.13% | 56.1% | 44.5% | 54.0% | HOLD |
| 8035.T | 2026-11-04 | UP | +0.32% | 53.4% | 47.0% | 45.5% | HOLD |
| 9984.T | 2026-11-04 | UP | +2.01% | 74.7% | 44.0% | 49.0% | HOLD |
| AAPL | 2026-11-03 | DOWN | -0.20% | 58.6% | 52.0% | 44.0% | HOLD |
| GOOGL | 2026-11-03 | UP | +1.39% | 81.5% | 48.5% | 48.5% | HOLD |
| MSFT | 2026-11-03 | UP | +0.13% | 50.8% | 48.5% | 50.0% | HOLD |
| NVDA | 2026-11-03 | UP | +0.37% | 60.9% | 47.5% | 54.5% | HOLD |
| TSLA | 2026-11-03 | DOWN | -0.69% | 69.0% | 54.5% | 50.0% | SELL |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
