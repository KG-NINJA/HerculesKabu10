# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-23T14:13:09.050901Z**
- Expanding time-series validation accuracy: **49.9%**
- Persistence baseline: **50.2%**
- Live 90-day direction accuracy: **47.1%** (119 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-18 | UP | +0.47% | 72.3% | 50.5% | 53.0% | HOLD |
| 6861.T | 2026-09-18 | DOWN | -0.25% | 57.1% | 51.0% | 49.5% | HOLD |
| 7203.T | 2026-09-18 | UP | +0.30% | 63.0% | 48.5% | 53.0% | HOLD |
| 8035.T | 2026-09-18 | DOWN | -0.10% | 52.6% | 49.0% | 48.0% | HOLD |
| 9984.T | 2026-09-18 | UP | +1.60% | 67.9% | 51.0% | 49.5% | BUY |
| AAPL | 2026-09-22 | UP | +0.42% | 68.9% | 55.5% | 46.5% | BUY |
| GOOGL | 2026-09-22 | UP | +0.10% | 58.2% | 50.5% | 49.5% | HOLD |
| MSFT | 2026-09-22 | UP | +0.29% | 59.8% | 49.0% | 51.0% | HOLD |
| NVDA | 2026-09-22 | UP | +0.41% | 58.9% | 46.5% | 52.5% | HOLD |
| TSLA | 2026-09-22 | DOWN | -0.47% | 61.8% | 47.5% | 50.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
