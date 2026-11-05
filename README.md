# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-05T17:21:04.968505Z**
- Expanding time-series validation accuracy: **49.0%**
- Persistence baseline: **49.8%**
- Live 90-day direction accuracy: **48.8%** (418 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-05 | DOWN | -0.51% | 60.8% | 52.0% | 54.5% | HOLD |
| 6861.T | 2026-11-05 | UP | +0.05% | 51.4% | 53.5% | 48.0% | HOLD |
| 7203.T | 2026-11-05 | DOWN | -0.83% | 70.3% | 44.5% | 53.5% | HOLD |
| 8035.T | 2026-11-05 | UP | +0.80% | 58.0% | 50.5% | 46.0% | HOLD |
| 9984.T | 2026-11-05 | UP | +1.91% | 69.9% | 44.5% | 49.0% | HOLD |
| AAPL | 2026-11-04 | UP | +0.71% | 76.4% | 52.0% | 43.5% | BUY |
| GOOGL | 2026-11-04 | UP | +0.69% | 56.6% | 46.5% | 48.5% | HOLD |
| MSFT | 2026-11-04 | UP | +0.36% | 63.9% | 46.0% | 50.5% | HOLD |
| NVDA | 2026-11-04 | UP | +1.11% | 68.9% | 46.5% | 54.0% | HOLD |
| TSLA | 2026-11-04 | UP | +0.92% | 63.2% | 54.0% | 50.0% | BUY |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
