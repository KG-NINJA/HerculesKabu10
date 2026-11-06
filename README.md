# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-06T16:43:35.337538Z**
- Expanding time-series validation accuracy: **49.5%**
- Persistence baseline: **49.9%**
- Live 90-day direction accuracy: **48.1%** (428 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-06 | UP | +0.24% | 54.6% | 49.0% | 54.0% | HOLD |
| 6861.T | 2026-11-06 | DOWN | -0.19% | 57.1% | 51.5% | 48.5% | HOLD |
| 7203.T | 2026-11-06 | DOWN | -0.05% | 51.9% | 44.5% | 53.5% | HOLD |
| 8035.T | 2026-11-06 | UP | +0.42% | 54.3% | 49.0% | 46.5% | HOLD |
| 9984.T | 2026-11-06 | UP | +1.56% | 60.3% | 46.0% | 49.0% | HOLD |
| AAPL | 2026-11-05 | UP | +0.05% | 50.2% | 53.5% | 43.5% | HOLD |
| GOOGL | 2026-11-05 | UP | +0.88% | 62.2% | 48.5% | 48.5% | HOLD |
| MSFT | 2026-11-05 | DOWN | -0.30% | 59.3% | 46.0% | 50.0% | HOLD |
| NVDA | 2026-11-05 | DOWN | -0.29% | 54.5% | 52.5% | 54.5% | HOLD |
| TSLA | 2026-11-05 | UP | +0.96% | 69.0% | 54.0% | 50.5% | BUY |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
