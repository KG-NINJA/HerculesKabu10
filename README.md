# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-18T13:34:27.738127Z**
- Expanding time-series validation accuracy: **50.0%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **47.1%** (104 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-18 | UP | +0.47% | 72.3% | 50.5% | 53.0% | HOLD |
| 6861.T | 2026-09-18 | DOWN | -0.25% | 57.1% | 51.0% | 49.5% | HOLD |
| 7203.T | 2026-09-18 | UP | +0.30% | 63.0% | 48.5% | 53.0% | HOLD |
| 8035.T | 2026-09-18 | DOWN | -0.10% | 52.6% | 49.0% | 48.0% | HOLD |
| 9984.T | 2026-09-18 | UP | +1.60% | 67.9% | 51.0% | 49.5% | BUY |
| AAPL | 2026-09-17 | UP | +0.27% | 62.4% | 53.5% | 47.0% | BUY |
| GOOGL | 2026-09-17 | UP | +0.50% | 64.7% | 49.0% | 49.0% | HOLD |
| MSFT | 2026-09-17 | UP | +0.16% | 55.1% | 45.5% | 51.0% | HOLD |
| NVDA | 2026-09-17 | DOWN | -0.22% | 52.1% | 50.0% | 51.5% | HOLD |
| TSLA | 2026-09-17 | UP | +0.45% | 57.8% | 52.0% | 50.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
