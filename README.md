# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-01T16:23:04.081485Z**
- Expanding time-series validation accuracy: **50.0%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **43.0%** (179 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-01 | DOWN | -0.06% | 50.4% | 50.5% | 53.5% | HOLD |
| 6861.T | 2026-10-01 | UP | +0.28% | 58.7% | 53.0% | 49.0% | HOLD |
| 7203.T | 2026-10-01 | DOWN | -0.46% | 71.7% | 46.5% | 52.5% | HOLD |
| 8035.T | 2026-10-01 | UP | +0.97% | 66.1% | 51.5% | 48.0% | BUY |
| 9984.T | 2026-10-01 | UP | +0.59% | 54.7% | 50.5% | 49.0% | HOLD |
| AAPL | 2026-09-30 | UP | +0.27% | 64.6% | 53.5% | 45.5% | BUY |
| GOOGL | 2026-09-30 | UP | +0.34% | 60.6% | 49.5% | 50.5% | HOLD |
| MSFT | 2026-09-30 | DOWN | -0.07% | 58.3% | 48.5% | 50.0% | HOLD |
| NVDA | 2026-09-30 | UP | +0.42% | 64.8% | 48.5% | 52.5% | HOLD |
| TSLA | 2026-09-30 | DOWN | -0.55% | 58.7% | 48.0% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
