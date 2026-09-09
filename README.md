# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-09T13:41:38.244713Z**
- Expanding time-series validation accuracy: **50.0%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **51.4%** (35 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-09 | UP | +0.12% | 58.1% | 50.5% | 54.0% | HOLD |
| 6861.T | 2026-09-09 | UP | +0.38% | 66.1% | 49.5% | 50.0% | HOLD |
| 7203.T | 2026-09-09 | DOWN | -0.22% | 58.1% | 48.5% | 53.0% | HOLD |
| 8035.T | 2026-09-09 | DOWN | -0.12% | 54.0% | 50.5% | 46.5% | HOLD |
| 9984.T | 2026-09-09 | UP | +1.32% | 61.3% | 47.5% | 49.5% | HOLD |
| AAPL | 2026-09-08 | DOWN | -0.17% | 57.7% | 56.5% | 47.0% | HOLD |
| GOOGL | 2026-09-08 | UP | +0.21% | 52.5% | 50.5% | 48.0% | HOLD |
| MSFT | 2026-09-08 | UP | +0.06% | 51.2% | 44.5% | 52.0% | HOLD |
| NVDA | 2026-09-08 | UP | +0.22% | 52.1% | 52.0% | 49.0% | HOLD |
| TSLA | 2026-09-08 | UP | +0.12% | 57.5% | 50.5% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
