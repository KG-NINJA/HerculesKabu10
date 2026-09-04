# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-04T13:30:04.060671Z**
- Expanding time-series validation accuracy: **50.3%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **40.0%** (10 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-04 | DOWN | -0.18% | 54.4% | 54.0% | 53.5% | HOLD |
| 6861.T | 2026-09-04 | UP | +0.11% | 51.1% | 51.5% | 49.5% | HOLD |
| 7203.T | 2026-09-04 | UP | +0.11% | 55.4% | 49.0% | 54.5% | HOLD |
| 8035.T | 2026-09-04 | UP | +0.53% | 69.5% | 50.0% | 46.0% | BUY |
| 9984.T | 2026-09-04 | DOWN | -0.26% | 53.9% | 49.5% | 49.0% | HOLD |
| AAPL | 2026-09-03 | UP | +0.20% | 55.1% | 51.5% | 47.5% | HOLD |
| GOOGL | 2026-09-03 | UP | +0.30% | 59.3% | 48.5% | 47.5% | HOLD |
| MSFT | 2026-09-03 | UP | +0.05% | 52.2% | 46.0% | 52.0% | HOLD |
| NVDA | 2026-09-03 | UP | +0.72% | 64.2% | 51.0% | 49.0% | BUY |
| TSLA | 2026-09-03 | UP | +0.14% | 58.6% | 52.0% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
