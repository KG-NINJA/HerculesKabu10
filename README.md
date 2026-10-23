# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-23T14:12:46.410378Z**
- Expanding time-series validation accuracy: **49.5%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **47.7%** (333 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-23 | DOWN | -0.22% | 54.4% | 53.0% | 54.5% | HOLD |
| 6861.T | 2026-10-23 | DOWN | -0.16% | 59.5% | 48.5% | 49.0% | HOLD |
| 7203.T | 2026-10-23 | DOWN | -0.13% | 52.2% | 45.5% | 53.5% | HOLD |
| 8035.T | 2026-10-23 | DOWN | -0.19% | 50.0% | 47.0% | 47.0% | HOLD |
| 9984.T | 2026-10-23 | UP | +1.13% | 54.7% | 47.5% | 48.0% | HOLD |
| AAPL | 2026-10-22 | UP | +0.11% | 51.6% | 49.5% | 45.5% | HOLD |
| GOOGL | 2026-10-22 | DOWN | -0.18% | 62.9% | 47.5% | 49.0% | HOLD |
| MSFT | 2026-10-22 | UP | +0.05% | 51.0% | 48.0% | 49.5% | HOLD |
| NVDA | 2026-10-22 | UP | +0.34% | 55.6% | 52.5% | 53.0% | HOLD |
| TSLA | 2026-10-22 | DOWN | -0.10% | 51.6% | 56.0% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
