# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-26T18:58:10.074657Z**
- Expanding time-series validation accuracy: **50.4%**
- Persistence baseline: **49.3%**
- Live 90-day direction accuracy: **48.9%** (558 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-26 | DOWN | -0.38% | 62.0% | 54.0% | 51.5% | SELL |
| 6861.T | 2026-11-26 | DOWN | -0.45% | 67.3% | 50.5% | 48.5% | SELL |
| 7203.T | 2026-11-26 | DOWN | -0.05% | 52.5% | 47.5% | 52.5% | HOLD |
| 8035.T | 2026-11-26 | UP | +0.32% | 53.3% | 52.5% | 46.5% | HOLD |
| 9984.T | 2026-11-26 | UP | +0.68% | 52.3% | 47.0% | 48.0% | HOLD |
| AAPL | 2026-11-25 | UP | +0.31% | 55.2% | 51.5% | 43.5% | HOLD |
| GOOGL | 2026-11-25 | DOWN | -0.83% | 71.4% | 49.0% | 48.0% | HOLD |
| MSFT | 2026-11-25 | UP | +0.05% | 53.4% | 47.0% | 49.0% | HOLD |
| NVDA | 2026-11-25 | DOWN | -1.55% | 66.8% | 48.5% | 54.0% | HOLD |
| TSLA | 2026-11-25 | DOWN | -0.33% | 53.7% | 56.5% | 51.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
