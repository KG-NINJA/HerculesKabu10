# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-16T13:55:47.577154Z**
- Expanding time-series validation accuracy: **50.3%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **46.6%** (283 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-16 | DOWN | -0.24% | 61.7% | 52.5% | 53.5% | HOLD |
| 6861.T | 2026-10-16 | DOWN | -0.05% | 51.7% | 52.0% | 47.5% | HOLD |
| 7203.T | 2026-10-16 | UP | +0.17% | 51.3% | 48.0% | 54.0% | HOLD |
| 8035.T | 2026-10-16 | DOWN | -0.59% | 58.9% | 52.0% | 47.0% | HOLD |
| 9984.T | 2026-10-16 | DOWN | -0.08% | 53.4% | 49.5% | 48.0% | HOLD |
| AAPL | 2026-10-15 | UP | +0.15% | 60.6% | 52.0% | 46.5% | BUY |
| GOOGL | 2026-10-15 | UP | +0.37% | 62.8% | 48.5% | 49.0% | HOLD |
| MSFT | 2026-10-15 | DOWN | -0.05% | 56.0% | 44.5% | 50.5% | HOLD |
| NVDA | 2026-10-15 | UP | +0.27% | 51.8% | 50.0% | 54.0% | HOLD |
| TSLA | 2026-10-15 | DOWN | -0.17% | 50.5% | 54.0% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
