# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-16T14:11:07.484499Z**
- Expanding time-series validation accuracy: **50.5%**
- Persistence baseline: **50.2%**
- Live 90-day direction accuracy: **46.4%** (84 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-16 | UP | +0.29% | 62.0% | 51.5% | 53.5% | HOLD |
| 6861.T | 2026-09-16 | UP | +0.06% | 51.5% | 53.5% | 49.0% | HOLD |
| 7203.T | 2026-09-16 | UP | +0.09% | 53.4% | 50.5% | 53.5% | HOLD |
| 8035.T | 2026-09-16 | DOWN | -0.06% | 53.0% | 47.0% | 48.0% | HOLD |
| 9984.T | 2026-09-16 | DOWN | -0.65% | 59.3% | 48.0% | 49.5% | HOLD |
| AAPL | 2026-09-15 | UP | +0.25% | 55.2% | 53.5% | 47.5% | HOLD |
| GOOGL | 2026-09-15 | UP | +0.09% | 52.0% | 48.5% | 48.5% | HOLD |
| MSFT | 2026-09-15 | UP | +0.05% | 55.9% | 44.5% | 51.5% | HOLD |
| NVDA | 2026-09-15 | DOWN | -0.27% | 58.1% | 55.0% | 50.5% | HOLD |
| TSLA | 2026-09-15 | UP | +0.17% | 57.0% | 53.0% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
