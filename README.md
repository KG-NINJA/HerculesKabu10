# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-15T14:08:38.658247Z**
- Expanding time-series validation accuracy: **50.0%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **46.5%** (273 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-15 | UP | +0.07% | 53.9% | 51.5% | 53.5% | HOLD |
| 6861.T | 2026-10-15 | DOWN | -0.09% | 51.2% | 53.0% | 48.0% | HOLD |
| 7203.T | 2026-10-15 | UP | +0.05% | 54.9% | 48.5% | 53.5% | HOLD |
| 8035.T | 2026-10-15 | UP | +0.18% | 52.0% | 48.5% | 47.0% | HOLD |
| 9984.T | 2026-10-15 | UP | +2.35% | 71.2% | 50.0% | 48.0% | BUY |
| AAPL | 2026-10-14 | UP | +0.28% | 62.4% | 54.0% | 46.5% | BUY |
| GOOGL | 2026-10-14 | DOWN | -0.08% | 53.6% | 48.5% | 49.0% | HOLD |
| MSFT | 2026-10-14 | UP | +0.24% | 57.0% | 44.5% | 51.0% | HOLD |
| NVDA | 2026-10-14 | UP | +0.94% | 72.4% | 49.5% | 53.5% | HOLD |
| TSLA | 2026-10-14 | UP | +0.05% | 53.2% | 52.0% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
