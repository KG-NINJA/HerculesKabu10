# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-20T14:16:50.282714Z**
- Expanding time-series validation accuracy: **49.9%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **48.2%** (303 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-20 | DOWN | -0.16% | 57.2% | 50.5% | 54.0% | HOLD |
| 6861.T | 2026-10-20 | UP | +0.16% | 60.9% | 52.0% | 48.0% | BUY |
| 7203.T | 2026-10-20 | DOWN | -0.36% | 62.8% | 48.5% | 53.5% | HOLD |
| 8035.T | 2026-10-20 | DOWN | -0.43% | 54.2% | 50.0% | 47.5% | HOLD |
| 9984.T | 2026-10-20 | UP | +1.08% | 62.4% | 49.5% | 48.5% | HOLD |
| AAPL | 2026-10-19 | DOWN | -0.12% | 58.1% | 53.5% | 46.0% | HOLD |
| GOOGL | 2026-10-19 | UP | +0.26% | 60.4% | 48.0% | 48.5% | HOLD |
| MSFT | 2026-10-19 | UP | +0.20% | 52.1% | 44.0% | 50.5% | HOLD |
| NVDA | 2026-10-19 | UP | +0.05% | 51.2% | 50.5% | 53.5% | HOLD |
| TSLA | 2026-10-19 | DOWN | -0.05% | 54.6% | 52.0% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
