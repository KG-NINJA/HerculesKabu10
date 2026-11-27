# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-27T18:54:26.464684Z**
- Expanding time-series validation accuracy: **49.7%**
- Persistence baseline: **49.3%**
- Live 90-day direction accuracy: **49.0%** (563 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-27 | DOWN | -0.42% | 59.1% | 54.5% | 51.5% | HOLD |
| 6861.T | 2026-11-27 | UP | +0.05% | 50.3% | 51.0% | 48.5% | HOLD |
| 7203.T | 2026-11-27 | UP | +0.39% | 57.3% | 48.0% | 52.0% | HOLD |
| 8035.T | 2026-11-27 | UP | +0.73% | 61.6% | 49.5% | 46.5% | HOLD |
| 9984.T | 2026-11-27 | DOWN | -0.29% | 55.1% | 46.5% | 48.5% | HOLD |
| AAPL | 2026-11-25 | UP | +0.33% | 61.7% | 52.5% | 43.5% | BUY |
| GOOGL | 2026-11-25 | DOWN | -1.03% | 74.0% | 48.0% | 48.0% | HOLD |
| MSFT | 2026-11-25 | DOWN | -0.08% | 51.0% | 45.0% | 49.0% | HOLD |
| NVDA | 2026-11-25 | DOWN | -1.19% | 62.6% | 48.5% | 54.0% | HOLD |
| TSLA | 2026-11-25 | DOWN | -0.07% | 52.0% | 53.5% | 51.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
