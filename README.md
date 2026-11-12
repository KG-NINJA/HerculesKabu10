# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-12T17:32:26.868255Z**
- Expanding time-series validation accuracy: **48.8%**
- Persistence baseline: **49.5%**
- Live 90-day direction accuracy: **49.1%** (468 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-12 | UP | +0.05% | 50.3% | 51.0% | 53.5% | HOLD |
| 6861.T | 2026-11-12 | UP | +0.42% | 63.4% | 54.5% | 47.5% | BUY |
| 7203.T | 2026-11-12 | DOWN | -0.16% | 52.0% | 47.0% | 53.0% | HOLD |
| 8035.T | 2026-11-12 | UP | +0.17% | 50.7% | 48.0% | 46.5% | HOLD |
| 9984.T | 2026-11-12 | DOWN | -0.12% | 53.5% | 45.5% | 49.0% | HOLD |
| AAPL | 2026-11-11 | UP | +0.20% | 51.5% | 51.0% | 43.0% | HOLD |
| GOOGL | 2026-11-11 | UP | +0.58% | 63.5% | 43.5% | 48.0% | HOLD |
| MSFT | 2026-11-11 | DOWN | -0.62% | 62.3% | 46.0% | 49.5% | HOLD |
| NVDA | 2026-11-11 | UP | +0.90% | 58.1% | 47.5% | 54.0% | HOLD |
| TSLA | 2026-11-11 | DOWN | -0.42% | 51.2% | 54.0% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
