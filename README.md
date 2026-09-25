# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-25T14:35:22.058237Z**
- Expanding time-series validation accuracy: **49.6%**
- Persistence baseline: **50.2%**
- Live 90-day direction accuracy: **44.6%** (139 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-25 | UP | +0.17% | 54.4% | 52.0% | 53.0% | HOLD |
| 6861.T | 2026-09-25 | UP | +0.33% | 62.2% | 54.0% | 49.5% | BUY |
| 7203.T | 2026-09-25 | UP | +0.62% | 75.8% | 45.5% | 53.5% | HOLD |
| 8035.T | 2026-09-25 | UP | +0.44% | 61.3% | 52.5% | 48.0% | BUY |
| 9984.T | 2026-09-25 | DOWN | -0.18% | 51.5% | 44.5% | 49.5% | HOLD |
| AAPL | 2026-09-24 | UP | +0.19% | 56.1% | 55.5% | 46.0% | HOLD |
| GOOGL | 2026-09-24 | UP | +0.77% | 69.9% | 49.0% | 50.0% | HOLD |
| MSFT | 2026-09-24 | UP | +0.11% | 61.5% | 46.0% | 50.5% | HOLD |
| NVDA | 2026-09-24 | DOWN | -0.35% | 63.9% | 47.0% | 53.0% | HOLD |
| TSLA | 2026-09-24 | DOWN | -1.00% | 71.4% | 50.5% | 49.5% | SELL |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
