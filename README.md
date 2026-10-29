# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-29T16:03:10.562687Z**
- Expanding time-series validation accuracy: **51.0%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **48.5%** (373 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-29 | UP | +0.11% | 51.2% | 52.5% | 55.0% | HOLD |
| 6861.T | 2026-10-29 | UP | +0.41% | 65.2% | 53.5% | 49.0% | BUY |
| 7203.T | 2026-10-29 | UP | +0.29% | 67.8% | 50.0% | 53.5% | HOLD |
| 8035.T | 2026-10-29 | DOWN | -0.55% | 67.3% | 50.5% | 47.0% | SELL |
| 9984.T | 2026-10-29 | UP | +2.12% | 77.4% | 46.5% | 48.5% | HOLD |
| AAPL | 2026-10-28 | UP | +0.09% | 51.1% | 52.5% | 44.5% | HOLD |
| GOOGL | 2026-10-28 | DOWN | -0.11% | 54.9% | 48.0% | 49.0% | HOLD |
| MSFT | 2026-10-28 | DOWN | -0.07% | 51.4% | 50.0% | 50.0% | HOLD |
| NVDA | 2026-10-28 | UP | +0.35% | 55.6% | 52.5% | 53.5% | HOLD |
| TSLA | 2026-10-28 | DOWN | -0.12% | 54.4% | 54.0% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
