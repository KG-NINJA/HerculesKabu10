# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-07T15:02:17.952534Z**
- Expanding time-series validation accuracy: **48.9%**
- Persistence baseline: **49.9%**
- Live 90-day direction accuracy: **45.0%** (20 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-07 | DOWN | -0.16% | 51.1% | 46.0% | 53.5% | HOLD |
| 6861.T | 2026-09-07 | DOWN | -0.24% | 55.3% | 48.5% | 49.5% | HOLD |
| 7203.T | 2026-09-07 | UP | +0.36% | 63.9% | 47.0% | 54.0% | HOLD |
| 8035.T | 2026-09-07 | DOWN | -0.42% | 62.9% | 53.0% | 46.5% | SELL |
| 9984.T | 2026-09-07 | UP | +1.63% | 67.1% | 47.5% | 49.0% | HOLD |
| AAPL | 2026-09-04 | DOWN | -0.10% | 53.0% | 54.0% | 47.0% | HOLD |
| GOOGL | 2026-09-04 | UP | +0.15% | 53.5% | 48.0% | 47.5% | HOLD |
| MSFT | 2026-09-04 | UP | +0.18% | 60.8% | 47.5% | 52.0% | HOLD |
| NVDA | 2026-09-04 | UP | +0.89% | 67.1% | 50.5% | 49.5% | BUY |
| TSLA | 2026-09-04 | UP | +1.25% | 72.3% | 47.5% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
