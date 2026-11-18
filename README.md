# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-18T17:59:33.497813Z**
- Expanding time-series validation accuracy: **50.9%**
- Persistence baseline: **49.5%**
- Live 90-day direction accuracy: **49.0%** (508 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-18 | DOWN | -0.63% | 65.9% | 53.0% | 53.0% | SELL |
| 6861.T | 2026-11-18 | DOWN | -1.65% | 66.8% | 57.5% | 47.5% | SELL |
| 7203.T | 2026-11-18 | DOWN | -0.44% | 59.5% | 48.5% | 52.5% | HOLD |
| 8035.T | 2026-11-18 | DOWN | -0.31% | 56.2% | 51.0% | 47.0% | HOLD |
| 9984.T | 2026-11-18 | DOWN | -0.65% | 59.5% | 45.0% | 49.0% | HOLD |
| AAPL | 2026-11-17 | UP | +0.30% | 61.4% | 51.5% | 42.5% | BUY |
| GOOGL | 2026-11-17 | DOWN | -1.06% | 75.7% | 47.5% | 48.5% | HOLD |
| MSFT | 2026-11-17 | DOWN | -0.06% | 51.3% | 53.0% | 49.0% | HOLD |
| NVDA | 2026-11-17 | UP | +0.94% | 53.6% | 49.5% | 54.0% | HOLD |
| TSLA | 2026-11-17 | DOWN | -0.24% | 54.0% | 52.5% | 51.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
