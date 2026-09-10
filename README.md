# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-10T13:34:12.707977Z**
- Expanding time-series validation accuracy: **49.9%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **50.0%** (44 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-10 | DOWN | -0.68% | 68.0% | 50.0% | 54.0% | HOLD |
| 6861.T | 2026-09-10 | DOWN | -0.15% | 56.3% | 51.0% | 49.5% | HOLD |
| 7203.T | 2026-09-10 | UP | +0.13% | 56.6% | 48.0% | 53.5% | HOLD |
| 8035.T | 2026-09-10 | UP | +0.32% | 61.0% | 49.5% | 47.0% | HOLD |
| 9984.T | 2026-09-10 | DOWN | -0.44% | 62.6% | 47.0% | 49.0% | HOLD |
| AAPL | 2026-09-09 | DOWN | -0.24% | 62.2% | 51.0% | 47.5% | SELL |
| GOOGL | 2026-09-09 | DOWN | -0.06% | 61.0% | 48.0% | 48.5% | HOLD |
| MSFT | 2026-09-09 | UP | +0.07% | 54.5% | 48.0% | 52.0% | HOLD |
| NVDA | 2026-09-09 | DOWN | -0.43% | 60.1% | 53.0% | 49.5% | SELL |
| TSLA | 2026-09-09 | DOWN | -0.17% | 53.4% | 53.0% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
