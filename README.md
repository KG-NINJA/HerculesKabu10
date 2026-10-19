# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-19T16:26:11.776254Z**
- Expanding time-series validation accuracy: **50.8%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **47.4%** (293 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-19 | UP | +0.34% | 63.9% | 53.5% | 53.5% | BUY |
| 6861.T | 2026-10-19 | UP | +0.05% | 53.5% | 54.5% | 48.0% | HOLD |
| 7203.T | 2026-10-19 | UP | +0.10% | 50.4% | 49.0% | 54.0% | HOLD |
| 8035.T | 2026-10-19 | DOWN | -0.10% | 55.3% | 52.5% | 47.0% | HOLD |
| 9984.T | 2026-10-19 | UP | +1.94% | 71.8% | 50.0% | 48.0% | BUY |
| AAPL | 2026-10-16 | UP | +0.18% | 57.7% | 55.0% | 46.5% | HOLD |
| GOOGL | 2026-10-16 | UP | +0.17% | 52.1% | 47.5% | 48.5% | HOLD |
| MSFT | 2026-10-16 | UP | +0.15% | 56.6% | 44.5% | 50.5% | HOLD |
| NVDA | 2026-10-16 | UP | +1.05% | 71.0% | 50.5% | 54.0% | HOLD |
| TSLA | 2026-10-16 | UP | +0.87% | 69.7% | 51.0% | 51.0% | BUY |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
