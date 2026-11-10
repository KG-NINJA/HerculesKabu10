# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-10T16:59:49.717153Z**
- Expanding time-series validation accuracy: **49.5%**
- Persistence baseline: **49.6%**
- Live 90-day direction accuracy: **49.1%** (448 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-10 | DOWN | -0.10% | 57.3% | 52.5% | 53.5% | HOLD |
| 6861.T | 2026-11-10 | UP | +0.19% | 53.9% | 54.0% | 48.0% | HOLD |
| 7203.T | 2026-11-10 | DOWN | -0.19% | 54.0% | 44.5% | 53.0% | HOLD |
| 8035.T | 2026-11-10 | DOWN | -0.77% | 63.8% | 48.5% | 46.0% | HOLD |
| 9984.T | 2026-11-10 | UP | +1.03% | 56.9% | 47.0% | 49.5% | HOLD |
| AAPL | 2026-11-09 | UP | +0.24% | 54.6% | 53.5% | 43.5% | HOLD |
| GOOGL | 2026-11-09 | UP | +1.17% | 73.2% | 48.5% | 48.5% | HOLD |
| MSFT | 2026-11-09 | DOWN | -0.15% | 53.8% | 47.0% | 50.0% | HOLD |
| NVDA | 2026-11-09 | DOWN | -0.23% | 56.3% | 47.5% | 53.5% | HOLD |
| TSLA | 2026-11-09 | DOWN | -0.40% | 58.3% | 52.5% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
