# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-27T15:19:02.172919Z**
- Expanding time-series validation accuracy: **49.9%**
- Persistence baseline: **49.9%**
- Live 90-day direction accuracy: **48.2%** (353 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-27 | UP | +0.10% | 54.1% | 50.5% | 54.5% | HOLD |
| 6861.T | 2026-10-27 | UP | +0.05% | 50.6% | 53.0% | 48.5% | HOLD |
| 7203.T | 2026-10-27 | UP | +0.18% | 62.1% | 46.5% | 53.5% | HOLD |
| 8035.T | 2026-10-27 | UP | +0.14% | 52.1% | 50.0% | 46.0% | HOLD |
| 9984.T | 2026-10-27 | UP | +0.78% | 54.6% | 44.5% | 48.0% | HOLD |
| AAPL | 2026-10-26 | UP | +0.15% | 54.2% | 54.5% | 45.0% | HOLD |
| GOOGL | 2026-10-26 | DOWN | -0.05% | 53.9% | 49.0% | 49.5% | HOLD |
| MSFT | 2026-10-26 | DOWN | -0.12% | 57.2% | 48.0% | 49.5% | HOLD |
| NVDA | 2026-10-26 | UP | +0.57% | 65.1% | 51.0% | 53.5% | HOLD |
| TSLA | 2026-10-26 | UP | +0.11% | 54.4% | 51.5% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
