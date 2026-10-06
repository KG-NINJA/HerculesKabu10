# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-06T15:54:31.998032Z**
- Expanding time-series validation accuracy: **49.7%**
- Persistence baseline: **49.9%**
- Live 90-day direction accuracy: **43.5%** (209 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-06 | UP | +0.41% | 67.0% | 49.0% | 53.5% | HOLD |
| 6861.T | 2026-10-06 | UP | +0.11% | 52.8% | 49.5% | 48.5% | HOLD |
| 7203.T | 2026-10-06 | DOWN | -0.13% | 56.7% | 45.5% | 53.0% | HOLD |
| 8035.T | 2026-10-06 | DOWN | -0.67% | 65.9% | 51.0% | 48.0% | SELL |
| 9984.T | 2026-10-06 | DOWN | -0.70% | 62.4% | 48.0% | 48.0% | HOLD |
| AAPL | 2026-10-05 | UP | +0.18% | 60.7% | 50.0% | 45.5% | BUY |
| GOOGL | 2026-10-05 | UP | +0.28% | 57.5% | 50.0% | 49.5% | HOLD |
| MSFT | 2026-10-05 | DOWN | -0.36% | 75.1% | 46.5% | 50.0% | HOLD |
| NVDA | 2026-10-05 | UP | +0.10% | 52.3% | 51.0% | 53.0% | HOLD |
| TSLA | 2026-10-05 | DOWN | -0.15% | 55.3% | 56.5% | 50.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
