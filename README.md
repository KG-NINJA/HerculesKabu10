# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-14T14:29:25.412667Z**
- Expanding time-series validation accuracy: **49.4%**
- Persistence baseline: **50.1%**
- Live 90-day direction accuracy: **45.6%** (263 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-14 | UP | +0.06% | 55.1% | 50.5% | 53.5% | HOLD |
| 6861.T | 2026-10-14 | UP | +0.11% | 55.2% | 49.5% | 48.0% | HOLD |
| 7203.T | 2026-10-14 | UP | +0.05% | 53.3% | 47.5% | 53.0% | HOLD |
| 8035.T | 2026-10-14 | DOWN | -0.30% | 60.0% | 50.0% | 47.0% | SELL |
| 9984.T | 2026-10-14 | UP | +1.28% | 69.3% | 48.5% | 48.5% | HOLD |
| AAPL | 2026-10-13 | UP | +0.22% | 60.9% | 54.5% | 46.0% | BUY |
| GOOGL | 2026-10-13 | UP | +0.24% | 54.6% | 48.0% | 49.5% | HOLD |
| MSFT | 2026-10-13 | DOWN | -0.44% | 71.0% | 45.0% | 51.0% | HOLD |
| NVDA | 2026-10-13 | DOWN | -0.16% | 52.5% | 51.5% | 53.5% | HOLD |
| TSLA | 2026-10-13 | DOWN | -0.69% | 72.9% | 48.5% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
