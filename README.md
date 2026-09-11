# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-11T13:31:22.910514Z**
- Expanding time-series validation accuracy: **50.6%**
- Persistence baseline: **50.2%**
- Live 90-day direction accuracy: **53.7%** (54 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-11 | DOWN | -0.47% | 66.6% | 53.0% | 54.0% | HOLD |
| 6861.T | 2026-09-11 | DOWN | -0.06% | 50.8% | 51.5% | 49.0% | HOLD |
| 7203.T | 2026-09-11 | DOWN | -0.36% | 61.1% | 46.5% | 53.5% | HOLD |
| 8035.T | 2026-09-11 | DOWN | -0.13% | 53.5% | 51.5% | 47.5% | HOLD |
| 9984.T | 2026-09-11 | UP | +0.75% | 59.0% | 48.5% | 49.5% | HOLD |
| AAPL | 2026-09-10 | UP | +0.17% | 52.7% | 55.5% | 47.5% | HOLD |
| GOOGL | 2026-09-10 | DOWN | -0.06% | 61.1% | 48.0% | 48.5% | HOLD |
| MSFT | 2026-09-10 | DOWN | -0.06% | 51.2% | 47.5% | 51.5% | HOLD |
| NVDA | 2026-09-10 | DOWN | -0.08% | 53.5% | 50.5% | 50.0% | HOLD |
| TSLA | 2026-09-10 | DOWN | -0.64% | 62.4% | 54.0% | 51.0% | SELL |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
