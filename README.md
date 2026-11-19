# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-19T18:13:11.483094Z**
- Expanding time-series validation accuracy: **49.4%**
- Persistence baseline: **49.5%**
- Live 90-day direction accuracy: **48.6%** (518 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-19 | DOWN | -0.32% | 60.9% | 54.5% | 52.5% | SELL |
| 6861.T | 2026-11-19 | UP | +0.17% | 56.2% | 57.0% | 47.5% | HOLD |
| 7203.T | 2026-11-19 | DOWN | -0.36% | 55.3% | 50.5% | 52.5% | HOLD |
| 8035.T | 2026-11-19 | DOWN | -0.16% | 53.2% | 49.0% | 47.5% | HOLD |
| 9984.T | 2026-11-19 | UP | +0.19% | 50.1% | 42.5% | 49.5% | HOLD |
| AAPL | 2026-11-18 | UP | +0.30% | 62.8% | 52.0% | 42.5% | BUY |
| GOOGL | 2026-11-18 | UP | +0.10% | 51.5% | 44.0% | 48.5% | HOLD |
| MSFT | 2026-11-18 | UP | +0.29% | 57.3% | 50.0% | 48.5% | HOLD |
| NVDA | 2026-11-18 | UP | +1.25% | 61.5% | 45.0% | 54.0% | HOLD |
| TSLA | 2026-11-18 | UP | +0.14% | 55.3% | 49.0% | 52.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
