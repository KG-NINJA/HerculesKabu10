# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-30T15:52:24.868962Z**
- Expanding time-series validation accuracy: **49.5%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **49.1%** (383 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-30 | UP | +0.09% | 55.8% | 51.0% | 55.0% | HOLD |
| 6861.T | 2026-10-30 | UP | +0.71% | 73.8% | 53.0% | 49.0% | BUY |
| 7203.T | 2026-10-30 | UP | +0.10% | 61.0% | 45.5% | 53.5% | HOLD |
| 8035.T | 2026-10-30 | DOWN | -0.33% | 60.6% | 50.5% | 46.5% | SELL |
| 9984.T | 2026-10-30 | UP | +0.84% | 61.0% | 45.5% | 48.5% | HOLD |
| AAPL | 2026-10-29 | UP | +0.28% | 61.9% | 52.5% | 44.0% | BUY |
| GOOGL | 2026-10-29 | UP | +0.95% | 66.2% | 49.0% | 48.5% | HOLD |
| MSFT | 2026-10-29 | UP | +0.20% | 55.2% | 46.0% | 50.0% | HOLD |
| NVDA | 2026-10-29 | DOWN | -0.07% | 56.5% | 49.0% | 53.5% | HOLD |
| TSLA | 2026-10-29 | DOWN | -1.06% | 81.7% | 53.5% | 51.0% | SELL |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
