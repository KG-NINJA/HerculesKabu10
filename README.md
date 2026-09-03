# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-09-03T15:29:37.838229Z**
- Expanding time-series validation accuracy: **50.0%**
- Persistence baseline: **49.6%**
- Live 90-day direction accuracy: **蓄積待ち** (0 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-09-03 | UP | +0.09% | 50.9% | 50.0% | 53.0% | HOLD |
| 6861.T | 2026-09-03 | DOWN | -0.45% | 65.7% | 53.5% | 49.0% | SELL |
| 7203.T | 2026-09-03 | UP | +0.05% | 50.4% | 52.0% | 54.0% | HOLD |
| 8035.T | 2026-09-03 | UP | +0.65% | 65.3% | 52.5% | 45.5% | BUY |
| 9984.T | 2026-09-03 | UP | +0.42% | 60.4% | 43.5% | 48.5% | HOLD |
| AAPL | 2026-09-02 | DOWN | -0.14% | 56.5% | 52.0% | 48.0% | HOLD |
| GOOGL | 2026-09-02 | DOWN | -0.05% | 53.1% | 48.5% | 47.5% | HOLD |
| MSFT | 2026-09-02 | UP | +0.31% | 62.3% | 47.0% | 52.0% | HOLD |
| NVDA | 2026-09-02 | DOWN | -0.09% | 50.9% | 50.0% | 48.5% | HOLD |
| TSLA | 2026-09-02 | UP | +0.09% | 56.9% | 50.5% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
