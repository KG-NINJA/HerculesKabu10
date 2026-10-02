# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-02T15:45:28.200629Z**
- Expanding time-series validation accuracy: **49.4%**
- Persistence baseline: **49.9%**
- Live 90-day direction accuracy: **43.4%** (189 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-02 | UP | +0.06% | 54.9% | 52.0% | 53.5% | HOLD |
| 6861.T | 2026-10-02 | UP | +0.39% | 63.3% | 50.5% | 48.5% | BUY |
| 7203.T | 2026-10-02 | UP | +0.31% | 58.3% | 47.0% | 52.5% | HOLD |
| 8035.T | 2026-10-02 | UP | +0.36% | 61.2% | 49.5% | 47.5% | HOLD |
| 9984.T | 2026-10-02 | DOWN | -0.49% | 53.8% | 49.0% | 48.5% | HOLD |
| AAPL | 2026-10-01 | DOWN | -0.05% | 50.4% | 49.0% | 45.5% | HOLD |
| GOOGL | 2026-10-01 | DOWN | -0.14% | 53.7% | 48.0% | 50.0% | HOLD |
| MSFT | 2026-10-01 | UP | +0.06% | 51.7% | 47.5% | 50.0% | HOLD |
| NVDA | 2026-10-01 | DOWN | -0.06% | 57.9% | 51.5% | 52.5% | HOLD |
| TSLA | 2026-10-01 | DOWN | -0.86% | 75.3% | 50.0% | 50.5% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
