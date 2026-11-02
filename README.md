# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-11-02T19:26:32.265358Z**
- Expanding time-series validation accuracy: **50.3%**
- Persistence baseline: **50.0%**
- Live 90-day direction accuracy: **48.6%** (393 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-11-02 | DOWN | -0.05% | 50.6% | 50.5% | 55.0% | HOLD |
| 6861.T | 2026-11-02 | UP | +0.50% | 64.2% | 54.0% | 49.0% | BUY |
| 7203.T | 2026-11-02 | UP | +0.21% | 63.3% | 45.0% | 53.5% | HOLD |
| 8035.T | 2026-11-02 | DOWN | -0.84% | 66.4% | 53.5% | 46.0% | SELL |
| 9984.T | 2026-11-02 | UP | +0.65% | 53.4% | 47.0% | 48.5% | HOLD |
| AAPL | 2026-10-30 | UP | +0.35% | 66.4% | 55.5% | 44.0% | BUY |
| GOOGL | 2026-10-30 | UP | +1.11% | 67.9% | 48.0% | 49.0% | HOLD |
| MSFT | 2026-10-30 | DOWN | -0.31% | 66.4% | 48.5% | 50.0% | HOLD |
| NVDA | 2026-10-30 | UP | +0.58% | 63.5% | 47.0% | 54.0% | HOLD |
| TSLA | 2026-10-30 | UP | +0.05% | 50.3% | 54.5% | 51.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
