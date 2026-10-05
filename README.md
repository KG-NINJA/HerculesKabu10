# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **healthy**
- Generated: **2026-10-05T18:22:23.804186Z**
- Expanding time-series validation accuracy: **49.6%**
- Persistence baseline: **49.8%**
- Live 90-day direction accuracy: **43.2%** (199 evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6758.T | 2026-10-05 | UP | +0.37% | 64.7% | 54.0% | 54.0% | BUY |
| 6861.T | 2026-10-05 | UP | +0.22% | 59.9% | 49.0% | 48.5% | HOLD |
| 7203.T | 2026-10-05 | UP | +0.05% | 53.0% | 45.5% | 52.5% | HOLD |
| 8035.T | 2026-10-05 | DOWN | -0.18% | 61.3% | 49.0% | 47.5% | HOLD |
| 9984.T | 2026-10-05 | DOWN | -0.19% | 53.7% | 46.5% | 48.0% | HOLD |
| AAPL | 2026-10-02 | UP | +0.10% | 58.6% | 51.5% | 45.5% | HOLD |
| GOOGL | 2026-10-02 | UP | +0.62% | 72.6% | 49.5% | 49.5% | HOLD |
| MSFT | 2026-10-02 | DOWN | -0.11% | 59.2% | 47.0% | 49.5% | HOLD |
| NVDA | 2026-10-02 | DOWN | -0.06% | 54.0% | 52.5% | 53.0% | HOLD |
| TSLA | 2026-10-02 | UP | +0.36% | 53.6% | 51.5% | 50.0% | HOLD |

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
