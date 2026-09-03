# NOROSHI / HerculesKabu10

Fresh-market-data, walk-forward-validated next-session stock research pipeline.

The legacy workflow repeatedly timestamped a frozen November 2025 cache as current data. Version 3 blocks stale publication, refreshes OHLCV before each run, retrains models, separates model probability from measured accuracy, and resolves saved predictions against the next observed market close.

Dashboard: https://kg-ninja.github.io/HerculesKabu10/

## Reliability policy

- At least 8 of 10 tickers, including both US and Japanese markets, must pass the freshness gate; partial success is marked degraded.
- Models are retrained from current five-year OHLCV on every run; legacy pickle files are ignored.
- Validation uses expanding-window time-series splits with a one-session gap.
- `confidence` is a model class probability, not a historical hit rate.
- Live accuracy begins with schema-v2 predictions; stale legacy predictions are excluded.
- The system is research software, not investment advice.

## Commands

```bash
python -m pip install -r requirements.txt
python simple_daily_prediction.py
python scripts/generate_dashboard.py
python scripts/validate_outputs.py
python -m unittest discover -s tests -v
```
