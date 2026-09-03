from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fmt_percent(value: Any, digits: int = 1, collecting: str = "蓄積待ち") -> str:
    if value is None:
        return collecting
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_points(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:+.{digits}f}pt"


def _fmt_number(value: Any, currency: str | None = None) -> str:
    if value is None:
        return "—"
    number = float(value)
    if currency == "JPY":
        return f"¥{number:,.0f}"
    if currency == "USD":
        return f"${number:,.2f}"
    return f"{number:,.2f}"


def _live_summary(metrics: Mapping[str, Any], window: str = "last_90_days") -> Mapping[str, Any]:
    return metrics.get("live", {}).get(window, {})


def generate_charts(root: Path, latest: dict[str, Any], metrics: dict[str, Any]) -> None:
    assets = root / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    predictions = latest.get("predictions", [])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if predictions:
        labels = [str(item["ticker"]) for item in predictions]
        values = [float(item["predicted_return_pct"]) for item in predictions]
        ax.bar(labels, values)
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Predicted next-session return (%)")
        ax.set_title("NOROSHI latest model estimate — not a guaranteed return")
        ax.tick_params(axis="x", rotation=35)
    else:
        ax.text(0.5, 0.5, "No valid predictions", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(assets / "latest_forecast.png", dpi=150)
    plt.close(fig)

    live_by_ticker = metrics.get("live", {}).get("by_ticker", {})
    eligible_live = {
        ticker: values
        for ticker, values in live_by_ticker.items()
        if int(values.get("direction_evaluable", 0)) >= 5
    }
    backtest_by_ticker = metrics.get("backtest", {}).get("by_ticker", {})

    if eligible_live:
        labels = sorted(eligible_live)
        model_values = [
            float(eligible_live[ticker].get("direction_accuracy") or 0) * 100
            for ticker in labels
        ]
        baseline_values = [
            float(eligible_live[ticker].get("baseline_direction_accuracy") or 0) * 100
            for ticker in labels
        ]
        title = "Live direction accuracy versus persistence baseline"
    else:
        labels = sorted(backtest_by_ticker)
        model_values = [
            float(backtest_by_ticker[ticker].get("direction_accuracy") or 0) * 100
            for ticker in labels
        ]
        baseline_values = [
            float(backtest_by_ticker[ticker].get("persistence_baseline_accuracy") or 0) * 100
            for ticker in labels
        ]
        title = "Walk-forward accuracy versus persistence baseline (live sample accumulating)"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if labels:
        positions = list(range(len(labels)))
        width = 0.38
        ax.bar([position - width / 2 for position in positions], model_values, width, label="Model")
        ax.bar(
            [position + width / 2 for position in positions],
            baseline_values,
            width,
            label="Persistence baseline",
        )
        ax.axhline(50, linewidth=1, linestyle="--")
        ax.set_xticks(positions, labels, rotation=35)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Direction accuracy (%)")
        ax.set_title(title)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Accuracy data is not available", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(assets / "accuracy.png", dpi=150)
    plt.close(fig)


def _prediction_rows(predictions: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for item in predictions:
        validation = item.get("model", {}).get("validation", {})
        signal_class = str(item.get("signal", "HOLD")).lower()
        gate = "PASS" if item.get("quality_gate_passed") else "CAUTION"
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(str(item.get('ticker', '')))}</strong><br>"
            f"<small>{html.escape(str(item.get('name', '')))}</small></td>"
            f"<td>{html.escape(str(item.get('as_of', '—')))}</td>"
            f"<td>{_fmt_number(item.get('current_price'), item.get('currency'))}</td>"
            f"<td>{_fmt_number(item.get('predicted_price'), item.get('currency'))}</td>"
            f"<td>{float(item.get('predicted_return_pct', 0.0)):+.2f}%</td>"
            f"<td>{html.escape(str(item.get('direction_ja', item.get('direction', ''))))}</td>"
            f"<td>{float(item.get('probability_up', 0.5)) * 100:.1f}%</td>"
            f"<td>{float(item.get('confidence', 0.5)) * 100:.1f}%<br><small>モデル確率</small></td>"
            f"<td>{_fmt_percent(validation.get('direction_accuracy'))}<br>"
            f"<small>baseline {_fmt_percent(validation.get('persistence_baseline_accuracy'))}</small></td>"
            f"<td>{html.escape(gate)}</td>"
            f"<td><span class='signal {signal_class}'>{html.escape(str(item.get('signal', 'HOLD')))}</span></td>"
            "</tr>"
        )
    return "".join(rows)


def _freshness_rows(status: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for item in status.get("tickers", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('ticker', '')))}</td>"
            f"<td>{html.escape(str(item.get('market', '')))}</td>"
            f"<td>{html.escape(str(item.get('as_of') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('source', '')))}</td>"
            f"<td>{html.escape(str(item.get('status', '')))}</td>"
            f"<td>{html.escape(str(item.get('age_days') if item.get('age_days') is not None else '—'))}</td>"
            f"<td>{html.escape(str(item.get('error') or ''))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _live_rows(metrics: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for ticker, values in sorted(metrics.get("live", {}).get("by_ticker", {}).items()):
        return_mae = values.get("return_mae_pct")
        return_mae_text = "—" if return_mae is None else f"{float(return_mae):.2f}%"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(ticker))}</td>"
            f"<td>{int(values.get('direction_evaluable', 0))}</td>"
            f"<td>{_fmt_percent(values.get('direction_accuracy'))}</td>"
            f"<td>{_fmt_percent(values.get('baseline_direction_accuracy'))}</td>"
            f"<td>{_fmt_points(values.get('skill_vs_baseline'))}</td>"
            f"<td>{return_mae_text}</td>"
            "</tr>"
        )
    if rows:
        return "".join(rows)
    return '<tr><td colspan="6">schema-v2予測の翌取引日実績を蓄積中です。</td></tr>'


def generate_dashboard(
    root: Path,
    latest: dict[str, Any],
    metrics: dict[str, Any],
    status: dict[str, Any],
) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    predictions = latest.get("predictions", [])
    backtest = metrics.get("backtest", {})
    live = _live_summary(metrics)
    warnings = status.get("warnings", [])

    health = str(status.get("health", "unknown"))
    health_label = {"healthy": "正常", "degraded": "一部縮退", "unhealthy": "停止"}.get(
        health, health
    )
    as_of_values = sorted({str(item.get("as_of")) for item in predictions if item.get("as_of")})
    as_of_text = " / ".join(as_of_values) if as_of_values else "データなし"
    warning_html = "".join(f"<li>{html.escape(str(value))}</li>" for value in warnings)
    if not warning_html:
        warning_html = "<li>現在、重大な警告はありません。</li>"

    document = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NOROSHI Auditable Forecast Dashboard</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --muted:#8b949e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }}
    main {{ width:min(1280px,94vw); margin:0 auto; padding:32px 0 64px; }}
    h1 {{ margin-bottom:4px; }} h2 {{ margin-top:34px; }}
    .subtle, small {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
    .metric {{ font-size:1.7rem; font-weight:700; }}
    .scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; }}
    table {{ width:100%; border-collapse:collapse; min-width:1120px; background:var(--panel); }}
    th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:right; vertical-align:top; }}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
    th {{ position:sticky; top:0; background:var(--panel); }}
    img {{ width:100%; max-width:1100px; background:white; border-radius:12px; }}
    .signal {{ display:inline-block; min-width:56px; text-align:center; border:1px solid var(--line); border-radius:999px; padding:2px 8px; }}
    .buy {{ border-color:#3fb950; }} .sell {{ border-color:#f85149; }} .hold {{ border-color:#d29922; }}
    code {{ background:var(--panel); padding:2px 5px; border-radius:5px; }} a {{ color:#58a6ff; }}
    .warning {{ border-left:4px solid #d29922; }}
  </style>
</head>
<body>
<main>
  <h1>NOROSHI Auditable Forecast Dashboard</h1>
  <p class="subtle">次営業日終値方向の研究用予測。売買の自動執行や利益保証は行いません。</p>

  <div class="grid">
    <section class="card"><div class="subtle">システム状態</div><div class="metric">{html.escape(health_label)}</div></section>
    <section class="card"><div class="subtle">生成日時 UTC</div><div class="metric" style="font-size:1rem">{html.escape(str(latest.get('generated_at', '—')))}</div></section>
    <section class="card"><div class="subtle">市場データ基準日</div><div class="metric" style="font-size:1rem">{html.escape(as_of_text)}</div></section>
    <section class="card"><div class="subtle">有効予測</div><div class="metric">{len(predictions)}</div></section>
    <section class="card"><div class="subtle">Walk-forward方向精度</div><div class="metric">{_fmt_percent(backtest.get('direction_accuracy'))}</div><small>baseline {_fmt_percent(backtest.get('persistence_baseline_accuracy'))}</small></section>
    <section class="card"><div class="subtle">直近90日Live方向精度</div><div class="metric">{_fmt_percent(live.get('direction_accuracy'))}</div><small>{int(live.get('direction_evaluable', 0))}件 / baseline {_fmt_percent(live.get('baseline_direction_accuracy'))}</small></section>
  </div>

  <section class="card warning">
    <h2 style="margin-top:0">状態と注意</h2>
    <ul>{warning_html}</ul>
    <p><strong>Confidenceは過去の的中率ではありません。</strong> 今回の判定に対するモデルのクラス確率です。実際の精度はWalk-forward検証と、保存後に確定したLive実績で別に測定します。</p>
    <p>30件未満のLive評価は暫定値であり、対外的な精度主張には使いません。</p>
  </section>

  <h2>最新予測</h2>
  <div class="scroll"><table>
    <thead><tr><th>銘柄</th><th>基準日</th><th>終値</th><th>予測値</th><th>予測騰落</th><th>方向</th><th>上昇確率</th><th>Confidence</th><th>時系列検証</th><th>品質</th><th>研究Signal</th></tr></thead>
    <tbody>{_prediction_rows(predictions) or '<tr><td colspan="11">有効な予測がありません。</td></tr>'}</tbody>
  </table></div>

  <h2>予測チャート</h2>
  <img src="assets/latest_forecast.png" alt="Latest forecast chart">
  <h2>精度とbaseline</h2>
  <img src="assets/accuracy.png" alt="Accuracy versus baseline chart">

  <h2>Live実績（銘柄別）</h2>
  <div class="scroll"><table>
    <thead><tr><th>銘柄</th><th>確定件数</th><th>方向精度</th><th>baseline</th><th>差</th><th>Return MAE</th></tr></thead>
    <tbody>{_live_rows(metrics)}</tbody>
  </table></div>

  <h2>データ鮮度</h2>
  <div class="scroll"><table>
    <thead><tr><th>Ticker</th><th>市場</th><th>最終日</th><th>取得元</th><th>状態</th><th>経過日</th><th>エラー</th></tr></thead>
    <tbody>{_freshness_rows(status)}</tbody>
  </table></div>

  <section class="card">
    <h2 style="margin-top:0">評価方法</h2>
    <p>毎回、最新OHLCVを取得し、未完了セッションを除外してからLightGBM分類器・回帰器を再学習します。モデル内部評価には、未来データを訓練側へ混ぜないexpanding <code>TimeSeriesSplit</code>と1期間のgapを使います。</p>
    <p>実運用評価は、変更不能なschema-v2予測を次の完成した市場セッションの終値と後日照合します。2025年11月で停止していた旧キャッシュ由来の予測は集計しません。</p>
    <p>機械可読データ: <a href="data/latest_predictions.json">latest_predictions.json</a> / <a href="data/metrics.json">metrics.json</a> / <a href="data/status.json">status.json</a> / <a href="data/live_results.json">live_results.json</a></p>
  </section>
</main>
</body>
</html>
"""
    (docs / "index.html").write_text(document, encoding="utf-8")


def generate_readme(root: Path, latest: dict[str, Any], metrics: dict[str, Any], status: dict[str, Any]) -> None:
    backtest = metrics.get("backtest", {})
    live = _live_summary(metrics)
    rows = []
    for item in latest.get("predictions", []):
        validation = item.get("model", {}).get("validation", {})
        rows.append(
            f"| {item['ticker']} | {item['as_of']} | {item['direction']} | "
            f"{item['predicted_return_pct']:+.2f}% | {item['confidence'] * 100:.1f}% | "
            f"{_fmt_percent(validation.get('direction_accuracy'))} | "
            f"{_fmt_percent(validation.get('persistence_baseline_accuracy'))} | {item['signal']} |"
        )
    markdown = f"""# NOROSHI / HerculesKabu10

Auditable next-session stock direction research pipeline for five US and five Japanese equities.

- Status: **{status.get('health', 'unknown')}**
- Generated: **{latest.get('generated_at', '—')}**
- Expanding time-series validation accuracy: **{_fmt_percent(backtest.get('direction_accuracy'))}**
- Persistence baseline: **{_fmt_percent(backtest.get('persistence_baseline_accuracy'))}**
- Live 90-day direction accuracy: **{_fmt_percent(live.get('direction_accuracy'))}** ({int(live.get('direction_evaluable', 0))} evaluated)
- Dashboard: <https://kg-ninja.github.io/HerculesKabu10/>

## Latest official forecast

| Ticker | Data as-of | Direction | Estimated return | Model confidence | Walk-forward | Baseline | Research signal |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows) if rows else '| — | — | — | — | — | — | — | — |'}

## Reliability policy

- Refresh OHLCV data before every forecast and reject stale downloaded data before it can overwrite cache.
- Exclude an unfinished same-day bar.
- Retrain deterministic LightGBM models on every run; legacy pickle files are ignored.
- Validate with expanding time-series splits and a one-session gap, against a persistence baseline.
- Archive predictions immutably and score them only after the next completed session is available.
- Treat `confidence` as a model class probability, not a historical hit rate.
- Exclude legacy stale-cache predictions from live accuracy.

This repository is research software. It does not provide investment advice, profit guarantees, or automatic trading.
"""
    (root / "README.md").write_text(markdown, encoding="utf-8")


def publish_machine_data(root: Path) -> None:
    docs_data = root / "docs" / "data"
    docs_data.mkdir(parents=True, exist_ok=True)
    mapping = {
        root / "data" / "latest_predictions.json": docs_data / "latest_predictions.json",
        root / "data" / "metrics.json": docs_data / "metrics.json",
        root / "data" / "status.json": docs_data / "status.json",
        root / "data" / "evaluations" / "live_results.json": docs_data / "live_results.json",
    }
    for source, destination in mapping.items():
        if source.exists():
            shutil.copyfile(source, destination)


def generate_reports(
    root: Path,
    latest: dict[str, Any],
    metrics: dict[str, Any],
    status: dict[str, Any],
) -> None:
    generate_charts(root, latest, metrics)
    generate_dashboard(root, latest, metrics, status)
    generate_readme(root, latest, metrics, status)
    publish_machine_data(root)
