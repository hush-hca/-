#!/usr/bin/env python3
"""
Web dashboard for the Crypto Pure Alpha Long/Short Screener.

Install:
    pip install ccxt pandas numpy statsmodels

Run:
    python crypto_pure_alpha_dashboard.py --host 127.0.0.1 --port 8080

Open:
    http://127.0.0.1:8080

This dashboard uses only Python's standard-library HTTP server for the web
layer. The quantitative logic is imported from crypto_pure_alpha_screener.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd

from crypto_pure_alpha_screener import (
    ScreenerConfig,
    attach_funding_and_scores,
    build_close_matrix,
    build_exchange,
    compute_returns,
    fetch_current_funding_rates,
    fetch_top_altcoin_symbols,
    regress_latest_residuals,
    select_signals,
)


LOGGER = logging.getLogger("pure_alpha_dashboard")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Crypto Pure Alpha Dashboard</title>
  <style>
    :root {
      --bg: #111417;
      --panel: #181d22;
      --panel-soft: #20262d;
      --text: #edf2f4;
      --muted: #aab4bd;
      --line: #303841;
      --green: #21c17a;
      --red: #ff5b6b;
      --blue: #63a4ff;
      --amber: #f6bd60;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      padding: 26px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #15191e;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 720;
      line-height: 1.2;
    }
    .subhead {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      max-width: 980px;
    }
    main {
      width: min(1480px, calc(100vw - 40px));
      margin: 22px auto 44px;
    }
    .layout {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .controls {
      padding: 18px;
      position: sticky;
      top: 16px;
    }
    .control {
      margin-bottom: 15px;
    }
    label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    input {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #101316;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    button {
      width: 100%;
      height: 42px;
      border: 0;
      border-radius: 6px;
      background: var(--blue);
      color: #07101c;
      font-weight: 760;
      cursor: pointer;
      font-size: 14px;
    }
    button:disabled {
      opacity: 0.62;
      cursor: wait;
    }
    .status {
      margin-top: 13px;
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric {
      padding: 15px 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 21px;
      line-height: 1.1;
    }
    .section {
      margin-bottom: 18px;
      overflow: hidden;
    }
    .section-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .section-title h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.25;
    }
    .section-title small {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .table-wrap {
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 920px;
      font-size: 13px;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child,
    th:nth-child(4), td:nth-child(4) {
      text-align: left;
    }
    th {
      color: var(--muted);
      background: #151a1f;
      font-size: 11px;
      text-transform: uppercase;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    tbody tr:hover { background: var(--panel-soft); }
    .long { color: var(--green); font-weight: 720; }
    .short { color: var(--red); font-weight: 720; }
    .positive { color: var(--green); }
    .negative { color: var(--red); }
    .bars {
      padding: 12px 16px 18px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 132px 1fr 88px;
      gap: 10px;
      align-items: center;
      min-height: 28px;
      font-size: 12px;
    }
    .bar-label {
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .bar-track {
      height: 10px;
      background: #0f1215;
      border: 1px solid #252c34;
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }
    .bar-fill {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 0;
      background: var(--green);
    }
    .bar-fill.neg {
      left: auto;
      right: 50%;
      background: var(--red);
    }
    .bar-value {
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .error {
      border-color: rgba(255, 91, 107, 0.55);
      color: #ffd3d8;
      padding: 14px 16px;
      line-height: 1.45;
    }
    .empty {
      padding: 32px 16px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 980px) {
      header { padding: 22px 20px 16px; }
      main { width: min(100vw - 24px, 920px); }
      .layout { grid-template-columns: 1fr; }
      .controls { position: static; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
      .grid { grid-template-columns: 1fr; }
      h1 { font-size: 21px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Crypto Pure Alpha Long/Short Screener</h1>
    <div class="subhead">
      Binance USD-M futures dashboard that removes BTC and ETH beta from hourly altcoin returns,
      then ranks the latest residuals with funding-rate filters.
    </div>
  </header>
  <main class="layout">
    <aside class="panel controls">
      <div class="control">
        <label for="universeSize"><span>Universe Size</span><span>Top volume</span></label>
        <input id="universeSize" type="number" min="10" max="150" step="1" value="100" />
      </div>
      <div class="control">
        <label for="lookbackHours"><span>Lookback Hours</span><span>Regression</span></label>
        <input id="lookbackHours" type="number" min="24" max="168" step="1" value="48" />
      </div>
      <div class="control">
        <label for="minObs"><span>Min Observations</span><span>OLS</span></label>
        <input id="minObs" type="number" min="12" max="120" step="1" value="24" />
      </div>
      <div class="control">
        <label for="topN"><span>Signals Per Side</span><span>Long / Short</span></label>
        <input id="topN" type="number" min="1" max="20" step="1" value="5" />
      </div>
      <div class="control">
        <label for="maxLongFunding"><span>Max Long Funding</span><span>Decimal</span></label>
        <input id="maxLongFunding" type="number" min="-0.01" max="0.01" step="0.0001" value="0.0005" />
      </div>
      <div class="control">
        <label for="minShortFunding"><span>Min Short Funding</span><span>Decimal</span></label>
        <input id="minShortFunding" type="number" min="-0.01" max="0.01" step="0.0001" value="-0.0005" />
      </div>
      <button id="runButton" type="button">Run Screener</button>
      <div id="status" class="status">Ready.</div>
    </aside>

    <section>
      <div class="grid">
        <div class="metric"><span>Universe</span><strong id="metricUniverse">-</strong></div>
        <div class="metric"><span>Regression Window</span><strong id="metricWindow">-</strong></div>
        <div class="metric"><span>Latest Hour UTC</span><strong id="metricHour">-</strong></div>
        <div class="metric"><span>Runtime</span><strong id="metricRuntime">-</strong></div>
      </div>

      <div id="errorBox" class="panel section error" style="display:none;"></div>

      <div class="panel section">
        <div class="section-title">
          <h2>Pair Trading Execution Table</h2>
          <small id="signalCount">No scan yet</small>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Residual</th>
                <th>Funding</th>
                <th>Action</th>
                <th>Actual Return</th>
                <th>Fitted Return</th>
                <th>Beta BTC</th>
                <th>Beta ETH</th>
                <th>R2</th>
                <th>Obs</th>
              </tr>
            </thead>
            <tbody id="signalsBody">
              <tr><td colspan="10" class="empty">Click Run Screener to fetch live Binance futures data.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="panel section">
        <div class="section-title">
          <h2>Residual Ranking</h2>
          <small>Top and bottom residuals</small>
        </div>
        <div id="bars" class="bars">
          <div class="empty">No residuals loaded.</div>
        </div>
      </div>

      <div class="panel section">
        <div class="section-title">
          <h2>Full Residual Table</h2>
          <small id="residualCount">No scan yet</small>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Residual</th>
                <th>Funding</th>
                <th>Actual Return</th>
                <th>Fitted Return</th>
                <th>Beta BTC</th>
                <th>Beta ETH</th>
                <th>R2</th>
                <th>Long Eligible</th>
                <th>Short Eligible</th>
              </tr>
            </thead>
            <tbody id="residualsBody">
              <tr><td colspan="10" class="empty">No residuals loaded.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);

    function pct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return `${(Number(value) * 100).toFixed(4)}%`;
    }

    function num(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }

    function cls(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
      return Number(value) >= 0 ? "positive" : "negative";
    }

    function rowHtml(row, signal = false) {
      const action = row.Action || "";
      return `<tr>
        <td>${row.Ticker ?? ""}</td>
        <td class="${cls(row.Residual)}">${pct(row.Residual)}</td>
        <td class="${cls(row.FundingRate)}">${pct(row.FundingRate)}</td>
        ${signal ? `<td class="${action === "Long" ? "long" : "short"}">${action}</td>` : ""}
        <td class="${cls(row.ActualReturn)}">${pct(row.ActualReturn)}</td>
        <td class="${cls(row.FittedReturn)}">${pct(row.FittedReturn)}</td>
        <td>${num(row.BetaBTC)}</td>
        <td>${num(row.BetaETH)}</td>
        <td>${num(row.RSquared)}</td>
        ${signal ? `<td>${row.Observations ?? ""}</td>` : `
          <td>${row.LongEligible ? "Yes" : "No"}</td>
          <td>${row.ShortEligible ? "Yes" : "No"}</td>
        `}
      </tr>`;
    }

    function renderBars(rows) {
      const chosen = [
        ...rows.slice(0, 10),
        ...rows.slice(Math.max(rows.length - 10, 10))
      ];
      const maxAbs = Math.max(...chosen.map((r) => Math.abs(Number(r.Residual || 0))), 0.000001);
      $("bars").innerHTML = chosen.map((row) => {
        const value = Number(row.Residual || 0);
        const width = Math.max(2, Math.abs(value) / maxAbs * 50);
        const neg = value < 0;
        return `<div class="bar-row">
          <div class="bar-label">${row.Ticker}</div>
          <div class="bar-track">
            <div class="bar-fill ${neg ? "neg" : ""}" style="width:${width}%"></div>
          </div>
          <div class="bar-value ${cls(value)}">${pct(value)}</div>
        </div>`;
      }).join("");
    }

    async function runScreener() {
      const button = $("runButton");
      const started = performance.now();
      button.disabled = true;
      $("status").textContent = "Fetching Binance futures data and running BTC/ETH beta regressions...";
      $("errorBox").style.display = "none";

      const params = new URLSearchParams({
        universe_size: $("universeSize").value,
        lookback_hours: $("lookbackHours").value,
        min_regression_obs: $("minObs").value,
        top_n: $("topN").value,
        max_long_funding: $("maxLongFunding").value,
        min_short_funding: $("minShortFunding").value,
      });

      try {
        const response = await fetch(`/api/run?${params.toString()}`);
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }

        $("metricUniverse").textContent = payload.meta.universe_size;
        $("metricWindow").textContent = `${payload.meta.lookback_hours}h`;
        $("metricHour").textContent = payload.meta.latest_hour_utc || "-";
        $("metricRuntime").textContent = `${payload.meta.runtime_seconds.toFixed(1)}s`;

        $("signalsBody").innerHTML = payload.signals.length
          ? payload.signals.map((row) => rowHtml(row, true)).join("")
          : `<tr><td colspan="10" class="empty">No signals passed the filters.</td></tr>`;
        $("residualsBody").innerHTML = payload.residuals.length
          ? payload.residuals.map((row) => rowHtml(row, false)).join("")
          : `<tr><td colspan="10" class="empty">No residuals produced.</td></tr>`;

        $("signalCount").textContent = `${payload.signals.length} selected`;
        $("residualCount").textContent = `${payload.residuals.length} assets`;
        renderBars(payload.residuals);
        $("status").textContent = "Scan complete.";
      } catch (error) {
        $("errorBox").style.display = "block";
        $("errorBox").textContent = error.message;
        $("status").textContent = "Scan failed.";
      } finally {
        button.disabled = false;
        const elapsed = (performance.now() - started) / 1000;
        if ($("metricRuntime").textContent === "-") {
          $("metricRuntime").textContent = `${elapsed.toFixed(1)}s`;
        }
      }
    }

    $("runButton").addEventListener("click", runScreener);
  </script>
</body>
</html>
"""


def parse_int(query: Dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def parse_float(query: Dict[str, list[str]], key: str, default: float) -> float:
    try:
        return float(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def sanitize_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def dataframe_records(frame: pd.DataFrame, columns: Optional[list[str]] = None) -> list[dict]:
    if columns is not None:
        frame = frame[columns].copy()
    records = []
    for row in frame.to_dict(orient="records"):
        records.append({key: sanitize_scalar(value) for key, value in row.items()})
    return records


def build_config_from_query(query: Dict[str, list[str]]) -> ScreenerConfig:
    universe_size = max(10, min(150, parse_int(query, "universe_size", 100)))
    lookback_hours = max(24, min(168, parse_int(query, "lookback_hours", 48)))
    min_obs = max(12, min(lookback_hours, parse_int(query, "min_regression_obs", 24)))
    signal_count = max(1, min(20, parse_int(query, "top_n", 5)))

    return ScreenerConfig(
        universe_size=universe_size,
        lookback_hours=lookback_hours,
        min_regression_obs=min_obs,
        signal_count=signal_count,
        max_long_funding=parse_float(query, "max_long_funding", 0.0005),
        min_short_funding=parse_float(query, "min_short_funding", -0.0005),
    )


def run_dashboard_scan(query: Dict[str, list[str]]) -> dict:
    started = time.perf_counter()
    config = build_config_from_query(query)

    exchange = build_exchange()
    universe = fetch_top_altcoin_symbols(exchange, config)
    closes = build_close_matrix(exchange, universe, config)
    returns = compute_returns(closes)
    residuals = regress_latest_residuals(returns, config)
    funding_rates = fetch_current_funding_rates(
        exchange, residuals["Ticker"].tolist(), config.request_pause_seconds
    )
    scored = attach_funding_and_scores(residuals, funding_rates, config)
    signals = select_signals(scored, config)

    residual_columns = [
        "Ticker",
        "Residual",
        "FundingRate",
        "ActualReturn",
        "FittedReturn",
        "Alpha",
        "BetaBTC",
        "BetaETH",
        "RSquared",
        "Observations",
        "LongEligible",
        "ShortEligible",
        "LongScore",
        "ShortScore",
        "LastReturnHourUTC",
    ]
    signal_columns = [
        "Ticker",
        "Residual",
        "FundingRate",
        "Action",
        "ActualReturn",
        "FittedReturn",
        "BetaBTC",
        "BetaETH",
        "RSquared",
        "Observations",
        "LastReturnHourUTC",
    ]

    latest_hour = None
    if "LastReturnHourUTC" in scored.columns and not scored.empty:
        latest_hour = sanitize_scalar(scored["LastReturnHourUTC"].dropna().max())

    return {
        "ok": True,
        "meta": {
            "universe_size": len(universe),
            "lookback_hours": config.lookback_hours,
            "min_regression_obs": config.min_regression_obs,
            "signals_per_side": config.signal_count,
            "latest_hour_utc": latest_hour,
            "runtime_seconds": time.perf_counter() - started,
        },
        "signals": dataframe_records(signals, signal_columns),
        "residuals": dataframe_records(scored.sort_values("Residual", ascending=False), residual_columns),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "PureAlphaDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/html") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML)
            return

        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            try:
                payload = run_dashboard_scan(query)
                self.send_json(payload)
            except Exception as exc:
                LOGGER.exception("Dashboard scan failed")
                self.send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        self.send_text("Not found", status=HTTPStatus.NOT_FOUND, content_type="text/plain")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Crypto Pure Alpha web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Crypto Pure Alpha dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
