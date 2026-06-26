from __future__ import annotations

import json
import math
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
if str(OUTPUTS) not in sys.path:
    sys.path.insert(0, str(OUTPUTS))

from crypto_pure_alpha_screener import (  # noqa: E402
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


def parse_int(query, key, default):
    try:
        return int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def parse_float(query, key, default):
    try:
        return float(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def sanitize_scalar(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def dataframe_records(frame, columns):
    frame = frame[columns].copy()
    records = []
    for row in frame.to_dict(orient="records"):
        records.append({key: sanitize_scalar(value) for key, value in row.items()})
    return records


def build_config(query):
    lookback_hours = max(24, min(96, parse_int(query, "lookback_hours", 48)))
    min_obs = max(12, min(lookback_hours, parse_int(query, "min_regression_obs", 24)))
    return ScreenerConfig(
        universe_size=max(10, min(100, parse_int(query, "universe_size", 50))),
        lookback_hours=lookback_hours,
        min_regression_obs=min_obs,
        signal_count=max(1, min(10, parse_int(query, "top_n", 5))),
        max_long_funding=parse_float(query, "max_long_funding", 0.0005),
        min_short_funding=parse_float(query, "min_short_funding", -0.0005),
        request_pause_seconds=0.02,
    )


def run_scan(query):
    started = time.perf_counter()
    config = build_config(query)
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
    residual_columns = [
        "Ticker",
        "Residual",
        "FundingRate",
        "ActualReturn",
        "FittedReturn",
        "BetaBTC",
        "BetaETH",
        "RSquared",
        "Observations",
        "LongEligible",
        "ShortEligible",
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
        "residuals": dataframe_records(
            scored.sort_values("Residual", ascending=False), residual_columns
        ),
    }


class handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        try:
            self.send_json(run_scan(query))
        except Exception as exc:
            self.send_json(
                {"ok": False, "error": str(exc)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
