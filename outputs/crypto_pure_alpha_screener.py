#!/usr/bin/env python3
"""
Crypto Pure Alpha Long/Short Screener

Public-data screener for Binance USD-M futures. It builds a top-volume altcoin
universe, estimates each coin's BTC/ETH beta from recent hourly returns, and
ranks the latest idiosyncratic residual for long/short candidates.

Install:
    pip install ccxt pandas numpy statsmodels

Run:
    python crypto_pure_alpha_screener.py --top-n 5 --lookback-hours 48

Notes:
    A literal single-hour cross-sectional regression cannot estimate individual
    BTC/ETH betas because BTC and ETH returns are common across all coins at that
    hour. This script uses the standard tradable interpretation: per-asset
    rolling hourly time-series regressions, then ranks the latest residuals
    cross-sectionally.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError as exc:
    raise SystemExit("Missing dependency: ccxt. Install with: pip install ccxt") from exc

try:
    import statsmodels.api as sm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: statsmodels. Install with: pip install statsmodels"
    ) from exc


LOGGER = logging.getLogger("pure_alpha_screener")

DEFAULT_STABLE_BASES = {
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "DAI",
    "TUSD",
    "USDP",
    "GUSD",
    "EUR",
    "EURI",
    "AEUR",
    "TRY",
    "BRL",
    "GBP",
    "AUD",
    "JPY",
}


@dataclass(frozen=True)
class ScreenerConfig:
    quote: str = "USDT"
    universe_size: int = 100
    lookback_hours: int = 48
    min_regression_obs: int = 24
    signal_count: int = 5
    max_long_funding: float = 0.0005
    min_short_funding: float = -0.0005
    funding_penalty_multiplier: float = 2.0
    request_pause_seconds: float = 0.05


def build_exchange() -> "ccxt.Exchange":
    """Create a Binance USD-M futures client for public endpoints."""
    exchange = ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }
    )
    exchange.load_markets()
    return exchange


def is_eligible_alt_market(market: dict, quote: str, stable_bases: set[str]) -> bool:
    """Return True for active linear USDT perpetual altcoin markets."""
    base = market.get("base")
    symbol = market.get("symbol", "")
    settle = market.get("settle")

    if not market.get("active", True):
        return False
    if market.get("quote") != quote:
        return False
    if settle != quote:
        return False
    if not market.get("swap", False):
        return False
    if not market.get("linear", False):
        return False
    if base in stable_bases or base in {"BTC", "ETH"}:
        return False
    if any(token in symbol.upper() for token in ("UP/", "DOWN/", "BULL/", "BEAR/")):
        return False
    return True


def fetch_top_altcoin_symbols(
    exchange: "ccxt.Exchange", config: ScreenerConfig
) -> List[str]:
    """Fetch and rank Binance futures altcoins by 24h quote trading volume."""
    markets = exchange.markets
    tickers = exchange.fetch_tickers()
    rows = []

    for symbol, market in markets.items():
        if not is_eligible_alt_market(market, config.quote, DEFAULT_STABLE_BASES):
            continue

        ticker = tickers.get(symbol) or {}
        quote_volume = ticker.get("quoteVolume")
        if quote_volume is None:
            base_volume = ticker.get("baseVolume")
            last = ticker.get("last")
            if base_volume is not None and last is not None:
                quote_volume = base_volume * last

        if quote_volume is None or not np.isfinite(float(quote_volume)):
            continue
        rows.append((symbol, float(quote_volume)))

    ranked = sorted(rows, key=lambda item: item[1], reverse=True)
    symbols = [symbol for symbol, _ in ranked[: config.universe_size]]
    if not symbols:
        raise RuntimeError("No eligible altcoin futures markets found.")

    LOGGER.info("Selected %d symbols by 24h quote volume.", len(symbols))
    return symbols


def fetch_hourly_close(
    exchange: "ccxt.Exchange",
    symbol: str,
    lookback_hours: int,
    request_pause_seconds: float,
) -> Optional[pd.Series]:
    """Fetch hourly close prices for one futures symbol."""
    limit = lookback_hours + 5
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=limit)
        time.sleep(request_pause_seconds)
    except Exception as exc:  # ccxt raises exchange-specific subclasses.
        LOGGER.warning("OHLCV fetch failed for %s: %s", symbol, exc)
        return None

    if not ohlcv:
        LOGGER.warning("No OHLCV returned for %s.", symbol)
        return None

    frame = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    close = frame["close"].astype(float).rename(symbol)
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    return close.tail(lookback_hours + 1)


def build_close_matrix(
    exchange: "ccxt.Exchange", symbols: Iterable[str], config: ScreenerConfig
) -> pd.DataFrame:
    """Fetch closes for BTC, ETH, and the selected altcoin universe."""
    all_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", *symbols]
    series = []

    for idx, symbol in enumerate(all_symbols, start=1):
        LOGGER.info("Fetching %s (%d/%d)", symbol, idx, len(all_symbols))
        close = fetch_hourly_close(
            exchange,
            symbol,
            config.lookback_hours,
            config.request_pause_seconds,
        )
        if close is not None and len(close) >= config.min_regression_obs + 1:
            series.append(close)
        else:
            LOGGER.warning("Skipping %s due to insufficient close history.", symbol)

    if len(series) < 3:
        raise RuntimeError("Insufficient price data to run regression.")

    closes = pd.concat(series, axis=1).sort_index()
    closes = closes.dropna(axis=1, thresh=config.min_regression_obs + 1)
    if "BTC/USDT:USDT" not in closes or "ETH/USDT:USDT" not in closes:
        raise RuntimeError("BTC/ETH benchmark data missing after cleaning.")
    return closes


def compute_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Compute simple hourly returns from closes."""
    returns = closes.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.dropna(how="all")


def regress_latest_residuals(
    returns: pd.DataFrame, config: ScreenerConfig
) -> pd.DataFrame:
    """
    Estimate each asset's BTC/ETH beta and latest residual.

    Regression:
        R_i = alpha_i + beta_BTC * R_BTC + beta_ETH * R_ETH + residual_i
    """
    btc_col = "BTC/USDT:USDT"
    eth_col = "ETH/USDT:USDT"
    factor_cols = [btc_col, eth_col]
    alt_cols = [col for col in returns.columns if col not in factor_cols]
    records = []

    for symbol in alt_cols:
        sample = returns[[symbol, *factor_cols]].dropna().tail(config.lookback_hours)
        if len(sample) < config.min_regression_obs:
            LOGGER.debug("Skipping %s: only %d aligned observations.", symbol, len(sample))
            continue

        y = sample[symbol]
        x = sm.add_constant(sample[factor_cols], has_constant="add")

        try:
            model = sm.OLS(y, x).fit()
        except Exception as exc:
            LOGGER.warning("Regression failed for %s: %s", symbol, exc)
            continue

        latest_x = x.iloc[[-1]]
        fitted_latest = float(model.predict(latest_x).iloc[0])
        actual_latest = float(y.iloc[-1])
        residual_latest = actual_latest - fitted_latest

        records.append(
            {
                "Ticker": symbol,
                "Residual": residual_latest,
                "ActualReturn": actual_latest,
                "FittedReturn": fitted_latest,
                "Alpha": float(model.params.get("const", np.nan)),
                "BetaBTC": float(model.params.get(btc_col, np.nan)),
                "BetaETH": float(model.params.get(eth_col, np.nan)),
                "RSquared": float(model.rsquared),
                "Observations": int(model.nobs),
                "LastReturnHourUTC": sample.index[-1],
            }
        )

    if not records:
        raise RuntimeError("No valid residuals produced.")

    return pd.DataFrame(records).sort_values("Residual", ascending=False)


def fetch_current_funding_rates(
    exchange: "ccxt.Exchange",
    symbols: Iterable[str],
    request_pause_seconds: float,
) -> Dict[str, float]:
    """Fetch latest funding rates, using batch endpoint when available."""
    symbols = list(symbols)
    rates: Dict[str, float] = {}

    try:
        funding_payload = exchange.fetch_funding_rates(symbols)
        for symbol, payload in funding_payload.items():
            rate = payload.get("fundingRate")
            if rate is not None and np.isfinite(float(rate)):
                rates[symbol] = float(rate)
        return rates
    except Exception as exc:
        LOGGER.info("Batch funding fetch unavailable; falling back per symbol: %s", exc)

    for symbol in symbols:
        try:
            payload = exchange.fetch_funding_rate(symbol)
            rate = payload.get("fundingRate")
            if rate is not None and np.isfinite(float(rate)):
                rates[symbol] = float(rate)
            time.sleep(request_pause_seconds)
        except Exception as exc:
            LOGGER.warning("Funding fetch failed for %s: %s", symbol, exc)

    return rates


def attach_funding_and_scores(
    residuals: pd.DataFrame, funding_rates: Dict[str, float], config: ScreenerConfig
) -> pd.DataFrame:
    """Attach funding and compute side-specific ranking scores."""
    frame = residuals.copy()
    frame["FundingRate"] = frame["Ticker"].map(funding_rates)
    frame["FundingRate"] = frame["FundingRate"].astype(float)

    long_penalty = config.funding_penalty_multiplier * frame["FundingRate"].clip(lower=0)
    short_penalty = config.funding_penalty_multiplier * (-frame["FundingRate"]).clip(lower=0)

    frame["LongScore"] = frame["Residual"] - long_penalty.fillna(0.0)
    frame["ShortScore"] = frame["Residual"] + short_penalty.fillna(0.0)
    frame["LongEligible"] = frame["FundingRate"].isna() | (
        frame["FundingRate"] <= config.max_long_funding
    )
    frame["ShortEligible"] = frame["FundingRate"].isna() | (
        frame["FundingRate"] >= config.min_short_funding
    )
    return frame


def select_signals(scored: pd.DataFrame, config: ScreenerConfig) -> pd.DataFrame:
    """Select long and short candidates using residual rank plus funding filters."""
    positive = scored[scored["Residual"] > 0]
    negative = scored[scored["Residual"] < 0]

    longs = (
        positive[positive["LongEligible"]]
        .sort_values(["LongScore", "Residual"], ascending=False)
        .head(config.signal_count)
        .copy()
    )
    shorts = (
        negative[negative["ShortEligible"]]
        .sort_values(["ShortScore", "Residual"], ascending=True)
        .head(config.signal_count)
        .copy()
    )

    if len(longs) < config.signal_count:
        LOGGER.warning(
            "Only %d long candidates passed residual/funding filters.", len(longs)
        )
    if len(shorts) < config.signal_count:
        LOGGER.warning(
            "Only %d short candidates passed residual/funding filters.", len(shorts)
        )

    longs["Action"] = "Long"
    shorts["Action"] = "Short"
    table = pd.concat([longs, shorts], axis=0)

    output_cols = [
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
    table = table[output_cols].reset_index(drop=True)
    return table


def format_signal_table(table: pd.DataFrame) -> str:
    """Return a readable fixed-width table for console output."""
    printable = table.copy()
    pct_cols = ["Residual", "FundingRate", "ActualReturn", "FittedReturn"]
    for col in pct_cols:
        printable[col] = printable[col].map(
            lambda value: "n/a" if pd.isna(value) else f"{value * 100: .4f}%"
        )
    for col in ["BetaBTC", "BetaETH", "RSquared"]:
        printable[col] = printable[col].map(
            lambda value: "n/a" if pd.isna(value) else f"{value: .3f}"
        )
    printable["LastReturnHourUTC"] = printable["LastReturnHourUTC"].astype(str)
    return printable.to_string(index=False)


def save_outputs(
    signal_table: pd.DataFrame,
    residual_table: pd.DataFrame,
    output_prefix: Optional[str],
) -> None:
    """Optionally save signal and full residual tables to CSV files."""
    if not output_prefix:
        return

    signal_path = f"{output_prefix}_signals.csv"
    residual_path = f"{output_prefix}_all_residuals.csv"
    signal_table.to_csv(signal_path, index=False)
    residual_table.to_csv(residual_path, index=False)
    LOGGER.info("Saved signal table to %s", signal_path)
    LOGGER.info("Saved full residual table to %s", residual_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crypto Pure Alpha Long/Short Screener for Binance Futures."
    )
    parser.add_argument("--universe-size", type=int, default=100)
    parser.add_argument("--lookback-hours", type=int, default=48)
    parser.add_argument("--min-regression-obs", type=int, default=24)
    parser.add_argument("--top-n", type=int, default=5, help="Longs and shorts per side.")
    parser.add_argument(
        "--max-long-funding",
        type=float,
        default=0.0005,
        help="Skip longs above this funding rate. 0.0005 = 5 bps.",
    )
    parser.add_argument(
        "--min-short-funding",
        type=float,
        default=-0.0005,
        help="Skip shorts below this funding rate. -0.0005 = -5 bps.",
    )
    parser.add_argument(
        "--funding-penalty-multiplier",
        type=float,
        default=2.0,
        help="Penalty applied to side-unfavorable funding when ranking.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional CSV prefix, e.g. outputs/pure_alpha.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = ScreenerConfig(
        universe_size=args.universe_size,
        lookback_hours=args.lookback_hours,
        min_regression_obs=args.min_regression_obs,
        signal_count=args.top_n,
        max_long_funding=args.max_long_funding,
        min_short_funding=args.min_short_funding,
        funding_penalty_multiplier=args.funding_penalty_multiplier,
    )

    if config.lookback_hours < config.min_regression_obs:
        raise ValueError("--lookback-hours must be >= --min-regression-obs")

    exchange = build_exchange()
    universe = fetch_top_altcoin_symbols(exchange, config)
    closes = build_close_matrix(exchange, universe, config)
    returns = compute_returns(closes)
    residuals = regress_latest_residuals(returns, config)

    funding_rates = fetch_current_funding_rates(
        exchange, residuals["Ticker"].tolist(), config.request_pause_seconds
    )
    scored = attach_funding_and_scores(residuals, funding_rates, config)
    signal_table = select_signals(scored, config)

    print("\nCrypto Pure Alpha Long/Short Screener")
    print(f"Universe: top {len(universe)} Binance USD-M altcoin futures by 24h volume")
    print(f"Regression window: {config.lookback_hours} hourly bars")
    print(f"Signals per side: {config.signal_count}")
    print("\nPair Trading Execution Table")
    print(format_signal_table(signal_table))

    save_outputs(signal_table, scored, args.output_prefix)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
