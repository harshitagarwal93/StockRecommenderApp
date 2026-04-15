"""Fetch stock price and fundamental data from Yahoo Finance (free, public)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from .config import get_full_universe
from .models import FundamentalData

logger = logging.getLogger(__name__)


def fetch_batch_prices(
    tickers: list[str], period: str = "1y"
) -> dict[str, pd.DataFrame]:
    """Download OHLCV history for multiple tickers in a single call."""
    if not tickers:
        return {}

    logger.info("Downloading price data for %d tickers …", len(tickers))

    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        logger.exception("Batch download failed")
        return {}

    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw
            else:
                df = raw[ticker]

            # Flatten multi-level columns if present (yfinance v2+)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1) if "Ticker" in df.columns.names else df[ticker]

            df = df.dropna()
            if df.shape[0] >= 50:
                result[ticker] = df
            else:
                logger.warning("%s: insufficient data (%d rows)", ticker, df.shape[0])
        except (KeyError, TypeError):
            logger.warning("%s: no data returned", ticker)

    logger.info("Received valid data for %d / %d tickers", len(result), len(tickers))
    return result


def fetch_fundamental_data(ticker: str, category: str = "LARGE_CAP") -> FundamentalData:
    """Fetch fundamental metrics for a single ticker via yfinance .info."""
    universe = get_full_universe()
    default_name = universe.get(ticker, (ticker, category))[0]

    try:
        t = yf.Ticker(ticker)
        info: dict[str, Any] = t.info
    except Exception:
        logger.warning("%s: failed to fetch fundamentals", ticker)
        return FundamentalData(ticker=ticker, name=default_name, category=category)

    # Compute historical PE averages from price history + trailing EPS
    pe_5d, pe_30d, pe_90d = 0.0, 0.0, 0.0
    trailing_eps = info.get("trailingEps") or 0
    if trailing_eps > 0:
        try:
            hist = t.history(period="6mo")["Close"]
            if len(hist) >= 5:
                pe_5d = round(float(hist.tail(5).mean() / trailing_eps), 2)
            if len(hist) >= 30:
                pe_30d = round(float(hist.tail(30).mean() / trailing_eps), 2)
            if len(hist) >= 90:
                pe_90d = round(float(hist.tail(90).mean() / trailing_eps), 2)
        except Exception:
            pass

    return FundamentalData(
        ticker=ticker,
        name=info.get("shortName") or default_name,
        sector=info.get("sector") or "Unknown",
        market_cap=info.get("marketCap") or 0,
        pe_ratio=info.get("trailingPE") or 0,
        pe_5d_avg=pe_5d,
        pe_30d_avg=pe_30d,
        pe_90d_avg=pe_90d,
        pb_ratio=info.get("priceToBook") or 0,
        dividend_yield=(info.get("dividendYield") or 0) * 100,
        roe=(info.get("returnOnEquity") or 0) * 100,
        debt_to_equity=info.get("debtToEquity") or 0,
        revenue_growth=(info.get("revenueGrowth") or 0) * 100,
        profit_margin=(info.get("profitMargins") or 0) * 100,
        category=category,
    )
