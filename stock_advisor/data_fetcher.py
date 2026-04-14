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
        info: dict[str, Any] = yf.Ticker(ticker).info
    except Exception:
        logger.warning("%s: failed to fetch fundamentals", ticker)
        return FundamentalData(ticker=ticker, name=default_name, category=category)

    return FundamentalData(
        ticker=ticker,
        name=info.get("shortName") or default_name,
        sector=info.get("sector") or "Unknown",
        market_cap=info.get("marketCap") or 0,
        pe_ratio=info.get("trailingPE") or 0,
        pb_ratio=info.get("priceToBook") or 0,
        dividend_yield=(info.get("dividendYield") or 0) * 100,
        roe=(info.get("returnOnEquity") or 0) * 100,
        debt_to_equity=info.get("debtToEquity") or 0,
        revenue_growth=(info.get("revenueGrowth") or 0) * 100,
        profit_margin=(info.get("profitMargins") or 0) * 100,
        category=category,
    )
