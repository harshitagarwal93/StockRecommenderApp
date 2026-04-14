"""Kite Connect API integration for fetching live portfolio holdings."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from .config import Config, kite_to_yfinance

logger = logging.getLogger(__name__)

KITE_BASE_URL = "https://api.kite.trade"


def _get_access_token(config: Config) -> str:
    """Get access token from config env var, or fall back to CosmosDB stored token."""
    if config.kite_access_token:
        return config.kite_access_token

    # Try loading from CosmosDB
    try:
        from .cosmos_store import CosmosStore

        store = CosmosStore(config)
        token_doc = store.read("kite-token", "kite")
        if token_doc:
            token_date = token_doc.get("date", "")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if token_date == today:
                logger.info("Loaded Kite access token from CosmosDB (today's session)")
                return token_doc.get("access_token", "")
            else:
                logger.warning("Stored Kite token is from %s (expired)", token_date)
    except Exception:
        logger.warning("Failed to load Kite token from CosmosDB", exc_info=True)

    return ""


def fetch_kite_holdings(config: Config) -> list[dict]:
    """Fetch current holdings from Kite Connect API."""
    access_token = _get_access_token(config)
    if not config.kite_api_key or not access_token:
        logger.info("Kite credentials not available — skipping holdings fetch")
        return []

    headers = {
        "X-Kite-Version": "3",
        "Authorization": f"token {config.kite_api_key}:{access_token}",
    }

    holdings = _fetch_holdings(headers)
    positions = _fetch_positions(headers)

    # Combine and deduplicate
    combined: dict[str, dict] = {}
    for h in holdings:
        ticker = kite_to_yfinance(h.get("tradingsymbol", ""))
        if not ticker:
            continue
        combined[ticker] = {
            "ticker": ticker,
            "name": h.get("tradingsymbol", ""),
            "quantity": h.get("quantity", 0),
            "avg_price": h.get("average_price", 0),
            "current_price": h.get("last_price", 0),
            "pnl": h.get("pnl", 0),
            "category": "LARGE_CAP",
        }

    # Add delivery positions not already in holdings
    for p in positions:
        if p.get("product", "") != "CNC":
            continue  # Only delivery positions for long-term
        ticker = kite_to_yfinance(p.get("tradingsymbol", ""))
        if ticker and ticker not in combined and p.get("quantity", 0) > 0:
            combined[ticker] = {
                "ticker": ticker,
                "name": p.get("tradingsymbol", ""),
                "quantity": p.get("quantity", 0),
                "avg_price": p.get("average_price", 0),
                "current_price": p.get("last_price", 0),
                "pnl": p.get("pnl", 0),
                "category": "LARGE_CAP",
            }

    logger.info("Fetched %d holdings from Kite", len(combined))
    return list(combined.values())


def _fetch_holdings(headers: dict) -> list[dict]:
    try:
        resp = requests.get(
            f"{KITE_BASE_URL}/portfolio/holdings",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception:
        logger.exception("Failed to fetch Kite holdings")
        return []


def _fetch_positions(headers: dict) -> list[dict]:
    try:
        resp = requests.get(
            f"{KITE_BASE_URL}/portfolio/positions",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        net = data.get("data", {}).get("net", [])
        return net
    except Exception:
        logger.exception("Failed to fetch Kite positions")
        return []


def get_kite_margin(config: Config) -> float:
    """Get available cash margin from Kite account."""
    access_token = _get_access_token(config)
    if not config.kite_api_key or not access_token:
        return 0.0

    headers = {
        "X-Kite-Version": "3",
        "Authorization": f"token {config.kite_api_key}:{access_token}",
    }

    try:
        resp = requests.get(
            f"{KITE_BASE_URL}/user/margins/equity",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        available = data.get("data", {}).get("available", {})
        return float(available.get("cash", 0))
    except Exception:
        logger.exception("Failed to fetch Kite margin")
        return 0.0
