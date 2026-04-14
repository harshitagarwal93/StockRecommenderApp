"""Portfolio tracking — load from Kite or local, update, persist."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from .config import Config
from .cosmos_store import CosmosStore
from .kite_client import fetch_kite_holdings, get_kite_margin
from .models import DailyRecommendation, Portfolio

logger = logging.getLogger(__name__)

PORTFOLIO_ID = "current-portfolio"
PARTITION_KEY = "portfolio"


class PortfolioManager:
    def __init__(self, config: Config, store: CosmosStore):
        self._config = config
        self._store = store

    def get_portfolio(self) -> Portfolio:
        """Load portfolio from Kite (live) or local storage."""
        # Try Kite first
        kite_holdings = fetch_kite_holdings(self._config)
        if kite_holdings:
            cash = get_kite_margin(self._config)
            invested = sum(h["quantity"] * h["avg_price"] for h in kite_holdings)
            portfolio = Portfolio(
                holdings=kite_holdings,
                cash_balance=cash if cash > 0 else self._config.initial_capital,
                initial_investment=invested + cash,
                last_updated=datetime.now(timezone.utc).isoformat(),
                source="kite",
            )
            self._store.upsert(portfolio.to_dict())
            return portfolio

        # Fallback to stored portfolio
        doc = self._store.read(PORTFOLIO_ID, PARTITION_KEY)
        if doc:
            return Portfolio.from_dict(doc)

        logger.info("No portfolio found — creating with Rs.%s capital", self._config.initial_capital)
        portfolio = Portfolio(
            cash_balance=self._config.initial_capital,
            initial_investment=self._config.initial_capital,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        self._store.upsert(portfolio.to_dict())
        return portfolio

    def save_portfolio(self, portfolio: Portfolio) -> None:
        portfolio.last_updated = datetime.now(timezone.utc).isoformat()
        self._store.upsert(portfolio.to_dict())

    def save_recommendation(self, rec: DailyRecommendation) -> None:
        self._store.upsert(rec.to_dict())

    def get_latest_recommendation(self) -> DailyRecommendation | None:
        today = date.today().isoformat()
        doc = self._store.read(f"rec-{today}", "recommendation")
        if doc:
            return DailyRecommendation.from_dict(doc)
        return None

    def get_recommendation_history(self, limit: int = 7) -> list[dict]:
        docs = self._store.query(
            "SELECT TOP @limit * FROM c WHERE c.type = 'recommendation' ORDER BY c.date DESC",
            [{"name": "@limit", "value": limit}],
        )
        return docs
