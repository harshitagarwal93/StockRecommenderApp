"""Main orchestrator — ties data fetching, analysis, LLM, and portfolio together."""

from __future__ import annotations

import logging
from datetime import date

from .config import Config, get_full_universe
from .cosmos_store import CosmosStore
from .data_fetcher import fetch_batch_prices, fetch_fundamental_data
from .llm_analyzer import analyze as llm_analyze
from .models import DailyRecommendation, StockAnalysis
from .notifier import send_recommendation_email
from .portfolio_manager import PortfolioManager
from .technical_analysis import composite_score, compute_indicators

logger = logging.getLogger(__name__)

TOP_CANDIDATES = 20


def run_daily_analysis(config: Config | None = None) -> DailyRecommendation:
    """Execute the full daily analysis pipeline."""
    if config is None:
        config = Config()

    store = CosmosStore(config)
    pm = PortfolioManager(config, store)
    portfolio = pm.get_portfolio()
    universe = get_full_universe()

    logger.info(
        "Starting analysis — %d universe stocks, %d holdings, Rs.%.0f cash, source=%s",
        len(universe), len(portfolio.holdings), portfolio.cash_balance, portfolio.source,
    )

    # Step 1: Batch price download
    all_tickers = list(universe.keys())
    for t in portfolio.holding_tickers():
        if t not in all_tickers:
            all_tickers.append(t)

    price_data = fetch_batch_prices(all_tickers)

    if not price_data:
        logger.error("No price data returned — aborting analysis")
        return _empty_recommendation("No market data available. Market may be closed.")

    # Step 2: Technical indicators
    analyses: list[StockAnalysis] = []
    for ticker, df in price_data.items():
        name, category = universe.get(ticker, (ticker, "LARGE_CAP"))
        try:
            tech = compute_indicators(ticker, df)
            score = composite_score(tech)
            analyses.append(
                StockAnalysis(
                    ticker=ticker, name=name, category=category,
                    technicals=tech, composite_score=score,
                )
            )
        except Exception:
            logger.warning("Failed indicators for %s", ticker, exc_info=True)

    # Step 3: Pre-screen top N + portfolio holdings
    analyses.sort(key=lambda a: a.composite_score, reverse=True)
    held_tickers = set(portfolio.holding_tickers())

    top = []
    count = 0
    for a in analyses:
        if a.ticker in held_tickers or count < TOP_CANDIDATES:
            top.append(a)
            if a.ticker not in held_tickers:
                count += 1

    # Step 4: Fundamental data for candidates
    for analysis in top:
        try:
            analysis.fundamentals = fetch_fundamental_data(analysis.ticker, analysis.category)
        except Exception:
            logger.warning("Fundamentals failed for %s", analysis.ticker)

    # Step 5: Update current prices in portfolio
    price_map = {a.ticker: a.technicals.current_price for a in analyses}
    for h in portfolio.holdings:
        if h["ticker"] in price_map:
            h["current_price"] = price_map[h["ticker"]]

    # Step 6: LLM analysis
    recommendation = llm_analyze(config, portfolio, top)

    # Step 7: Persist
    pm.save_recommendation(recommendation)
    logger.info("Recommendation saved: %s", recommendation.id)

    # Step 8: Email
    send_recommendation_email(config, recommendation)

    return recommendation


def _empty_recommendation(reason: str) -> DailyRecommendation:
    today = date.today().isoformat()
    return DailyRecommendation(
        id=f"rec-{today}", date=today,
        market_outlook=reason, recommendations=[],
        analysis_summary=reason, risk_assessment="N/A",
    )
