"""Data models used across the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StockHolding:
    ticker: str
    name: str
    quantity: int
    avg_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    category: str = "LARGE_CAP"


@dataclass
class Portfolio:
    id: str = "current-portfolio"
    type: str = "portfolio"
    holdings: list[dict] = field(default_factory=list)
    cash_balance: float = 0.0
    initial_investment: float = 0.0
    last_updated: str = ""
    source: str = "manual"  # "kite" or "manual"

    def total_invested_value(self) -> float:
        return sum(h["quantity"] * h["avg_price"] for h in self.holdings)

    def total_current_value(self) -> float:
        return sum(
            h["quantity"] * h.get("current_price", h["avg_price"])
            for h in self.holdings
        )

    def holding_tickers(self) -> list[str]:
        return [h["ticker"] for h in self.holdings]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "holdings": self.holdings,
            "cash_balance": self.cash_balance,
            "initial_investment": self.initial_investment,
            "last_updated": self.last_updated,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Portfolio:
        return cls(
            id=data.get("id", "current-portfolio"),
            type=data.get("type", "portfolio"),
            holdings=data.get("holdings", []),
            cash_balance=data.get("cash_balance", 0.0),
            initial_investment=data.get("initial_investment", 0.0),
            last_updated=data.get("last_updated", ""),
            source=data.get("source", "manual"),
        )


@dataclass
class TechnicalIndicators:
    ticker: str
    current_price: float = 0.0
    rsi_14: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_20: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    volume_avg_20: float = 0.0
    current_volume: float = 0.0
    pct_from_52w_high: float = 0.0
    pct_from_52w_low: float = 0.0
    price_change_1m: float = 0.0
    price_change_3m: float = 0.0
    price_change_6m: float = 0.0


@dataclass
class FundamentalData:
    ticker: str
    name: str = ""
    sector: str = "Unknown"
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    pe_5d_avg: float = 0.0
    pe_30d_avg: float = 0.0
    pe_90d_avg: float = 0.0
    pb_ratio: float = 0.0
    dividend_yield: float = 0.0
    roe: float = 0.0
    debt_to_equity: float = 0.0
    revenue_growth: float = 0.0
    profit_margin: float = 0.0
    category: str = "LARGE_CAP"


@dataclass
class StockAnalysis:
    ticker: str
    name: str
    category: str
    technicals: TechnicalIndicators
    fundamentals: FundamentalData | None = None
    composite_score: float = 0.0


@dataclass
class Recommendation:
    ticker: str
    name: str
    action: str
    quantity: int
    current_price: float
    target_price: float
    stop_loss: float
    reason: str
    confidence: str
    citations: list[str] = field(default_factory=list)
    technicals_url: str = ""
    fundamentals_url: str = ""


@dataclass
class DailyRecommendation:
    id: str
    type: str = "recommendation"
    date: str = ""
    market_outlook: str = ""
    portfolio_value: float = 0.0
    portfolio_returns_pct: float = 0.0
    recommendations: list[dict] = field(default_factory=list)
    analysis_summary: str = ""
    risk_assessment: str = ""
    created_at: str = ""
    kite_holdings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "date": self.date,
            "market_outlook": self.market_outlook,
            "portfolio_value": self.portfolio_value,
            "portfolio_returns_pct": self.portfolio_returns_pct,
            "recommendations": self.recommendations,
            "analysis_summary": self.analysis_summary,
            "risk_assessment": self.risk_assessment,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "kite_holdings": self.kite_holdings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DailyRecommendation:
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "recommendation"),
            date=data.get("date", ""),
            market_outlook=data.get("market_outlook", ""),
            portfolio_value=data.get("portfolio_value", 0.0),
            portfolio_returns_pct=data.get("portfolio_returns_pct", 0.0),
            recommendations=data.get("recommendations", []),
            analysis_summary=data.get("analysis_summary", ""),
            risk_assessment=data.get("risk_assessment", ""),
            created_at=data.get("created_at", ""),
            kite_holdings=data.get("kite_holdings", []),
        )
