"""LLM-based stock analysis via Azure AI Foundry or GitHub Models API."""

from __future__ import annotations

import json
import logging
from datetime import date

from openai import OpenAI, AzureOpenAI

from .config import Config
from .models import DailyRecommendation, Portfolio, StockAnalysis

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert financial analyst specializing in Indian equity markets (NSE/BSE). Your role is to evaluate stocks for a long-term portfolio (6-24 month horizon) using both technical and fundamental analysis, then generate structured investment recommendations.

## Context
- Market: Indian equities (NSE)
- Currency: INR (Rs.)
- Benchmark valuations: Indian sector peers, not global averages
- Regulatory context: SEBI-listed companies
- Max single stock allocation: {max_alloc}% of total portfolio

## Analysis Framework

### Technical Analysis (weight: 40%)
Evaluate these signals from the data provided. Omit gracefully if not available:
- Trend: Price vs 50/200-day SMA; trend direction (uptrend, downtrend, sideways)
- Momentum: RSI(14) — overbought >70, oversold <30; MACD crossover signals
- Volume: Confirm strength with volume vs 20-day average
- Support/Resistance: 52-week high/low proximity, Bollinger Band position
- Price momentum: 1M, 3M, 6M returns for trend confirmation

### Fundamental Analysis (weight: 60%)
Evaluate these from the data provided. Omit gracefully if not available:
- Valuation: P/E, P/B vs sector peers
- Profitability: ROE, profit margin
- Growth: Revenue growth YoY
- Balance sheet: Debt-to-equity ratio
- Dividend: Yield as income indicator
- Business quality: Sector positioning, market cap category

## Scoring (mandatory for every stock analyzed)
- Technical score: 1 (very bearish) to 10 (very bullish)
- Fundamental score: 1 (very weak) to 10 (very strong)
- Composite score: (Technical x 0.4) + (Fundamental x 0.6), rounded to 1 decimal

## Recommendation Logic (deterministic — follow strictly)
- Composite >= 7.0 → BUY (if within budget and position limits)
- Composite < 5.0 → SELL (for existing holdings only)
- Composite 5.0-6.9 → OMIT (do not include — this is effectively HOLD)
- Confidence: HIGH if data is complete and signals align, MEDIUM if partial data, LOW if conflicting signals

## Critical Rules
- Total cost of ALL BUY recommendations must NOT exceed the TOTAL_INVESTMENT_BUDGET
- You can ONLY recommend SELL for stocks currently held in the portfolio
- Do NOT include stocks with composite 5.0-6.9 — omit them entirely (HOLD is implicit)
- Never fabricate or assume financial data not explicitly provided
- Every metric you cite MUST come from the data provided
- If no stock meets BUY/SELL criteria, return ZERO recommendations (empty array)
- For BUY: set target based on PE re-rating potential; stop loss at SMA200 or nearest support
- For SELL: cite specific deterioration trigger from the data
- Risk:Reward ratio must be at least 1:2 for BUY recommendations
"""

USER_PROMPT_TEMPLATE = """\
Date: {date}
Analysis Mode: {mode}

=== CURRENT PORTFOLIO ===
Cash Available: Rs.{cash:,.0f}
Total Holdings Value: Rs.{holdings_value:,.0f}
Positions: {num_positions}

{holdings_table}

=== CONSTRAINTS ===
- Total investment budget: Rs.{max_buy_amount:,.0f} (total across all BUY recommendations)
- Max portfolio positions: {max_positions}
- Max single stock allocation: {max_alloc}% of total portfolio
- Portfolio value: Rs.{total_value:,.0f}

=== CANDIDATES (ranked by pre-screen composite score) ===

{stock_data}

=== TASK ===
{mode_instruction}

Respond ONLY with valid JSON. No preamble, no markdown fences, no text outside JSON:

{{
  "market_outlook": "2-3 sentence assessment citing specific macro factors or index context",
  "portfolio_assessment": "Assessment noting specific weak and strong performers by name",
  "recommendations": [
    {{
      "ticker": "SYMBOL.NS",
      "name": "Company Name",
      "action": "BUY or SELL",
      "quantity": 5,
      "current_price": 1234.56,
      "target_price": 1400.00,
      "stop_loss": 1100.00,
      "reason": "2-3 sentences citing SPECIFIC metrics from the data",
      "confidence": "HIGH or MEDIUM or LOW",
      "fundamental_score": 7,
      "technical_score": 8,
      "composite_score": 7.4,
      "risk_reward_ratio": "1:2.5",
      "data_quality": "Complete / Partial — note any missing data",
      "citations": [
        "PE 18.5x vs sector avg 22x — 16% discount",
        "RSI 38 oversold + MACD histogram positive at 0.5",
        "Revenue growth 15% YoY with margin at 14%"
      ]
    }}
  ],
  "analysis_summary": "Key factors driving recommendations with composite scores cited",
  "risk_assessment": "Specific risks with trigger levels"
}}
"""

MODE_INSTRUCTIONS = {
    "all": "Score ALL candidates using the composite formula. Include only stocks with composite >= 7.0 (BUY) or composite < 5.0 for holdings (SELL). Omit everything in between. Budget applies to total BUY cost.",
    "buy": "Score ALL candidates. Include only stocks with composite >= 7.0 as BUY. Do NOT include any SELL. Allocate within the investment budget proportionally by conviction.",
    "sell": "Score ALL current holdings. Include only stocks with composite < 5.0 as SELL. Do NOT include any BUY. For each SELL, cite the specific deterioration trigger.",
}


def _build_client(config: Config) -> tuple[AzureOpenAI | OpenAI, str]:
    """Create the appropriate OpenAI client based on provider config."""
    if config.llm_provider == "azure_foundry" and config.azure_ai_endpoint:
        client = AzureOpenAI(
            azure_endpoint=config.azure_ai_endpoint,
            api_key=config.azure_ai_key,
            api_version="2025-03-01-preview",
        )
        return client, config.azure_ai_deployment

    # Fallback: GitHub Models (free)
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=config.github_token,
    )
    return client, "gpt-4o"


def _format_holdings(portfolio: Portfolio) -> str:
    if not portfolio.holdings:
        return "(No current holdings — fresh portfolio)"

    lines = [
        f"{'Ticker':<18} {'Name':<25} {'Qty':>5} {'Avg Price':>12} {'CMP':>12} {'P&L':>12}"
    ]
    lines.append("-" * 90)
    for h in portfolio.holdings:
        cmp = h.get("current_price", h["avg_price"])
        pnl = (cmp - h["avg_price"]) * h["quantity"]
        lines.append(
            f"{h['ticker']:<18} {h.get('name', '')[:24]:<25} {h['quantity']:>5} "
            f"Rs.{h['avg_price']:>9,.2f} Rs.{cmp:>9,.2f} Rs.{pnl:>9,.2f}"
        )
    return "\n".join(lines)


def _format_stock_data(analyses: list[StockAnalysis]) -> str:
    sections = []
    for a in analyses:
        t = a.technicals
        section = f"""--- {a.ticker} | {a.name} | {a.category} | Score: {a.composite_score:.0f}/100 ---
Price: Rs.{t.current_price:,.2f} | RSI(14): {t.rsi_14} | MACD Hist: {t.macd_histogram}
SMA50: Rs.{t.sma_50:,.2f} | SMA200: Rs.{t.sma_200:,.2f} | EMA20: Rs.{t.ema_20:,.2f}
Bollinger: [{t.bollinger_lower:,.2f} - {t.bollinger_middle:,.2f} - {t.bollinger_upper:,.2f}]
52W High: {t.pct_from_52w_high:+.1f}% | 52W Low: {t.pct_from_52w_low:+.1f}%
1M: {t.price_change_1m:+.1f}% | 3M: {t.price_change_3m:+.1f}% | 6M: {t.price_change_6m:+.1f}%
Volume: {t.current_volume:,.0f} (20d avg: {t.volume_avg_20:,.0f})"""

        if a.fundamentals:
            f = a.fundamentals
            section += f"""
PE: {f.pe_ratio:.1f} | PB: {f.pb_ratio:.1f} | Div Yield: {f.dividend_yield:.1f}%
ROE: {f.roe:.1f}% | D/E: {f.debt_to_equity:.1f} | Rev Growth: {f.revenue_growth:.1f}%
Profit Margin: {f.profit_margin:.1f}% | Sector: {f.sector} | MCap: Rs.{f.market_cap/1e7:,.0f} Cr"""

        sections.append(section)
    return "\n\n".join(sections)


def analyze(
    config: Config,
    portfolio: Portfolio,
    candidates: list[StockAnalysis],
    mode: str = "all",
) -> DailyRecommendation:
    """Send portfolio + candidate data to the LLM and parse recommendations.

    mode: "all" (default), "buy" (buy-only), "sell" (sell-only)
    """
    client, model = _build_client(config)

    holdings_value = portfolio.total_current_value()
    total_value = portfolio.cash_balance + holdings_value
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["all"])

    system_prompt = SYSTEM_PROMPT.format(max_alloc=config.max_single_allocation_pct)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        date=date.today().isoformat(),
        mode=mode.upper(),
        cash=portfolio.cash_balance,
        holdings_value=holdings_value,
        num_positions=len(portfolio.holdings),
        holdings_table=_format_holdings(portfolio),
        max_buy_amount=config.max_buy_amount,
        max_positions=config.max_portfolio_positions,
        max_alloc=config.max_single_allocation_pct,
        total_value=total_value,
        stock_data=_format_stock_data(candidates),
        mode_instruction=mode_instruction,
    )

    logger.info("Sending %s analysis to %s (%s) …", mode, config.llm_provider, model)

    content = ""
    try:
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            temperature=0.2,
            max_output_tokens=8192,
        )

        content = response.output_text.strip()

        # Robust markdown fence stripping
        if "```" in content:
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        # Try to extract JSON if wrapped in extra text
        if not content.startswith("{"):
            start = content.find("{")
            if start >= 0:
                content = content[start:]
        if not content.endswith("}"):
            end = content.rfind("}")
            if end >= 0:
                content = content[:end + 1]

        result = json.loads(content)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON:\n%s", content[:1000])
        result = {
            "market_outlook": "Analysis failed — LLM returned invalid response.",
            "recommendations": [],
            "analysis_summary": content[:500] if content else "No response",
            "risk_assessment": "Unable to assess",
        }
    except Exception:
        logger.exception("LLM API call failed")
        result = {
            "market_outlook": "Analysis failed — API error.",
            "recommendations": [],
            "analysis_summary": "LLM call failed. Check logs.",
            "risk_assessment": "Unable to assess",
        }

    # Filter out any HOLD recommendations and add analysis URLs
    recs = [r for r in result.get("recommendations", []) if r.get("action", "").upper() != "HOLD"]
    result["recommendations"] = recs

    for rec in recs:
        symbol = rec.get("ticker", "").replace(".NS", "")
        rec["technicals_url"] = f"https://www.tradingview.com/chart/?symbol=NSE:{symbol}"
        rec["fundamentals_url"] = f"https://www.screener.in/company/{symbol}/"
        if "citations" not in rec:
            rec["citations"] = []

    today = date.today().isoformat()
    return DailyRecommendation(
        id=f"rec-{today}",
        date=today,
        market_outlook=result.get("market_outlook", ""),
        portfolio_value=total_value,
        portfolio_returns_pct=(
            round(
                ((total_value - portfolio.initial_investment) / portfolio.initial_investment) * 100,
                2,
            )
            if portfolio.initial_investment > 0
            else 0.0
        ),
        recommendations=result.get("recommendations", []),
        analysis_summary=result.get("analysis_summary", ""),
        risk_assessment=result.get("risk_assessment", ""),
        kite_holdings=portfolio.holdings,
    )
