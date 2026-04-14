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
You are a SEBI-registered-grade Indian equity research analyst specializing in NSE Large Cap and Mid Cap stocks for long-term investment (6-24 month horizon).

## ANALYSIS METHODOLOGY (apply in strict order)

### Step 1: Fundamental Quality Screen
Score each stock 0-10 on fundamentals using these HARD thresholds:
- ROE > 15% = good (2pts), > 20% = excellent (3pts)
- Debt/Equity < 1.0 = good (2pts), < 0.5 = excellent (3pts)
- Revenue Growth > 10% YoY = good (1pt), > 20% = excellent (2pts)
- Profit Margin > 10% = good (1pt), > 15% = excellent (2pts)
- PE < sector average = undervalued (1pt), PE < 15 = deep value (2pts)
- Reject stocks scoring below 4/10 for BUY consideration

### Step 2: Technical Timing Confirmation
Only recommend BUY if at least 2 of these conditions are met:
- RSI(14) between 30-55 (not overbought)
- Price above SMA200 (long-term uptrend intact)
- MACD histogram positive or showing bullish crossover
- Price near or below Bollinger middle band (not extended)
- Volume above 20-day average (institutional interest)

### Step 3: SELL Trigger Conditions
Recommend SELL for existing holdings if ANY of these apply:
- RSI > 75 (severely overbought)
- Price dropped > 15% below SMA200 (broken long-term trend)
- Fundamental deterioration: ROE < 10% or Debt/Equity > 2.0
- Stock at 52-week high with declining volume (distribution)
- Better reallocation opportunity exists with available budget
- Stop loss hit (current price below recommended stop loss)

### Step 4: Position Sizing
- Allocate proportionally based on conviction (HIGH=40%, MEDIUM=30%, LOW=20% of budget)
- Never put more than {max_alloc}% of total portfolio in a single stock
- Round to nearest tradeable lot size

### Step 5: Target and Stop Loss Calculation
- Target price: based on historical PE re-rating potential, not arbitrary %
- Stop loss: below nearest strong support level or SMA200, whichever is tighter
- Risk:Reward ratio must be at least 1:2 (potential upside >= 2x potential downside)

## CRITICAL RULES
- Total cost of ALL BUY recommendations must NOT exceed the TOTAL_INVESTMENT_BUDGET
- You can ONLY recommend SELL for stocks currently held in the portfolio
- If no compelling opportunity exists, return ZERO recommendations (holding is a valid strategy)
- Every metric you cite MUST come from the data provided — do not hallucinate numbers
- Be specific: cite exact PE, RSI, SMA values from the data, not vague statements
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

=== CANDIDATES (ranked by composite technical score) ===

{stock_data}

=== TASK ===
{mode_instruction}

Return ONLY valid JSON (no markdown fencing) matching this schema:

{{
  "market_outlook": "2-3 sentence assessment citing specific index levels or macro factors",
  "portfolio_assessment": "Assessment of current holdings with specific weak/strong performers",
  "recommendations": [
    {{
      "ticker": "SYMBOL.NS",
      "name": "Company Name",
      "action": "BUY or SELL",
      "quantity": 5,
      "current_price": 1234.56,
      "target_price": 1400.00,
      "stop_loss": 1100.00,
      "reason": "2-3 sentences with SPECIFIC metrics from the data provided",
      "confidence": "HIGH or MEDIUM or LOW",
      "fundamental_score": 7,
      "technical_score": 8,
      "risk_reward_ratio": "1:2.5",
      "citations": [
        "PE 18.5x vs sector avg 22x — 16% discount to peers",
        "RSI 38 oversold + MACD histogram turning positive at -0.5",
        "Revenue growth 15% YoY with margin expansion from 12% to 14%"
      ]
    }}
  ],
  "analysis_summary": "Key factors driving today's recommendations",
  "risk_assessment": "Specific risks with trigger levels (e.g., 'Nifty below 22000 would invalidate bullish thesis')"
}}
"""

MODE_INSTRUCTIONS = {
    "all": "Analyze ALL candidates. Provide BUY recommendations for the best new opportunities AND SELL recommendations for any holdings that meet sell criteria. Apply the full 5-step methodology.",
    "buy": "Focus ONLY on BUY opportunities. Identify the strongest candidates for purchase within the investment budget. Do NOT recommend any SELL actions. Apply Steps 1-2 and 4-5 of the methodology.",
    "sell": "Focus ONLY on SELL decisions for current holdings. Evaluate each holding against the SELL trigger conditions in Step 3. Do NOT recommend any BUY actions. For each holding, state whether to HOLD or SELL with specific reasoning.",
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
            max_output_tokens=4096,
        )

        content = response.output_text.strip()
        # Strip markdown fencing if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]

        result = json.loads(content)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON:\n%s", content[:500])
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

    # Add analysis URLs to each recommendation
    for rec in result.get("recommendations", []):
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
