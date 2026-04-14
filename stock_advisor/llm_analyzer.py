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
You are an expert Indian stock market analyst and portfolio manager.
You specialize in long-term value investing in NSE-listed Large Cap and Mid Cap stocks.

Your investment philosophy:
- Long-term wealth creation with a minimum 6-12 month holding horizon
- Quality businesses with strong fundamentals: high ROE, low debt, consistent growth
- Buy at reasonable valuations when technicals confirm entry points
- Maintain diversification across sectors (no more than 25% in any single sector)
- Strict risk management with stop losses and position sizing
- Prefer gradual portfolio building — do not overhaul portfolio in a single day

Your analysis approach:
1. FUNDAMENTAL: PE, PB, ROE, debt/equity, revenue growth, profit margins, dividend yield
2. TECHNICAL: RSI, MACD, moving averages (SMA 50/200), Bollinger Bands, volume, momentum
3. VALUATION: Compare current PE/PB vs sector averages and historical ranges
4. RISK: Consider 52-week highs/lows, volatility, sector concentration

CRITICAL RULES:
- Each BUY recommendation total cost (quantity × price) must NOT exceed the MAX_BUY_AMOUNT
- Respect the buy/sell limits strictly
- Never recommend intraday or short-term trades
- Only recommend sells for portfolio holdings
- Set realistic target prices (10-30% upside over 6-12 months) and stop losses (8-15% below entry)
- If market conditions are uncertain, it's OK to recommend no action (HOLD all)
- Provide clear, concise reasoning for each recommendation
- Include citations: mention specific financial metrics, news items, or data sources
- For each stock, generate analysis URLs for the user to verify your claims
"""

USER_PROMPT_TEMPLATE = """\
Date: {date}

=== CURRENT PORTFOLIO (from Kite Connect) ===
Cash Available: Rs.{cash:,.0f}
Total Holdings Value: Rs.{holdings_value:,.0f}
Number of Positions: {num_positions}

Holdings:
{holdings_table}

=== CONSTRAINTS ===
- Max BUY amount per stock: Rs.{max_buy_amount:,.0f}
- Max BUY recommendations today: {max_buys}
- Max SELL recommendations today: {max_sells}
- Max portfolio positions: {max_positions}
- Max single stock allocation: {max_alloc}% of total portfolio value
- Total portfolio value (cash + holdings): Rs.{total_value:,.0f}

=== STOCK ANALYSIS DATA ===
Top candidates ranked by technical + fundamental composite score:

{stock_data}

=== INSTRUCTIONS ===
Analyze the above data and provide your daily recommendation.
Return your response as valid JSON matching this exact schema:

{{
  "market_outlook": "2-3 sentence market assessment for today",
  "portfolio_assessment": "assessment of current holdings",
  "recommendations": [
    {{
      "ticker": "SYMBOL.NS",
      "name": "Company Name",
      "action": "BUY or SELL",
      "quantity": 5,
      "current_price": 1234.56,
      "target_price": 1400.00,
      "stop_loss": 1100.00,
      "reason": "Detailed 2-3 sentence rationale with specific metrics cited",
      "confidence": "HIGH or MEDIUM or LOW",
      "citations": [
        "PE ratio of 18.5x vs sector average 22x indicates undervaluation",
        "RSI at 38 suggests oversold conditions with MACD showing bullish crossover",
        "Q3 revenue growth of 15% YoY with improving margins"
      ]
    }}
  ],
  "analysis_summary": "Overall summary of today's analysis and rationale",
  "risk_assessment": "Key risks to monitor"
}}

IMPORTANT:
- Each BUY total cost (quantity * current_price) must not exceed Rs.{max_buy_amount:,.0f}
- Include 2-4 specific citations per recommendation
- If no action is warranted, return an empty recommendations array
- Return ONLY the JSON, no markdown fencing
"""


def _build_client(config: Config) -> tuple[AzureOpenAI | OpenAI, str]:
    """Create the appropriate OpenAI client based on provider config."""
    if config.llm_provider == "azure_foundry" and config.azure_ai_endpoint:
        client = AzureOpenAI(
            azure_endpoint=config.azure_ai_endpoint,
            api_key=config.azure_ai_key,
            api_version="2024-12-01-preview",
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
) -> DailyRecommendation:
    """Send portfolio + candidate data to the LLM and parse recommendations."""
    client, model = _build_client(config)

    holdings_value = portfolio.total_current_value()
    total_value = portfolio.cash_balance + holdings_value

    user_prompt = USER_PROMPT_TEMPLATE.format(
        date=date.today().isoformat(),
        cash=portfolio.cash_balance,
        holdings_value=holdings_value,
        num_positions=len(portfolio.holdings),
        holdings_table=_format_holdings(portfolio),
        max_buy_amount=config.max_buy_amount,
        max_buys=config.max_buys_per_day,
        max_sells=config.max_sells_per_day,
        max_positions=config.max_portfolio_positions,
        max_alloc=config.max_single_allocation_pct,
        total_value=total_value,
        stock_data=_format_stock_data(candidates),
    )

    logger.info("Sending analysis request to %s (%s) …", config.llm_provider, model)

    content = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )

        content = response.choices[0].message.content.strip()
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
