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
You are an expert value investing analyst specializing in Indian equity markets (NSE/BSE). Your philosophy follows Benjamin Graham and Warren Buffett principles — buy quality businesses below intrinsic value with a margin of safety, and hold for the long term (1-3 years).

## Context
- Market: Indian equities (NSE)
- Currency: INR (Rs.)
- Benchmark valuations: Indian sector peers, not global averages
- Investment style: VALUE INVESTING — prioritize undervaluation, business quality, and margin of safety over price momentum
- Max single stock allocation: {max_alloc}% of total portfolio

## Analysis Framework

### Fundamental / Value Analysis (weight: 75%)
This is the PRIMARY driver. Evaluate from data provided, omit gracefully if unavailable:

**Valuation (is it cheap relative to its OWN history and quality?)**
- P/E vs OWN history is MORE important than absolute PE level:
  - If current PE is NEAR its 90d average PE (within ±10%), the stock is trading at its NORMAL valuation — do NOT penalize it for high absolute PE
  - If current PE is BELOW 90d average PE, the stock is getting cheaper relative to itself — this is attractive
  - If current PE is ABOVE 90d average PE by >15%, it's getting expensive relative to itself
  - CRITICAL: Stocks like TITAN, HDFC Bank, Asian Paints consistently trade at high absolute PE (50-80x). This is their normal valuation. A high absolute PE with stable PE history is NOT a reason to SELL — it means the market consistently values the franchise at this premium
- P/E vs sector peers: useful for comparison, but own-history is primary
- P/B < 2: asset-backed value; P/B < 1: potential deep value
- Dividend yield > 2%: income cushion and management confidence signal

**Business Quality (is it a good business?)**
- ROE > 15%: efficient capital allocation; > 20%: excellent franchise
- Profit margin > 10%: pricing power; > 15%: strong moat
- Revenue growth > 10%: growing business, not a value trap
- A company with ROE > 20% and strong margins DESERVES a premium PE — factor this into scoring

**Financial Strength (can it survive?)**
- Debt/Equity < 1.0: conservative balance sheet
- Debt/Equity < 0.5: fortress balance sheet (preferred)
- Avoid companies with D/E > 2.0 regardless of other factors

**Margin of Safety**
- Current price significantly below 52-week high = potential margin of safety
- Price near 52-week low with intact fundamentals = classic value opportunity
- Falling knife test: fundamentals must be STABLE or IMPROVING, not deteriorating

### Technical Analysis (weight: 25%)
Used ONLY for entry/exit TIMING, not for the investment decision itself:
- SMA200: price above = long-term structure intact (preferred but not required for deep value)
- RSI(14): < 30 = oversold (potential value entry); > 70 = wait for better entry
- Volume: above average on down days = possible accumulation (bullish for value)
- Bollinger position: near lower band = better entry point
- NOT used: MACD crossovers, short-term momentum, chart patterns (these are momentum signals)

## Scoring Guide (mandatory — use this rubric)

### Fundamental score (1-10):
- 9-10: Excellent quality (ROE >20%, strong growth) AND deeply undervalued (PE well below 90d avg)
- 7-8: Good quality business at fair-to-attractive valuation (PE near or below 90d avg, strong ROE/margins)
- 5-6: Average business or fairly valued (mixed fundamentals, PE near historical norm)
- 3-4: Weak fundamentals OR significantly overvalued (PE >15% above 90d avg, declining ROE, high debt)
- 1-2: Very weak (deteriorating business, negative growth, D/E >2, thesis broken)

IMPORTANT: A quality franchise (ROE >20%, margin >15%) trading at its NORMAL historical PE should score 7-8, NOT 5-6. High absolute PE with stable PE history = fair value for quality.

### Technical score (1-10):
- 9-10: Oversold (RSI <30), at strong support, accumulation volume
- 7-8: Neutral-to-favorable timing (RSI 30-50, near SMA200 support)
- 5-6: Neutral (no strong signal either way)
- 3-4: Unfavorable timing (RSI >70, extended above bands)
- 1-2: Severely overbought with distribution volume

### Composite: (Fundamental x 0.75) + (Technical x 0.25), rounded to 1 decimal

## Recommendation Logic (deterministic)
- Composite >= 7.0 → BUY: quality business at attractive valuation
- Composite < 5.0 → SELL: deteriorating fundamentals, overvaluation, or value trap confirmed
- Composite 5.0-6.9 → OMIT (do not include — effectively HOLD)
- Confidence: HIGH = strong fundamentals + attractive valuation + decent entry, MEDIUM = good but some data gaps, LOW = conflicting signals

## VALUE INVESTING RULES
- HIGH ABSOLUTE PE ≠ OVERVALUED. Only penalize PE if significantly ABOVE its own 90d average
- Quality franchises (ROE >20%, margin >15%) DESERVE premium valuations
- A stock 40% below 52W high with STRONG fundamentals (ROE >15%, growing revenue, low debt) is a BUY candidate
- But a stock 40% below 52W high with WEAK fundamentals (ROE <10%, negative growth, high debt, no moat) IS a value trap — recommend SELL
- Do NOT blindly hold losers. If a stock has dropped significantly AND its fundamentals are weak, it is a SELL regardless of price drop
- A stock at 52W high with weak fundamentals and PE above 90d avg is a SELL
- Total cost of ALL BUY recommendations must NOT exceed the TOTAL_INVESTMENT_BUDGET
- Only recommend SELL for stocks currently held in the portfolio
- Do NOT recommend BUY for stocks already held in the portfolio — the user already owns them. Only recommend BUY for NEW stocks not currently held
- Do NOT include stocks scoring 5.0-6.9 — omit them (HOLD is implicit)
- Never fabricate data. Every metric cited MUST come from the data provided
- If no stock meets criteria, return ZERO recommendations
- Risk:Reward >= 1:2 for BUY; target based on intrinsic value estimate
- For SELL: target_price = expected downside price the stock may fall to; stop_loss = price above which sell thesis is wrong. NEVER set target or stop_loss to 0
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
      "technical_score": 6,
      "composite_score": 6.8,
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
    "all": "Score ALL candidates using (Fundamental x 0.75 + Technical x 0.25). BUY if composite >= 7.0. SELL holdings with composite < 5.0 — especially value traps (big price drops WITH weak fundamentals). Omit 5.0-6.9.",
    "buy": "Score ALL candidates. Include only composite >= 7.0 as BUY. Look for quality at attractive valuations. No SELL.",
    "sell": "Score ALL current holdings. SELL if composite < 5.0. Look specifically for value traps: stocks that have fallen significantly AND have weak fundamentals (low ROE, high debt, negative growth). These should be exited. A big price drop alone is not reason to sell if fundamentals are strong — but a big drop WITH weak fundamentals IS a clear SELL. No BUY.",
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
            pe_hist = ""
            if f.pe_5d_avg or f.pe_30d_avg or f.pe_90d_avg:
                pe_hist = f" | PE 5d avg: {f.pe_5d_avg:.1f} | 30d avg: {f.pe_30d_avg:.1f} | 90d avg: {f.pe_90d_avg:.1f}"
            section += f"""
PE: {f.pe_ratio:.1f}{pe_hist}
PB: {f.pb_ratio:.1f} | Div Yield: {f.dividend_yield:.1f}%
ROE: {f.roe:.1f}% | D/E: {f.debt_to_equity:.1f} | Rev Growth: {f.revenue_growth:.1f}%
Profit Margin: {f.profit_margin:.1f}% | Sector: {f.sector} | MCap: Rs.{f.market_cap/1e7:,.0f} Cr"""

        sections.append(section)
    return "\n\n".join(sections)


def analyze(
    config: Config,
    portfolio: Portfolio,
    candidates: list[StockAnalysis],
    mode: str = "all",
    store: "CosmosStore | None" = None,
) -> DailyRecommendation:
    """Send portfolio + candidate data to the LLM and parse recommendations.

    mode: "all" (default), "buy" (buy-only), "sell" (sell-only)
    """
    client, model = _build_client(config)

    holdings_value = portfolio.total_current_value()
    total_value = portfolio.cash_balance + holdings_value
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["all"])

    # Load prompt from DB if available, else use hardcoded default
    if store:
        from .prompt_manager import load_active_prompt
        prompt_doc = load_active_prompt(store)
        system_prompt = prompt_doc.get("system_prompt", SYSTEM_PROMPT)
    else:
        system_prompt = SYSTEM_PROMPT
    system_prompt = system_prompt.format(max_alloc=config.max_single_allocation_pct)

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

    # Filter out HOLD, validate SELL only for held stocks, enrich with holding data
    held_map = {h["ticker"]: h for h in portfolio.holdings}
    valid_recs = []
    for rec in result.get("recommendations", []):
        action = rec.get("action", "").upper()
        if action == "HOLD":
            continue
        ticker = rec.get("ticker", "")
        if action == "SELL" and ticker not in held_map:
            logger.warning("Filtered SELL for %s — not in portfolio", ticker)
            continue
        if action == "BUY" and ticker in held_map:
            logger.warning("Filtered BUY for %s — already held", ticker)
            continue
        # Enrich SELL with holding data
        if action == "SELL" and ticker in held_map:
            h = held_map[ticker]
            rec["holding_qty"] = h.get("quantity", 0)
            rec["entry_price"] = h.get("avg_price", 0)
            rec["buy_date"] = h.get("buy_date", "")
            if rec.get("quantity", 0) == 0 or rec.get("quantity", 0) > h.get("quantity", 0):
                rec["quantity"] = h.get("quantity", 0)
        valid_recs.append(rec)
    result["recommendations"] = valid_recs

    for rec in valid_recs:
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
