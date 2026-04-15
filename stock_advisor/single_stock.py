"""Single-stock deep analysis via LLM."""

from __future__ import annotations

import json
import logging
from datetime import date

from openai import AzureOpenAI, OpenAI

from .config import Config
from .models import TechnicalIndicators, FundamentalData

logger = logging.getLogger(__name__)

SINGLE_STOCK_PROMPT = """\
You are an expert value investing analyst specializing in Indian equity markets (NSE/BSE). Your philosophy follows Graham/Buffett principles — buy quality businesses below intrinsic value with margin of safety.

## Context
- Market: Indian equities (NSE)
- Currency: INR (Rs.)
- Benchmark valuations: Indian sector peers, not global averages
- Style: VALUE INVESTING — prioritize undervaluation, business quality, margin of safety

## Analysis Framework

### Fundamental / Value Analysis (weight: 75%)
PRIMARY driver. Evaluate from data provided, omit gracefully if unavailable:

**Valuation:** Compare current PE to its OWN 5d/30d/90d PE averages — this is MORE important than absolute PE. If current PE is NEAR 90d avg (±10%), stock is at NORMAL valuation — do NOT penalize high absolute PE. Quality franchises (TITAN, HDFC Bank, Asian Paints) always trade at premium PE — that IS their fair value. Only flag expensive if PE is >15% ABOVE its own 90d avg. Also compare P/B vs peers.
**Business Quality:** ROE (>15% good, >20% excellent franchise), profit margin (>10% good, >15% moat), revenue growth. Companies with ROE >20% DESERVE premium PE — factor this in.
**Financial Strength:** Debt/Equity (<1.0 good, <0.5 excellent). Avoid D/E > 2.0
**Dividend:** Yield > 2% signals management confidence
**Margin of Safety:** Price well below 52W high with intact fundamentals = opportunity, not risk

### Technical Analysis (weight: 25%)
Used ONLY for entry/exit TIMING:
- SMA200: above = structure intact (preferred, not required for deep value)
- RSI: <30 = oversold value entry; >70 = wait for better price
- Volume: above average = institutional interest
- Bollinger: near lower band = better entry point
- NOT used: MACD crossovers, momentum signals, chart patterns

## Scoring Guide (mandatory — use this rubric)

### Fundamental score:
- 9-10: Excellent quality (ROE >20%, strong growth) AND deeply undervalued (PE well below 90d avg)
- 7-8: Good quality at fair-to-attractive valuation (PE near or below 90d avg, strong ROE/margins)
- 5-6: Average business or fairly valued (mixed fundamentals)
- 3-4: Weak fundamentals OR significantly overvalued (PE >15% above 90d avg, declining ROE)
- 1-2: Very weak (deteriorating, negative growth, D/E >2)

IMPORTANT: Quality franchise (ROE >20%, margin >15%) at its NORMAL historical PE = score 7-8, NOT 5-6.

### Technical score:
- 9-10: Oversold (RSI <30), at strong support, accumulation volume
- 7-8: Neutral-to-favorable (RSI 30-50, near SMA200)
- 5-6: Neutral
- 3-4: Unfavorable (RSI >70, extended)
- 1-2: Severely overbought

### Composite: (Fundamental x 0.75) + (Technical x 0.25), rounded to 1 decimal

## Recommendation Logic (deterministic)
- Composite >= 7.0 → BUY
- Composite 5.0–6.9 → HOLD
- Composite < 5.0 → SELL
- Confidence: HIGH = strong fundamentals + undervalued, MEDIUM = partial data, LOW = conflicting

## Value Rules
- HIGH ABSOLUTE PE ≠ OVERVALUED. If current PE is near 90d avg PE, it's at fair value
- Quality franchises with ROE >20% deserve premium PE — do NOT sell them for high PE alone
- A stock down 40% with STRONG fundamentals is a BUY candidate, not a SELL
- A stock at 52W high with WEAK fundamentals is a SELL, not a HOLD
- NEVER sell just because price dropped — only if thesis is broken (ROE collapsed, debt surged)
- Use ONLY the data provided — never fabricate data
- If data missing, note in data_quality
- Target based on intrinsic value. R:R >= 1:2 for BUY
"""

USER_STOCK_PROMPT = """\
Date: {date}

=== STOCK: {ticker} ({name}) ===
Sector: {sector} | Category: {category} | Market Cap: Rs.{mcap} Cr

=== PRICE & TECHNICAL DATA ===
Current Price: Rs.{price:,.2f}
RSI(14): {rsi} | MACD Histogram: {macd_hist}
SMA50: Rs.{sma50:,.2f} | SMA200: Rs.{sma200:,.2f} | EMA20: Rs.{ema20:,.2f}
Bollinger Bands: [{bb_lower:,.2f} — {bb_mid:,.2f} — {bb_upper:,.2f}]
52W from High: {pct_high:+.1f}% | 52W from Low: {pct_low:+.1f}%
1M Return: {ret_1m:+.1f}% | 3M: {ret_3m:+.1f}% | 6M: {ret_6m:+.1f}%
Volume: {volume:,.0f} (20d avg: {vol_avg:,.0f})

=== FUNDAMENTAL DATA ===
PE Ratio: {pe:.1f} | PE 5d avg: {pe_5d:.1f} | PE 30d avg: {pe_30d:.1f} | PE 90d avg: {pe_90d:.1f}
PB Ratio: {pb:.1f} | Dividend Yield: {div_yield:.1f}%
ROE: {roe:.1f}% | Debt/Equity: {de:.1f}
Revenue Growth: {rev_growth:.1f}% | Profit Margin: {margin:.1f}%

=== CURRENT HOLDING ===
{holding_info}

=== TASK ===
Provide comprehensive analysis and a clear recommendation.

Respond ONLY with valid JSON (no preamble, no markdown fences):
{{
  "recommendation": "BUY or SELL or HOLD",
  "confidence": "HIGH or MEDIUM or LOW",
  "target_price": 0.00,
  "stop_loss": 0.00,
  "expected_holding_period": "6-12 months",
  "risk_reward_ratio": "1:2.5",
  "fundamental_score": 7,
  "technical_score": 6,
  "composite_score": 6.8,
  "fundamental_analysis": "2-3 sentences with specific metrics from data",
  "technical_analysis": "2-3 sentences with specific indicator readings from data",
  "valuation_assessment": "Is the stock cheap, fair, or expensive and why",
  "key_risks": ["Risk 1 with specific detail", "Risk 2"],
  "data_quality": "Complete / Partial — note any missing data fields",
  "verdict_summary": "2-3 sentence actionable summary citing composite score",
  "citations": ["Specific metric 1 from data", "Specific metric 2", "Specific metric 3"]
}}
"""


def analyze_single_stock(
    config: Config,
    ticker: str,
    technicals: TechnicalIndicators,
    fundamentals: FundamentalData,
    holding_info: str = "Not currently held",
) -> dict:
    """Run deep LLM analysis on a single stock."""
    from .llm_analyzer import _build_client

    client, model = _build_client(config)

    mcap_cr = f"{fundamentals.market_cap / 1e7:,.0f}" if fundamentals.market_cap else "N/A"

    user_prompt = USER_STOCK_PROMPT.format(
        date=date.today().isoformat(),
        ticker=ticker,
        name=fundamentals.name or ticker,
        sector=fundamentals.sector,
        category=fundamentals.category,
        mcap=mcap_cr,
        price=technicals.current_price,
        rsi=technicals.rsi_14,
        macd_hist=technicals.macd_histogram,
        sma50=technicals.sma_50,
        sma200=technicals.sma_200,
        ema20=technicals.ema_20,
        bb_lower=technicals.bollinger_lower,
        bb_mid=technicals.bollinger_middle,
        bb_upper=technicals.bollinger_upper,
        pct_high=technicals.pct_from_52w_high,
        pct_low=technicals.pct_from_52w_low,
        ret_1m=technicals.price_change_1m,
        ret_3m=technicals.price_change_3m,
        ret_6m=technicals.price_change_6m,
        volume=technicals.current_volume,
        vol_avg=technicals.volume_avg_20,
        pe=fundamentals.pe_ratio,
        pe_5d=fundamentals.pe_5d_avg,
        pe_30d=fundamentals.pe_30d_avg,
        pe_90d=fundamentals.pe_90d_avg,
        pb=fundamentals.pb_ratio,
        div_yield=fundamentals.dividend_yield,
        roe=fundamentals.roe,
        de=fundamentals.debt_to_equity,
        rev_growth=fundamentals.revenue_growth,
        margin=fundamentals.profit_margin,
        holding_info=holding_info,
    )

    try:
        response = client.responses.create(
            model=model,
            instructions=SINGLE_STOCK_PROMPT,
            input=user_prompt,
            temperature=0.2,
            max_output_tokens=4096,
        )

        content = response.output_text.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]

        result = json.loads(content)

        # Add analysis URLs
        symbol = ticker.replace(".NS", "")
        result["ticker"] = ticker
        result["name"] = fundamentals.name or ticker
        result["current_price"] = technicals.current_price
        result["sector"] = fundamentals.sector
        result["technicals_url"] = f"https://www.tradingview.com/chart/?symbol=NSE:{symbol}"
        result["fundamentals_url"] = f"https://www.screener.in/company/{symbol}/"
        result["google_finance_url"] = f"https://www.google.com/finance/quote/{symbol}:NSE"
        result["trendlyne_url"] = f"https://trendlyne.com/equity/{symbol}/"

        return result

    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON for %s", ticker)
        return {"error": "LLM returned invalid response", "ticker": ticker}
    except Exception:
        logger.exception("Single stock analysis failed for %s", ticker)
        return {"error": "Analysis failed", "ticker": ticker}
