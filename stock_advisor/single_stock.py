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
You are an expert financial analyst specializing in Indian equity markets (NSE/BSE). Provide a comprehensive investment analysis for the stock below.

## Context
- Market: Indian equities (NSE)
- Currency: INR (Rs.)
- Benchmark valuations: Indian sector peers, not global averages

## Analysis Framework

### Technical Analysis (weight: 40%)
Evaluate from the data provided. Omit gracefully if not available:
- Trend: Price vs 50/200-day SMA; trend direction (uptrend, downtrend, sideways)
- Momentum: RSI(14) — overbought >70, oversold <30; MACD crossover signals
- Volume: Current vs 20-day average — confirms conviction
- Support/Resistance: 52-week high/low proximity, Bollinger Band position
- Price momentum: 1M, 3M, 6M returns for trend confirmation

### Fundamental Analysis (weight: 60%)
Evaluate from the data provided. Omit gracefully if not available:
- Valuation: P/E, P/B vs sector peers
- Profitability: ROE, profit margin
- Growth: Revenue growth YoY
- Balance sheet: Debt-to-equity ratio
- Dividend: Yield as income signal
- Business quality: Sector positioning, market cap

## Scoring (mandatory)
- Technical score: 1 (very bearish) to 10 (very bullish)
- Fundamental score: 1 (very weak) to 10 (very strong)
- Composite score: (Technical x 0.4) + (Fundamental x 0.6), rounded to 1 decimal

## Recommendation Logic (deterministic)
- Composite >= 7.0 → BUY
- Composite 5.0–6.9 → HOLD
- Composite < 5.0 → SELL
- Confidence: HIGH if data complete and signals align, MEDIUM if partial, LOW if conflicting

## Rules
- Use ONLY the data provided — never fabricate financial data
- If a data field is missing, set it to null and note in data_quality
- Be direct and actionable — this determines real investment decisions
- For BUY: target based on PE re-rating; stop loss at SMA200 or support; R:R >= 1:2
- Keep reasoning concise — 2-3 sentences per section
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
PE Ratio: {pe:.1f} | PB Ratio: {pb:.1f} | Dividend Yield: {div_yield:.1f}%
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
  "composite_score": 6.6,
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
