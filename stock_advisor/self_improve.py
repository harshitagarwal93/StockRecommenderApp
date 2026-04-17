"""Post-market prompt self-improvement engine.

Runs daily after market close:
1. Load past 7 days of recommendations
2. Fetch actual price movements for recommended stocks
3. Ask LLM to analyze accuracy and suggest prompt improvements
4. Apply improvements if valid, save changelog
5. Clean up recommendations older than 7 days
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import yfinance as yf

from .config import Config
from .cosmos_store import CosmosStore
from .prompt_manager import load_active_prompt, save_prompt, get_prompt_changelog

logger = logging.getLogger(__name__)

REVIEW_PROMPT = """\
You are a meta-analyst improving a stock recommendation system's prompt. Your job is to analyze past recommendations against actual outcomes and suggest prompt improvements.

## YOUR TASK
Below are recommendations made over the past week and how stocks actually moved. Analyze the accuracy and suggest improvements to the system prompt.

## RULES FOR IMPROVEMENTS
- Only suggest changes that are GENERALIZABLE — not specific to any single stock
- Focus on SYSTEMATIC biases: consistently wrong scoring, incorrect thresholds, missing factors
- Do NOT overfit: if a recommendation was wrong due to unexpected news/events (earnings surprise, government policy, global crash), that is NOT a prompt problem — skip it
- Do NOT change the core value investing philosophy
- Do NOT change the scoring formula (75/25 weighting)
- Suggest at most 3 changes per review cycle
- Each change must be a specific, actionable text edit to the prompt
- Consider macro trends (market-wide movements) — if the whole market dropped 5%, individual stock declines are not prompt errors

## PAST RECOMMENDATIONS AND OUTCOMES

{review_data}

## CURRENT SYSTEM PROMPT

{current_prompt}

## RESPOND WITH JSON ONLY (no markdown fencing):

{{
  "accuracy_assessment": "Brief overall assessment of recommendation accuracy",
  "market_context": "Were there macro factors (market crash, sector rotation, policy changes) that explain outcomes?",
  "systematic_issues": ["Issue 1 description", "Issue 2 if any"],
  "prompt_changes": [
    {{
      "section": "Which section of the prompt to modify",
      "current_text": "Exact text to find in the current prompt (must exist)",
      "new_text": "Replacement text",
      "reason": "Why this change improves accuracy (in layman terms)",
      "layman_summary": "Simple 1-sentence explanation of what changed"
    }}
  ],
  "no_change_reason": "If no changes needed, explain why here. Set prompt_changes to empty array."
}}

IMPORTANT:
- If recommendations were mostly correct, or errors were due to external events, return empty prompt_changes
- The current_text field MUST be an exact substring of the current prompt — if you can't find exact text to change, skip it
- Never remove safety rules or risk management guidelines
"""


def run_self_improvement(config: Config) -> dict:
    """Analyze past week's recommendations vs actual outcomes and improve prompt."""
    store = CosmosStore(config)

    # 1. Load past 7 days of recommendations
    today = date.today()
    recs = []
    for i in range(7):
        d = (today - timedelta(days=i)).isoformat()
        doc = store.read(f"rec-{d}", "recommendation")
        if doc and doc.get("recommendations"):
            recs.append(doc)

    if not recs:
        logger.info("No recommendations found for review")
        return {"status": "no_data", "message": "No recommendations to review"}

    # 2. Build review data — compare recommendation prices to current prices
    review_entries = []
    seen_tickers = set()

    for rec_doc in recs:
        rec_date = rec_doc.get("date", "")
        for r in rec_doc.get("recommendations", []):
            ticker = r.get("ticker", "")
            action = r.get("action", "")
            rec_price = r.get("current_price", 0)
            target = r.get("target_price", 0)
            stop_loss = r.get("stop_loss", 0)

            if not ticker or not rec_price:
                continue

            key = f"{ticker}-{action}-{rec_date}"
            if key in seen_tickers:
                continue
            seen_tickers.add(key)

            # Fetch current price
            try:
                current = yf.Ticker(ticker).info.get("currentPrice") or yf.Ticker(ticker).info.get("regularMarketPrice") or 0
            except Exception:
                current = 0

            if current <= 0:
                continue

            pct_move = ((current - rec_price) / rec_price * 100) if rec_price else 0
            correct = (action == "BUY" and current > rec_price) or (action == "SELL" and current < rec_price)

            review_entries.append({
                "date": rec_date,
                "ticker": ticker,
                "action": action,
                "rec_price": round(rec_price, 2),
                "target": round(target, 2),
                "stop_loss": round(stop_loss, 2),
                "current_price": round(current, 2),
                "pct_move": round(pct_move, 1),
                "outcome": "CORRECT" if correct else "INCORRECT",
                "reason": r.get("reason", "")[:100],
            })

    if not review_entries:
        logger.info("No actionable review data")
        return {"status": "no_data", "message": "No price data for review"}

    # 3. Load current prompt
    prompt_doc = load_active_prompt(store)
    current_prompt = prompt_doc.get("system_prompt", "")
    current_version = prompt_doc.get("version", 1)

    # 4. Ask LLM to analyze and suggest improvements
    review_text = json.dumps(review_entries, indent=2)

    from .llm_analyzer import _build_client
    client, model = _build_client(config)

    try:
        response = client.responses.create(
            model=model,
            instructions="You are a meta-analyst improving a stock recommendation system. Respond with valid JSON only.",
            input=REVIEW_PROMPT.format(
                review_data=review_text,
                current_prompt=current_prompt[:3000],  # Truncate to fit
            ),
            temperature=0.2,
            max_output_tokens=4096,
        )

        content = response.output_text.strip()
        if "```" in content:
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()
        if not content.startswith("{"):
            start = content.find("{")
            if start >= 0:
                content = content[start:]
        if not content.endswith("}"):
            end = content.rfind("}")
            if end >= 0:
                content = content[:end + 1]

        result = json.loads(content)
    except Exception:
        logger.exception("Self-improvement LLM call failed")
        return {"status": "error", "message": "LLM analysis failed"}

    # 5. Apply changes if any
    changes = result.get("prompt_changes", [])
    applied_changes = []

    if changes:
        updated_prompt = current_prompt
        for change in changes:
            old_text = change.get("current_text", "")
            new_text = change.get("new_text", "")
            if old_text and new_text and old_text in updated_prompt:
                updated_prompt = updated_prompt.replace(old_text, new_text, 1)
                applied_changes.append({
                    "section": change.get("section", ""),
                    "old": old_text[:200],
                    "new": new_text[:200],
                    "reason": change.get("reason", ""),
                    "summary": change.get("layman_summary", ""),
                })

        if applied_changes:
            new_version = current_version + 1
            save_prompt(
                store, updated_prompt, new_version,
                applied_changes,
                result.get("accuracy_assessment", "Automated review"),
            )
            logger.info("Applied %d prompt changes (v%d → v%d)", len(applied_changes), current_version, new_version)

            # Cleanup: keep only last 5 changelog entries
            all_logs = get_prompt_changelog(store, limit=50)
            if len(all_logs) > 5:
                for old_log in all_logs[5:]:
                    try:
                        store._container.delete_item(item=old_log["id"], partition_key="prompt-changelog")
                    except Exception:
                        pass
                logger.info("Cleaned up %d old changelog entries", len(all_logs) - 5)

    # 6. Cleanup old recommendations (>7 days)
    cleanup_count = 0
    for i in range(8, 30):
        d = (today - timedelta(days=i)).isoformat()
        old_doc = store.read(f"rec-{d}", "recommendation")
        if old_doc:
            try:
                store._container.delete_item(item=f"rec-{d}", partition_key="recommendation")
                cleanup_count += 1
            except Exception:
                pass

    if cleanup_count:
        logger.info("Cleaned up %d old recommendations", cleanup_count)

    return {
        "status": "completed",
        "recommendations_reviewed": len(review_entries),
        "accuracy": sum(1 for r in review_entries if r["outcome"] == "CORRECT") / len(review_entries) * 100 if review_entries else 0,
        "changes_applied": len(applied_changes),
        "changes": applied_changes,
        "assessment": result.get("accuracy_assessment", ""),
        "market_context": result.get("market_context", ""),
        "cleanup_count": cleanup_count,
        "prompt_version": current_version + (1 if applied_changes else 0),
    }
