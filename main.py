"""Local CLI runner for testing."""

import logging
import json
import sys

from stock_advisor.config import Config
from stock_advisor.orchestrator import run_daily_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    config = Config()

    if config.llm_provider == "azure_foundry" and not config.azure_ai_endpoint:
        if not config.github_token:
            print("ERROR: No LLM configured. Set AZURE_AI_ENDPOINT or GITHUB_TOKEN in .env")
            sys.exit(1)
        config.llm_provider = "github"

    rec = run_daily_analysis(config)

    print(f"\n{'='*60}")
    print(f"  Date: {rec.date}")
    print(f"  Portfolio: Rs.{rec.portfolio_value:,.0f} ({rec.portfolio_returns_pct:+.1f}%)")
    print(f"  Outlook: {rec.market_outlook}")
    print(f"{'='*60}")

    for r in rec.recommendations:
        print(f"\n  [{r['action']}] {r.get('quantity',0)} x {r['ticker']} @ Rs.{r.get('current_price',0):,.2f}")
        print(f"    Target: Rs.{r.get('target_price',0):,.2f} | SL: Rs.{r.get('stop_loss',0):,.2f}")
        print(f"    {r.get('reason','')}")
        for c in r.get("citations", []):
            print(f"    - {c}")

    if not rec.recommendations:
        print("\n  No action recommended today.")

    with open(f"recommendation-{rec.date}.json", "w") as f:
        json.dump(rec.to_dict(), f, indent=2, default=str)
    print(f"\n  Saved to recommendation-{rec.date}.json")


if __name__ == "__main__":
    main()
