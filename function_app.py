"""Azure Function entry point — Timer trigger (daily) + HTTP passthrough."""

import json
import logging

import azure.functions as func

from stock_advisor.config import Config
from stock_advisor.cosmos_store import CosmosStore
from stock_advisor.orchestrator import run_daily_analysis
from stock_advisor.portfolio_manager import PortfolioManager

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.timer_trigger(schedule="0 30 3 * * 1-5", arg_name="timer", run_on_startup=False)
def daily_stock_analysis(timer: func.TimerRequest) -> None:
    """Runs Mon-Fri at 09:00 IST (03:30 UTC) — before Indian market open."""
    if timer.past_due:
        logger.warning("Timer trigger is past due")
    logger.info("=== Daily Stock Analysis Started ===")
    try:
        rec = run_daily_analysis()
        logger.info("Done: %d recommendations", len(rec.recommendations))
    except Exception:
        logger.exception("Daily analysis failed")


@app.route(route="recommendation", methods=["GET"])
def get_recommendation(req: func.HttpRequest) -> func.HttpResponse:
    config = Config()
    store = CosmosStore(config)
    pm = PortfolioManager(config, store)
    rec = pm.get_latest_recommendation()
    if rec:
        return func.HttpResponse(json.dumps(rec.to_dict(), default=str), mimetype="application/json")
    return func.HttpResponse(json.dumps({"message": "No recommendation"}), status_code=404, mimetype="application/json")


@app.route(route="portfolio", methods=["GET"])
def get_portfolio(req: func.HttpRequest) -> func.HttpResponse:
    config = Config()
    store = CosmosStore(config)
    pm = PortfolioManager(config, store)
    portfolio = pm.get_portfolio()
    return func.HttpResponse(json.dumps(portfolio.to_dict(), default=str), mimetype="application/json")


@app.route(route="analyze", methods=["POST"])
def trigger_analysis(req: func.HttpRequest) -> func.HttpResponse:
    try:
        max_buy_amount = None
        try:
            body = req.get_json()
            max_buy_amount = body.get("max_buy_amount")
            if max_buy_amount is not None:
                max_buy_amount = float(max_buy_amount)
        except ValueError:
            pass
        rec = run_daily_analysis(max_buy_amount=max_buy_amount)
        return func.HttpResponse(json.dumps(rec.to_dict(), default=str), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500, mimetype="application/json")


@app.route(route="settings", methods=["GET"])
def get_settings(req: func.HttpRequest) -> func.HttpResponse:
    config = Config()
    return func.HttpResponse(json.dumps({
        "max_buy_amount": config.max_buy_amount,
        "max_buys_per_day": config.max_buys_per_day,
        "max_sells_per_day": config.max_sells_per_day,
        "max_portfolio_positions": config.max_portfolio_positions,
        "max_single_allocation_pct": config.max_single_allocation_pct,
        "llm_provider": config.llm_provider,
        "llm_model": config.azure_ai_deployment if config.llm_provider == "azure_foundry" else "gpt-4o",
        "kite_connected": bool(config.kite_api_key and config.kite_access_token),
    }), mimetype="application/json")
