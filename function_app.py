"""Azure Function entry point — Timer trigger (daily) + HTTP passthrough."""

import hashlib
import json
import logging
from datetime import datetime, timezone

import azure.functions as func
import requests

from stock_advisor.config import Config
from stock_advisor.cosmos_store import CosmosStore
from stock_advisor.orchestrator import run_daily_analysis
from stock_advisor.portfolio_manager import PortfolioManager

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_URL = "https://stockadvisorsa.z29.web.core.windows.net"

# ---- Singleton instances (created once per cold start, reused across requests) ----
_config = Config()
_store = CosmosStore(_config)


def _pm() -> PortfolioManager:
    return PortfolioManager(_config, _store)


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
    rec = _pm().get_latest_recommendation()
    if rec:
        return func.HttpResponse(json.dumps(rec.to_dict(), default=str), mimetype="application/json")
    return func.HttpResponse(json.dumps({"message": "No recommendation"}), status_code=404, mimetype="application/json")


@app.route(route="portfolio", methods=["GET"])
def get_portfolio(req: func.HttpRequest) -> func.HttpResponse:
    portfolio = _pm().get_portfolio()
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
        rec = run_daily_analysis(config=_config, max_buy_amount=max_buy_amount)
        return func.HttpResponse(json.dumps(rec.to_dict(), default=str), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500, mimetype="application/json")


@app.route(route="settings", methods=["GET"])
def get_settings(req: func.HttpRequest) -> func.HttpResponse:
    token_doc = _store.read("kite-token", "kite")
    token_valid = False
    if token_doc:
        token_date = token_doc.get("date", "")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        token_valid = token_date == today

    return func.HttpResponse(json.dumps({
        "max_buy_amount": _config.max_buy_amount,
        "max_buys_per_day": _config.max_buys_per_day,
        "max_sells_per_day": _config.max_sells_per_day,
        "max_portfolio_positions": _config.max_portfolio_positions,
        "max_single_allocation_pct": _config.max_single_allocation_pct,
        "llm_provider": _config.llm_provider,
        "llm_model": _config.azure_ai_deployment if _config.llm_provider == "azure_foundry" else "gpt-4o",
        "kite_connected": token_valid,
        "kite_login_url": f"https://kite.zerodha.com/connect/login?v=3&api_key={_config.kite_api_key}" if _config.kite_api_key else "",
    }), mimetype="application/json")


# ---------------------------------------------------------------------------
# Kite OAuth Flow
# ---------------------------------------------------------------------------

@app.route(route="kite/callback", methods=["GET"])
def kite_callback(req: func.HttpRequest) -> func.HttpResponse:
    """Kite redirects here after login with ?request_token=...&status=success."""
    request_token = req.params.get("request_token", "")
    status = req.params.get("status", "")

    if status != "success" or not request_token:
        return func.HttpResponse(
            _html_redirect(FRONTEND_URL, "Kite login failed or was cancelled."),
            mimetype="text/html", status_code=400,
        )

    if not _config.kite_api_key or not _config.kite_api_secret:
        return func.HttpResponse(
            _html_redirect(FRONTEND_URL, "Kite API key/secret not configured."),
            mimetype="text/html", status_code=500,
        )

    # Exchange request_token for access_token
    checksum = hashlib.sha256(
        (_config.kite_api_key + request_token + _config.kite_api_secret).encode()
    ).hexdigest()

    try:
        resp = requests.post(
            "https://api.kite.trade/session/token",
            data={
                "api_key": _config.kite_api_key,
                "request_token": request_token,
                "checksum": checksum,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        access_token = data.get("data", {}).get("access_token", "")

        if not access_token:
            logger.error("No access_token in Kite response: %s", data)
            return func.HttpResponse(
                _html_redirect(FRONTEND_URL, "Kite returned no access token."),
                mimetype="text/html", status_code=500,
            )

        # Store token in CosmosDB
        _store.upsert({
            "id": "kite-token",
            "type": "kite",
            "access_token": access_token,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_id": data.get("data", {}).get("user_id", ""),
        })

        logger.info("Kite access token stored successfully")
        return func.HttpResponse(
            _html_redirect(FRONTEND_URL, None),
            mimetype="text/html",
        )

    except Exception as e:
        logger.exception("Kite token exchange failed")
        return func.HttpResponse(
            _html_redirect(FRONTEND_URL, f"Token exchange failed: {e}"),
            mimetype="text/html", status_code=500,
        )


@app.route(route="kite/status", methods=["GET"])
def kite_status(req: func.HttpRequest) -> func.HttpResponse:
    """Check if today's Kite token is available."""
    token_doc = _store.read("kite-token", "kite")

    if not token_doc:
        return func.HttpResponse(json.dumps({"connected": False, "reason": "No token stored"}), mimetype="application/json")

    token_date = token_doc.get("date", "")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if token_date != today:
        return func.HttpResponse(json.dumps({
            "connected": False,
            "reason": f"Token expired (from {token_date}). Please login again.",
            "last_login": token_date,
        }), mimetype="application/json")

    return func.HttpResponse(json.dumps({
        "connected": True,
        "user_id": token_doc.get("user_id", ""),
        "last_login": token_date,
    }), mimetype="application/json")


def _html_redirect(url: str, error: str | None) -> str:
    """Generate an HTML page that redirects to the frontend."""
    if error:
        return f"""<html><body style="font-family:sans-serif;background:#0d1117;color:#e6edf3;display:flex;align-items:center;justify-content:center;height:100vh">
<div style="text-align:center"><h2 style="color:#f85149">Kite Connection Failed</h2><p>{error}</p>
<a href="{url}" style="color:#58a6ff">Return to Stock Advisor</a></div></body></html>"""

    return f"""<html><body style="font-family:sans-serif;background:#0d1117;color:#e6edf3;display:flex;align-items:center;justify-content:center;height:100vh">
<div style="text-align:center"><h2 style="color:#3fb950">Kite Connected Successfully!</h2>
<p>Redirecting to Stock Advisor...</p>
<script>setTimeout(function(){{window.location.href="{url}"}},1500)</script>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Keep-warm timer — runs every 10 min during IST trading hours (Mon-Fri)
# Prevents cold starts for instant API responses
# ---------------------------------------------------------------------------

@app.timer_trigger(schedule="0 */10 3-11 * * 1-5", arg_name="timer", run_on_startup=False)
def keep_warm(timer: func.TimerRequest) -> None:
    """Ping every 10 min during 08:30-17:00 IST to prevent cold starts."""
    logger.debug("Keep-warm ping")
