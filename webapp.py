"""Flask web application — responsive UI for stock recommendations."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from stock_advisor.config import Config
from stock_advisor.cosmos_store import CosmosStore
from stock_advisor.orchestrator import run_daily_analysis
from stock_advisor.portfolio_manager import PortfolioManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")


def _get_pm() -> PortfolioManager:
    config = Config()
    store = CosmosStore(config)
    return PortfolioManager(config, store)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recommendation")
def api_recommendation():
    pm = _get_pm()
    rec = pm.get_latest_recommendation()
    if rec:
        return jsonify(rec.to_dict())
    return jsonify({"message": "No recommendation found for today"}), 404


@app.route("/api/portfolio")
def api_portfolio():
    pm = _get_pm()
    portfolio = pm.get_portfolio()
    return jsonify(portfolio.to_dict())


@app.route("/api/history")
def api_history():
    limit = request.args.get("limit", 7, type=int)
    pm = _get_pm()
    history = pm.get_recommendation_history(limit)
    return jsonify(history)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        rec = run_daily_analysis()
        return jsonify(rec.to_dict())
    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings")
def api_settings():
    config = Config()
    return jsonify({
        "max_buy_amount": config.max_buy_amount,
        "max_buys_per_day": config.max_buys_per_day,
        "max_sells_per_day": config.max_sells_per_day,
        "max_portfolio_positions": config.max_portfolio_positions,
        "max_single_allocation_pct": config.max_single_allocation_pct,
        "llm_provider": config.llm_provider,
        "llm_model": config.azure_ai_deployment if config.llm_provider == "azure_foundry" else "gpt-4o",
        "kite_connected": bool(config.kite_api_key and config.kite_access_token),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
