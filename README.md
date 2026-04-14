# Stock Advisor — AI-Powered Indian Stock Recommendation Engine

An AI-powered stock market recommendation engine for NSE-listed Large Cap and Mid Cap Indian stocks. Provides daily BUY/SELL recommendations based on fundamental and technical analysis.

## Architecture

```
Timer (9 AM IST daily)  →  Yahoo Finance (free)  →  Technical Analysis
                                                      RSI, MACD, SMA, Bollinger
                                                            ↓
Kite Connect API  →  Live Portfolio  →  Azure OpenAI (GPT-4.1)  →  Recommendations
                                                            ↓
                                        CosmosDB + Email + Web UI
```

## Features

- **Daily AI Analysis**: Automated daily stock recommendations at market open
- **Live Portfolio**: Integrates with Kite Connect for real-time holdings
- **Technical Analysis**: RSI, MACD, SMA/EMA, Bollinger Bands, Volume analysis
- **Fundamental Data**: PE, PB, ROE, D/E, Revenue Growth via Yahoo Finance
- **LLM-Powered**: GPT-4.1 via Azure AI Foundry for intelligent stock picking
- **Responsive Web UI**: Mobile-friendly dashboard with reasoning and citations
- **Risk Management**: Configurable buy limits, position sizing, stop losses
- **Persistence**: Azure CosmosDB for recommendation history

## Setup

### Prerequisites
- Python 3.11+
- Azure subscription (for deployment)
- Kite Connect API credentials (optional, for live portfolio)

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the web UI
python webapp.py

# Or run CLI analysis
python main.py
```

### Configuration (via .env or App Settings)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_BUY_AMOUNT` | 10000 | Max ₹ per BUY recommendation |
| `MAX_BUYS_PER_DAY` | 3 | Max BUY recommendations per day |
| `MAX_SELLS_PER_DAY` | 2 | Max SELL recommendations per day |
| `LLM_PROVIDER` | azure_foundry | `azure_foundry` or `github` |
| `KITE_API_KEY` | — | Kite Connect API key |
| `KITE_ACCESS_TOKEN` | — | Kite session access token |

## Azure Deployment

- **Web App**: `stock-advisor-webapp.azurewebsites.net`
- **AI Model**: GPT-4.1 on Azure OpenAI (East US)
- **Database**: CosmosDB `StockAdvisor` database
- **Resource Group**: `StockRecommenderApp`

## Disclaimer

This is an AI-generated recommendation tool for **informational purposes only**. Not financial advice. Always do your own research before making investment decisions.
