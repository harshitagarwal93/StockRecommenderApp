"""Configuration and stock universe definitions."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Azure AI Foundry (primary LLM)
    azure_ai_endpoint: str = os.getenv("AZURE_AI_ENDPOINT", "")
    azure_ai_key: str = os.getenv("AZURE_AI_KEY", "")
    azure_ai_deployment: str = os.getenv("AZURE_AI_DEPLOYMENT", "gpt-4.1")

    # Fallback: GitHub Models API (free)
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "azure_foundry")

    # Kite Connect
    kite_api_key: str = os.getenv("KITE_API_KEY", "")
    kite_access_token: str = os.getenv("KITE_ACCESS_TOKEN", "")
    kite_api_secret: str = os.getenv("KITE_API_SECRET", "")

    # Cosmos DB
    cosmos_endpoint: str = os.getenv("COSMOS_ENDPOINT", "")
    cosmos_database: str = os.getenv("COSMOS_DATABASE", "StockAdvisor")
    cosmos_container: str = os.getenv("COSMOS_CONTAINER", "recommendations")
    cosmos_key: str = os.getenv("COSMOS_KEY", "")

    # Notification
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    notification_email: str = os.getenv("NOTIFICATION_EMAIL", "")

    # Portfolio constraints
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "500000"))
    max_buy_amount: float = float(os.getenv("MAX_BUY_AMOUNT", "10000"))
    max_buys_per_day: int = int(os.getenv("MAX_BUYS_PER_DAY", "3"))
    max_sells_per_day: int = int(os.getenv("MAX_SELLS_PER_DAY", "2"))
    max_portfolio_positions: int = int(os.getenv("MAX_PORTFOLIO_POSITIONS", "15"))
    max_single_allocation_pct: int = int(os.getenv("MAX_SINGLE_ALLOCATION_PCT", "15"))


# ---------------------------------------------------------------------------
# Indian stock universe (NSE tickers for yfinance)
# ---------------------------------------------------------------------------

LARGE_CAP_STOCKS: dict[str, str] = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Limited",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "TITAN.NS": "Titan Company",
    "SUNPHARMA.NS": "Sun Pharma",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "NESTLEIND.NS": "Nestle India",
    "WIPRO.NS": "Wipro",
    "TATAMOTORS.NS": "Tata Motors",
    "M&M.NS": "Mahindra & Mahindra",
    "HCLTECH.NS": "HCL Technologies",
    "POWERGRID.NS": "Power Grid Corp",
    "NTPC.NS": "NTPC Limited",
    "TATASTEEL.NS": "Tata Steel",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ONGC.NS": "ONGC",
    "JSWSTEEL.NS": "JSW Steel",
    "ADANIENT.NS": "Adani Enterprises",
}

MID_CAP_STOCKS: dict[str, str] = {
    "PERSISTENT.NS": "Persistent Systems",
    "TRENT.NS": "Trent Limited",
    "POLYCAB.NS": "Polycab India",
    "PIIND.NS": "PI Industries",
    "ASTRAL.NS": "Astral Limited",
    "MPHASIS.NS": "Mphasis",
    "COFORGE.NS": "Coforge",
    "JUBLFOOD.NS": "Jubilant FoodWorks",
    "FEDERALBNK.NS": "Federal Bank",
    "VOLTAS.NS": "Voltas",
    "PAGEIND.NS": "Page Industries",
    "AUROPHARMA.NS": "Aurobindo Pharma",
    "OBEROIRLTY.NS": "Oberoi Realty",
    "LICHSGFIN.NS": "LIC Housing Finance",
    "CUMMINSIND.NS": "Cummins India",
    "CROMPTON.NS": "Crompton Greaves CE",
    "KPITTECH.NS": "KPIT Technologies",
    "LTIM.NS": "LTIMindtree",
    "MAXHEALTH.NS": "Max Healthcare",
    "INDUSTOWER.NS": "Indus Towers",
}


def kite_to_yfinance(tradingsymbol: str) -> str:
    """Convert a Kite tradingsymbol like 'RELIANCE' to yfinance ticker 'RELIANCE.NS'."""
    return f"{tradingsymbol}.NS"


def get_full_universe() -> dict[str, tuple[str, str]]:
    """Return combined universe: ticker -> (name, category)."""
    universe = {}
    for ticker, name in LARGE_CAP_STOCKS.items():
        universe[ticker] = (name, "LARGE_CAP")
    for ticker, name in MID_CAP_STOCKS.items():
        universe[ticker] = (name, "MID_CAP")
    return universe
