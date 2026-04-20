---
description: "Use when committing code, pushing to GitHub, or deploying the StockApp to Azure. Covers Git workflow, Azure deployment commands, and project conventions."
applyTo: "**"
---

# StockApp Workflow Instructions

## Git Workflow

- **Remote**: `origin` → `https://github.com/harshitagarwal93/StockRecommenderApp.git`
- **Branch**: `main`
- When asked to commit, stage all relevant changed files, write a concise commit message, and run `git add` + `git commit`. **Always ask the user for confirmation before running `git push`.**
- When asked to push or check in, run `git push origin main` after confirming with the user.

## Azure Deployment

- **Account**: `harshit93@outlook.com`
- **Resource Group**: `StockRecommenderApp`
- **Function App**: `stock-advisor-func` (Python 3.11, serverless)
  - URL: `https://stock-advisor-func.azurewebsites.net`
  - Timer-triggered daily analysis + HTTP API endpoints
- **Web App**: `stock-advisor-webapp` (Docker container, Gunicorn)
  - URL: `https://stock-advisor-webapp.azurewebsites.net`
- **Cosmos DB**: database `StockAdvisor`, container `recommendations`
- **Storage Account**: `stockadvisorsa` (static frontend hosting)
  - URL: `https://stockadvisorsa.z29.web.core.windows.net`
- **AI Model**: GPT-4.1 on Azure AI Foundry (East US)

### Deployment Commands

When asked to deploy, **confirm with the user first**, then use these commands:

**Function App** (zip deploy):
```bash
cd /d/POCs/StockApp
az login --username harshit93@outlook.com
func azure functionapp publish stock-advisor-func --python
```

**Web App** (Docker):
```bash
az webapp up --name stock-advisor-webapp --resource-group StockRecommenderApp --runtime "PYTHON:3.11"
```

**Frontend** (static site to blob storage):
```bash
az storage blob upload-batch --account-name stockadvisorsa --destination '$web' --source frontend/ --overwrite
```

## Project Conventions

- Python 3.11, Flask for web UI, Azure Functions for serverless
- All stock tickers use NSE format with `.NS` suffix (e.g., `SBIN.NS`)
- Environment variables for all secrets — never hardcode credentials
- Cosmos DB for persistence, Yahoo Finance for market data
- LLM provider: Azure AI Foundry (primary), GitHub Models (fallback)
- Portfolio constraints: MAX_BUY_AMOUNT=₹10k, MAX_BUYS_PER_DAY=3, MAX_SINGLE_ALLOCATION_PCT=15%
