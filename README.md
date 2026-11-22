# Stock Price Forecast

A Streamlit application that downloads historical stock prices, trains an LSTM model, and plots a short-term forecast. Enter a Yahoo Finance ticker such as `AAPL`, `MSFT`, or `BRK-B` to get started.

## Live demo

Try the deployed app: [Stock Price Forecast on Railway](https://web-production-1bc1aa24.up.railway.app/).

The interface lets you enter a ticker and choose a forecast horizon. After training, it displays the latest adjusted close, the final forecast estimate, the estimated percentage change, and a chart comparing recent historical prices with the model forecast.

## Features

- Downloads daily, adjusted historical prices from Yahoo Finance.
- Trains a two-layer LSTM model using the latest 60 observations as its lookback window.
- Forecasts 5 to 60 weekdays ahead; the default is 30.
- Displays recent history, the model forecast, the latest observed price, the final forecast estimate, and the estimated percentage change.
- Requires no API key. Data is retrieved when a forecast is requested.

## Run locally

Python 3.11 is used by the deployment configuration. From the project root, create and activate a virtual environment, install dependencies, and start Streamlit:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Deploy on Railway

1. Push the project to a GitHub repository.
2. In Railway, create a project and deploy from that repository.
3. Set the service root directory to the repository root, where `app.py` and `requirements.txt` are located.
4. Railway uses `requirements.txt` to install dependencies and the `Procfile` to start Streamlit. The start command listens on Railway's `$PORT`.
5. Once the service is deployed, open the service's **Settings** and generate or enable a public domain.

The `.python-version` file pins Python to 3.11.9. The `mise.toml` file disables Python GitHub artifact-attestation verification to work around the Railway build error `No GitHub artifact attestations found for python@3.11.9`. This setting weakens verification of the Python runtime artifact; remove it when the builder no longer requires the workaround.

## Project files

- `app.py`: Streamlit interface, Yahoo Finance data download, model training, forecasting, and chart.
- `notebooks/StockPrice.ipynb`: original exploratory notebook and source for the LSTM workflow.
- `requirements.txt`: pinned Python dependencies.
- `Procfile`: Railway web process command.
- `.python-version` and `mise.toml`: Railway Python version and build configuration.

## Forecasting notes

The model is trained when a forecast is requested. It uses adjusted closing prices from Yahoo Finance, scales the full available history, and recursively feeds each prediction into the next forecast step. Forecast dates advance by weekdays and do not account for exchange-specific holidays. Returned price units and currency follow the Yahoo Finance ticker data.

This is a simple educational forecasting example, not a validated trading system. Long-range recursive predictions can drift substantially, historical prices do not determine future returns, and the displayed percentage is a model estimate rather than a probability or guarantee. Forecasts are not investment advice.

## Troubleshooting

- **No history returned:** confirm the ticker is valid on Yahoo Finance and try again later; the app needs network access to fetch market data.
- **Railway build fails on Python attestations:** confirm `mise.toml` is present in the repository root and that Railway is building the latest commit.
- **Railway service builds but is unavailable:** check the deployment logs and confirm the service has a public domain enabled.