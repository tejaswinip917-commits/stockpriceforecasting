# Stock Price Forecast

A Streamlit app that downloads daily adjusted closing prices from Yahoo Finance, trains the notebook's LSTM model, and plots a configurable forecast of 5 to 60 future trading days.

## Run locally

Use Python 3.11, install the packages, then start Streamlit:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Railway

1. Push this project to a GitHub repository.
2. In Railway, create a project from that GitHub repository.
3. Railway installs `requirements.txt` and starts the web process from `Procfile`.
4. In the service settings, enable a public domain.

The app binds to Railway's `PORT` environment variable. No API key is required; price history is fetched from Yahoo Finance at forecast time. Forecasts are experimental and are not investment advice.