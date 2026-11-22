from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential


LOOKBACK_DAYS = 60
HISTORY_START = "2000-01-01"


def get_close_prices(symbol: str) -> pd.Series:
    history = yf.download(
        symbol,
        start=HISTORY_START,
        end=date.today().isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if history.empty:
        raise ValueError(f"No price history was returned for {symbol}.")

    close_columns = [
        column
        for column in history.columns
        if (column == "Close" if isinstance(column, str) else "Close" in column)
    ]
    if not close_columns:
        raise ValueError(f"Yahoo Finance did not return closing prices for {symbol}.")

    prices = history[close_columns[0]]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    prices.name = "Close"
    if len(prices) <= LOOKBACK_DAYS:
        raise ValueError("This symbol does not have enough price history to train the model.")
    return prices


def train_and_forecast(prices: pd.Series, forecast_days: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.to_numpy().reshape(-1, 1))

    features = np.array(
        [scaled_prices[index : index + LOOKBACK_DAYS] for index in range(len(scaled_prices) - LOOKBACK_DAYS)]
    )
    targets = scaled_prices[LOOKBACK_DAYS:]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = Sequential(
        [
            Input(shape=(LOOKBACK_DAYS, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dropout(0.3),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(features, targets, epochs=2, batch_size=8, shuffle=False, verbose=0)

    window = scaled_prices[-LOOKBACK_DAYS:].copy()
    predictions = []
    for _ in range(forecast_days):
        scaled_prediction = model.predict(window.reshape(1, LOOKBACK_DAYS, 1), verbose=0)
        predictions.append(scaler.inverse_transform(scaled_prediction)[0, 0])
        window = np.concatenate((window[1:], scaled_prediction), axis=0)

    forecast_dates = pd.bdate_range(
        start=prices.index[-1] + pd.offsets.BDay(),
        periods=forecast_days,
    )
    return forecast_dates, np.asarray(predictions)


def make_forecast_chart(prices: pd.Series, forecast_dates: pd.DatetimeIndex, forecast: np.ndarray) -> go.Figure:
    chart_start = prices.index[-1] - pd.DateOffset(years=2)
    visible_prices = prices.loc[prices.index >= chart_start]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=visible_prices.index,
            y=visible_prices,
            name="Historical close",
            mode="lines",
            line={"color": "#176b5b", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[prices.index[-1], *forecast_dates],
            y=[prices.iloc[-1], *forecast],
            name="Model forecast",
            mode="lines+markers",
            line={"color": "#e17b3a", "width": 2, "dash": "dash"},
            marker={"size": 5},
        )
    )
    figure.update_layout(
        height=470,
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#263b36", "family": "DM Sans, sans-serif"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={"title": None, "showgrid": False, "linecolor": "#d8dfdc"},
        yaxis={"title": "Adjusted close", "gridcolor": "#e7ece9", "zeroline": False},
    )
    return figure


st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      .stApp { background: #f5f7f4; }
      .main .block-container { max-width: 1120px; padding-top: 3rem; padding-bottom: 3rem; }
      h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #173b32; }
      p, label, input, button { font-family: 'DM Sans', sans-serif; }
      div[data-testid="stForm"] { background: #ffffff; padding: 1.2rem 1.4rem; border: 1px solid #dfe7e2; border-radius: 8px; }
      div[data-testid="stForm"] button { background: #176b5b; border: 0; color: #ffffff; }
      div[data-testid="stForm"] button:hover { background: #125447; color: #ffffff; }
      div[data-testid="stMetric"] { background: #ffffff; padding: 1rem 1.1rem; border: 1px solid #dfe7e2; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stock price forecast")
st.caption("Train a recurrent model on historical market data and explore its next-month estimate.")

with st.form("forecast_form"):
    left, right = st.columns([3, 1])
    with left:
        ticker_input = st.text_input("Ticker symbol", value="", placeholder="For example: AAPL")
    with right:
        forecast_days = st.number_input("Forecast days", min_value=5, max_value=60, value=30, step=5)
    submitted = st.form_submit_button("Train and forecast", use_container_width=True)

if submitted:
    ticker = ticker_input.strip().upper()
    if not ticker:
        st.error("Enter a ticker symbol to continue.")
    elif not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", ticker):
        st.error("Use a valid Yahoo Finance ticker, such as AAPL, MSFT, or BRK-B.")
    else:
        try:
            with st.spinner(f"Downloading {ticker} history and training the LSTM model..."):
                close_prices = get_close_prices(ticker)
                forecast_index, forecast_values = train_and_forecast(close_prices, int(forecast_days))

            latest_close = float(close_prices.iloc[-1])
            final_forecast = float(forecast_values[-1])
            forecast_change = (final_forecast / latest_close - 1) * 100

            st.subheader(f"{ticker} forecast")
            metrics = st.columns(3)
            metrics[0].metric("Latest adjusted close", f"{latest_close:,.2f}")
            metrics[1].metric(f"Estimate in {int(forecast_days)} trading days", f"{final_forecast:,.2f}")
            metrics[2].metric("Change vs. latest close", f"{forecast_change:+.2f}%")
            st.plotly_chart(
                make_forecast_chart(close_prices, forecast_index, forecast_values),
                use_container_width=True,
            )
            st.caption(
                f"History through {close_prices.index[-1].date()} · Forecast through {forecast_index[-1].date()} · "
                "Model estimates are experimental and are not investment advice."
            )
        except Exception as error:
            st.error(f"Could not create a forecast for {ticker}: {error}")