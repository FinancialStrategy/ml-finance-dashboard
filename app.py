import dash
from dash import dcc, html
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pypfopt import EfficientFrontier, expected_returns, risk_models
import os

# Dash uygulaması
app = dash.Dash(__name__)
server = app.server  # Render için gerekli

# Varsayılan parametreler
TICKERS = ["GC=F", "SI=F", "CL=F", "^GSPC"]
START_DATE = "2018-01-01"
INITIAL_BUDGET = 1_000_000

# Veri çekme fonksiyonu
def get_data(tickers, start):
    df = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    return df.dropna()

# ML tahmin fonksiyonu
def forecast_lstm(series, n_steps=20, epochs=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i - n_steps:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, n_steps, 1)
    y = np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)
    idx = series.index[n_steps:]
    return pd.Series(pred.flatten(), index=idx)

# Portföy optimizasyonu
def optimize_portfolio(prices, forecast_mu=None):
    mu = expected_returns.mean_historical_return(prices)
    if forecast_mu is not None:
        mu["^GSPC"] = forecast_mu.mean() * 252
        mu = mu * 0.7 + forecast_mu.mean() * 0.3 * 252
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu.drop("^GSPC"), S)
    weights = ef.max_sharpe()
    return ef.clean_weights(), ef.portfolio_performance()

# Layout
app.layout = html.Div([
    html.H1("ML Destekli Portföy Analizi"),
    dcc.Interval(id="startup-trigger", interval=1000, n_intervals=0),
    dcc.Graph(id="ml-graph"),
    dcc.Graph(id="ef-graph"),
    dcc.Graph(id="perf-graph")
])

# Callback
@app.callback(
    dash.dependencies.Output("ml-graph", "figure"),
    dash.dependencies.Output("ef-graph", "figure"),
    dash.dependencies.Output("perf-graph", "figure"),
    dash.dependencies.Input("startup-trigger", "n_intervals")
)
def update_graphs(_):
    prices = get_data(TICKERS, START_DATE)
    returns = np.log(prices / prices.shift(1)).dropna()
    sp500 = returns["^GSPC"].dropna()
    forecast = forecast_lstm(sp500)

    # ML grafiği
    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(x=sp500.index, y=sp500, name="Gerçek"))
    fig_ml.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Tahmin"))
    fig_ml.update_layout(title="S&P 500 LSTM Tahmini")

    # Optimizasyon
    weights, perf = optimize_portfolio(prices, forecast)
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Bar(x=list(weights.keys()), y=list(weights.values())))
    fig_ef.update_layout(title="Portföy Ağırlıkları")

    # Performans
    pf_returns = returns.drop("^GSPC", axis=1).dot(pd.Series(weights))
    pf_value = (1 + pf_returns).cumprod() * INITIAL_BUDGET
    bm_value = (1 + sp500).cumprod() * INITIAL_BUDGET
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=pf_value.index, y=pf_value, name="Portföy"))
    fig_perf.add_trace(go.Scatter(x=bm_value.index, y=bm_value, name="S&P 500"))
    fig_perf.update_layout(title="Portföy vs Benchmark")

    return fig_ml, fig_ef, fig_perf

# Port bind ve sunucu başlatma
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
import os

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
