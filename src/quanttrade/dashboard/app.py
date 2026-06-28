"""Streamlit dashboard.

A thin presentation layer over the core library. It only *reads* artifacts and
runs deterministic backtests; it does not train in a background thread or mutate
session state from threads (both legacy bugs). Streamlit and Plotly are optional
dependencies (the ``dashboard`` extra).

Data source: a local CSV (``<csv_dir>/<symbol>_<interval>.csv``) is used when
present; otherwise the app falls back to fetching from Yahoo Finance (requires
the ``data`` extra). Any trained models found under ``artifacts/models`` are
loaded and added to the comparison automatically.

Run with::

    streamlit run src/quanttrade/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from quanttrade.agents.base import HoldAgent, RandomAgent
from quanttrade.config import AppConfig
from quanttrade.dashboard.models import discover_trained_agents
from quanttrade.data.cache import OhlcvCache
from quanttrade.data.loader import CsvProvider, DataLoader, MarketDataProvider, YFinanceProvider
from quanttrade.evaluation.backtester import Backtester, BacktestResult
from quanttrade.inference.predictor import Predictor
from quanttrade.training.datasets import prepare_datasets
from quanttrade.utils.exceptions import QuantTradeError


@st.cache_data(show_spinner=False)
def _load_ohlcv(symbol: str, interval: str, cache_dir: str, csv_dir: str) -> pd.DataFrame:
    config = AppConfig.from_dict({"data": {"symbol": symbol, "interval": interval}})
    csv_path = Path(csv_dir) / f"{symbol}_{interval}.csv"
    provider: MarketDataProvider = (
        CsvProvider(csv_dir) if csv_path.exists() else YFinanceProvider()
    )
    loader = DataLoader(config.data, provider=provider, cache=OhlcvCache(cache_dir))
    return loader.load(period="10y")


def main() -> None:
    st.set_page_config(page_title="QuantTrade", layout="wide")
    st.title("QuantTrade — RL Trading Dashboard")

    with st.sidebar:
        st.header("Configuration")
        symbol = st.text_input("Symbol", value="^NSEI")
        interval = st.selectbox("Interval", ["1d", "60m", "30m", "15m", "5m"], index=0)
        csv_dir = st.text_input("CSV directory", value="data")
        st.caption("If the CSV is missing, the app fetches from Yahoo Finance.")
        run = st.button("Run backtest", type="primary")

    if not run:
        st.info("Configure inputs in the sidebar and click **Run backtest**.")
        return

    config = AppConfig.from_dict({"data": {"symbol": symbol, "interval": interval}})

    with st.spinner("Loading market data..."):
        try:
            ohlcv = _load_ohlcv(symbol, interval, config.data.cache_dir, csv_dir)
        except QuantTradeError as exc:
            st.error(f"Could not load data: {exc}")
            return

    with st.spinner("Preparing leak-free datasets..."):
        try:
            bundle = prepare_datasets(ohlcv, config)
        except QuantTradeError as exc:
            st.error(f"Could not prepare datasets: {exc}")
            return

    bt = Backtester(
        bundle.test.features,
        bundle.test.prices,
        config.env,
        config.evaluation,
        bundle.bars_per_year,
    )

    results: list[BacktestResult] = [bt.run_buy_and_hold(), bt.run_agent(HoldAgent())]

    trained = discover_trained_agents(config)
    if trained:
        with st.spinner(f"Evaluating {len(trained)} trained model(s)..."):
            for label, agent in trained:
                results.append(bt.run_agent(agent, label=label))
    else:
        results.append(bt.run_agent(RandomAgent()))
        st.caption(
            "No trained models found. Train one with "
            "`python -m quanttrade train --algo ppo` (and install the `agents` extra)."
        )

    st.subheader("Equity curves (test split)")
    equity_df = pd.DataFrame({r.label: pd.Series(r.equity_curve) for r in results})
    st.line_chart(equity_df)

    st.subheader("Performance comparison")
    table = pd.DataFrame(
        [
            {
                "strategy": r.label,
                "total_return_%": round(r.metrics.total_return * 100, 2),
                "cagr_%": round(r.metrics.cagr * 100, 2),
                "sharpe": round(r.metrics.sharpe, 2),
                "sortino": round(r.metrics.sortino, 2),
                "max_drawdown_%": round(r.metrics.max_drawdown * 100, 2),
                "win_rate_%": round(r.metrics.win_rate * 100, 1),
                "trades": r.metrics.trade_count,
            }
            for r in results
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Latest recommendation")
    st.caption(
        "Each trained agent's action for the most recent bar, assuming a flat "
        "starting position. This is a model output, not financial advice."
    )
    if trained:
        rows = []
        for label, agent in trained:
            try:
                predictor = Predictor.from_artifacts(agent, config.train.models_dir, config)
                pred = predictor.predict(ohlcv)
                rows.append(
                    {
                        "agent": label,
                        "recommendation": pred.action,
                        "as_of": str(pred.timestamp.date()),
                        "price": round(pred.price, 2),
                    }
                )
            except QuantTradeError as exc:
                rows.append(
                    {"agent": label, "recommendation": f"error: {exc}", "as_of": "-", "price": 0.0}
                )
        cols = st.columns(len(rows)) if rows else []
        for col, row in zip(cols, rows, strict=True):
            col.metric(label=str(row["agent"]).upper(), value=str(row["recommendation"]))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Train a model to see a live BUY / SELL / HOLD recommendation here.")


if __name__ == "__main__":
    main()
