# QuantTrade

A reinforcement-learning trading research framework for a single asset (e.g. the
NIFTY 50 index). It emphasizes the things that actually make an RL trading system
trustworthy: **causal features, leak-free evaluation, honest metrics, and
reproducibility** — not just a pile of algorithms.

The core library has no heavy dependencies (NumPy, pandas, Gymnasium), so it is
fast to install and fully unit-tested. PyTorch and Stable-Baselines3 are optional
extras used only by the learning agents.

## Why this design

* **No data fabrication.** The data pipeline fails closed if real data is
  unavailable; it never invents prices.
* **Causal, scale-free features.** Returns, RSI, ATR%, MACD histogram, Bollinger
  width/position, realized volatility and volume z-scores — no raw price levels.
* **Leak-free normalization.** Statistics are fit on the *training split only*,
  persisted as JSON, and applied unchanged at validation, test and inference. The
  environment and the predictor build observations through the same helper, so
  there is no train/serve skew.
* **Realistic execution.** Fractional position sizing (a ₹24,000 index is
  tradable with any reasonable balance), transaction costs, slippage, a trade
  ledger with round-trip PnL, and stop-loss / take-profit.
* **Correct metrics.** Sharpe, Sortino, CAGR, max drawdown, Calmar, profit
  factor, trade-level win rate — annualized using a factor derived from the bar
  interval, with degenerate inputs returning zeros rather than exploding.

## Architecture

```
src/quanttrade/
  config.py              frozen, validated config (no global singleton)
  utils/                 logging (rotating), seeding, structured exceptions
  data/                  schema validation, metadata cache, provider-injected loader
  features/              causal indicators, feature pipeline, train-only normalizer
  env/                   portfolio accounting + Gymnasium TradingEnv + shared observation
  evaluation/            metrics + backtester + walk-forward splits
  agents/                Agent protocol, baselines, replay buffers, DDQN, SB3 wrapper
  training/              leak-free dataset prep + trainer
  ensemble/              confidence-weighted ensemble
  inference/             predictor (uses the loaded train-fit normalizer)
  dashboard/             thin Streamlit UI
  cli.py                 train / backtest / predict entry points
tests/                   one suite per module
```

## Installation

```bash
pip install -e ".[dev]"            # core + tooling
pip install -e ".[dev,agents]"     # + PyTorch / Stable-Baselines3
pip install -e ".[dev,agents,data,dashboard]"  # everything
```

## Quickstart

Using a local CSV (`data/<symbol>_<interval>.csv`, e.g. `data/^NSEI_1d.csv`) for
fully reproducible, offline runs:

```bash
python -m quanttrade --csv-dir data --timesteps 200000 train --algo ppo
python -m quanttrade --csv-dir data backtest --algo ppo
python -m quanttrade --csv-dir data predict  --algo ppo
```

Omit `--csv-dir` to fetch from Yahoo Finance (requires the `data` extra).

Dashboard:

```bash
streamlit run src/quanttrade/dashboard/app.py
```

The dashboard uses a local CSV when present (`<csv_dir>/<symbol>_<interval>.csv`)
and otherwise fetches from Yahoo Finance. Any trained models found under
`artifacts/models/` are loaded and added to the comparison automatically; with
none present it shows buy-and-hold plus baseline agents.

## Deployment (Streamlit Community Cloud)

1. Train locally and commit the artifacts you want served (models are gitignored
   by default, so force-add them):
   ```bash
   python -m quanttrade --csv-dir data --timesteps 200000 train --algo ppo
   git add -f data/^NSEI_1d.csv artifacts/models/
   git commit -m "Add data and trained models" && git push
   ```
2. The repo ships a root `requirements.txt`. By default it serves the benchmark +
   baseline view (lean, reliable on the free tier). To also serve trained PPO/DQN/
   A2C/DDQN models on the live site, uncomment the `torch` / `stable-baselines3`
   lines in `requirements.txt` — but note these are large and may exceed free-tier
   memory; viewing trained agents is most reliable on a local run or a paid host.
3. On share.streamlit.io: New app → your repo → branch `main` → main file
   `src/quanttrade/dashboard/app.py` → Deploy.

## Agents

* `ppo`, `dqn`, `a2c` via Stable-Baselines3.
* `ddqn` — a from-scratch Double DQN with a Dueling head, factorized NoisyNet
  exploration, prioritized replay, and n-step returns. The replay buffer is
  pure NumPy and unit-tested, including a regression for the sparse-buffer
  sampling crash that affected the predecessor implementation.

Model checkpoints are loaded with `weights_only=True` to avoid arbitrary-code
execution during deserialization.

## Development

```bash
ruff check src tests     # lint + import order
mypy                     # strict type checking
pytest                   # unit + integration tests
```

CI runs all three on Python 3.11 and 3.12, plus a second job with the `agents`
extra installed so the DDQN/SB3 paths are exercised.

## Disclaimer

This is a research and educational framework. It is not financial advice and is
not intended for live trading.

## License

MIT — see `LICENSE`.
