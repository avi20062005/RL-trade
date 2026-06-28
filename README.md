# QuantTrade — Reinforcement Learning for Index Trading

A reinforcement-learning trading framework I built around the NIFTY 50 index (`^NSEI`).
It trains RL agents to make a daily **Buy / Sell / Hold** decision, evaluates them
honestly against a buy-and-hold benchmark, and serves both the backtest and a live
recommendation through a Streamlit dashboard.

I want to be upfront about the goal: this is a **research and learning project**, and the
thing I cared about most was getting the *methodology* right — no data leakage, realistic
costs, correct metrics, reproducible runs — rather than chasing an impressive-looking
return. A trading system that lies to you with a pretty backtest is worse than useless, so
I optimised for honesty first.

---

## What it does

- Fetches/loads **NIFTY 50 daily** OHLCV data (Yahoo Finance or a local CSV).
- Turns it into **causal, stationary features** (returns, RSI, ATR%, MACD, Bollinger
  width/position, realized volatility, volume z-score) — no raw price levels.
- Trains four RL agents: **PPO, DQN, A2C** (via Stable-Baselines3) and a **Double DQN I
  implemented from scratch** (Dueling network + NoisyNet exploration + prioritized
  experience replay + n-step returns).
- Backtests each agent on a held-out test split with proper trading economics
  (fractional position sizing, transaction costs, slippage, stop-loss / take-profit) and
  a full trade ledger.
- Reports the metrics that actually matter: Sharpe, Sortino, CAGR, max drawdown, Calmar,
  profit factor, trade-level win rate — all annualized correctly for the bar interval.
- Serves everything in a **dashboard** with two views: a performance comparison + equity
  curves, and a **"latest recommendation"** panel that shows each agent's live call for
  the most recent bar.

---

## Screenshots

> Drop your own screenshots into a `docs/` folder and they'll show up here.

![Equity curves](docs/equity_curves.png)
![Performance comparison](docs/performance.png)
![Latest recommendation](docs/recommendation.png)

---

## The design decisions I'm most proud of

These are the parts where I deliberately did the harder, correct thing instead of the easy
thing.

**1. No data leakage.** The normalization statistics (mean/std for each feature) are
computed on the **training split only**, saved to disk, and then applied unchanged to the
validation set, the test set, and live inference. This sounds obvious but it's the single
most common way backtests get inflated — if you normalize using the whole dataset, you've
leaked information about the future into the past. The training environment and the live
predictor build their observations through the *same* function, so what the model sees at
serving time is exactly what it saw during training (no train/serve skew).

**2. Causal, scale-free features.** I don't feed the raw price into the model. Price level
is non-stationary — a model trained at ₹15,000 means nothing at ₹24,000. Instead I use
returns and ratio/oscillator features that are comparable across time, and every indicator
only ever looks backwards (no peeking at future bars).

**3. Realistic execution.** Positions are sized as a fraction of equity (so a ₹24,000
index is tradable with any sensible balance — fractional units allowed), and every fill
pays slippage and a transaction cost. There's a proper trade ledger, so the win rate is
computed over actual round-trip trades, not "number of green days".

**4. Correct, non-exploding metrics.** Sharpe and volatility are annualized using a factor
derived from the bar interval, and degenerate cases (a flat equity curve, zero variance)
return 0 instead of dividing by ~0 and producing nonsense.

**5. Reproducibility.** A single seed seeds Python, NumPy and PyTorch, and the environment
resets are seeded, so a training run can be reproduced.

---

## How it's structured

I used a `src/` layout and split the project into small, single-responsibility modules so
each piece can be tested on its own and swapped out without touching the rest.

```
src/quanttrade/
  config.py        # frozen, validated config (no global state)
  utils/           # logging, seeding, exceptions
  data/            # schema validation, caching, provider-injected loader (fails closed)
  features/        # causal indicators, feature pipeline, train-only normalizer
  env/             # portfolio accounting + Gymnasium trading environment
  evaluation/      # metrics + backtester + walk-forward splits
  agents/          # Agent protocol, baselines, replay buffers, DDQN, SB3 wrapper
  training/        # leak-free dataset prep + trainer
  ensemble/        # confidence-weighted ensemble of agents
  inference/       # predictor (uses the saved train-fit normalizer)
  dashboard/       # Streamlit app
  cli.py           # train / backtest / predict entry points
tests/             # one test suite per module
```

The data flow is: **load -> validate -> causal features -> normalize (train stats) ->
environment -> agent -> backtest / predict.**

---

## Getting started

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,agents,data]"
```

The core library only needs NumPy / pandas / Gymnasium. PyTorch and Stable-Baselines3 are
optional extras (`agents`) used only for training, so the rest of the project stays light.

### 2. Get the data

The project defaults to `^NSEI` daily. Either let it fetch from Yahoo Finance (omit
`--csv-dir`), or download the daily history from Yahoo and save it as
`data/^NSEI_1d.csv` and pass `--csv-dir data`.

### 3. Train

```bash
python -m quanttrade --csv-dir data --timesteps 200000 train --algo ppo
python -m quanttrade --csv-dir data --timesteps 200000 train --algo dqn
python -m quanttrade --csv-dir data --timesteps 200000 train --algo a2c
python -m quanttrade --csv-dir data --timesteps 200000 train --algo ddqn
```

Each run saves the model and the normalizer into `artifacts/models/`.

### 4. Backtest and predict

```bash
python -m quanttrade --csv-dir data backtest --algo ppo
python -m quanttrade --csv-dir data predict  --algo ppo
```

### 5. Run the dashboard

```bash
streamlit run src/quanttrade/dashboard/app.py
```

Set the CSV directory in the sidebar and click **Run backtest**. You'll see the equity
curves, the performance table, and each trained agent's latest Buy/Sell/Hold call.

---

## Results (and an honest reading of them)

On my held-out `^NSEI` daily test split, the RL agents traded actively but **did not beat
buy-and-hold** — buy-and-hold returned roughly +1.7% while the agents landed slightly above
or below flat, with negative Sharpe ratios across the board (though generally with smaller
drawdowns).

I'm reporting that as-is on purpose. Beating a long-only benchmark on a single index with
daily bars is genuinely hard, and a lightly-trained agent on one historical slice has no
reason to. What this project demonstrates isn't a money-printing strategy — it's a
**correct, leak-free pipeline** where the numbers can actually be trusted, and where the
agents *do* place real, sensible trades (unlike my first attempt, which couldn't trade at
all). That's the part I set out to get right.

---

## Tech stack

- **Python 3.11+**
- **NumPy, pandas** — data and numerics
- **Gymnasium** — the RL environment interface
- **PyTorch** — my custom Double DQN
- **Stable-Baselines3** — PPO / DQN / A2C
- **Streamlit** — the dashboard
- **pytest, Ruff, mypy** — testing, linting, strict type-checking

---

## Code quality

The whole codebase is type-hinted and checked with `mypy --strict`, linted with `Ruff`,
and covered by a `pytest` suite (environment accounting, reward/PnL, feature causality,
normalizer round-trips, the replay buffer, metrics, inference). CI runs lint + types +
tests on every push.

```bash
ruff check src tests
mypy
pytest
```

---

## Limitations & future work

- **Single index, daily bars, long-only.** No shorting and no leverage by default.
- **One train/test split** in the default flow. The framework includes walk-forward /
  expanding-window splits, which would give a more robust evaluation across regimes.
- **No live sentiment feature.** An earlier version faked this; I removed it rather than
  ship a feature that didn't actually work. Adding a real, time-aligned news-sentiment
  signal is the most interesting extension I'd like to try next.
- **Not a trading bot.** The predictor outputs a signal; it does not connect to a broker
  or place orders.

---
