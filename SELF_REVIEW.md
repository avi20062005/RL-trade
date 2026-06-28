# Self-Review: how the rebuild addresses the original audit

This document is the final engineering pass. The left column is the defect from
the audit of the legacy repository; the right column is how the rewrite resolves
it and where to verify.

## Critical findings

| Legacy defect | Resolution | Verify |
|---|---|---|
| Agent could not place a single trade (1 whole share of a ₹24k index vs ₹10k balance) | Fractional notional position sizing (`position_fraction` of equity); a high-priced index is tradable | `tests/test_env.py::test_buy_then_liquidate_accounting`; e2e PPO made 14 trades |
| Fabricated sine-wave "intraday" data; silent synthetic fallback poisoning the cache | Pipeline fails closed (`DataUnavailableError`); no synthetic generation anywhere | `tests/test_data.py::test_loader_fails_closed_on_outage` |
| Unresolved Git merge conflicts broke the whole `src/` tree | Single clean source tree; CI step greps for and rejects conflict markers | `.github/workflows/ci.yml` "Reject merge-conflict markers" |
| Train/serve normalization skew + look-ahead leakage (stats recomputed per slice) | Normalizer fit on train only, persisted, reused everywhere; env + predictor share one observation builder | `tests/test_features.py`, `tests/test_training.py::test_normalizer_fit_on_train_only` |
| Raw price level used as a feature, normalized non-causally | All features causal and scale-free; raw price excluded | `tests/test_features.py::test_features_are_causal` |
| Prioritized replay sampled uninitialized slots → `'int' object has no attribute 'state'` | Sampling validates every drawn leaf and re-draws from the filled region | `tests/test_replay.py::test_per_never_returns_empty_slot_when_sparse` |
| Insecure `torch.load` / pickle deserialization (RCE) | `torch.load(..., weights_only=True)`; SB3 native save; normalizer/cache via JSON | `agents/ddqn.py::DoubleDQNAgent.load` |

## High findings

| Legacy defect | Resolution |
|---|---|
| Duplicate, diverging root vs `src/` codebase | One installable package under `src/` |
| Sharpe annualized for daily bars on 5-minute data; daily RF on per-bar returns | Annualization factor = `bars_per_day * trading_days_per_year` derived from interval; per-bar RF |
| "Win rate" counted up-bars, not trades | Win rate / profit factor computed from round-trip PnL in the trade ledger |
| No global seeding → non-reproducible | `set_global_seeds` (python/numpy/torch); `env.reset(seed=...)`; SB3 `seed=` |
| `session_state` mutated from background threads; in-process Socket.IO server | Dashboard only reads artifacts and runs deterministic backtests; no threads, no global server |
| Unbounded logs | `RotatingFileHandler` with size + retention caps |
| Ensemble ties default to HOLD; ignores confidence | Weighted continuous scoring with `argmax`; `weights_from_sharpe` via softmax |
| Blanket `except Exception` hiding failures | Structured `QuantTradeError` hierarchy; narrow excepts; failures surface |
| Zero tests / no CI | 56 unit + integration tests; ruff + mypy(strict) + pytest in CI |

## Final Staff-Engineer checklist

* **Bugs / runtime errors** — full suite green; CLI exercised end-to-end (train → backtest → predict).
* **Type issues** — `mypy --strict` clean across 37 modules (ML-glue modules relax only the "Any from untyped library" checks, documented in `pyproject.toml`).
* **Lint / style** — `ruff` clean (pyflakes, bugbear, import order, pathlib, naming, comprehensions).
* **Security** — no pickle loads of untrusted data; `weights_only=True`; file paths sanitized in the cache key; no secrets in code.
* **RL correctness** — Double DQN target decoupling, Dueling aggregation, factorized NoisyNet, n-step bootstrap discounted by `gamma**n` with tail draining, PER with IS-weight correction and beta annealing.
* **Data leakage / train-serve skew** — single normalizer fit on train; single observation builder; causal features (regression-tested).
* **Performance** — vectorized indicator computation; float32 observations; no per-step DataFrame copies in the hot path; replay is O(log n).
* **Scalability** — pure functions and injected dependencies; no global mutable state; agents/data providers are swappable.

## Known limitations (honest scope)

* The bundled quickstart CSV is illustrative; real research requires a real data
  feed (Yahoo daily, or an intraday vendor for sub-daily bars).
* Long-only by default (`max_leverage = 1.0`); shorting is not implemented.
* The DDQN is a solid, tested implementation but not a full Rainbow (no
  distributional/categorical head); this is intentional to keep the codebase at
  a strong-student, reviewable scale rather than a research-lab artifact.
