"""Gymnasium trading environment.

The environment consumes an **already-normalized** feature matrix and the
aligned execution price. It deliberately does *not* compute normalization
itself: that responsibility lives in :class:`FeatureNormalizer`, fit on the
training split only. This separation is what prevents train/serve skew.

Timing (no look-ahead): at step ``t`` the agent observes features computed from
information up to and including ``t``, the order executes at price ``t`` (with
slippage), and the reward is the log growth of equity from ``t`` to ``t+1``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quanttrade.config import EnvConfig
from quanttrade.env.observation import POSITION_FEATURES, assemble_observation
from quanttrade.env.portfolio import Action, Portfolio
from quanttrade.utils.exceptions import TradingEnvError
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)

_POSITION_FEATURES = POSITION_FEATURES


class TradingEnv(gym.Env[np.ndarray, np.intp]):
    """A single-asset, discrete-action trading environment."""

    metadata: dict[str, list[str]] = {"render_modes": []}  # noqa: RUF012  gym base contract

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        config: EnvConfig,
    ) -> None:
        super().__init__()
        if features.ndim != 2:
            raise TradingEnvError("features must be a 2D array [timesteps, n_features]")
        if len(features) != len(prices):
            raise TradingEnvError("features and prices must have equal length")
        if len(prices) < 2:
            raise TradingEnvError("need at least 2 timesteps")
        if np.any(prices <= 0):
            raise TradingEnvError("prices must be positive")

        self._features = features.astype(np.float32)
        self._prices = prices.astype(np.float64)
        self._config = config
        self._n_steps = len(prices)
        self._portfolio = Portfolio(config)

        obs_dim = features.shape[1] + _POSITION_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(Action))

        self._step = 0
        self._equity_curve: list[float] = []

    # ------------------------------------------------------------------ API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._portfolio.reset()
        self._step = 0
        self._equity_curve = [self._portfolio.equity(self._prices[0])]
        return self._observe(), self._info()

    def step(self, action: np.intp | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(int(action)):
            raise TradingEnvError(f"invalid action: {action!r}")

        price = self._prices[self._step]
        equity_before = self._portfolio.equity(price)

        effective = self._apply_risk_overrides(int(action), price)
        if effective == Action.BUY:
            self._portfolio.buy(self._step, price)
        elif effective == Action.SELL:
            self._portfolio.liquidate(self._step, price)

        self._step += 1
        terminated = self._step >= self._n_steps - 1
        next_price = self._prices[self._step]
        equity_after = self._portfolio.equity(next_price)
        self._equity_curve.append(equity_after)

        reward = self._reward(equity_before, equity_after)
        info = self._info()
        info["action"] = Action(effective).name
        return self._observe(), reward, terminated, False, info

    # -------------------------------------------------------------- helpers

    def _apply_risk_overrides(self, action: int, price: float) -> int:
        """Force a flatten if stop-loss or take-profit is breached."""
        pnl = self._portfolio.unrealized_pnl_pct(price)
        if self._portfolio.units > 0 and (
            pnl <= -self._config.stop_loss_pct or pnl >= self._config.take_profit_pct
        ):
            return int(Action.SELL)
        return action

    def _reward(self, equity_before: float, equity_after: float) -> float:
        if equity_before <= 0 or equity_after <= 0:
            return 0.0
        return self._config.reward_scaling * float(np.log(equity_after / equity_before))

    def _observe(self) -> np.ndarray:
        idx = min(self._step, self._n_steps - 1)
        price = self._prices[idx]
        return assemble_observation(
            self._features[idx],
            self._portfolio.exposure(price),
            self._portfolio.unrealized_pnl_pct(price),
        )

    def _info(self) -> dict[str, Any]:
        idx = min(self._step, self._n_steps - 1)
        price = self._prices[idx]
        return {
            "step": self._step,
            "price": float(price),
            "cash": self._portfolio.cash,
            "units": self._portfolio.units,
            "equity": self._portfolio.equity(price),
            "exposure": self._portfolio.exposure(price),
            "trade_count": len(self._portfolio.ledger),
        }

    # ---------------------------------------------------------- evaluation

    @property
    def equity_curve(self) -> np.ndarray:
        return np.asarray(self._equity_curve, dtype=np.float64)

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio
