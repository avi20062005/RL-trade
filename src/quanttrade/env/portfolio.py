"""Portfolio accounting and trade ledger.

Models a long-only (optionally leveraged) book with fractional position sizing.
Every fill pays slippage and a proportional transaction cost. The ledger records
each fill and computes realized PnL on each round trip, which is the correct
basis for a trade-level win rate (the legacy system counted up-bars instead).

Invariants enforced:
* cash never goes negative;
* exposure never exceeds ``max_leverage * equity``;
* a BUY that cannot afford a minimum quantity is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from quanttrade.config import EnvConfig

_BPS = 1e-4
_MIN_QTY = 1e-9


class Action(IntEnum):
    """Discrete trading actions."""

    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass(frozen=True, slots=True)
class Fill:
    """A single executed order."""

    step: int
    action: Action
    price: float  # execution price including slippage
    quantity: float
    fee: float
    realized_pnl: float  # non-zero only on closing fills


@dataclass(slots=True)
class Portfolio:
    """Mutable portfolio state with explicit, testable accounting."""

    config: EnvConfig
    cash: float = field(init=False)
    units: float = field(init=False)
    avg_price: float = field(init=False)
    ledger: list[Fill] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cash = self.config.initial_cash
        self.units = 0.0
        self.avg_price = 0.0
        self.ledger = []

    @property
    def cost_rate(self) -> float:
        return self.config.transaction_cost_bps * _BPS

    @property
    def slippage_rate(self) -> float:
        return self.config.slippage_bps * _BPS

    def equity(self, mark_price: float) -> float:
        """Total portfolio value marked at ``mark_price``."""
        return self.cash + self.units * mark_price

    def exposure(self, mark_price: float) -> float:
        """Fraction of equity currently held in the position."""
        eq = self.equity(mark_price)
        return (self.units * mark_price) / eq if eq > 0 else 0.0

    def unrealized_pnl_pct(self, mark_price: float) -> float:
        """Unrealized return of the open position versus average cost."""
        if self.units <= 0 or self.avg_price <= 0:
            return 0.0
        return (mark_price - self.avg_price) / self.avg_price

    def buy(self, step: int, mark_price: float) -> Fill | None:
        """Add one lot (``position_fraction`` of equity) of exposure.

        Returns the :class:`Fill`, or ``None`` if the trade was not affordable.
        """
        exec_price = mark_price * (1.0 + self.slippage_rate)
        equity = self.equity(mark_price)

        target_notional = self.config.position_fraction * equity
        # Respect the leverage cap on total exposure.
        max_total_exposure = self.config.max_leverage * equity
        current_exposure_value = self.units * mark_price
        headroom = max_total_exposure - current_exposure_value
        notional = min(target_notional, max(headroom, 0.0))

        # Respect available cash (cost = notional + fee).
        affordable_notional = self.cash / (1.0 + self.cost_rate)
        notional = min(notional, affordable_notional)
        if notional <= 0:
            return None

        quantity = notional / exec_price
        if quantity < _MIN_QTY:
            return None

        fee = notional * self.cost_rate
        self.avg_price = (self.avg_price * self.units + exec_price * quantity) / (
            self.units + quantity
        )
        self.units += quantity
        self.cash -= notional + fee

        fill = Fill(step, Action.BUY, exec_price, quantity, fee, realized_pnl=0.0)
        self.ledger.append(fill)
        return fill

    def liquidate(self, step: int, mark_price: float) -> Fill | None:
        """Close the entire position. Returns the closing fill, or ``None``."""
        if self.units < _MIN_QTY:
            return None
        exec_price = mark_price * (1.0 - self.slippage_rate)
        quantity = self.units
        proceeds = quantity * exec_price
        fee = proceeds * self.cost_rate
        cost_basis = self.avg_price * quantity
        realized = proceeds - fee - cost_basis

        self.cash += proceeds - fee
        self.units = 0.0
        self.avg_price = 0.0

        fill = Fill(step, Action.SELL, exec_price, quantity, fee, realized_pnl=realized)
        self.ledger.append(fill)
        return fill

    @property
    def closed_trades(self) -> list[Fill]:
        """Fills that closed a position (carry realized PnL)."""
        return [f for f in self.ledger if f.action == Action.SELL]
