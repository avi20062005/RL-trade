"""Tests for prioritized replay and the n-step buffer."""

from __future__ import annotations

import numpy as np
import pytest

from quanttrade.agents.replay import NStepBuffer, PrioritizedReplayBuffer, Transition


def _tr(i: int, *, done: bool = False) -> Transition:
    return Transition(
        state=np.full(3, i, dtype=np.float32),
        action=i % 3,
        reward=float(i),
        next_state=np.full(3, i + 1, dtype=np.float32),
        done=done,
    )


def test_per_never_returns_empty_slot_when_sparse() -> None:
    """Regression for the legacy 'int has no attribute state' crash."""
    rng = np.random.default_rng(0)
    buf = PrioritizedReplayBuffer(capacity=1024, beta_frames=1000)
    # Fill far fewer entries than capacity, so most leaves are empty.
    for i in range(40):
        buf.push(_tr(i))
    for _ in range(200):
        batch, indices, weights = buf.sample(32, rng)
        assert all(isinstance(t, Transition) for t in batch)
        assert len(indices) == 32
        assert weights.shape == (32,)
        buf.update_priorities(indices, np.abs(rng.normal(size=32)))


def test_per_requires_enough_samples() -> None:
    rng = np.random.default_rng(0)
    buf = PrioritizedReplayBuffer(capacity=16)
    buf.push(_tr(0))
    with pytest.raises(Exception):  # noqa: B017 - QuantTradeError subclass
        buf.sample(8, rng)


def test_per_beta_anneals_to_one() -> None:
    buf = PrioritizedReplayBuffer(capacity=16, beta_start=0.4, beta_frames=10)
    rng = np.random.default_rng(0)
    for i in range(16):
        buf.push(_tr(i))
    for _ in range(50):
        buf.sample(4, rng)
    assert buf.beta == pytest.approx(1.0)


def test_nstep_returns_discounted_sum() -> None:
    nb = NStepBuffer(n_steps=3, gamma=0.5)
    out: list[Transition] = []
    for i in range(1, 4):  # rewards 1, 2, 3
        out.extend(nb.push(_tr(i)))
    assert len(out) == 1
    # R = 1 + 0.5*2 + 0.25*3 = 2.75
    assert out[0].reward == pytest.approx(2.75)


def test_nstep_drains_tail_on_done() -> None:
    nb = NStepBuffer(n_steps=4, gamma=1.0)
    emitted: list[Transition] = []
    emitted.extend(nb.push(_tr(1)))
    emitted.extend(nb.push(_tr(2)))
    emitted.extend(nb.push(_tr(3, done=True)))
    # Episode ended before the window filled; all 3 must be emitted, none lost.
    assert len(emitted) == 3
    assert all(t.done for t in emitted)
