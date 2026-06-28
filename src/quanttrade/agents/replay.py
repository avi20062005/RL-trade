"""Replay buffers for value-based agents.

These structures are pure Python/NumPy (no torch), so they are fully unit
tested. Two correctness fixes over the legacy implementation:

* :meth:`PrioritizedReplayBuffer.sample` never returns an uninitialized slot.
  The legacy SumTree could route a sample to an empty leaf and then crash with
  ``'int' object has no attribute 'state'``; here every draw is validated and
  re-drawn from the filled region if necessary.
* :class:`NStepBuffer` *drains* its tail on episode end, emitting the shorter
  n-step returns instead of silently discarding the final transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanttrade.utils.exceptions import QuantTradeError


@dataclass(frozen=True, slots=True)
class Transition:
    """A single (possibly n-step) environment transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class _SumTree:
    """Binary indexed tree storing priorities; O(log n) sample/update."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: list[Transition | None] = [None] * capacity
        self._write = 0
        self.size = 0

    def total(self) -> float:
        return float(self._tree[0])

    def max_leaf(self) -> float:
        return float(self._tree[self.capacity - 1 :].max())

    def add(self, priority: float, data: Transition) -> None:
        idx = self._write + self.capacity - 1
        self._data[self._write] = data
        self.update(idx, priority)
        self._write = (self._write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self._tree[idx]
        self._tree[idx] = priority
        parent = (idx - 1) // 2
        while parent >= 0:
            self._tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, s: float) -> tuple[int, float, Transition | None]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self._tree):
                break
            if s <= self._tree[left]:
                idx = left
            else:
                s -= self._tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, float(self._tree[idx]), self._data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay with IS-weight correction."""

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-5,
    ) -> None:
        if capacity < 1:
            raise QuantTradeError("capacity must be >= 1")
        self._tree = _SumTree(capacity)
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta_frames = max(1, beta_frames)
        self._epsilon = epsilon
        self._frame = 0

    def __len__(self) -> int:
        return self._tree.size

    @property
    def beta(self) -> float:
        frac = min(1.0, self._frame / self._beta_frames)
        return self._beta_start + frac * (1.0 - self._beta_start)

    def push(self, transition: Transition) -> None:
        max_priority = self._tree.max_leaf() or 1.0
        self._tree.add(max_priority, transition)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[list[Transition], list[int], np.ndarray]:
        """Sample a prioritized minibatch. Validates every drawn leaf."""
        if len(self) < batch_size:
            raise QuantTradeError("not enough samples in buffer")
        self._frame += 1

        batch: list[Transition] = []
        indices: list[int] = []
        priorities: list[float] = []
        total = self._tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            data: Transition | None = None
            idx: int = 0
            priority: float = 0.0
            attempts = 0
            while not isinstance(data, Transition) and attempts < 8:
                lo, hi = segment * i, segment * (i + 1)
                s = rng.uniform(lo, hi) if hi > lo else rng.uniform(0.0, total)
                idx, priority, data = self._tree.get(s)
                attempts += 1
            if not isinstance(data, Transition):  # fall back to any filled slot
                idx, priority, data = self._tree.get(rng.uniform(0.0, total))
                if not isinstance(data, Transition):
                    raise QuantTradeError("replay buffer returned an empty slot")
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        probs = np.asarray(priorities, dtype=np.float64) / (total + 1e-12)
        weights = (len(self) * probs) ** (-self.beta)
        weights /= weights.max()
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors, strict=True):
            priority = (abs(float(err)) + self._epsilon) ** self._alpha
            self._tree.update(idx, priority)


class NStepBuffer:
    """Accumulates n-step returns; drains its tail on episode end."""

    def __init__(self, n_steps: int, gamma: float) -> None:
        if n_steps < 1:
            raise QuantTradeError("n_steps must be >= 1")
        self._n = n_steps
        self._gamma = gamma
        self._buffer: list[Transition] = []

    def push(self, transition: Transition) -> list[Transition]:
        """Append a 1-step transition; return any completed n-step transitions."""
        self._buffer.append(transition)
        emitted: list[Transition] = []
        if len(self._buffer) >= self._n:
            emitted.append(self._make_nstep(0))
            self._buffer.pop(0)
        if transition.done:
            # Drain the tail with progressively shorter horizons.
            while self._buffer:
                emitted.append(self._make_nstep(0))
                self._buffer.pop(0)
        return emitted

    def _make_nstep(self, start: int) -> Transition:
        reward = 0.0
        next_state = self._buffer[start].next_state
        done = self._buffer[start].done
        for offset, tr in enumerate(self._buffer[start:]):
            reward += (self._gamma**offset) * tr.reward
            next_state = tr.next_state
            done = tr.done
            if tr.done:
                break
        first = self._buffer[start]
        return Transition(first.state, first.action, reward, next_state, done)
