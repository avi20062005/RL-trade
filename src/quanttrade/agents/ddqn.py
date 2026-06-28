"""Double DQN agent (Dueling + NoisyNet + PER + n-step).

This module requires PyTorch and is imported lazily by callers, so the core
package remains usable without the ``agents`` extra. The implementation follows
the published algorithms:

* Double DQN (van Hasselt et al., 2016): the online net selects the next action,
  the target net evaluates it.
* Dueling architecture (Wang et al., 2016): ``Q = V + (A - mean(A))``.
* Factorized NoisyNet (Fortunato et al., 2018) for exploration in place of
  epsilon-greedy.
* Prioritized replay (Schaul et al., 2016) with IS-weight correction.
* N-step returns with the bootstrap discounted by ``gamma**n``.

Model checkpoints are saved/loaded with ``weights_only=True`` to avoid
arbitrary-code execution during deserialization.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from quanttrade.agents.replay import NStepBuffer, PrioritizedReplayBuffer, Transition
from quanttrade.config import TrainConfig
from quanttrade.env.trading_env import TradingEnv
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer (Fortunato et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self._sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self._sigma0 / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self._sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _scale(size: int) -> Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale(self.in_features)
        eps_out = self._scale(self.out_features)
        cast(Tensor, self.weight_eps).copy_(eps_out.outer(eps_in))
        cast(Tensor, self.bias_eps).copy_(eps_out)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * cast(Tensor, self.weight_eps)
            bias = self.bias_mu + self.bias_sigma * cast(Tensor, self.bias_eps)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, weight, bias)


class DuelingNoisyNet(nn.Module):
    """Dueling Q-network with noisy output streams."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.value = nn.Sequential(nn.ReLU(), NoisyLinear(hidden, 1))
        self.advantage = nn.Sequential(nn.ReLU(), NoisyLinear(hidden, n_actions))

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DoubleDQNAgent:
    """A Double DQN agent implementing the :class:`Agent` protocol."""

    name = "ddqn"

    def __init__(
        self,
        state_dim: int,
        config: TrainConfig,
        *,
        n_actions: int = 3,
        n_step: int = 3,
        tau: float = 0.005,
        buffer_size: int = 50_000,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_step = n_step
        self._tau = tau
        self._gamma_n = config.gamma**n_step
        self._batch_size = config.batch_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = DuelingNoisyNet(state_dim, n_actions).to(self._device)
        self.target = DuelingNoisyNet(state_dim, n_actions).to(self._device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self._optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self._replay = PrioritizedReplayBuffer(buffer_size, beta_frames=config.total_timesteps)
        self._nstep = NStepBuffer(n_step, config.gamma)
        self._rng = np.random.default_rng(config.seed)

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        self.online.eval()
        with torch.no_grad():
            obs = torch.as_tensor(
                observation, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            action = int(self.online(obs).argmax(dim=1).item())
        return action

    def learn(self, env: TradingEnv, total_timesteps: int) -> DoubleDQNAgent:
        obs, _ = env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        for step in range(total_timesteps):
            self.online.train()
            action = self._select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            for tr in self._nstep.push(Transition(obs, action, reward, next_obs, done)):
                self._replay.push(tr)
            self._optimize()
            obs = next_obs
            if done:
                obs, _ = env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
            if (step + 1) % max(1, total_timesteps // 10) == 0:
                logger.info(
                    "DDQN step %d/%d buffer=%d", step + 1, total_timesteps, len(self._replay)
                )
        return self

    def _select_action(self, obs: np.ndarray) -> int:
        self.online.train()  # noise active -> exploration
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def _optimize(self) -> None:
        if len(self._replay) < self._batch_size:
            return
        batch, indices, is_weights = self._replay.sample(self._batch_size, self._rng)
        states = torch.as_tensor(np.stack([t.state for t in batch]), device=self._device)
        actions = torch.as_tensor([t.action for t in batch], device=self._device).unsqueeze(1)
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self._device
        ).unsqueeze(1)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), device=self._device)
        dones = torch.as_tensor(
            [float(t.done) for t in batch], dtype=torch.float32, device=self._device
        ).unsqueeze(1)
        weights = torch.as_tensor(is_weights, device=self._device).unsqueeze(1)

        self.online.train()
        current_q = self.online(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target_q = rewards + self._gamma_n * next_q * (1.0 - dones)

        td_errors = (current_q - target_q).detach().abs().cpu().numpy().flatten()
        self._replay.update_priorities(indices, td_errors)

        per_sample = nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (weights * per_sample).mean()
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self._optimizer.step()
        self._soft_update()
        self.online.reset_noise()
        self.target.reset_noise()

    def _soft_update(self) -> None:
        for online_p, target_p in zip(
            self.online.parameters(), self.target.parameters(), strict=True
        ):
            target_p.data.mul_(1.0 - self._tau).add_(self._tau * online_p.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "n_step": self.n_step,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, config: TrainConfig) -> DoubleDQNAgent:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(
            state_dim=int(checkpoint["state_dim"]),
            config=config,
            n_actions=int(checkpoint["n_actions"]),
            n_step=int(checkpoint["n_step"]),
        )
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        return agent
