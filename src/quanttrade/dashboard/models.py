"""Discovery of trained-model artifacts for the dashboard.

Kept separate from the Streamlit view so it can be unit-tested without Streamlit.
Loading an agent requires the optional ``agents`` extra (torch / SB3); if those
libraries are absent or a checkpoint is missing, discovery simply returns fewer
agents instead of crashing the page.
"""

from __future__ import annotations

from pathlib import Path

from quanttrade.agents.base import Agent
from quanttrade.config import AppConfig
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)

# Maps an algo name to its checkpoint filename under the models directory.
_MODEL_FILES: dict[str, str] = {
    "ppo": "ppo.zip",
    "dqn": "dqn.zip",
    "a2c": "a2c.zip",
    "ddqn": "ddqn.pt",
}


def discover_trained_agents(config: AppConfig) -> list[tuple[str, Agent]]:
    """Load every trained agent whose checkpoint exists and whose libs import.

    Returns a list of ``(label, agent)`` pairs; an empty list is a valid result
    (e.g. nothing trained yet, or the ``agents`` extra not installed).
    """
    models_dir = Path(config.train.models_dir)
    found: list[tuple[str, Agent]] = []
    for algo, filename in _MODEL_FILES.items():
        path = models_dir / filename
        if not path.exists():
            continue
        agent = _load_one(algo, path, config)
        if agent is not None:
            found.append((algo, agent))
    return found


def _load_one(algo: str, path: Path, config: AppConfig) -> Agent | None:
    try:
        if algo == "ddqn":
            from quanttrade.agents.ddqn import DoubleDQNAgent

            return DoubleDQNAgent.load(path, config.train)
        from quanttrade.agents.sb3_agent import SB3Agent

        return SB3Agent.load(algo, path)
    except ImportError:
        logger.info("Skipping %s: agents extra not installed", algo)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Could not load %s from %s: %s", algo, path, exc)
    return None
