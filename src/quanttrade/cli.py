"""Command-line interface.

Provides reproducible entry points::

    python -m quanttrade train    --algo ppo
    python -m quanttrade backtest --algo ppo
    python -m quanttrade predict  --algo ppo

Data can come from Yahoo Finance (default) or a local CSV directory via
``--csv-dir`` for fully offline, reproducible runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quanttrade.agents.base import Agent
from quanttrade.config import AppConfig
from quanttrade.data.cache import OhlcvCache
from quanttrade.data.loader import CsvProvider, DataLoader, MarketDataProvider, YFinanceProvider
from quanttrade.evaluation.backtester import Backtester
from quanttrade.training.datasets import DatasetBundle
from quanttrade.training.trainer import Trainer
from quanttrade.utils.exceptions import QuantTradeError
from quanttrade.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)
_SB3_ALGOS = {"ppo", "dqn", "a2c"}


def _build_config(args: argparse.Namespace) -> AppConfig:
    overrides: dict[str, object] = {}
    data: dict[str, object] = {}
    if args.symbol:
        data["symbol"] = args.symbol
    if args.interval:
        data["interval"] = args.interval
    if data:
        overrides["data"] = data
    if args.timesteps:
        overrides["train"] = {"total_timesteps": args.timesteps}
    return AppConfig.from_dict(overrides)


def _make_loader(config: AppConfig, csv_dir: str | None) -> DataLoader:
    provider: MarketDataProvider = (
        CsvProvider(csv_dir) if csv_dir else YFinanceProvider()
    )
    return DataLoader(config.data, provider=provider, cache=OhlcvCache(config.data.cache_dir))


def _load_agent(algo: str, config: AppConfig, state_dim: int) -> Agent:
    models_dir = Path(config.train.models_dir)
    if algo in _SB3_ALGOS:
        from quanttrade.agents.sb3_agent import SB3Agent

        return SB3Agent.load(algo, models_dir / f"{algo}.zip")
    if algo == "ddqn":
        from quanttrade.agents.ddqn import DoubleDQNAgent

        return DoubleDQNAgent.load(models_dir / "ddqn.pt", config.train)
    raise QuantTradeError(f"unknown algo {algo!r}")


def _prepare(
    config: AppConfig, csv_dir: str | None, period: str
) -> tuple[pd.DataFrame, DatasetBundle]:
    loader = _make_loader(config, csv_dir)
    ohlcv = loader.load(period=period)
    bundle = Trainer(config).prepare(ohlcv)
    return ohlcv, bundle


def cmd_train(args: argparse.Namespace) -> None:
    config = _build_config(args)
    _, bundle = _prepare(config, args.csv_dir, args.period)
    path = Trainer(config).train(args.algo, bundle)
    logger.info("Training complete: %s", path)


def cmd_backtest(args: argparse.Namespace) -> None:
    config = _build_config(args)
    _, bundle = _prepare(config, args.csv_dir, args.period)
    state_dim = len(bundle.feature_names) + 2
    agent = _load_agent(args.algo, config, state_dim)
    bt = Backtester(
        bundle.test.features,
        bundle.test.prices,
        config.env,
        config.evaluation,
        bundle.bars_per_year,
    )
    result = bt.run_agent(agent)
    benchmark = bt.run_buy_and_hold()
    for r in (result, benchmark):
        m = r.metrics
        logger.info(
            "%-14s return=%.2f%% sharpe=%.2f sortino=%.2f maxDD=%.2f%% trades=%d",
            r.label,
            m.total_return * 100,
            m.sharpe,
            m.sortino,
            m.max_drawdown * 100,
            m.trade_count,
        )


def cmd_predict(args: argparse.Namespace) -> None:
    config = _build_config(args)
    ohlcv, bundle = _prepare(config, args.csv_dir, args.period)
    state_dim = len(bundle.feature_names) + 2
    agent = _load_agent(args.algo, config, state_dim)
    from quanttrade.inference.predictor import Predictor

    predictor = Predictor.from_artifacts(agent, config.train.models_dir, config)
    prediction = predictor.predict(ohlcv)
    logger.info(
        "Recommendation: %s @ %.2f (%s)",
        prediction.action,
        prediction.price,
        prediction.timestamp,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quanttrade", description="RL trading framework")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--period", default="2y")
    parser.add_argument("--csv-dir", default=None, help="use local CSVs instead of Yahoo Finance")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("train", cmd_train), ("backtest", cmd_backtest), ("predict", cmd_predict)):
        p = sub.add_parser(name)
        p.add_argument("--algo", required=True, choices=["ppo", "dqn", "a2c", "ddqn"])
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
