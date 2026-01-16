#!/usr/bin/env python3
"""
Backtesting Module for Trading Strategies

Features:
- Vectorized backtesting for speed
- Walk-forward analysis
- Monte Carlo simulation
- Transaction costs and slippage modeling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from .metrics import TradingMetrics, MetricsCalculator, calculate_sharpe_ratio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    max_position_size: float = 1.0  # 100% of capital
    allow_short: bool = True
    risk_free_rate: float = 0.0
    annualization_factor: float = 252 * 24 * 60  # 1-min bars


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    trades: List[Dict]
    metrics: TradingMetrics
    config: BacktestConfig

    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate_pct: float = 0.0

    def __post_init__(self):
        if len(self.equity_curve) > 0:
            self.total_return_pct = (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100
            self.sharpe_ratio = self.metrics.sharpe_ratio
            self.max_drawdown_pct = self.metrics.max_drawdown * 100
            self.win_rate_pct = self.metrics.win_rate * 100


class VectorizedBacktester:
    """
    Fast vectorized backtester for trading strategies.

    Example:
        backtester = VectorizedBacktester(config)
        result = backtester.run(prices, signals)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.metrics_calc = MetricsCalculator(
            annualization_factor=self.config.annualization_factor,
            risk_free_rate=self.config.risk_free_rate,
        )

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Args:
            prices: Price series
            signals: Trading signals (-1 to 1)
            timestamps: Optional timestamps for trades

        Returns:
            BacktestResult with equity curve and metrics
        """
        prices = np.asarray(prices)
        signals = np.asarray(signals)
        n = len(prices)

        if len(signals) != n:
            raise ValueError("Prices and signals must have same length")

        # Clip signals to max position size
        signals = np.clip(signals, -self.config.max_position_size, self.config.max_position_size)

        if not self.config.allow_short:
            signals = np.maximum(signals, 0)

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        price_returns = np.insert(price_returns, 0, 0)

        # Position changes for transaction costs
        position_changes = np.diff(signals)
        position_changes = np.insert(position_changes, 0, signals[0])

        # Transaction costs (commission + slippage)
        transaction_costs = np.abs(position_changes) * (
            self.config.commission_pct + self.config.slippage_pct
        )

        # Strategy returns (shifted signals to avoid look-ahead bias)
        shifted_signals = np.roll(signals, 1)
        shifted_signals[0] = 0

        strategy_returns = shifted_signals * price_returns - transaction_costs

        # Equity curve
        equity = self.config.initial_capital * np.cumprod(1 + strategy_returns)

        # Extract trades
        trades = self._extract_trades(signals, prices, timestamps)

        # Calculate metrics
        metrics = self.metrics_calc.calculate_trading_metrics(strategy_returns)

        return BacktestResult(
            equity_curve=equity,
            returns=strategy_returns,
            positions=signals,
            trades=trades,
            metrics=metrics,
            config=self.config,
        )

    def _extract_trades(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Extract individual trades from signals."""
        trades = []
        position = 0
        entry_price = 0
        entry_idx = 0

        for i in range(1, len(signals)):
            new_position = signals[i]

            # Position change
            if new_position != position:
                # Close existing position
                if position != 0:
                    pnl = (prices[i] - entry_price) * position
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": prices[i],
                        "position": position,
                        "pnl": pnl,
                        "return_pct": (prices[i] / entry_price - 1) * 100 * np.sign(position),
                        "entry_time": timestamps[entry_idx] if timestamps is not None else None,
                        "exit_time": timestamps[i] if timestamps is not None else None,
                    })

                # Open new position
                if new_position != 0:
                    entry_price = prices[i]
                    entry_idx = i

                position = new_position

        return trades


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.

    Implements rolling window optimization and out-of-sample testing.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: Optional[int] = None,
        config: Optional[BacktestConfig] = None,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.config = config or BacktestConfig()
        self.backtester = VectorizedBacktester(self.config)

    def run(
        self,
        prices: np.ndarray,
        signal_generator: Callable[[np.ndarray], np.ndarray],
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run walk-forward analysis.

        Args:
            prices: Price series
            signal_generator: Function that takes prices and returns signals
            timestamps: Optional timestamps

        Returns:
            Dictionary with fold results and aggregated metrics
        """
        prices = np.asarray(prices)
        n = len(prices)

        fold_results = []
        all_oos_returns = []

        start = 0
        fold_idx = 0

        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            # Training data
            train_prices = prices[start:train_end]
            train_signals = signal_generator(train_prices)
            train_result = self.backtester.run(train_prices, train_signals)

            # Testing data (out-of-sample)
            test_prices = prices[train_end:test_end]
            test_signals = signal_generator(test_prices)
            test_result = self.backtester.run(test_prices, test_signals)

            all_oos_returns.extend(test_result.returns)

            fold_results.append({
                "fold": fold_idx,
                "train_start": start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "train_sharpe": train_result.metrics.sharpe_ratio,
                "test_sharpe": test_result.metrics.sharpe_ratio,
                "train_return": train_result.total_return_pct,
                "test_return": test_result.total_return_pct,
                "train_max_dd": train_result.max_drawdown_pct,
                "test_max_dd": test_result.max_drawdown_pct,
            })

            logger.info(
                f"Fold {fold_idx}: Train Sharpe={train_result.metrics.sharpe_ratio:.3f}, "
                f"Test Sharpe={test_result.metrics.sharpe_ratio:.3f}"
            )

            start += self.step_size
            fold_idx += 1

        # Aggregate OOS metrics
        oos_returns = np.array(all_oos_returns)
        oos_metrics = self.backtester.metrics_calc.calculate_trading_metrics(oos_returns)

        # Summary statistics
        test_sharpes = [f["test_sharpe"] for f in fold_results]

        return {
            "folds": fold_results,
            "n_folds": len(fold_results),
            "oos_metrics": oos_metrics,
            "mean_test_sharpe": np.mean(test_sharpes),
            "std_test_sharpe": np.std(test_sharpes),
            "min_test_sharpe": np.min(test_sharpes),
            "max_test_sharpe": np.max(test_sharpes),
            "oos_total_return": float(np.sum(oos_returns)) * 100,
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.

    Generates synthetic scenarios through:
    - Bootstrap resampling
    - Return shuffling
    - Noise injection
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        block_size: int = 20,
        config: Optional[BacktestConfig] = None,
    ):
        self.n_simulations = n_simulations
        self.block_size = block_size
        self.config = config or BacktestConfig()
        self.metrics_calc = MetricsCalculator(
            annualization_factor=self.config.annualization_factor,
            risk_free_rate=self.config.risk_free_rate,
        )

    def run_bootstrap(
        self,
        returns: np.ndarray,
        seed: int = 42,
    ) -> Dict:
        """
        Run bootstrap Monte Carlo simulation.

        Args:
            returns: Historical returns
            seed: Random seed

        Returns:
            Dictionary with simulation results
        """
        np.random.seed(seed)
        returns = np.asarray(returns)
        n = len(returns)

        # Block bootstrap for time series
        n_blocks = n // self.block_size

        sim_sharpes = []
        sim_max_dds = []
        sim_returns = []

        for _ in range(self.n_simulations):
            # Sample blocks with replacement
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)

            sim_ret = []
            for bi in block_indices:
                start = bi * self.block_size
                end = min(start + self.block_size, n)
                sim_ret.extend(returns[start:end])

            sim_ret = np.array(sim_ret[:n])  # Truncate to original length

            metrics = self.metrics_calc.calculate_trading_metrics(sim_ret)
            sim_sharpes.append(metrics.sharpe_ratio)
            sim_max_dds.append(metrics.max_drawdown)
            sim_returns.append(np.sum(sim_ret))

        sim_sharpes = np.array(sim_sharpes)
        sim_max_dds = np.array(sim_max_dds)
        sim_returns = np.array(sim_returns)

        # Original metrics
        orig_metrics = self.metrics_calc.calculate_trading_metrics(returns)

        # Confidence intervals
        def ci(arr, alpha=0.05):
            return np.percentile(arr, [alpha/2*100, (1-alpha/2)*100])

        return {
            "original_sharpe": orig_metrics.sharpe_ratio,
            "original_max_dd": orig_metrics.max_drawdown,
            "original_return": float(np.sum(returns)),

            "sharpe_mean": float(np.mean(sim_sharpes)),
            "sharpe_std": float(np.std(sim_sharpes)),
            "sharpe_ci_95": ci(sim_sharpes).tolist(),
            "sharpe_percentile_5": float(np.percentile(sim_sharpes, 5)),

            "max_dd_mean": float(np.mean(sim_max_dds)),
            "max_dd_std": float(np.std(sim_max_dds)),
            "max_dd_ci_95": ci(sim_max_dds).tolist(),
            "max_dd_percentile_95": float(np.percentile(sim_max_dds, 95)),

            "return_mean": float(np.mean(sim_returns)),
            "return_std": float(np.std(sim_returns)),
            "return_ci_95": ci(sim_returns).tolist(),

            "prob_positive_sharpe": float(np.mean(sim_sharpes > 0)),
            "prob_sharpe_above_1": float(np.mean(sim_sharpes > 1)),

            "n_simulations": self.n_simulations,
        }

    def run_noise_injection(
        self,
        returns: np.ndarray,
        noise_levels: List[float] = [0.01, 0.02, 0.05, 0.10],
        seed: int = 42,
    ) -> Dict:
        """
        Test strategy robustness to noise in returns.

        Args:
            returns: Historical returns
            noise_levels: Standard deviations of noise to inject
            seed: Random seed

        Returns:
            Dictionary with noise sensitivity results
        """
        np.random.seed(seed)
        returns = np.asarray(returns)

        orig_metrics = self.metrics_calc.calculate_trading_metrics(returns)

        results = {
            "original_sharpe": orig_metrics.sharpe_ratio,
            "noise_sensitivity": []
        }

        for noise_std in noise_levels:
            noisy_sharpes = []

            for _ in range(100):  # Reduced simulations per noise level
                noise = np.random.normal(0, noise_std, len(returns))
                noisy_returns = returns + noise
                metrics = self.metrics_calc.calculate_trading_metrics(noisy_returns)
                noisy_sharpes.append(metrics.sharpe_ratio)

            noisy_sharpes = np.array(noisy_sharpes)

            results["noise_sensitivity"].append({
                "noise_std": noise_std,
                "mean_sharpe": float(np.mean(noisy_sharpes)),
                "std_sharpe": float(np.std(noisy_sharpes)),
                "sharpe_degradation_pct": float(
                    (orig_metrics.sharpe_ratio - np.mean(noisy_sharpes))
                    / abs(orig_metrics.sharpe_ratio) * 100
                    if orig_metrics.sharpe_ratio != 0 else 0
                ),
            })

        return results


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "VectorizedBacktester",
    "WalkForwardAnalyzer",
    "MonteCarloSimulator",
]
