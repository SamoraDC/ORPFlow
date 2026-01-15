"""
RL Training Infrastructure for Quantitative Trading
=====================================================

This module provides comprehensive reinforcement learning training infrastructure
with proper historical replay simulation (NO lookahead bias).

Key Features:
- Proper fill simulation: Market orders fill at next bar OPEN, not close
- Limit order simulation using high/low bars
- Walk-forward validation
- Probability of Backtest Overfitting (PBO) calculation
- Comprehensive trading metrics
- Checkpoint management
- ONNX export

Author: ORPFlow Team
"""

import logging
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random
from collections import deque
from datetime import datetime

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

if not TORCH_AVAILABLE:
    raise ImportError(
        "PyTorch is required for RL training. "
        "Install with: pip install torch>=2.0.0"
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ORDER TYPES AND FILL SIMULATION
# =============================================================================

class OrderType(Enum):
    """Order types supported by the simulation."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    order_type: OrderType
    side: OrderSide
    size: float
    price: Optional[float] = None  # For limit/stop orders
    stop_price: Optional[float] = None  # For stop orders
    timestamp: int = 0
    filled: bool = False
    fill_price: float = 0.0
    fill_timestamp: int = 0


@dataclass
class FillSimulatorConfig:
    """Configuration for realistic fill simulation.

    CRITICAL: These parameters prevent lookahead bias.
    """
    # Transaction costs
    commission_rate: float = 0.0005  # 0.05% per trade
    slippage_bps: float = 1.0  # 1 basis point slippage

    # Latency simulation
    latency_bars: int = 1  # Minimum 1 bar delay for market orders

    # Market impact
    market_impact_factor: float = 0.0001  # Price impact per unit volume

    # Partial fills
    enable_partial_fills: bool = False
    max_fill_pct_of_volume: float = 0.1  # Max 10% of bar volume

    # Spread simulation
    spread_bps: float = 2.0  # 2 basis points bid-ask spread


class FillSimulator:
    """
    Realistic order fill simulation with NO lookahead bias.

    CRITICAL RULES:
    1. Market orders fill at NEXT bar's OPEN price (not current close)
    2. Limit orders check if price was touched using HIGH/LOW
    3. Slippage is applied AGAINST the trader
    4. Latency is simulated (minimum 1 bar delay)
    """

    def __init__(self, config: FillSimulatorConfig):
        self.config = config
        self.pending_orders: List[Order] = []
        self._rng = np.random.default_rng()

    def submit_order(self, order: Order, current_bar_idx: int) -> None:
        """Submit an order for execution."""
        order.timestamp = current_bar_idx
        self.pending_orders.append(order)

    def process_bar(
        self,
        bar_idx: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
    ) -> List[Order]:
        """
        Process pending orders against the current bar.

        This is called AFTER the bar is complete, simulating real-time execution.
        Orders submitted on bar N can only execute on bar N+1 or later.

        Args:
            bar_idx: Current bar index
            open_price: Bar open price (used for market order fills)
            high_price: Bar high price (used for limit order checks)
            low_price: Bar low price (used for limit order checks)
            close_price: Bar close price (NOT used for fills - prevents lookahead)
            volume: Bar volume

        Returns:
            List of filled orders
        """
        filled_orders = []
        remaining_orders = []

        for order in self.pending_orders:
            # Check latency requirement
            bars_since_submit = bar_idx - order.timestamp
            if bars_since_submit < self.config.latency_bars:
                remaining_orders.append(order)
                continue

            fill_result = self._try_fill_order(
                order, open_price, high_price, low_price, volume
            )

            if fill_result is not None:
                order.filled = True
                order.fill_price = fill_result
                order.fill_timestamp = bar_idx
                filled_orders.append(order)
            else:
                remaining_orders.append(order)

        self.pending_orders = remaining_orders
        return filled_orders

    def _try_fill_order(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        volume: float,
    ) -> Optional[float]:
        """
        Attempt to fill an order.

        Returns fill price if filled, None otherwise.
        """
        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(order, open_price, volume)

        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit_order(order, open_price, high_price, low_price, volume)

        elif order.order_type == OrderType.STOP:
            return self._fill_stop_order(order, open_price, high_price, low_price, volume)

        elif order.order_type == OrderType.STOP_LIMIT:
            return self._fill_stop_limit_order(order, open_price, high_price, low_price, volume)

        return None

    def _fill_market_order(
        self,
        order: Order,
        open_price: float,
        volume: float,
    ) -> float:
        """
        Fill market order at bar OPEN price with slippage.

        CRITICAL: Uses OPEN price, not close, to avoid lookahead.
        """
        # Base fill price is the open
        fill_price = open_price

        # Apply slippage (against the trader)
        slippage = open_price * (self.config.slippage_bps / 10000)
        if order.side == OrderSide.BUY:
            fill_price += slippage  # Buy at higher price
        else:
            fill_price -= slippage  # Sell at lower price

        # Apply market impact
        impact = order.size * self.config.market_impact_factor * open_price
        if order.side == OrderSide.BUY:
            fill_price += impact
        else:
            fill_price -= impact

        # Add random noise
        noise = self._rng.normal(0, slippage * 0.5)
        fill_price += noise

        return fill_price

    def _fill_limit_order(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        volume: float,
    ) -> Optional[float]:
        """
        Fill limit order if price was touched.

        Uses HIGH/LOW to check if limit was reached.
        """
        if order.price is None:
            return None

        limit_price = order.price
        half_spread = open_price * (self.config.spread_bps / 20000)

        if order.side == OrderSide.BUY:
            # Buy limit: Fill if low went below our limit
            effective_low = low_price + half_spread  # Adjust for spread
            if effective_low <= limit_price:
                # Fill at limit price or worse
                fill_price = min(limit_price, open_price)
                return fill_price + half_spread
        else:
            # Sell limit: Fill if high went above our limit
            effective_high = high_price - half_spread
            if effective_high >= limit_price:
                fill_price = max(limit_price, open_price)
                return fill_price - half_spread

        return None

    def _fill_stop_order(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        volume: float,
    ) -> Optional[float]:
        """Fill stop order when stop price is triggered."""
        if order.stop_price is None:
            return None

        stop_price = order.stop_price

        if order.side == OrderSide.BUY:
            # Buy stop: Triggered when price rises to stop level
            if high_price >= stop_price:
                # Fill at open or stop price, whichever is higher
                fill_price = max(open_price, stop_price)
                return self._apply_slippage(fill_price, order.side, open_price)
        else:
            # Sell stop: Triggered when price falls to stop level
            if low_price <= stop_price:
                fill_price = min(open_price, stop_price)
                return self._apply_slippage(fill_price, order.side, open_price)

        return None

    def _fill_stop_limit_order(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        volume: float,
    ) -> Optional[float]:
        """Fill stop-limit order: stop triggers, then limit applies."""
        if order.stop_price is None or order.price is None:
            return None

        # First check if stop is triggered
        stop_triggered = False
        if order.side == OrderSide.BUY:
            stop_triggered = high_price >= order.stop_price
        else:
            stop_triggered = low_price <= order.stop_price

        if not stop_triggered:
            return None

        # Then check limit price
        return self._fill_limit_order(order, open_price, high_price, low_price, volume)

    def _apply_slippage(
        self,
        price: float,
        side: OrderSide,
        reference_price: float,
    ) -> float:
        """Apply slippage against the trader."""
        slippage = reference_price * (self.config.slippage_bps / 10000)
        if side == OrderSide.BUY:
            return price + slippage
        return price - slippage

    def calculate_transaction_cost(
        self,
        fill_price: float,
        size: float,
    ) -> float:
        """Calculate transaction cost for a fill."""
        notional = abs(fill_price * size)
        return notional * self.config.commission_rate

    def clear_pending_orders(self) -> None:
        """Clear all pending orders."""
        self.pending_orders = []


# =============================================================================
# REPLAY ENVIRONMENT
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Configuration for the replay environment."""
    initial_balance: float = 100_000.0
    max_position: float = 1.0
    transaction_cost: float = 0.0005
    slippage_bps: float = 1.0
    latency_bars: int = 1

    # Episode configuration
    episode_length: Optional[int] = None  # None = use full data
    random_start: bool = True  # Random start point in data
    min_start_idx: int = 100  # Minimum start index (for warmup features)

    # Walk-forward settings
    train_pct: float = 0.7
    validation_pct: float = 0.15
    test_pct: float = 0.15


class ReplayEnvironment:
    """
    Historical replay environment for RL training.

    CRITICAL: This environment ensures NO lookahead bias:
    1. Actions taken at time T affect fills at time T+1
    2. Fill prices use NEXT bar's OPEN, not current CLOSE
    3. State only includes information available at decision time
    """

    def __init__(
        self,
        data: np.ndarray,
        features: np.ndarray,
        config: Optional[EnvironmentConfig] = None,
    ):
        """
        Initialize replay environment.

        Args:
            data: OHLCV data array (N, 5) - [open, high, low, close, volume]
            features: Feature array (N, F) - preprocessed features
            config: Environment configuration
        """
        self.data = data
        self.features = features
        self.config = config or EnvironmentConfig()

        # Validate data
        assert data.shape[0] == features.shape[0], "Data and features must have same length"
        assert data.shape[1] >= 5, "Data must have at least 5 columns (OHLCV)"

        self.n_samples = len(data)
        self.state_dim = features.shape[1] + 4  # features + portfolio state
        self.action_dim = 1  # Target position

        # Initialize fill simulator
        fill_config = FillSimulatorConfig(
            commission_rate=self.config.transaction_cost,
            slippage_bps=self.config.slippage_bps,
            latency_bars=self.config.latency_bars,
        )
        self.fill_simulator = FillSimulator(fill_config)

        # Episode state
        self._reset_state()

        logger.info(
            f"ReplayEnvironment initialized: {self.n_samples} bars, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}"
        )

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.step_idx = 0
        self.start_idx = 0
        self.end_idx = self.n_samples - 1
        self.portfolio_values: List[float] = []
        self.returns: List[float] = []
        self.trades: List[Dict] = []
        self.pending_action: Optional[float] = None

    def reset(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reset environment for a new episode.

        Args:
            start_idx: Optional start index (for walk-forward validation)
            end_idx: Optional end index

        Returns:
            Initial state observation
        """
        self._reset_state()
        self.fill_simulator.clear_pending_orders()

        # Determine episode boundaries
        if start_idx is not None:
            self.start_idx = max(start_idx, self.config.min_start_idx)
        elif self.config.random_start:
            max_start = self.n_samples - (self.config.episode_length or 1000)
            max_start = max(max_start, self.config.min_start_idx + 1)
            self.start_idx = np.random.randint(self.config.min_start_idx, max_start)
        else:
            self.start_idx = self.config.min_start_idx

        if end_idx is not None:
            self.end_idx = min(end_idx, self.n_samples - 1)
        elif self.config.episode_length is not None:
            self.end_idx = min(
                self.start_idx + self.config.episode_length,
                self.n_samples - 1
            )
        else:
            self.end_idx = self.n_samples - 1

        self.step_idx = self.start_idx
        self.portfolio_values = [self.config.initial_balance]

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Get current state observation.

        State includes:
        - Market features (from preprocessed data)
        - Portfolio state (position, balance, recent performance)

        CRITICAL: Only uses information available at current time.
        """
        market_state = self.features[self.step_idx]

        # Recent returns statistics
        if len(self.returns) >= 20:
            recent_returns = np.array(self.returns[-20:])
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
        elif len(self.returns) > 0:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns) if len(self.returns) > 1 else 0.0
        else:
            mean_return = 0.0
            std_return = 0.0

        portfolio_state = np.array([
            self.position / self.config.max_position,  # Normalized position
            self.balance / self.config.initial_balance - 1,  # Return vs initial
            mean_return,  # Recent average return
            std_return,  # Recent volatility
        ], dtype=np.float32)

        state = np.concatenate([market_state, portfolio_state])
        return state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.

        CRITICAL TIMING:
        1. Action is received at time T (based on state at T)
        2. Order is submitted at end of bar T
        3. Order fills at bar T+1's OPEN price
        4. Next state is returned (at T+1)

        Args:
            action: Target position [-1, 1]

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Clip action to valid range
        target_position = float(np.clip(action[0] if len(action.shape) > 0 else action,
                                        -self.config.max_position,
                                        self.config.max_position))

        # Store pending action (will execute next bar)
        position_change = target_position - self.position

        # Get current bar data
        current_bar = self.data[self.step_idx]
        current_open = current_bar[0]
        current_close = current_bar[3]

        # Submit order for next bar execution
        if abs(position_change) > 1e-6:
            order = Order(
                order_type=OrderType.MARKET,
                side=OrderSide.BUY if position_change > 0 else OrderSide.SELL,
                size=abs(position_change),
            )
            self.fill_simulator.submit_order(order, self.step_idx)

        # Move to next bar
        self.step_idx += 1
        done = self.step_idx >= self.end_idx

        if done:
            # Episode complete - calculate final metrics
            next_state = self._get_state()
            reward = 0.0
            info = self._get_info()
            return next_state, reward, done, info

        # Process fills at next bar's OPEN
        next_bar = self.data[self.step_idx]
        next_open = next_bar[0]
        next_high = next_bar[1]
        next_low = next_bar[2]
        next_close = next_bar[3]
        next_volume = next_bar[4]

        filled_orders = self.fill_simulator.process_bar(
            self.step_idx, next_open, next_high, next_low, next_close, next_volume
        )

        # Process fills and update portfolio
        transaction_costs = 0.0
        for order in filled_orders:
            # Calculate transaction cost
            cost = self.fill_simulator.calculate_transaction_cost(
                order.fill_price, order.size
            )
            transaction_costs += cost

            # Update position
            if order.side == OrderSide.BUY:
                self.position += order.size
            else:
                self.position -= order.size

            # Record trade
            self.trades.append({
                'bar_idx': order.fill_timestamp,
                'side': order.side.value,
                'size': order.size,
                'fill_price': order.fill_price,
                'cost': cost,
            })

        # Calculate PnL based on position and price change
        # CRITICAL: Use price change from FILL price to current close
        # This represents the realistic PnL after execution
        price_return = (next_close - next_open) / next_open
        position_pnl = self.position * price_return * self.balance
        total_pnl = position_pnl - transaction_costs

        self.balance += total_pnl
        step_return = total_pnl / self.portfolio_values[-1]
        self.returns.append(step_return)
        self.portfolio_values.append(self.balance)

        # Calculate reward
        reward = self._calculate_reward(step_return, transaction_costs)

        # Get next state
        next_state = self._get_state()
        info = self._get_info()

        return next_state, reward, done, info

    def _calculate_reward(
        self,
        step_return: float,
        transaction_costs: float,
    ) -> float:
        """
        Calculate reward for the step.

        Uses risk-adjusted return (similar to Sharpe ratio).
        """
        if len(self.returns) < 2:
            return step_return * 100

        # Risk-adjusted return
        recent_returns = np.array(self.returns[-20:])
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-8
        sharpe_component = mean_return / std_return

        # Drawdown penalty
        peak = max(self.portfolio_values)
        drawdown = (peak - self.balance) / peak
        drawdown_penalty = -drawdown * 0.5

        # Transaction cost penalty
        cost_penalty = -transaction_costs / self.config.initial_balance * 100

        reward = sharpe_component + drawdown_penalty + cost_penalty + step_return * 10

        return float(reward)

    def _get_info(self) -> Dict:
        """Get info dictionary with current state."""
        return {
            'balance': self.balance,
            'position': self.position,
            'step_idx': self.step_idx,
            'return': self.returns[-1] if self.returns else 0.0,
            'total_return': (self.balance - self.config.initial_balance) / self.config.initial_balance,
            'n_trades': len(self.trades),
        }

    def get_train_val_test_indices(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get indices for train/validation/test split.

        Used for walk-forward validation.
        """
        n = self.n_samples - self.config.min_start_idx
        train_end = int(n * self.config.train_pct) + self.config.min_start_idx
        val_end = int(n * (self.config.train_pct + self.config.validation_pct)) + self.config.min_start_idx

        train_indices = (self.config.min_start_idx, train_end)
        val_indices = (train_end, val_end)
        test_indices = (val_end, self.n_samples - 1)

        return train_indices, val_indices, test_indices


# =============================================================================
# VALIDATION METRICS
# =============================================================================

@dataclass
class TradingMetrics:
    """Comprehensive trading performance metrics."""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # In bars

    # Trading metrics
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0

    # Consistency metrics
    positive_periods_pct: float = 0.0
    best_period_return: float = 0.0
    worst_period_return: float = 0.0

    # Robustness metrics (across seeds/windows)
    return_std_across_seeds: float = 0.0
    sharpe_std_across_seeds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class RLValidationMetrics:
    """
    Comprehensive validation metrics calculator for RL trading agents.

    Features:
    - Standard trading metrics (Sharpe, Sortino, MaxDD, etc.)
    - Robustness metrics across seeds and time windows
    - Reward hacking detection
    - Comparison vs baseline (buy & hold)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,  # Annualized
        periods_per_year: int = 252 * 24,  # For crypto (hourly data)
    ):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._rng = np.random.default_rng()

    def calculate_metrics(
        self,
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        trades: List[Dict],
        prices: Optional[np.ndarray] = None,
    ) -> TradingMetrics:
        """
        Calculate comprehensive trading metrics.

        Args:
            returns: Array of period returns
            portfolio_values: Array of portfolio values over time
            trades: List of trade dictionaries
            prices: Optional price array for buy&hold comparison

        Returns:
            TradingMetrics dataclass
        """
        metrics = TradingMetrics()

        if len(returns) == 0 or len(portfolio_values) < 2:
            return metrics

        returns = np.array(returns)
        portfolio_values = np.array(portfolio_values)

        # Total return
        metrics.total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Annualized return
        n_periods = len(returns)
        years = n_periods / self.periods_per_year
        if years > 0 and (1 + metrics.total_return) > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1

        # Risk-adjusted metrics
        metrics.sharpe_ratio = self._calculate_sharpe(returns)
        metrics.sortino_ratio = self._calculate_sortino(returns)

        # Drawdown metrics
        dd_info = self._calculate_drawdown(portfolio_values)
        metrics.max_drawdown = dd_info['max_drawdown']
        metrics.max_drawdown_duration = dd_info['max_duration']

        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

        # Trading metrics
        if trades:
            trade_returns = self._calculate_trade_returns(trades)
            metrics.n_trades = len(trades)
            metrics.win_rate = np.mean(np.array(trade_returns) > 0) if trade_returns else 0.0

            gains = sum(r for r in trade_returns if r > 0)
            losses = abs(sum(r for r in trade_returns if r < 0))
            metrics.profit_factor = gains / losses if losses > 0 else float('inf')
            metrics.avg_trade_return = np.mean(trade_returns) if trade_returns else 0.0

        # Consistency metrics
        if len(returns) > 0:
            metrics.positive_periods_pct = np.mean(returns > 0)
            metrics.best_period_return = np.max(returns)
            metrics.worst_period_return = np.min(returns)

        return metrics

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-8:
            return 0.0

        # Annualize
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess_return = mean_return - rf_per_period
        annualized_sharpe = (excess_return / std_return) * np.sqrt(self.periods_per_year)

        return float(annualized_sharpe)

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)

        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 2:
            return 0.0 if mean_return <= 0 else float('inf')

        downside_std = np.std(downside_returns)
        if downside_std < 1e-8:
            return float('inf') if mean_return > 0 else 0.0

        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess_return = mean_return - rf_per_period
        annualized_sortino = (excess_return / downside_std) * np.sqrt(self.periods_per_year)

        return float(annualized_sortino)

    def _calculate_drawdown(
        self,
        portfolio_values: np.ndarray,
    ) -> Dict:
        """Calculate maximum drawdown and duration."""
        if len(portfolio_values) < 2:
            return {'max_drawdown': 0.0, 'max_duration': 0}

        # Running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max

        max_drawdown = np.max(drawdowns)

        # Calculate max duration
        in_drawdown = drawdowns > 0
        max_duration = 0
        current_duration = 0

        for dd in in_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return {
            'max_drawdown': float(max_drawdown),
            'max_duration': max_duration,
        }

    def _calculate_trade_returns(self, trades: List[Dict]) -> List[float]:
        """Calculate returns for each trade."""
        trade_returns = []

        # Simplified: use trade costs and sizes
        for trade in trades:
            if 'fill_price' in trade and 'cost' in trade:
                # Estimate return from cost ratio
                cost_ratio = trade.get('cost', 0) / (trade['fill_price'] * trade.get('size', 1) + 1e-8)
                trade_returns.append(-cost_ratio)  # Cost is negative return

        return trade_returns

    def calculate_robustness_metrics(
        self,
        results_across_seeds: List[TradingMetrics],
    ) -> Dict:
        """
        Calculate robustness metrics across multiple seeds/runs.

        Args:
            results_across_seeds: List of TradingMetrics from different seeds

        Returns:
            Dictionary with robustness metrics
        """
        if not results_across_seeds:
            return {}

        returns = [r.total_return for r in results_across_seeds]
        sharpes = [r.sharpe_ratio for r in results_across_seeds]

        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'min_sharpe': np.min(sharpes),
            'max_sharpe': np.max(sharpes),
            'return_consistency': 1 - np.std(returns) / (np.abs(np.mean(returns)) + 1e-8),
        }

    def detect_reward_hacking(
        self,
        training_returns: List[float],
        validation_returns: List[float],
        threshold: float = 0.5,
    ) -> Dict:
        """
        Detect potential reward hacking/gaming.

        Signs of reward hacking:
        1. Large gap between training and validation performance
        2. Unrealistic consistency in training
        3. Sudden drops in validation

        Args:
            training_returns: Returns during training
            validation_returns: Returns during validation
            threshold: Performance gap threshold

        Returns:
            Dictionary with hacking detection results
        """
        if not training_returns or not validation_returns:
            return {'hacking_detected': False, 'confidence': 0.0}

        train_mean = np.mean(training_returns)
        val_mean = np.mean(validation_returns)

        # Performance gap
        if abs(train_mean) > 1e-8:
            performance_gap = (train_mean - val_mean) / abs(train_mean)
        else:
            performance_gap = 0.0

        # Consistency check (too good to be true)
        train_std = np.std(training_returns)
        train_consistency = train_std / (abs(train_mean) + 1e-8)
        unrealistic_consistency = train_consistency < 0.1 and train_mean > 0.01

        # Sudden drops in validation
        val_array = np.array(validation_returns)
        if len(val_array) > 10:
            val_changes = np.diff(val_array)
            sudden_drops = np.sum(val_changes < -0.05 * abs(val_mean)) > len(val_array) * 0.1
        else:
            sudden_drops = False

        # Aggregate detection
        hacking_score = 0.0
        if performance_gap > threshold:
            hacking_score += 0.4
        if unrealistic_consistency:
            hacking_score += 0.3
        if sudden_drops:
            hacking_score += 0.3

        return {
            'hacking_detected': hacking_score > 0.5,
            'confidence': hacking_score,
            'performance_gap': performance_gap,
            'unrealistic_consistency': unrealistic_consistency,
            'sudden_drops': sudden_drops,
        }

    def compare_vs_baseline(
        self,
        agent_returns: np.ndarray,
        prices: np.ndarray,
        baseline: str = 'buy_and_hold',
    ) -> Dict:
        """
        Compare agent performance vs baseline strategy.

        Args:
            agent_returns: Agent's returns
            prices: Price array for baseline calculation
            baseline: Baseline strategy ('buy_and_hold', 'random')

        Returns:
            Dictionary with comparison metrics
        """
        if len(prices) < 2:
            return {}

        if baseline == 'buy_and_hold':
            # Buy and hold returns
            price_returns = np.diff(prices) / prices[:-1]
            baseline_cumret = np.cumprod(1 + price_returns)[-1] - 1
        elif baseline == 'random':
            # Random trading baseline (100 simulations)
            baseline_rets = []
            for _ in range(100):
                random_positions = self._rng.uniform(-1, 1, len(prices) - 1)
                price_returns = np.diff(prices) / prices[:-1]
                random_ret = np.sum(random_positions * price_returns)
                baseline_rets.append(random_ret)
            baseline_cumret = np.mean(baseline_rets)
        else:
            baseline_cumret = 0.0

        agent_cumret = np.cumprod(1 + agent_returns)[-1] - 1 if len(agent_returns) > 0 else 0.0

        return {
            'agent_return': float(agent_cumret),
            'baseline_return': float(baseline_cumret),
            'excess_return': float(agent_cumret - baseline_cumret),
            'outperformed': agent_cumret > baseline_cumret,
        }

    def calculate_pbo(
        self,
        fold_results: List[TradingMetrics],
        n_combinations: int = 1000,
    ) -> float:
        """
        Calculate Probability of Backtest Overfitting (PBO).

        Uses combinatorial symmetric cross-validation to estimate
        the probability that an optimal in-sample strategy
        underperforms out-of-sample.

        Args:
            fold_results: Results from different folds/windows
            n_combinations: Number of combinations to sample

        Returns:
            PBO probability (0 = no overfitting, 1 = severe overfitting)
        """
        if len(fold_results) < 4:
            logger.warning("PBO requires at least 4 folds for meaningful estimate")
            return 0.0

        n_folds = len(fold_results)
        returns = np.array([r.total_return for r in fold_results])

        # Generate combinations for CSCV
        underperform_count = 0

        for _ in range(n_combinations):
            # Randomly split folds into in-sample and out-of-sample
            indices = self._rng.permutation(n_folds)
            half = n_folds // 2

            is_indices = indices[:half]
            oos_indices = indices[half:]

            # Find best in-sample strategy
            is_returns = returns[is_indices]
            best_is_idx = np.argmax(is_returns)

            # Check if it underperforms out-of-sample
            oos_returns = returns[oos_indices]
            best_oos_return = oos_returns[best_is_idx] if best_is_idx < len(oos_returns) else oos_returns[0]
            median_oos_return = np.median(oos_returns)

            if best_oos_return < median_oos_return:
                underperform_count += 1

        pbo = underperform_count / n_combinations
        return float(pbo)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for RL trainers."""
    # Training parameters
    n_episodes: int = 500
    max_steps_per_episode: int = 5000

    # Walk-forward parameters
    n_folds: int = 5
    train_ratio: float = 0.7

    # Validation
    validate_every_n_episodes: int = 50
    n_validation_episodes: int = 5

    # Early stopping
    early_stopping_patience: int = 50
    min_improvement: float = 0.001

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_episodes: int = 100
    keep_n_checkpoints: int = 5

    # Logging
    log_every_n_episodes: int = 10

    # Seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])


# =============================================================================
# D4PG TRAINER
# =============================================================================

class D4PGTrainer:
    """
    Trainer for D4PG+EVT agents.

    Features:
    - Episode-based training with experience collection
    - Walk-forward validation
    - PBO calculation
    - Checkpoint management
    - ONNX export
    """

    def __init__(
        self,
        env: ReplayEnvironment,
        agent: Any,  # D4PGAgent
        config: Optional[TrainerConfig] = None,
        metrics_calculator: Optional[RLValidationMetrics] = None,
    ):
        self.env = env
        self.agent = agent
        self.config = config or TrainerConfig()
        self.metrics_calculator = metrics_calculator or RLValidationMetrics()

        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_validation_return = -float('inf')
        self.patience_counter = 0

        # History
        self.training_history: List[Dict] = []
        self.validation_history: List[TradingMetrics] = []
        self.fold_results: List[TradingMetrics] = []

        # Checkpoint management
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints: List[Path] = []

        logger.info(f"D4PGTrainer initialized with config: {self.config}")

    def train_episode(self) -> Dict:
        """
        Train for a single episode with proper experience collection.

        Returns:
            Dictionary with episode statistics
        """
        state = self.env.reset()
        episode_return = 0.0
        episode_steps = 0
        episode_rewards = []

        train_infos = []

        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(state, evaluate=False)

            # Execute in environment
            next_state, reward, done, info = self.env.step(action)

            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            train_info = self.agent.train()
            if train_info:
                train_infos.append(train_info)

            state = next_state
            episode_return += reward
            episode_rewards.append(reward)
            episode_steps += 1
            self.total_steps += 1

            if done:
                break

        # Calculate episode metrics
        total_return = info.get('total_return', 0.0)
        n_trades = info.get('n_trades', 0)

        # Aggregate training info
        avg_actor_loss = np.mean([t.get('actor_loss', 0) for t in train_infos]) if train_infos else 0
        avg_critic_loss = np.mean([t.get('critic_loss', 0) for t in train_infos]) if train_infos else 0

        episode_stats = {
            'episode': self.episode_count,
            'steps': episode_steps,
            'total_steps': self.total_steps,
            'episode_return': episode_return,
            'total_return': total_return,
            'n_trades': n_trades,
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'avg_actor_loss': avg_actor_loss,
            'avg_critic_loss': avg_critic_loss,
            'var': self.agent.evt_model.var() if hasattr(self.agent, 'evt_model') else 0,
            'cvar': self.agent.evt_model.cvar() if hasattr(self.agent, 'evt_model') else 0,
        }

        self.training_history.append(episode_stats)
        self.episode_count += 1

        return episode_stats

    def train(
        self,
        n_episodes: Optional[int] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Dict:
        """
        Full training loop.

        Args:
            n_episodes: Number of episodes (default from config)
            start_idx: Optional start index for walk-forward
            end_idx: Optional end index for walk-forward

        Returns:
            Dictionary with training results
        """
        n_episodes = n_episodes or self.config.n_episodes

        # Set environment bounds if specified
        if start_idx is not None:
            self.env.config.random_start = True
            self.env.config.min_start_idx = start_idx

        logger.info(f"Starting training for {n_episodes} episodes")
        start_time = time.time()

        returns_window = deque(maxlen=100)

        for episode in range(n_episodes):
            # Train episode
            stats = self.train_episode()
            returns_window.append(stats['total_return'])

            # Logging
            if (episode + 1) % self.config.log_every_n_episodes == 0:
                avg_return = np.mean(returns_window)
                logger.info(
                    f"Episode {episode + 1}/{n_episodes} - "
                    f"Return: {stats['total_return']:.2%}, "
                    f"Avg100: {avg_return:.2%}, "
                    f"Steps: {stats['steps']}, "
                    f"VaR: {stats['var']:.4f}"
                )

            # Validation
            if (episode + 1) % self.config.validate_every_n_episodes == 0:
                val_metrics = self.validate()
                self.validation_history.append(val_metrics)

                logger.info(
                    f"Validation - Return: {val_metrics.total_return:.2%}, "
                    f"Sharpe: {val_metrics.sharpe_ratio:.2f}"
                )

                # Early stopping check
                if val_metrics.total_return > self.best_validation_return + self.config.min_improvement:
                    self.best_validation_return = val_metrics.total_return
                    self.patience_counter = 0
                    self._save_checkpoint('best')
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at episode {episode + 1}")
                    break

            # Periodic checkpointing
            if (episode + 1) % self.config.save_every_n_episodes == 0:
                self._save_checkpoint(f'episode_{episode + 1}')

        training_time = time.time() - start_time

        # Final validation
        final_metrics = self.validate()

        results = {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'training_time_seconds': training_time,
            'best_validation_return': self.best_validation_return,
            'final_metrics': final_metrics.to_dict(),
            'early_stopped': self.patience_counter >= self.config.early_stopping_patience,
        }

        logger.info(f"Training complete: {results}")
        return results

    def validate(
        self,
        n_episodes: Optional[int] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> TradingMetrics:
        """
        Run validation episodes.

        Args:
            n_episodes: Number of validation episodes
            start_idx: Optional start index
            end_idx: Optional end index

        Returns:
            TradingMetrics for validation
        """
        n_episodes = n_episodes or self.config.n_validation_episodes

        all_returns = []
        all_portfolio_values = []
        all_trades = []

        for _ in range(n_episodes):
            state = self.env.reset(start_idx=start_idx, end_idx=end_idx)

            while True:
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, info = self.env.step(action)
                state = next_state

                if done:
                    break

            all_returns.extend(self.env.returns)
            all_portfolio_values.append(self.env.portfolio_values[-1])
            all_trades.extend(self.env.trades)

        # Calculate metrics
        returns_array = np.array(all_returns)
        portfolio_values = np.array([self.env.config.initial_balance] + all_portfolio_values)

        metrics = self.metrics_calculator.calculate_metrics(
            returns_array,
            portfolio_values,
            all_trades,
        )

        return metrics

    def walk_forward_train(self) -> Dict:
        """
        Walk-forward training and validation.

        Trains on multiple time windows and validates on subsequent windows.

        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Starting walk-forward training with {self.config.n_folds} folds")

        n_samples = self.env.n_samples - self.env.config.min_start_idx
        fold_size = n_samples // self.config.n_folds

        fold_results = []

        for fold in range(self.config.n_folds - 1):
            train_start = self.env.config.min_start_idx + fold * fold_size
            train_end = train_start + int(fold_size * self.config.train_ratio)
            val_start = train_end
            val_end = train_start + fold_size

            logger.info(f"Fold {fold + 1}/{self.config.n_folds - 1}: "
                       f"Train [{train_start}, {train_end}], Val [{val_start}, {val_end}]")

            # Reset agent for each fold (optional - can also use warm start)
            # self.agent.reset()

            # Train on this fold
            self.train(
                n_episodes=self.config.n_episodes // self.config.n_folds,
                start_idx=train_start,
                end_idx=train_end,
            )

            # Validate on next window
            val_metrics = self.validate(start_idx=val_start, end_idx=val_end)
            fold_results.append(val_metrics)

            logger.info(f"Fold {fold + 1} validation - Return: {val_metrics.total_return:.2%}, "
                       f"Sharpe: {val_metrics.sharpe_ratio:.2f}")

        self.fold_results = fold_results

        # Calculate PBO
        pbo = self.calculate_pbo(fold_results)

        # Aggregate results
        robustness = self.metrics_calculator.calculate_robustness_metrics(fold_results)

        results = {
            'n_folds': len(fold_results),
            'fold_returns': [m.total_return for m in fold_results],
            'fold_sharpes': [m.sharpe_ratio for m in fold_results],
            'pbo': pbo,
            'robustness': robustness,
            'mean_return': np.mean([m.total_return for m in fold_results]),
            'std_return': np.std([m.total_return for m in fold_results]),
        }

        logger.info(f"Walk-forward complete: PBO={pbo:.2%}, "
                   f"Mean Return={results['mean_return']:.2%}")

        return results

    def calculate_pbo(self, fold_results: List[TradingMetrics]) -> float:
        """Calculate Probability of Backtest Overfitting."""
        return self.metrics_calculator.calculate_pbo(fold_results)

    def export_onnx(self, path: str) -> None:
        """Export trained actor network to ONNX format."""
        logger.info(f"Exporting model to ONNX: {path}")
        self.agent.export_onnx(path)

    def _save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"d4pg_{name}.pt"

        checkpoint = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_validation_return': self.best_validation_return,
            'training_history': self.training_history[-100:],  # Last 100 episodes
        }

        # Save agent state
        self.agent.save(str(checkpoint_path.with_suffix('.agent.pt')))

        # Save trainer state
        torch.save(checkpoint, checkpoint_path)

        self.saved_checkpoints.append(checkpoint_path)

        # Cleanup old checkpoints
        while len(self.saved_checkpoints) > self.config.keep_n_checkpoints:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists() and 'best' not in str(old_checkpoint):
                old_checkpoint.unlink()
                agent_checkpoint = old_checkpoint.with_suffix('.agent.pt')
                if agent_checkpoint.exists():
                    agent_checkpoint.unlink()

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path)

        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        self.best_validation_return = checkpoint['best_validation_return']
        self.training_history = checkpoint['training_history']

        # Load agent state
        agent_path = checkpoint_path.with_suffix('.agent.pt')
        if agent_path.exists():
            self.agent.load(str(agent_path))

        logger.info(f"Checkpoint loaded: {path}")


# =============================================================================
# MARL TRAINER
# =============================================================================

class MARLTrainer:
    """
    Trainer for Multi-Agent RL systems.

    Similar to D4PGTrainer but handles multiple agents.
    """

    def __init__(
        self,
        env: ReplayEnvironment,
        system: Any,  # MARLSystem
        config: Optional[TrainerConfig] = None,
        metrics_calculator: Optional[RLValidationMetrics] = None,
    ):
        self.env = env
        self.system = system
        self.config = config or TrainerConfig()
        self.metrics_calculator = metrics_calculator or RLValidationMetrics()

        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_validation_return = -float('inf')
        self.patience_counter = 0

        # History
        self.training_history: List[Dict] = []
        self.validation_history: List[TradingMetrics] = []
        self.fold_results: List[TradingMetrics] = []

        # Checkpoint management
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints: List[Path] = []

        logger.info(f"MARLTrainer initialized for {self.system.n_agents} agents")

    def train_episode(self) -> Dict:
        """Train for a single multi-agent episode."""
        # Get initial states for all agents
        # Note: For MARL, we need to broadcast the environment state to all agents
        base_state = self.env.reset()

        # Create states for each agent (with slight variations for exploration)
        states = self._create_agent_states(base_state)

        episode_return = 0.0
        episode_steps = 0
        train_infos = []

        for step in range(self.config.max_steps_per_episode):
            # Select actions for all agents
            actions, messages = self.system.select_actions(states, evaluate=False)

            # Aggregate actions
            aggregated_action = self.system.get_aggregated_action(actions)

            # Execute in environment
            next_base_state, base_reward, done, info = self.env.step(aggregated_action)

            # Create next states for agents
            next_states = self._create_agent_states(next_base_state)

            # Shape rewards per agent role
            market_state = self._get_market_state()
            rewards = self.system.shape_rewards(base_reward, market_state)

            # Store transitions for all agents
            dones = np.full(self.system.n_agents, done)
            self.system.store_transition(states, actions, rewards, next_states, dones, messages)

            # Train system
            train_info = self.system.train()
            if train_info:
                train_infos.append(train_info)

            states = next_states
            episode_return += base_reward
            episode_steps += 1
            self.total_steps += 1

            if done:
                break

        # Episode stats
        avg_critic_loss = np.mean([t.get('critic_loss', 0) for t in train_infos]) if train_infos else 0
        avg_actor_loss = np.mean([t.get('mean_actor_loss', 0) for t in train_infos]) if train_infos else 0

        episode_stats = {
            'episode': self.episode_count,
            'steps': episode_steps,
            'total_steps': self.total_steps,
            'episode_return': episode_return,
            'total_return': info.get('total_return', 0.0),
            'n_trades': info.get('n_trades', 0),
            'avg_critic_loss': avg_critic_loss,
            'avg_actor_loss': avg_actor_loss,
        }

        self.training_history.append(episode_stats)
        self.episode_count += 1

        return episode_stats

    def _create_agent_states(self, base_state: np.ndarray) -> np.ndarray:
        """Create per-agent states with slight variations."""
        states = []
        for i in range(self.system.n_agents):
            # Add agent-specific noise for exploration
            noise = np.random.normal(0, 0.01, size=base_state.shape)
            agent_state = base_state + noise
            states.append(agent_state)
        return np.array(states, dtype=np.float32)

    def _get_market_state(self) -> Dict:
        """Get market state for reward shaping."""
        returns = self.env.returns

        if len(returns) < 20:
            return {
                'trend_strength': 0,
                'momentum': 0,
                'deviation': 0,
                'volatility': 0,
            }

        recent_returns = np.array(returns[-20:])

        return {
            'trend_strength': np.mean(recent_returns) / (np.std(recent_returns) + 1e-8),
            'momentum': recent_returns[-1] - np.mean(recent_returns),
            'deviation': 0,  # Would need price data
            'volatility': np.std(recent_returns) * np.sqrt(252 * 24),
        }

    def train(
        self,
        n_episodes: Optional[int] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Dict:
        """Full training loop for MARL."""
        n_episodes = n_episodes or self.config.n_episodes

        logger.info(f"Starting MARL training for {n_episodes} episodes")
        start_time = time.time()

        returns_window = deque(maxlen=100)

        for episode in range(n_episodes):
            stats = self.train_episode()
            returns_window.append(stats['total_return'])

            # Logging
            if (episode + 1) % self.config.log_every_n_episodes == 0:
                avg_return = np.mean(returns_window)
                logger.info(
                    f"Episode {episode + 1}/{n_episodes} - "
                    f"Return: {stats['total_return']:.2%}, "
                    f"Avg100: {avg_return:.2%}"
                )

            # Validation
            if (episode + 1) % self.config.validate_every_n_episodes == 0:
                val_metrics = self.validate()
                self.validation_history.append(val_metrics)

                logger.info(
                    f"Validation - Return: {val_metrics.total_return:.2%}, "
                    f"Sharpe: {val_metrics.sharpe_ratio:.2f}"
                )

                # Early stopping
                if val_metrics.total_return > self.best_validation_return + self.config.min_improvement:
                    self.best_validation_return = val_metrics.total_return
                    self.patience_counter = 0
                    self._save_checkpoint('best')
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at episode {episode + 1}")
                    break

            # Checkpointing
            if (episode + 1) % self.config.save_every_n_episodes == 0:
                self._save_checkpoint(f'episode_{episode + 1}')

        training_time = time.time() - start_time
        final_metrics = self.validate()

        results = {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'training_time_seconds': training_time,
            'best_validation_return': self.best_validation_return,
            'final_metrics': final_metrics.to_dict(),
            'n_agents': self.system.n_agents,
        }

        logger.info(f"MARL training complete: {results}")
        return results

    def validate(
        self,
        n_episodes: Optional[int] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> TradingMetrics:
        """Run validation episodes for MARL."""
        n_episodes = n_episodes or self.config.n_validation_episodes

        all_returns = []
        all_portfolio_values = []
        all_trades = []

        for _ in range(n_episodes):
            base_state = self.env.reset(start_idx=start_idx, end_idx=end_idx)
            states = self._create_agent_states(base_state)

            while True:
                actions, messages = self.system.select_actions(states, evaluate=True)
                aggregated_action = self.system.get_aggregated_action(actions)
                next_base_state, reward, done, info = self.env.step(aggregated_action)
                states = self._create_agent_states(next_base_state)

                if done:
                    break

            all_returns.extend(self.env.returns)
            all_portfolio_values.append(self.env.portfolio_values[-1])
            all_trades.extend(self.env.trades)

        returns_array = np.array(all_returns)
        portfolio_values = np.array([self.env.config.initial_balance] + all_portfolio_values)

        return self.metrics_calculator.calculate_metrics(
            returns_array,
            portfolio_values,
            all_trades,
        )

    def walk_forward_train(self) -> Dict:
        """Walk-forward training for MARL."""
        logger.info(f"Starting MARL walk-forward with {self.config.n_folds} folds")

        n_samples = self.env.n_samples - self.env.config.min_start_idx
        fold_size = n_samples // self.config.n_folds

        fold_results = []

        for fold in range(self.config.n_folds - 1):
            train_start = self.env.config.min_start_idx + fold * fold_size
            train_end = train_start + int(fold_size * self.config.train_ratio)
            val_start = train_end
            val_end = train_start + fold_size

            logger.info(f"MARL Fold {fold + 1}: Train [{train_start}, {train_end}], "
                       f"Val [{val_start}, {val_end}]")

            self.train(
                n_episodes=self.config.n_episodes // self.config.n_folds,
                start_idx=train_start,
                end_idx=train_end,
            )

            val_metrics = self.validate(start_idx=val_start, end_idx=val_end)
            fold_results.append(val_metrics)

            logger.info(f"MARL Fold {fold + 1} validation - "
                       f"Return: {val_metrics.total_return:.2%}")

        self.fold_results = fold_results
        pbo = self.calculate_pbo(fold_results)
        robustness = self.metrics_calculator.calculate_robustness_metrics(fold_results)

        return {
            'n_folds': len(fold_results),
            'fold_returns': [m.total_return for m in fold_results],
            'pbo': pbo,
            'robustness': robustness,
        }

    def calculate_pbo(self, fold_results: List[TradingMetrics]) -> float:
        """Calculate PBO for MARL."""
        return self.metrics_calculator.calculate_pbo(fold_results)

    def export_onnx(self, path: str) -> None:
        """Export all agent networks to ONNX."""
        logger.info(f"Exporting MARL agents to ONNX: {path}")
        self.system.export_onnx(path)

    def _save_checkpoint(self, name: str) -> None:
        """Save MARL checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"marl_{name}.pt"

        checkpoint = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_validation_return': self.best_validation_return,
            'training_history': self.training_history[-100:],
        }

        self.system.save(str(checkpoint_path.with_suffix('.system.pt')))
        torch.save(checkpoint, checkpoint_path)

        self.saved_checkpoints.append(checkpoint_path)

        # Cleanup
        while len(self.saved_checkpoints) > self.config.keep_n_checkpoints:
            old = self.saved_checkpoints.pop(0)
            if old.exists() and 'best' not in str(old):
                old.unlink()
                sys_path = old.with_suffix('.system.pt')
                if sys_path.exists():
                    sys_path.unlink()

        logger.info(f"MARL checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """Load MARL checkpoint."""
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path)

        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        self.best_validation_return = checkpoint['best_validation_return']
        self.training_history = checkpoint['training_history']

        system_path = checkpoint_path.with_suffix('.system.pt')
        if system_path.exists():
            self.system.load(str(system_path))

        logger.info(f"MARL checkpoint loaded: {path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_walk_forward_folds(
    n_samples: int,
    n_folds: int,
    train_ratio: float = 0.7,
    min_start: int = 100,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Create walk-forward fold indices.

    Returns list of ((train_start, train_end), (val_start, val_end)) tuples.
    """
    effective_samples = n_samples - min_start
    fold_size = effective_samples // n_folds

    folds = []
    for i in range(n_folds - 1):
        train_start = min_start + i * fold_size
        train_end = train_start + int(fold_size * train_ratio)
        val_start = train_end
        val_end = train_start + fold_size

        folds.append(((train_start, train_end), (val_start, val_end)))

    return folds


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_multi_seed_training(
    trainer_class: type,
    env: ReplayEnvironment,
    agent_factory: Callable,
    seeds: List[int],
    config: TrainerConfig,
) -> Dict:
    """
    Run training across multiple seeds for robustness analysis.

    Args:
        trainer_class: D4PGTrainer or MARLTrainer
        env: ReplayEnvironment
        agent_factory: Function that creates a new agent
        seeds: List of random seeds
        config: Training configuration

    Returns:
        Dictionary with multi-seed results
    """
    logger.info(f"Running multi-seed training with {len(seeds)} seeds")

    all_results = []

    for seed in seeds:
        logger.info(f"Training with seed {seed}")
        seed_everything(seed)

        # Create fresh agent
        agent = agent_factory()

        # Create trainer
        trainer = trainer_class(env, agent, config)

        # Train
        results = trainer.train()
        results['seed'] = seed
        all_results.append(results)

    # Aggregate results
    final_returns = [r['final_metrics']['total_return'] for r in all_results]
    sharpes = [r['final_metrics']['sharpe_ratio'] for r in all_results]

    aggregated = {
        'n_seeds': len(seeds),
        'seeds': seeds,
        'individual_results': all_results,
        'mean_return': np.mean(final_returns),
        'std_return': np.std(final_returns),
        'mean_sharpe': np.mean(sharpes),
        'std_sharpe': np.std(sharpes),
        'min_return': np.min(final_returns),
        'max_return': np.max(final_returns),
    }

    logger.info(f"Multi-seed training complete: Mean Return={aggregated['mean_return']:.2%} "
               f"(+/- {aggregated['std_return']:.2%})")

    return aggregated


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Example usage of the RL training infrastructure."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Try to load data
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "features.parquet"

    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        logger.info("Creating synthetic data for demonstration...")

        # Create synthetic data for testing
        n_samples = 10000
        data = np.random.randn(n_samples, 5)  # OHLCV
        data[:, 0] = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)  # Open
        data[:, 3] = data[:, 0] + np.random.randn(n_samples) * 0.1  # Close
        data[:, 1] = np.maximum(data[:, 0], data[:, 3]) + np.abs(np.random.randn(n_samples) * 0.2)  # High
        data[:, 2] = np.minimum(data[:, 0], data[:, 3]) - np.abs(np.random.randn(n_samples) * 0.2)  # Low
        data[:, 4] = np.abs(np.random.randn(n_samples) * 1000) + 100  # Volume

        features = np.random.randn(n_samples, 20)  # 20 features
    else:
        import pandas as pd
        from sklearn.preprocessing import RobustScaler

        df = pd.read_parquet(data_path)

        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        exclude_cols = set(ohlcv_cols) | {'open_time', 'close_time', 'symbol', 'ignore'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        data = df[ohlcv_cols].values
        features = df[feature_cols].values

        scaler = RobustScaler()
        features = scaler.fit_transform(features)

        mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
        data = data[mask]
        features = features[mask]

    logger.info(f"Data shape: {data.shape}, Features shape: {features.shape}")

    # Create environment
    env_config = EnvironmentConfig(
        initial_balance=100_000,
        transaction_cost=0.0005,
        slippage_bps=1.0,
        latency_bars=1,
        random_start=True,
        episode_length=1000,
    )

    env = ReplayEnvironment(data, features, env_config)

    # Example: Create and train D4PG agent
    from models.rl.d4pg_evt import D4PGAgent

    state_dim = features.shape[1] + 4
    agent = D4PGAgent(state_dim=state_dim, batch_size=128)

    trainer_config = TrainerConfig(
        n_episodes=100,
        validate_every_n_episodes=25,
        log_every_n_episodes=10,
        checkpoint_dir=str(Path(__file__).parent.parent.parent / "checkpoints"),
    )

    trainer = D4PGTrainer(env, agent, trainer_config)

    # Run training
    results = trainer.train(n_episodes=50)

    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Episodes: {results['total_episodes']}")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Best Validation Return: {results['best_validation_return']:.2%}")
    print(f"Final Sharpe: {results['final_metrics']['sharpe_ratio']:.2f}")

    # Export model
    onnx_path = Path(__file__).parent.parent.parent / "trained" / "onnx" / "d4pg_actor.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.export_onnx(str(onnx_path))

    return trainer


if __name__ == "__main__":
    main()
