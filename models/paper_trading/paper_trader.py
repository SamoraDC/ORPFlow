#!/usr/bin/env python3
"""
Paper Trading Engine for Strategy Validation

Features:
- Real-time simulation with live/historical data
- Multi-model ensemble support
- Risk management and position sizing
- Performance tracking and reporting
- Webhook notifications
"""

import json
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enum."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionSide(Enum):
    """Position side enum."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "commission": self.commission,
            "slippage": self.slippage,
        }


@dataclass
class Position:
    """Position representation."""
    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

    def update_price(self, price: float):
        self.current_price = price
        self.updated_at = datetime.now()
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Trade:
    """Trade record."""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    pnl: float
    executed_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "pnl": self.pnl,
            "executed_at": self.executed_at.isoformat(),
        }


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading."""

    # Account
    initial_capital: float = 10000.0
    leverage: float = 1.0

    # Costs
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%

    # Risk Management
    max_position_size: float = 0.1  # 10% of capital per position
    max_total_exposure: float = 0.5  # 50% total exposure
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    enable_stop_loss: bool = True
    enable_take_profit: bool = True

    # Trading
    symbol: str = "BTCUSDT"
    min_order_size: float = 0.001
    price_decimals: int = 2
    quantity_decimals: int = 8

    # Data
    data_source: str = "historical"  # historical, live, simulation
    historical_data_path: Optional[str] = None

    # Output
    output_dir: str = "paper_trading_results"
    log_trades: bool = True
    save_interval: int = 100  # Save every N trades

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class RiskManager:
    """Risk management module."""

    def __init__(self, config: PaperTradingConfig):
        self.config = config

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate position size based on risk parameters."""
        max_value = capital * self.config.max_position_size
        position_value = max_value * abs(signal_strength)
        quantity = position_value / price

        # Round to decimals
        quantity = round(quantity, self.config.quantity_decimals)

        return max(quantity, self.config.min_order_size)

    def check_risk_limits(
        self,
        current_exposure: float,
        new_position_value: float,
        capital: float,
    ) -> Tuple[bool, str]:
        """Check if new position violates risk limits."""
        # Check individual position size
        if new_position_value > capital * self.config.max_position_size:
            return False, "Position size exceeds maximum"

        # Check total exposure
        total_exposure = current_exposure + new_position_value
        if total_exposure > capital * self.config.max_total_exposure:
            return False, "Total exposure exceeds maximum"

        return True, "OK"

    def calculate_stop_loss(self, entry_price: float, side: PositionSide) -> float:
        """Calculate stop loss price."""
        if side == PositionSide.LONG:
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, side: PositionSide) -> float:
        """Calculate take profit price."""
        if side == PositionSide.LONG:
            return entry_price * (1 + self.config.take_profit_pct)
        else:
            return entry_price * (1 - self.config.take_profit_pct)


class PerformanceTracker:
    """Track and calculate performance metrics."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trades: List[Trade] = []
        self.daily_returns: List[float] = []

    def record_equity(self, timestamp: datetime, equity: float):
        """Record equity point."""
        self.equity_curve.append((timestamp, equity))

        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2][1]
            daily_return = (equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)

    def record_trade(self, trade: Trade):
        """Record a trade."""
        self.trades.append(trade)

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return {}

        equity = [e[1] for e in self.equity_curve]
        returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])

        # Basic metrics
        final_equity = equity[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Sharpe Ratio (annualized)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (
            mean_return / std_return * np.sqrt(252 * 24 * 60)
            if std_return > 0 else 0.0
        )

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = (
            mean_return / downside_std * np.sqrt(252 * 24 * 60)
            if downside_std > 0 else 0.0
        )

        # Max Drawdown
        cumulative = np.array(equity)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdown)

        # Trade statistics
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]

            win_rate = len(winning_trades) / len(pnls)
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(np.abs(losing_trades)) if losing_trades else 0
            profit_factor = (
                sum(winning_trades) / abs(sum(losing_trades))
                if losing_trades else float("inf")
            )
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": len(self.trades),
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_commission": sum(t.commission for t in self.trades),
        }


class DataFeed(ABC):
    """Abstract data feed interface."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price."""
        pass

    @abstractmethod
    def get_ohlcv(self, symbol: str, limit: int) -> pd.DataFrame:
        """Get OHLCV data."""
        pass


class HistoricalDataFeed(DataFeed):
    """Historical data feed for backtesting."""

    def __init__(self, data_path: str):
        self.df = pd.read_parquet(data_path)
        self.current_idx = 0

        # Ensure we have required columns
        required = ["open", "high", "low", "close"]
        if not all(c in self.df.columns for c in required):
            raise ValueError(f"Data must contain columns: {required}")

        logger.info(f"Loaded {len(self.df)} bars from {data_path}")

    def get_current_price(self, symbol: str) -> float:
        if self.current_idx >= len(self.df):
            raise StopIteration("No more data")
        return float(self.df.iloc[self.current_idx]["close"])

    def get_ohlcv(self, symbol: str, limit: int) -> pd.DataFrame:
        start_idx = max(0, self.current_idx - limit)
        return self.df.iloc[start_idx:self.current_idx + 1].copy()

    def advance(self):
        """Move to next bar."""
        self.current_idx += 1
        return self.current_idx < len(self.df)

    def reset(self):
        """Reset to beginning."""
        self.current_idx = 0


class SimulatedDataFeed(DataFeed):
    """Simulated data feed for testing."""

    def __init__(self, initial_price: float = 50000.0, volatility: float = 0.02):
        self.price = initial_price
        self.volatility = volatility
        self.history: List[Dict] = []

    def get_current_price(self, symbol: str) -> float:
        # Random walk
        change = np.random.normal(0, self.volatility)
        self.price *= (1 + change)

        bar = {
            "timestamp": datetime.now(),
            "open": self.price * (1 - abs(change) / 2),
            "high": self.price * (1 + abs(change)),
            "low": self.price * (1 - abs(change)),
            "close": self.price,
            "volume": np.random.uniform(100, 1000),
        }
        self.history.append(bar)

        return self.price

    def get_ohlcv(self, symbol: str, limit: int) -> pd.DataFrame:
        return pd.DataFrame(self.history[-limit:])


class PaperTrader:
    """Paper trading engine."""

    def __init__(
        self,
        config: PaperTradingConfig,
        signal_generator: Optional[Callable[[pd.DataFrame], float]] = None,
    ):
        self.config = config
        self.signal_generator = signal_generator

        # Initialize components
        self.risk_manager = RiskManager(config)
        self.performance = PerformanceTracker(config.initial_capital)

        # Account state
        self.capital = config.initial_capital
        self.position = Position(symbol=config.symbol)
        self.orders: List[Order] = []
        self.pending_orders: List[Order] = []

        # Counters
        self.order_counter = 0
        self.trade_counter = 0

        # Data feed
        self.data_feed: Optional[DataFeed] = None
        self._setup_data_feed()

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Running state
        self._running = False

        logger.info(f"Paper trader initialized with {config.initial_capital} capital")

    def _setup_data_feed(self):
        """Set up data feed based on configuration."""
        if self.config.data_source == "historical":
            if self.config.historical_data_path:
                self.data_feed = HistoricalDataFeed(self.config.historical_data_path)
            else:
                logger.warning("No historical data path provided")
        elif self.config.data_source == "simulation":
            self.data_feed = SimulatedDataFeed()

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"ORDER-{self.order_counter:08d}"

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        return f"TRADE-{self.trade_counter:08d}"

    def get_equity(self) -> float:
        """Calculate current equity."""
        return self.capital + self.position.unrealized_pnl

    def get_exposure(self) -> float:
        """Calculate current exposure."""
        return abs(self.position.market_value)

    def submit_order(
        self,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Submit a new order."""
        order = Order(
            id=self._generate_order_id(),
            symbol=self.config.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )

        self.pending_orders.append(order)
        self.orders.append(order)

        logger.info(f"Order submitted: {order.id} {side.value} {quantity} @ {order_type.value}")

        return order

    def _execute_order(self, order: Order, current_price: float) -> Optional[Trade]:
        """Execute an order at current price."""
        # Apply slippage
        slippage = current_price * self.config.slippage_pct
        if order.side == OrderSide.BUY:
            execution_price = current_price + slippage
        else:
            execution_price = current_price - slippage

        execution_price = round(execution_price, self.config.price_decimals)

        # Calculate commission
        commission = order.quantity * execution_price * self.config.commission_pct

        # Calculate PnL for closing trades
        pnl = 0.0
        if order.side == OrderSide.SELL and self.position.side == PositionSide.LONG:
            pnl = (execution_price - self.position.entry_price) * order.quantity - commission
        elif order.side == OrderSide.BUY and self.position.side == PositionSide.SHORT:
            pnl = (self.position.entry_price - execution_price) * order.quantity - commission

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        order.commission = commission
        order.slippage = slippage

        # Update position
        self._update_position(order, execution_price)

        # Update capital
        self.capital -= commission
        if pnl != 0:
            self.capital += pnl
            self.position.realized_pnl += pnl

        # Create trade record
        trade = Trade(
            id=self._generate_trade_id(),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            pnl=pnl,
            executed_at=datetime.now(),
        )

        self.performance.record_trade(trade)

        logger.info(
            f"Order filled: {order.id} @ {execution_price:.2f}, "
            f"PnL: {pnl:.2f}, Commission: {commission:.4f}"
        )

        return trade

    def _update_position(self, order: Order, price: float):
        """Update position based on filled order."""
        if order.side == OrderSide.BUY:
            if self.position.side == PositionSide.SHORT:
                # Close short or reduce
                if order.quantity >= self.position.quantity:
                    remaining = order.quantity - self.position.quantity
                    self.position.quantity = remaining
                    self.position.side = PositionSide.LONG if remaining > 0 else PositionSide.FLAT
                    self.position.entry_price = price if remaining > 0 else 0
                else:
                    self.position.quantity -= order.quantity
            else:
                # Add to long or open new long
                total_cost = self.position.quantity * self.position.entry_price + order.quantity * price
                self.position.quantity += order.quantity
                self.position.entry_price = total_cost / self.position.quantity if self.position.quantity > 0 else 0
                self.position.side = PositionSide.LONG

        else:  # SELL
            if self.position.side == PositionSide.LONG:
                # Close long or reduce
                if order.quantity >= self.position.quantity:
                    remaining = order.quantity - self.position.quantity
                    self.position.quantity = remaining
                    self.position.side = PositionSide.SHORT if remaining > 0 else PositionSide.FLAT
                    self.position.entry_price = price if remaining > 0 else 0
                else:
                    self.position.quantity -= order.quantity
            else:
                # Add to short or open new short
                total_cost = self.position.quantity * self.position.entry_price + order.quantity * price
                self.position.quantity += order.quantity
                self.position.entry_price = total_cost / self.position.quantity if self.position.quantity > 0 else 0
                self.position.side = PositionSide.SHORT

        if self.position.side == PositionSide.FLAT:
            self.position.opened_at = None
        elif self.position.opened_at is None:
            self.position.opened_at = datetime.now()

        self.position.current_price = price
        self.position.update_price(price)

    def _check_stop_orders(self, current_price: float):
        """Check and execute stop loss / take profit orders."""
        if self.position.side == PositionSide.FLAT:
            return

        # Stop Loss
        if self.config.enable_stop_loss:
            stop_price = self.risk_manager.calculate_stop_loss(
                self.position.entry_price, self.position.side
            )

            if self.position.side == PositionSide.LONG and current_price <= stop_price:
                logger.info(f"Stop loss triggered at {current_price:.2f}")
                self.submit_order(OrderSide.SELL, self.position.quantity)

            elif self.position.side == PositionSide.SHORT and current_price >= stop_price:
                logger.info(f"Stop loss triggered at {current_price:.2f}")
                self.submit_order(OrderSide.BUY, self.position.quantity)

        # Take Profit
        if self.config.enable_take_profit:
            tp_price = self.risk_manager.calculate_take_profit(
                self.position.entry_price, self.position.side
            )

            if self.position.side == PositionSide.LONG and current_price >= tp_price:
                logger.info(f"Take profit triggered at {current_price:.2f}")
                self.submit_order(OrderSide.SELL, self.position.quantity)

            elif self.position.side == PositionSide.SHORT and current_price <= tp_price:
                logger.info(f"Take profit triggered at {current_price:.2f}")
                self.submit_order(OrderSide.BUY, self.position.quantity)

    def process_signal(self, signal: float, current_price: float):
        """Process trading signal and generate orders."""
        # signal: -1 to 1 (negative = short, positive = long)

        if abs(signal) < 0.1:  # Neutral zone
            return

        # Determine target position
        target_side = PositionSide.LONG if signal > 0 else PositionSide.SHORT

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.capital, current_price, abs(signal)
        )

        # Check risk limits
        position_value = position_size * current_price
        allowed, reason = self.risk_manager.check_risk_limits(
            self.get_exposure(), position_value, self.capital
        )

        if not allowed:
            logger.warning(f"Risk limit violation: {reason}")
            return

        # Generate orders based on current position
        if self.position.side == PositionSide.FLAT:
            # Open new position
            side = OrderSide.BUY if target_side == PositionSide.LONG else OrderSide.SELL
            self.submit_order(side, position_size)

        elif self.position.side != target_side:
            # Reverse position
            close_side = OrderSide.SELL if self.position.side == PositionSide.LONG else OrderSide.BUY
            self.submit_order(close_side, self.position.quantity)
            open_side = OrderSide.BUY if target_side == PositionSide.LONG else OrderSide.SELL
            self.submit_order(open_side, position_size)

        # Same direction - could scale in/out (not implemented for simplicity)

    def step(self) -> bool:
        """Execute one trading step."""
        if self.data_feed is None:
            return False

        try:
            current_price = self.data_feed.get_current_price(self.config.symbol)
        except StopIteration:
            return False

        # Update position price
        self.position.update_price(current_price)

        # Check stop orders
        self._check_stop_orders(current_price)

        # Execute pending market orders
        for order in self.pending_orders[:]:
            if order.order_type == OrderType.MARKET and order.status == OrderStatus.PENDING:
                self._execute_order(order, current_price)
                self.pending_orders.remove(order)

        # Generate signal if we have a signal generator
        if self.signal_generator:
            ohlcv = self.data_feed.get_ohlcv(self.config.symbol, limit=100)
            if len(ohlcv) > 0:
                signal = self.signal_generator(ohlcv)
                self.process_signal(signal, current_price)

        # Record equity
        self.performance.record_equity(datetime.now(), self.get_equity())

        # Advance data feed
        if isinstance(self.data_feed, HistoricalDataFeed):
            return self.data_feed.advance()

        return True

    def run(self, max_steps: Optional[int] = None):
        """Run paper trading loop."""
        logger.info("Starting paper trading...")
        self._running = True

        step_count = 0

        while self._running:
            if max_steps and step_count >= max_steps:
                break

            if not self.step():
                break

            step_count += 1

            # Periodic logging
            if step_count % 1000 == 0:
                logger.info(
                    f"Step {step_count}: Equity={self.get_equity():.2f}, "
                    f"Position={self.position.side.value} {self.position.quantity:.6f}"
                )

            # Periodic save
            if self.config.save_interval and step_count % self.config.save_interval == 0:
                self.save_state()

        self._running = False
        logger.info(f"Paper trading completed after {step_count} steps")

        # Final save
        self.save_state()

        return self.get_results()

    def stop(self):
        """Stop paper trading."""
        self._running = False

    def get_results(self) -> Dict[str, Any]:
        """Get trading results."""
        metrics = self.performance.calculate_metrics()

        return {
            "config": self.config.to_dict(),
            "metrics": metrics,
            "position": self.position.to_dict(),
            "total_orders": len(self.orders),
            "total_trades": len(self.performance.trades),
        }

    def save_state(self):
        """Save current state to disk."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "capital": self.capital,
            "position": self.position.to_dict(),
            "orders": [o.to_dict() for o in self.orders[-100:]],  # Last 100 orders
            "trades": [t.to_dict() for t in self.performance.trades[-100:]],
            "metrics": self.performance.calculate_metrics(),
        }

        state_path = self.output_dir / "paper_trading_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        # Save equity curve
        equity_df = pd.DataFrame(
            self.performance.equity_curve,
            columns=["timestamp", "equity"]
        )
        equity_df.to_parquet(self.output_dir / "equity_curve.parquet", index=False)

    def generate_report(self) -> str:
        """Generate text report."""
        metrics = self.performance.calculate_metrics()

        report = []
        report.append("=" * 60)
        report.append("PAPER TRADING REPORT")
        report.append("=" * 60)
        report.append(f"Symbol: {self.config.symbol}")
        report.append(f"Initial Capital: ${metrics.get('initial_capital', 0):,.2f}")
        report.append(f"Final Equity: ${metrics.get('final_equity', 0):,.2f}")
        report.append(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        report.append("")
        report.append("Performance Metrics:")
        report.append("-" * 40)
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        report.append("")
        report.append("Trade Statistics:")
        report.append("-" * 40)
        report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        report.append(f"Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
        report.append(f"Avg Win: ${metrics.get('avg_win', 0):,.2f}")
        report.append(f"Avg Loss: ${metrics.get('avg_loss', 0):,.2f}")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"Total Commission: ${metrics.get('total_commission', 0):,.2f}")
        report.append("=" * 60)

        return "\n".join(report)


def example_signal_generator(ohlcv: pd.DataFrame) -> float:
    """Example signal generator using simple moving averages."""
    if len(ohlcv) < 20:
        return 0.0

    close = ohlcv["close"].values

    # Simple MA crossover
    short_ma = np.mean(close[-5:])
    long_ma = np.mean(close[-20:])

    if short_ma > long_ma * 1.001:  # Short MA above long MA
        return 0.5  # Buy signal
    elif short_ma < long_ma * 0.999:  # Short MA below long MA
        return -0.5  # Sell signal

    return 0.0  # Neutral


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trading Engine")
    parser.add_argument("--data", type=str, help="Path to historical data")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--output", type=str, default="paper_trading_results", help="Output dir")
    parser.add_argument("--steps", type=int, help="Max steps to run")

    args = parser.parse_args()

    config = PaperTradingConfig(
        initial_capital=args.capital,
        output_dir=args.output,
        data_source="historical" if args.data else "simulation",
        historical_data_path=args.data,
    )

    trader = PaperTrader(config, signal_generator=example_signal_generator)
    results = trader.run(max_steps=args.steps)

    print(trader.generate_report())
    print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
