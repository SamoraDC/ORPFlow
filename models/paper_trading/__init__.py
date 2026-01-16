"""
Paper Trading Module

Provides paper trading engine for strategy validation:
- Real-time simulation with live/historical data
- Multi-model ensemble support
- Risk management and position sizing
- Performance tracking and reporting
"""

from .paper_trader import (
    PaperTradingConfig,
    PaperTrader,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    PositionSide,
    Trade,
    RiskManager,
    PerformanceTracker,
    DataFeed,
    HistoricalDataFeed,
    SimulatedDataFeed,
)

__all__ = [
    "PaperTradingConfig",
    "PaperTrader",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Position",
    "PositionSide",
    "Trade",
    "RiskManager",
    "PerformanceTracker",
    "DataFeed",
    "HistoricalDataFeed",
    "SimulatedDataFeed",
]
