"""
Evaluation Module for Trading Strategies and ML Models

Provides comprehensive tools for:
- Trading metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
- ML metrics (MSE, R2, F1, ROC-AUC)
- Backtesting with transaction costs
- Walk-forward analysis
- Monte Carlo simulation
"""

from .metrics import (
    TradingMetrics,
    MLMetrics,
    MetricsCalculator,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
)

from .backtesting import (
    BacktestConfig,
    BacktestResult,
    VectorizedBacktester,
    WalkForwardAnalyzer,
    MonteCarloSimulator,
)

__all__ = [
    # Metrics
    "TradingMetrics",
    "MLMetrics",
    "MetricsCalculator",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_var",
    "calculate_cvar",
    # Backtesting
    "BacktestConfig",
    "BacktestResult",
    "VectorizedBacktester",
    "WalkForwardAnalyzer",
    "MonteCarloSimulator",
]
