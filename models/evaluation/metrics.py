#!/usr/bin/env python3
"""
Trading and ML Metrics Module

Comprehensive metrics for evaluating trading strategies and ML models:
- Trading metrics: Sharpe, Sortino, Calmar, VaR, CVaR
- ML metrics: MSE, R2, F1, ROC-AUC
- Risk metrics: Max Drawdown, Win Rate, Profit Factor
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class TradingMetrics:
    """Container for trading-specific metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0

    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0

    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    alpha: float = 0.0
    beta: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class MLMetrics:
    """Container for ML model metrics."""
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None

    direction_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MetricsCalculator:
    """Calculate comprehensive trading and ML metrics."""

    def __init__(
        self,
        annualization_factor: float = 252 * 24 * 60,  # 1-minute bars
        risk_free_rate: float = 0.0,
    ):
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate

    def calculate_trading_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> TradingMetrics:
        """Calculate all trading metrics from returns series."""
        returns = np.asarray(returns)
        metrics = TradingMetrics()

        if len(returns) == 0:
            return metrics

        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Total and annual returns
        metrics.total_return = float(np.sum(returns))
        metrics.annual_return = float(mean_return * self.annualization_factor)
        metrics.annual_volatility = float(std_return * np.sqrt(self.annualization_factor))

        # Sharpe Ratio
        excess_return = mean_return - self.risk_free_rate / self.annualization_factor
        metrics.sharpe_ratio = float(
            (excess_return / std_return * np.sqrt(self.annualization_factor))
            if std_return > 0 else 0.0
        )

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        metrics.sortino_ratio = float(
            (excess_return / downside_std * np.sqrt(self.annualization_factor))
            if downside_std > 0 else 0.0
        )

        # Drawdown analysis
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative

        metrics.max_drawdown = float(np.max(drawdown))
        metrics.avg_drawdown = float(np.mean(drawdown[drawdown > 0])) if np.any(drawdown > 0) else 0.0

        # Drawdown duration
        in_drawdown = drawdown > 0
        if np.any(in_drawdown):
            dd_periods = []
            current_dd = 0
            for i in range(len(in_drawdown)):
                if in_drawdown[i]:
                    current_dd += 1
                elif current_dd > 0:
                    dd_periods.append(current_dd)
                    current_dd = 0
            if current_dd > 0:
                dd_periods.append(current_dd)
            metrics.max_drawdown_duration = max(dd_periods) if dd_periods else 0

        # Calmar Ratio
        metrics.calmar_ratio = float(
            metrics.annual_return / metrics.max_drawdown
            if metrics.max_drawdown > 0 else 0.0
        )

        # Win/Loss analysis
        winning = returns > 0
        losing = returns < 0

        metrics.total_trades = len(returns)
        metrics.winning_trades = int(np.sum(winning))
        metrics.losing_trades = int(np.sum(losing))
        metrics.win_rate = float(np.mean(winning))

        wins = returns[winning]
        losses = np.abs(returns[losing])

        metrics.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        metrics.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        metrics.win_loss_ratio = float(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss > 0 else 0.0

        # Profit Factor
        total_wins = np.sum(wins)
        total_losses = np.sum(losses)
        metrics.profit_factor = float(total_wins / total_losses) if total_losses > 0 else 0.0

        # Omega Ratio
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses_omega = threshold - returns[returns < threshold]
        metrics.omega_ratio = float(
            np.sum(gains) / np.sum(losses_omega)
            if np.sum(losses_omega) > 0 else 0.0
        )

        # VaR and CVaR
        metrics.var_95 = float(-np.percentile(returns, 5))
        metrics.var_99 = float(-np.percentile(returns, 1))

        tail_95 = returns[returns <= np.percentile(returns, 5)]
        tail_99 = returns[returns <= np.percentile(returns, 1)]
        metrics.cvar_95 = float(-np.mean(tail_95)) if len(tail_95) > 0 else 0.0
        metrics.cvar_99 = float(-np.mean(tail_99)) if len(tail_99) > 0 else 0.0

        # Alpha and Beta (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_returns = np.asarray(benchmark_returns)
            covariance = np.cov(returns, benchmark_returns)

            if covariance.shape == (2, 2):
                beta = covariance[0, 1] / covariance[1, 1] if covariance[1, 1] > 0 else 0.0
                alpha = mean_return - beta * np.mean(benchmark_returns)
                metrics.beta = float(beta)
                metrics.alpha = float(alpha * self.annualization_factor)

                # Information Ratio
                tracking_error = np.std(returns - benchmark_returns)
                metrics.information_ratio = float(
                    (mean_return - np.mean(benchmark_returns)) / tracking_error
                    * np.sqrt(self.annualization_factor)
                    if tracking_error > 0 else 0.0
                )

        return metrics

    def calculate_ml_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task_type: str = "regression",
    ) -> MLMetrics:
        """Calculate ML model metrics."""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        metrics = MLMetrics()

        # Regression metrics
        metrics.mse = float(mean_squared_error(y_true, y_pred))
        metrics.rmse = float(np.sqrt(metrics.mse))
        metrics.mae = float(mean_absolute_error(y_true, y_pred))
        metrics.r2 = float(r2_score(y_true, y_pred))

        # MAPE (avoiding division by zero)
        mask = y_true != 0
        if np.any(mask):
            metrics.mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

        # Direction accuracy (trading-specific)
        metrics.direction_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

        # Classification metrics
        if task_type == "classification":
            y_pred_class = (y_pred > 0.5).astype(int) if y_pred.max() <= 1 else y_pred.astype(int)
            y_true_class = y_true.astype(int)

            metrics.accuracy = float(accuracy_score(y_true_class, y_pred_class))
            metrics.precision = float(precision_score(y_true_class, y_pred_class, zero_division=0))
            metrics.recall = float(recall_score(y_true_class, y_pred_class, zero_division=0))
            metrics.f1 = float(f1_score(y_true_class, y_pred_class, zero_division=0))

            if y_proba is not None:
                try:
                    proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    metrics.roc_auc = float(roc_auc_score(y_true_class, proba))
                except Exception:
                    pass

        return metrics


def calculate_sharpe_ratio(
    returns: np.ndarray,
    annualization_factor: float = 252 * 24 * 60,
    risk_free_rate: float = 0.0,
) -> float:
    """Standalone Sharpe ratio calculation."""
    if len(returns) == 0:
        return 0.0
    std = np.std(returns)
    if std == 0:
        return 0.0
    excess_return = np.mean(returns) - risk_free_rate / annualization_factor
    return float(excess_return / std * np.sqrt(annualization_factor))


def calculate_sortino_ratio(
    returns: np.ndarray,
    annualization_factor: float = 252 * 24 * 60,
    risk_free_rate: float = 0.0,
) -> float:
    """Standalone Sortino ratio calculation."""
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0
    downside_std = np.std(downside)
    if downside_std == 0:
        return 0.0
    excess_return = np.mean(returns) - risk_free_rate / annualization_factor
    return float(excess_return / downside_std * np.sqrt(annualization_factor))


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Standalone max drawdown calculation."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(np.max(drawdown))


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Calculate Value at Risk."""
    if len(returns) == 0:
        return 0.0
    percentile = (1 - confidence) * 100
    return float(-np.percentile(returns, percentile))


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    if len(returns) == 0:
        return 0.0
    var = calculate_var(returns, confidence)
    tail = returns[returns <= -var]
    return float(-np.mean(tail)) if len(tail) > 0 else var


__all__ = [
    "TradingMetrics",
    "MLMetrics",
    "MetricsCalculator",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_var",
    "calculate_cvar",
]
