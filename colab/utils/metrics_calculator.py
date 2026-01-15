"""
Trading Metrics Calculator
Comprehensive evaluation metrics for trading models

Includes:
1. Regression metrics (MSE, MAE, R2)
2. Trading-specific metrics (Sharpe, Sortino, Profit Factor)
3. Risk metrics (VaR, CVaR, Max Drawdown)
4. Classification metrics (Direction Accuracy, Win Rate)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TradingMetrics:
    """Container for all trading metrics"""
    # Regression
    mse: float
    mae: float
    rmse: float
    r2: float

    # Trading
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float

    # Risk
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float

    # Classification
    direction_accuracy: float
    win_rate: float
    total_return: float


class MetricsCalculator:
    """
    Calculate comprehensive metrics for model evaluation.

    REALISTIC EXPECTATIONS:
    - Sharpe Ratio: 0.5-3.0 is good, >5 is suspicious
    - Win Rate: 48-55% is realistic, >60% is suspicious
    - Direction Accuracy: ~50% is random, 52-55% is good
    """

    def __init__(self, annualization_factor: float = np.sqrt(252 * 24 * 60)):
        """
        Args:
            annualization_factor: For minute data = sqrt(252 * 24 * 60)
        """
        self.annualization_factor = annualization_factor
        self.realistic_bounds = {
            "sharpe_ratio": (-2.0, 5.0),
            "sortino_ratio": (-2.0, 8.0),
            "win_rate": (0.40, 0.65),
            "direction_accuracy": (0.45, 0.60),
            "profit_factor": (0.5, 3.0)
        }

    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Calculate standard regression metrics"""
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }

    def calculate_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate trading-specific metrics.

        Strategy: Trade in direction of prediction
        """
        # Strategy returns = actual return * sign of prediction
        strategy_returns = y_true * np.sign(y_pred)

        # Direction accuracy
        direction_correct = np.sign(y_true) == np.sign(y_pred)
        direction_accuracy = np.mean(direction_correct)

        # Win rate (positive strategy returns)
        win_rate = np.mean(strategy_returns > 0)

        # Sharpe ratio (annualized)
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)
        sharpe_ratio = (mean_ret / (std_ret + 1e-8)) * self.annualization_factor

        # Sortino ratio (only downside volatility)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = (mean_ret / (downside_std + 1e-8)) * self.annualization_factor
        else:
            sortino_ratio = sharpe_ratio * 1.5  # No downside = great

        # Profit factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = np.abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gains / (losses + 1e-8)

        # Total return
        cumulative_returns = np.cumsum(strategy_returns)
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Calmar ratio
        if max_drawdown > 0:
            calmar_ratio = (total_return / max_drawdown) * self.annualization_factor / 252
        else:
            calmar_ratio = sharpe_ratio

        return {
            "direction_accuracy": float(direction_accuracy),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "profit_factor": float(profit_factor),
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown)
        }

    def calculate_risk_metrics(
        self,
        returns: np.ndarray,
        confidence_levels: list = [0.95, 0.99]
    ) -> Dict:
        """Calculate Value at Risk and Conditional VaR"""
        metrics = {}

        for conf in confidence_levels:
            # VaR (historical)
            var = np.percentile(returns, (1 - conf) * 100)
            metrics[f"var_{int(conf*100)}"] = float(-var)

            # CVaR (Expected Shortfall)
            cvar = np.mean(returns[returns <= var])
            metrics[f"cvar_{int(conf*100)}"] = float(-cvar) if not np.isnan(cvar) else float(-var)

        return metrics

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> TradingMetrics:
        """Calculate all metrics"""
        reg_metrics = self.calculate_regression_metrics(y_true, y_pred)
        trade_metrics = self.calculate_trading_metrics(y_true, y_pred)

        strategy_returns = y_true * np.sign(y_pred)
        risk_metrics = self.calculate_risk_metrics(strategy_returns)

        return TradingMetrics(
            mse=reg_metrics["mse"],
            mae=reg_metrics["mae"],
            rmse=reg_metrics["rmse"],
            r2=reg_metrics["r2"],
            sharpe_ratio=trade_metrics["sharpe_ratio"],
            sortino_ratio=trade_metrics["sortino_ratio"],
            calmar_ratio=trade_metrics["calmar_ratio"],
            profit_factor=trade_metrics["profit_factor"],
            var_95=risk_metrics["var_95"],
            var_99=risk_metrics["var_99"],
            cvar_95=risk_metrics["cvar_95"],
            max_drawdown=trade_metrics["max_drawdown"],
            direction_accuracy=trade_metrics["direction_accuracy"],
            win_rate=trade_metrics["win_rate"],
            total_return=trade_metrics["total_return"]
        )

    def validate_metrics(self, metrics: Dict) -> Dict:
        """
        Validate that metrics are realistic.

        Returns validation results with warnings for suspicious values.
        """
        validation = {
            "passed": True,
            "warnings": [],
            "analysis": {}
        }

        for metric, bounds in self.realistic_bounds.items():
            if metric not in metrics:
                continue

            value = metrics[metric]
            low, high = bounds

            validation["analysis"][metric] = {
                "value": value,
                "expected_range": bounds,
                "within_bounds": low <= value <= high
            }

            if value < low:
                validation["warnings"].append(
                    f"{metric}={value:.4f} below expected minimum {low}"
                )
            elif value > high:
                validation["passed"] = False
                validation["warnings"].append(
                    f" {metric}={value:.4f} ABOVE maximum {high} - SUSPICIOUS!"
                )

        # Special checks
        if "sharpe_ratio" in metrics and metrics["sharpe_ratio"] > 5:
            validation["passed"] = False
            validation["warnings"].append(
                " Sharpe > 5 is HIGHLY SUSPICIOUS - likely data leakage!"
            )

        if "direction_accuracy" in metrics and metrics["direction_accuracy"] > 0.55:
            validation["warnings"].append(
                f" Direction accuracy {metrics['direction_accuracy']:.1%} > 55% - verify no leakage"
            )

        return validation

    def print_report(
        self,
        metrics: TradingMetrics,
        model_name: str,
        split: str = "test"
    ):
        """Print formatted metrics report"""
        print(f"\n{'='*60}")
        print(f"{model_name.upper()} - {split.upper()} METRICS")
        print("=" * 60)

        print("\nRegression Metrics:")
        print(f"  MSE:  {metrics.mse:.6f}")
        print(f"  MAE:  {metrics.mae:.6f}")
        print(f"  RMSE: {metrics.rmse:.6f}")
        print(f"  R2:   {metrics.r2:.4f}")

        print("\nTrading Metrics:")
        print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:.4f}")
        print(f"  Sortino Ratio:     {metrics.sortino_ratio:.4f}")
        print(f"  Profit Factor:     {metrics.profit_factor:.2f}")
        print(f"  Direction Accuracy: {metrics.direction_accuracy:.2%}")
        print(f"  Win Rate:          {metrics.win_rate:.2%}")

        print("\nRisk Metrics:")
        print(f"  VaR 95%:       {metrics.var_95:.6f}")
        print(f"  VaR 99%:       {metrics.var_99:.6f}")
        print(f"  CVaR 95%:      {metrics.cvar_95:.6f}")
        print(f"  Max Drawdown:  {metrics.max_drawdown:.6f}")
        print(f"  Total Return:  {metrics.total_return:.4f}")

        # Validate
        metrics_dict = {
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "profit_factor": metrics.profit_factor,
            "direction_accuracy": metrics.direction_accuracy,
            "win_rate": metrics.win_rate
        }

        validation = self.validate_metrics(metrics_dict)

        if validation["warnings"]:
            print("\nValidation Warnings:")
            for w in validation["warnings"]:
                print(f"  {w}")

        if not validation["passed"]:
            print("\n METRICS VALIDATION FAILED - Check for data leakage!")
        else:
            print("\n Metrics are within realistic bounds")
