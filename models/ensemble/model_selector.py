"""
Model Ensemble and Dynamic Selection
Walk-forward validation and performance-based model weighting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types"""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    CNN = "cnn"
    D4PG = "d4pg"
    MARL = "marl"


@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    model_type: ModelType
    mse: float
    mae: float
    r2: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    direction_accuracy: float
    total_return: float
    validation_period: str = ""

    def score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted composite score"""
        if weights is None:
            weights = {
                "sharpe_ratio": 0.25,
                "sortino_ratio": 0.15,
                "win_rate": 0.15,
                "profit_factor": 0.15,
                "direction_accuracy": 0.15,
                "max_drawdown": -0.15,  # Negative weight (lower is better)
            }

        score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0)
            if metric == "max_drawdown":
                # Penalize high drawdown
                score += weight * (1 - min(value, 1))
            else:
                score += weight * value

        return score


@dataclass
class EnsembleConfig:
    """Configuration for ensemble"""
    min_models: int = 2
    max_models: int = 4
    rebalance_frequency: int = 100  # Steps between weight updates
    lookback_periods: int = 20  # Periods for rolling performance
    min_weight: float = 0.05  # Minimum model weight
    decay_factor: float = 0.95  # Exponential decay for older performance


class WalkForwardValidator:
    """Walk-forward validation for time series"""

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap = gap

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits"""
        splits = []

        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            train_end = int(fold_size * (i + 1) * self.train_ratio)
            test_start = train_end + self.gap
            test_end = min(fold_size * (i + 2), n_samples)

            if test_start >= test_end:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        return splits


class ModelEvaluator:
    """Evaluate and compare model performance"""

    def __init__(self):
        self.metrics_history: Dict[ModelType, List[ModelMetrics]] = {
            mt: [] for mt in ModelType
        }

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: ModelType,
        period: str = "",
    ) -> ModelMetrics:
        """Calculate comprehensive metrics"""

        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # Trading metrics
        direction_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        strategy_returns = y_true * np.sign(y_pred)

        # Sharpe ratio (annualized, assuming minute data)
        sharpe_ratio = float(
            (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8))
            * np.sqrt(252 * 24 * 60)
        )

        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        sortino_ratio = float(
            (np.mean(strategy_returns) / (np.std(downside_returns) + 1e-8))
            * np.sqrt(252 * 24 * 60)
        ) if len(downside_returns) > 0 else 0.0

        # Win rate
        win_rate = float(np.mean(strategy_returns > 0))

        # Profit factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = np.abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = float(gains / (losses + 1e-8))

        # Max drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = float(np.max(drawdown))

        # Total return
        total_return = float(cumulative_returns[-1]) if len(cumulative_returns) > 0 else 0.0

        metrics = ModelMetrics(
            model_type=model_type,
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            direction_accuracy=direction_accuracy,
            total_return=total_return,
            validation_period=period,
        )

        self.metrics_history[model_type].append(metrics)

        return metrics

    def get_best_models(
        self,
        n: int = 3,
        metric: str = "score",
    ) -> List[Tuple[ModelType, float]]:
        """Get top N models by performance"""

        model_scores = []

        for model_type in ModelType:
            if self.metrics_history[model_type]:
                recent_metrics = self.metrics_history[model_type][-5:]

                if metric == "score":
                    avg_score = np.mean([m.score() for m in recent_metrics])
                else:
                    avg_score = np.mean([getattr(m, metric) for m in recent_metrics])

                model_scores.append((model_type, avg_score))

        model_scores.sort(key=lambda x: x[1], reverse=True)

        return model_scores[:n]


class DynamicEnsemble:
    """Dynamic model ensemble with adaptive weights"""

    def __init__(
        self,
        models: Dict[ModelType, Any],
        config: Optional[EnsembleConfig] = None,
    ):
        self.models = models
        self.config = config or EnsembleConfig()
        self.evaluator = ModelEvaluator()

        # Initialize equal weights
        self.weights: Dict[ModelType, float] = {
            mt: 1.0 / len(models) for mt in models.keys()
        }

        # Performance tracking
        self.predictions_history: Dict[ModelType, List[float]] = {
            mt: [] for mt in models.keys()
        }
        self.actuals_history: List[float] = []

        self.step_count = 0

    def predict(
        self,
        X: np.ndarray,
        rebalance: bool = True,
    ) -> np.ndarray:
        """Make ensemble prediction"""

        predictions = {}

        for model_type, model in self.models.items():
            pred = model.predict(X)
            predictions[model_type] = pred

        # Weighted average
        ensemble_pred = np.zeros_like(list(predictions.values())[0])

        for model_type, pred in predictions.items():
            ensemble_pred += self.weights[model_type] * pred

        return ensemble_pred

    def update(
        self,
        y_true: float,
        model_predictions: Dict[ModelType, float],
    ):
        """Update ensemble with new observation"""

        self.actuals_history.append(y_true)

        for model_type, pred in model_predictions.items():
            self.predictions_history[model_type].append(pred)

        self.step_count += 1

        # Rebalance weights periodically
        if self.step_count % self.config.rebalance_frequency == 0:
            self._rebalance_weights()

    def _rebalance_weights(self):
        """Rebalance model weights based on recent performance"""

        if len(self.actuals_history) < self.config.lookback_periods:
            return

        recent_actuals = np.array(self.actuals_history[-self.config.lookback_periods:])

        scores = {}

        for model_type in self.models.keys():
            recent_preds = np.array(
                self.predictions_history[model_type][-self.config.lookback_periods:]
            )

            metrics = self.evaluator.calculate_metrics(
                recent_actuals,
                recent_preds,
                model_type,
                period=f"step_{self.step_count}",
            )

            scores[model_type] = max(metrics.score(), 0.01)

        # Normalize weights
        total_score = sum(scores.values())

        for model_type in self.models.keys():
            new_weight = scores[model_type] / total_score

            # Apply exponential smoothing
            self.weights[model_type] = (
                self.config.decay_factor * self.weights[model_type] +
                (1 - self.config.decay_factor) * new_weight
            )

        # Enforce minimum weight
        for model_type in self.models.keys():
            self.weights[model_type] = max(
                self.weights[model_type],
                self.config.min_weight,
            )

        # Re-normalize
        total_weight = sum(self.weights.values())
        for model_type in self.models.keys():
            self.weights[model_type] /= total_weight

        logger.info(f"Rebalanced weights: {self.weights}")

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights"""
        return {mt.value: w for mt, w in self.weights.items()}

    def save_weights(self, path: str):
        """Save weights to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        weights_dict = {
            "weights": {mt.value: w for mt, w in self.weights.items()},
            "step_count": self.step_count,
        }

        with open(path, "w") as f:
            json.dump(weights_dict, f, indent=2)

        logger.info(f"Weights saved to {path}")

    def load_weights(self, path: str):
        """Load weights from file"""
        with open(path, "r") as f:
            weights_dict = json.load(f)

        for model_type_str, weight in weights_dict["weights"].items():
            model_type = ModelType(model_type_str)
            if model_type in self.weights:
                self.weights[model_type] = weight

        self.step_count = weights_dict.get("step_count", 0)

        logger.info(f"Weights loaded from {path}")


class ModelSelector:
    """Select best model(s) for deployment"""

    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.validator = WalkForwardValidator()
        self.validation_results: Dict[ModelType, List[ModelMetrics]] = {}

    def validate_models(
        self,
        models: Dict[ModelType, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> Dict[ModelType, ModelMetrics]:
        """Run walk-forward validation on all models"""

        results = {}
        splits = self.validator.split(len(X))

        for model_type, model in models.items():
            logger.info(f"Validating {model_type.value}...")

            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Retrain model on this fold
                if hasattr(model, "train"):
                    # For ML/DL models
                    split_point = int(len(X_train) * 0.85)
                    model.train(
                        X_train[:split_point],
                        y_train[:split_point],
                        X_train[split_point:],
                        y_train[split_point:],
                    )

                y_pred = model.predict(X_test)

                metrics = self.evaluator.calculate_metrics(
                    y_test,
                    y_pred,
                    model_type,
                    period=f"fold_{fold_idx}",
                )

                fold_metrics.append(metrics)

            # Average metrics across folds
            avg_metrics = self._average_metrics(fold_metrics, model_type)
            results[model_type] = avg_metrics
            self.validation_results[model_type] = fold_metrics

            logger.info(
                f"{model_type.value}: Sharpe={avg_metrics.sharpe_ratio:.3f}, "
                f"WinRate={avg_metrics.win_rate:.2%}, "
                f"PF={avg_metrics.profit_factor:.2f}"
            )

        return results

    def _average_metrics(
        self,
        metrics_list: List[ModelMetrics],
        model_type: ModelType,
    ) -> ModelMetrics:
        """Average metrics across folds"""

        return ModelMetrics(
            model_type=model_type,
            mse=np.mean([m.mse for m in metrics_list]),
            mae=np.mean([m.mae for m in metrics_list]),
            r2=np.mean([m.r2 for m in metrics_list]),
            sharpe_ratio=np.mean([m.sharpe_ratio for m in metrics_list]),
            sortino_ratio=np.mean([m.sortino_ratio for m in metrics_list]),
            win_rate=np.mean([m.win_rate for m in metrics_list]),
            profit_factor=np.mean([m.profit_factor for m in metrics_list]),
            max_drawdown=np.mean([m.max_drawdown for m in metrics_list]),
            direction_accuracy=np.mean([m.direction_accuracy for m in metrics_list]),
            total_return=np.mean([m.total_return for m in metrics_list]),
            validation_period="average",
        )

    def select_best(
        self,
        validation_results: Dict[ModelType, ModelMetrics],
        n_models: int = 3,
        min_sharpe: float = 0.5,
        min_win_rate: float = 0.45,
    ) -> List[ModelType]:
        """Select best models based on criteria"""

        qualified_models = []

        for model_type, metrics in validation_results.items():
            if metrics.sharpe_ratio >= min_sharpe and metrics.win_rate >= min_win_rate:
                qualified_models.append((model_type, metrics.score()))

        # Sort by score
        qualified_models.sort(key=lambda x: x[1], reverse=True)

        selected = [m[0] for m in qualified_models[:n_models]]

        # If not enough qualified, take top performers anyway
        if len(selected) < n_models:
            all_models = [(mt, m.score()) for mt, m in validation_results.items()]
            all_models.sort(key=lambda x: x[1], reverse=True)

            for model_type, _ in all_models:
                if model_type not in selected:
                    selected.append(model_type)
                if len(selected) >= n_models:
                    break

        logger.info(f"Selected models: {[m.value for m in selected]}")

        return selected

    def generate_report(
        self,
        validation_results: Dict[ModelType, ModelMetrics],
        output_path: str,
    ):
        """Generate validation report"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        report = {
            "summary": {},
            "detailed_results": {},
            "recommendations": [],
        }

        # Summary
        for model_type, metrics in validation_results.items():
            report["summary"][model_type.value] = {
                "score": metrics.score(),
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "max_drawdown": metrics.max_drawdown,
                "direction_accuracy": metrics.direction_accuracy,
            }

        # Detailed results
        for model_type, fold_metrics in self.validation_results.items():
            report["detailed_results"][model_type.value] = [
                {
                    "period": m.validation_period,
                    "sharpe_ratio": m.sharpe_ratio,
                    "win_rate": m.win_rate,
                    "profit_factor": m.profit_factor,
                }
                for m in fold_metrics
            ]

        # Recommendations
        sorted_models = sorted(
            validation_results.items(),
            key=lambda x: x[1].score(),
            reverse=True,
        )

        for rank, (model_type, metrics) in enumerate(sorted_models, 1):
            recommendation = {
                "rank": rank,
                "model": model_type.value,
                "score": metrics.score(),
                "notes": [],
            }

            if metrics.sharpe_ratio > 1.0:
                recommendation["notes"].append("Strong risk-adjusted returns")
            if metrics.win_rate > 0.55:
                recommendation["notes"].append("High win rate")
            if metrics.max_drawdown < 0.1:
                recommendation["notes"].append("Low drawdown risk")
            if metrics.profit_factor > 1.5:
                recommendation["notes"].append("Excellent profit factor")

            report["recommendations"].append(recommendation)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")

        return report


def main():
    """Run model selection and ensemble creation"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    logger.info("Model Selector initialized")
    logger.info("Use validate_models() with trained models to run walk-forward validation")
    logger.info("Use select_best() to choose top performing models")
    logger.info("Use DynamicEnsemble for adaptive model weighting")


if __name__ == "__main__":
    main()
