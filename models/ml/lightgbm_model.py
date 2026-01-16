"""
LightGBM Model for Trading Signal Prediction
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMModel:
    """LightGBM model for return/signal prediction"""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.model = None
        self.feature_importance = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> Dict:
        """Train the LightGBM model"""

        logger.info("Training LightGBM model...")

        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=feature_names,
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            feature_name=feature_names,
            reference=train_data,
        )

        # Train with callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            "feature": feature_names or [f"f_{i}" for i in range(X_train.shape[1])],
            "importance": self.model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
            "val_mse": mean_squared_error(y_val, y_pred_val),
            "val_mae": mean_absolute_error(y_val, y_pred_val),
            "val_r2": r2_score(y_val, y_pred_val),
            "best_iteration": self.model.best_iteration,
        }

        logger.info(f"Training complete. Val MSE: {metrics['val_mse']:.6f}, Val R2: {metrics['val_r2']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Evaluate on test data"""
        y_pred = self.predict(X_test)

        metrics = {
            "test_mse": mean_squared_error(y_test, y_pred),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

        # Trading metrics
        metrics.update(self._calculate_trading_metrics(y_test, y_pred))

        logger.info(f"Test MSE: {metrics['test_mse']:.6f}, Sharpe: {metrics.get('sharpe_ratio', 0):.4f}")

        return metrics

    def _calculate_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Calculate trading-specific metrics"""

        # Directional accuracy
        direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

        # Strategy returns (trade in direction of prediction)
        strategy_returns = y_true * np.sign(y_pred)

        # Sharpe ratio (annualized, assuming minute data)
        sharpe_ratio = (
            np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
        ) * np.sqrt(252 * 24 * 60)

        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        sortino_ratio = (
            np.mean(strategy_returns) / (np.std(downside_returns) + 1e-8)
        ) * np.sqrt(252 * 24 * 60) if len(downside_returns) > 0 else 0

        # Win rate
        win_rate = np.mean(strategy_returns > 0)

        # Profit factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = np.abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gains / (losses + 1e-8)

        # Max drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = np.max(drawdown)

        return {
            "direction_accuracy": direction_accuracy,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "total_return": cumulative_returns[-1] if len(cumulative_returns) > 0 else 0,
        }

    def save(self, path: str):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "params": self.params,
            "feature_importance": self.feature_importance,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        data = joblib.load(path)
        self.model = data["model"]
        self.params = data["params"]
        self.feature_importance = data["feature_importance"]
        logger.info(f"Model loaded from {path}")

    def export_onnx(self, path: str, feature_names: list):
        """Export model to ONNX format"""
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx

        # LightGBM to ONNX requires onnxmltools
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType

            initial_types = [("input", OnnxFloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_lightgbm(
                self.model,
                initial_types=initial_types,
                target_opset=17,
            )

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            onnx.save_model(onnx_model, path)
            logger.info(f"ONNX model exported to {path}")

        except ImportError:
            logger.warning("onnxmltools not installed. Install with: pip install onnxmltools")
            # Fallback: save as native format
            native_path = path.replace(".onnx", ".lgb")
            self.model.save_model(native_path)
            logger.info(f"Saved native LightGBM model to {native_path}")

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Get top N important features"""
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.feature_importance.head(n)


def main():
    """Train and evaluate LightGBM model"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.preprocessor import FeatureEngineer
    from data.collector import BinanceDataCollector

    # Load processed data
    data_path = Path(__file__).parent.parent / "data" / "processed" / "features.parquet"

    if not data_path.exists():
        logger.error("Processed features not found. Run preprocessor.py first.")
        return

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")

    # Prepare data
    engineer = FeatureEngineer()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = engineer.prepare_ml_data(
        df,
        target_col="target_return_5",
    )

    # Train model
    model = LightGBMModel()
    train_metrics = model.train(
        X_train, y_train, X_val, y_val,
        feature_names=feature_names,
    )

    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)

    # Save
    model_dir = Path(__file__).parent.parent / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(model_dir / "lightgbm_model.pkl"))
    model.export_onnx(str(model_dir / "onnx" / "lightgbm_model.onnx"), feature_names)

    # Print results
    print("\n" + "=" * 50)
    print("LightGBM Model Results")
    print("=" * 50)
    print(f"Validation MSE: {train_metrics['val_mse']:.6f}")
    print(f"Test MSE: {test_metrics['test_mse']:.6f}")
    print(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
    print(f"Win Rate: {test_metrics['win_rate']:.2%}")
    print(f"Profit Factor: {test_metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {test_metrics['max_drawdown']:.4f}")
    print("\nTop 10 Features:")
    print(model.get_top_features(10))


if __name__ == "__main__":
    main()
