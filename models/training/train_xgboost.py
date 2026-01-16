#!/usr/bin/env python3
"""
XGBoost Training Script with CPCV Validation and Optuna Hyperparameter Tuning

Features:
- Combinatorial Purged Cross-Validation (CPCV)
- Optuna hyperparameter optimization
- MLflow/Weights&Biases tracking
- ONNX export for production
- Trading-specific metrics (Sharpe, Sortino, Calmar)
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import optuna
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(returns: np.ndarray, annualization_factor: float = 252 * 24 * 60) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor))


def calculate_sortino_ratio(returns: np.ndarray, annualization_factor: float = 252 * 24 * 60) -> float:
    """Calculate annualized Sortino ratio."""
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(annualization_factor))


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    direction_accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    best_iteration: int = 0
    training_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class XGBoostConfig:
    """XGBoost training configuration."""
    data_path: str = "data/raw/klines_90d.parquet"
    output_dir: str = "trained/xgboost"
    target_column: str = "target_return_5"
    seed: int = 42
    use_gpu: bool = False
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    cpcv_splits: int = 5
    embargo_pct: float = 0.01
    purge_pct: float = 0.01
    enable_optuna: bool = True
    optuna_trials: int = 50
    optuna_timeout: int = 3600
    enable_mlflow: bool = False
    enable_wandb: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "XGBoostConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("xgboost", {}))


class CombinatorialPurgedKFold:
    """Combinatorial Purged K-Fold Cross-Validation for time series."""

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def get_n_splits(self) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_groups)

    def split(self, X, y=None, times=None):
        """Generate train/test indices for CPCV."""
        n_samples = len(X)
        group_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        # Create group boundaries
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups.append((start, end))

        # Generate all combinations of test groups
        from itertools import combinations
        for test_group_indices in combinations(range(self.n_splits), self.n_test_groups):
            test_indices = []
            for gi in test_group_indices:
                start, end = groups[gi]
                test_indices.extend(range(start, end))

            # Training indices with purge and embargo
            train_indices = []
            for i in range(n_samples):
                # Check if index is too close to any test group
                is_purged = False
                for gi in test_group_indices:
                    g_start, g_end = groups[gi]
                    # Purge before test
                    if g_start - purge_size <= i < g_start:
                        is_purged = True
                        break
                    # Embargo after test
                    if g_end <= i < g_end + embargo_size:
                        is_purged = True
                        break

                if i not in test_indices and not is_purged:
                    train_indices.append(i)

            yield np.array(train_indices), np.array(test_indices)


class XGBoostTrainer:
    """Complete XGBoost training pipeline."""

    def __init__(self, config: XGBoostConfig):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(config.seed)

        self.scaler = RobustScaler()
        self.model = None
        self.feature_names: List[str] = []
        self.best_params: Dict[str, Any] = {}
        self._mlflow = None
        self._wandb = None

        self._init_tracking()

    def _init_tracking(self):
        """Initialize experiment tracking."""
        if self.config.enable_mlflow:
            try:
                import mlflow
                mlflow.set_experiment("xgboost_trading")
                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow not available")

        if self.config.enable_wandb:
            try:
                import wandb
                wandb.init(project="orpflow", config=asdict(self.config))
                self._wandb = wandb
            except ImportError:
                logger.warning("W&B not available")

    def load_data(self) -> pd.DataFrame:
        """Load data from parquet file."""
        logger.info(f"Loading data from {self.config.data_path}")
        df = pd.read_parquet(self.config.data_path)

        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

        logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series]:
        """Prepare features for training."""
        logger.info("Engineering features...")

        # Generate target if not exists
        if self.config.target_column not in df.columns:
            df["target_return_5"] = df["close"].pct_change(5).shift(-5)
            df = df.dropna()

        # Select numeric columns as features (exclude target and time)
        exclude_cols = [self.config.target_column, "open_time", "close_time", "symbol"]
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in exclude_cols and not c.startswith("target")]

        self.feature_names = feature_cols
        target = df[self.config.target_column].values
        features = df[feature_cols].values
        times = df["open_time"] if "open_time" in df.columns else pd.Series(range(len(df)))

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Features: {len(feature_cols)}, Samples: {len(features)}")
        return features, target, feature_cols, times

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate all metrics."""
        strategy_returns = y_true * np.sign(y_pred)

        return ModelMetrics(
            mse=float(mean_squared_error(y_true, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mae=float(mean_absolute_error(y_true, y_pred)),
            r2=float(r2_score(y_true, y_pred)),
            direction_accuracy=float(np.mean(np.sign(y_true) == np.sign(y_pred))),
            sharpe_ratio=calculate_sharpe_ratio(strategy_returns),
            sortino_ratio=calculate_sortino_ratio(strategy_returns),
            win_rate=float(np.mean(strategy_returns > 0)),
            max_drawdown=calculate_max_drawdown(strategy_returns),
            total_return=float(np.sum(strategy_returns)),
        )

    def create_optuna_objective(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ):
        """Create Optuna objective function."""
        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "seed": self.config.seed,
                "tree_method": "hist",
                "device": "cuda" if self.config.use_gpu else "cpu",
            }

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

            model = xgb.train(
                params, dtrain,
                num_boost_round=500,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )

            y_pred = model.predict(dval)
            metrics = self._calculate_metrics(y_val, y_pred)
            return metrics.sharpe_ratio

        return objective

    def run_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Run Optuna hyperparameter optimization."""
        logger.info(f"Running Optuna with {self.config.optuna_trials} trials...")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.seed)
        )

        objective = self.create_optuna_objective(X_train, y_train, X_val, y_val)
        study.optimize(
            objective,
            n_trials=self.config.optuna_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True
        )

        logger.info(f"Best Sharpe: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        self.best_params = study.best_params
        return study.best_params

    def train_with_cpcv(
        self, X: np.ndarray, y: np.ndarray, times: pd.Series,
        params: Optional[Dict] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train model with CPCV validation."""
        logger.info(f"Training with CPCV ({self.config.cpcv_splits} splits)...")

        cpcv = CombinatorialPurgedKFold(
            n_splits=self.config.cpcv_splits,
            n_test_groups=2,
            embargo_pct=self.config.embargo_pct,
            purge_pct=self.config.purge_pct
        )

        fold_sharpes = []
        best_model = None
        best_sharpe = -np.inf

        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": self.config.seed,
            "tree_method": "hist",
            "device": "cuda" if self.config.use_gpu else "cpu",
            **(params or {})
        }

        for fold_idx, (train_idx, test_idx) in enumerate(cpcv.split(X, y, times)):
            logger.info(f"  Fold {fold_idx + 1}/{cpcv.get_n_splits()}")

            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # Split train into train/val
            val_split = int(len(X_train_fold) * 0.85)
            X_train, X_val = X_train_fold[:val_split], X_train_fold[val_split:]
            y_train, y_val = y_train_fold[:val_split], y_train_fold[val_split:]

            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test_fold)

            dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=self.feature_names)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test_fold, feature_names=self.feature_names)

            model = xgb.train(
                xgb_params, dtrain,
                num_boost_round=self.config.n_estimators,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=False,
            )

            y_pred = model.predict(dtest)
            sharpe = calculate_sharpe_ratio(y_test_fold * np.sign(y_pred))
            fold_sharpes.append(sharpe)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model = model

            logger.info(f"    Fold {fold_idx + 1} Sharpe: {sharpe:.4f}")

        results = {
            "mean_sharpe": float(np.mean(fold_sharpes)),
            "std_sharpe": float(np.std(fold_sharpes)),
            "min_sharpe": float(np.min(fold_sharpes)),
            "max_sharpe": float(np.max(fold_sharpes)),
            "folds": len(fold_sharpes),
        }

        logger.info(f"CPCV Mean Sharpe: {results['mean_sharpe']:.4f} ± {results['std_sharpe']:.4f}")
        return best_model, results

    def train(self) -> Dict[str, Any]:
        """Run complete training pipeline."""
        start_time = time.time()

        if self._mlflow:
            self._mlflow.start_run()

        # Load and prepare data
        df = self.load_data()
        X, y, feature_names, times = self.prepare_features(df)

        # Split data
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X_train_full = X[:train_end]
        X_val_full = X[train_end:val_end]
        X_test = X[val_end:]

        y_train_full = y[:train_end]
        y_val_full = y[train_end:val_end]
        y_test = y[val_end:]

        # Scale for Optuna
        X_train_scaled = self.scaler.fit_transform(X_train_full)
        X_val_scaled = self.scaler.transform(X_val_full)
        X_test_scaled = self.scaler.transform(X_test)

        # Optuna optimization
        if self.config.enable_optuna:
            self.best_params = self.run_optuna(X_train_scaled, y_train_full, X_val_scaled, y_val_full)

        # CPCV training
        X_combined = np.vstack([X_train_full, X_val_full])
        y_combined = np.concatenate([y_train_full, y_val_full])
        times_combined = times.iloc[:val_end] if hasattr(times, 'iloc') else times[:val_end]

        self.model, cpcv_results = self.train_with_cpcv(
            X_combined, y_combined, times_combined, self.best_params
        )

        # Final test evaluation
        X_test_scaled = self.scaler.transform(X_test)
        dtest = xgb.DMatrix(X_test_scaled, feature_names=self.feature_names)
        y_pred = self.model.predict(dtest)
        test_metrics = self._calculate_metrics(y_test, y_pred)

        training_time = time.time() - start_time
        test_metrics.training_time_seconds = training_time
        test_metrics.best_iteration = self.model.best_iteration

        # Save model
        model_path = self.output_dir / "xgboost_model.ubj"
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Export ONNX
        try:
            self._export_onnx()
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

        # Save results
        results = {
            "training_time_seconds": training_time,
            "best_params": self.best_params,
            "cpcv_results": cpcv_results,
            "test_metrics": test_metrics.to_dict(),
            "model_path": str(model_path),
            "feature_count": len(feature_names),
            "train_samples": len(X_train_full),
            "test_samples": len(X_test),
        }

        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Log to trackers
        if self._mlflow:
            self._mlflow.log_params(self.best_params)
            self._mlflow.log_metrics(test_metrics.to_dict())
            self._mlflow.log_artifact(str(model_path))
            self._mlflow.end_run()

        if self._wandb:
            self._wandb.log(test_metrics.to_dict())
            self._wandb.log({"cpcv_mean_sharpe": cpcv_results["mean_sharpe"]})

        self._print_report(results, test_metrics)
        return results

    def _export_onnx(self):
        """Export model to ONNX format."""
        try:
            import onnx
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_types = [("input", FloatTensorType([None, len(self.feature_names)]))]
            onnx_model = convert_xgboost(self.model, initial_types=initial_types, target_opset=17)

            onnx_path = self.output_dir / "xgboost_model.onnx"
            onnx.save_model(onnx_model, str(onnx_path))
            logger.info(f"ONNX model saved to {onnx_path}")
        except ImportError:
            logger.warning("ONNX export requires: pip install onnx onnxmltools")

    def _print_report(self, results: Dict, test_metrics: ModelMetrics):
        """Print training report."""
        print("\n" + "=" * 60)
        print("XGBOOST TRAINING REPORT")
        print("=" * 60)
        print(f"Training Time: {results['training_time_seconds']:.1f}s")
        print(f"Features: {results['feature_count']}")
        print(f"Train/Test: {results['train_samples']}/{results['test_samples']}")
        print(f"\nCPCV Results:")
        print(f"  Mean Sharpe: {results['cpcv_results']['mean_sharpe']:.4f}")
        print(f"  Std Sharpe: {results['cpcv_results']['std_sharpe']:.4f}")
        print(f"\nTest Metrics:")
        print(f"  MSE: {test_metrics.mse:.6f}")
        print(f"  R2: {test_metrics.r2:.4f}")
        print(f"  Sharpe: {test_metrics.sharpe_ratio:.4f}")
        print(f"  Sortino: {test_metrics.sortino_ratio:.4f}")
        print(f"  Win Rate: {test_metrics.win_rate:.2%}")
        print(f"  Max DD: {test_metrics.max_drawdown:.4f}")
        print(f"\nModel saved to: {results['model_path']}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="XGBoost Training")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--data", type=str, default="data/raw/klines_90d.parquet")
    parser.add_argument("--output", type=str, default="trained/xgboost")
    parser.add_argument("--target", type=str, default="target_return_5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.config:
        config = XGBoostConfig.from_yaml(args.config)
    else:
        config = XGBoostConfig(
            data_path=args.data,
            output_dir=args.output,
            target_column=args.target,
            seed=args.seed,
            use_gpu=args.gpu,
            enable_optuna=not args.no_optuna,
            optuna_trials=args.optuna_trials,
            enable_mlflow=args.mlflow,
            enable_wandb=args.wandb
        )

    trainer = XGBoostTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
