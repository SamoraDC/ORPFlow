#!/usr/bin/env python3
"""
LightGBM Training Script with CPCV, Optuna, and DART/GOSS boosting support.

Features:
- GBDT, DART, and GOSS boosting types
- Combinatorial Purged Cross-Validation (CPCV)
- Optuna hyperparameter optimization
- Categorical feature support
- ONNX export
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
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(returns: np.ndarray, annualization_factor: float = 252 * 24 * 60) -> float:
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor))


def calculate_sortino_ratio(returns: np.ndarray, annualization_factor: float = 252 * 24 * 60) -> float:
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(annualization_factor))


def calculate_max_drawdown(returns: np.ndarray) -> float:
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


@dataclass
class ModelMetrics:
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
class LightGBMConfig:
    data_path: str = "data/raw/klines_90d.parquet"
    output_dir: str = "trained/lightgbm"
    target_column: str = "target_return_5"
    seed: int = 42
    use_gpu: bool = False
    boosting_type: str = "gbdt"  # gbdt, dart, goss
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    cpcv_splits: int = 5
    embargo_pct: float = 0.01
    purge_pct: float = 0.01
    categorical_features: List[str] = None
    enable_optuna: bool = True
    optuna_trials: int = 50
    optuna_timeout: int = 3600
    enable_mlflow: bool = False
    enable_wandb: bool = False

    def __post_init__(self):
        if self.categorical_features is None:
            self.categorical_features = []

    @classmethod
    def from_yaml(cls, path: str) -> "LightGBMConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("lightgbm", {}))


class CombinatorialPurgedKFold:
    def __init__(self, n_splits: int = 5, n_test_groups: int = 2, embargo_pct: float = 0.01, purge_pct: float = 0.01):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def get_n_splits(self) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_groups)

    def split(self, X, y=None, times=None):
        n_samples = len(X)
        group_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        groups = [(i * group_size, (i + 1) * group_size if i < self.n_splits - 1 else n_samples)
                  for i in range(self.n_splits)]

        from itertools import combinations
        for test_group_indices in combinations(range(self.n_splits), self.n_test_groups):
            test_indices = []
            for gi in test_group_indices:
                start, end = groups[gi]
                test_indices.extend(range(start, end))

            train_indices = []
            for i in range(n_samples):
                is_purged = False
                for gi in test_group_indices:
                    g_start, g_end = groups[gi]
                    if g_start - purge_size <= i < g_start or g_end <= i < g_end + embargo_size:
                        is_purged = True
                        break
                if i not in test_indices and not is_purged:
                    train_indices.append(i)

            yield np.array(train_indices), np.array(test_indices)


class LightGBMTrainer:
    def __init__(self, config: LightGBMConfig):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

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
        if self.config.enable_mlflow:
            try:
                import mlflow
                mlflow.set_experiment("lightgbm_trading")
                self._mlflow = mlflow
            except ImportError:
                pass
        if self.config.enable_wandb:
            try:
                import wandb
                wandb.init(project="orpflow", config=asdict(self.config))
                self._wandb = wandb
            except ImportError:
                pass

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.config.data_path}")
        df = pd.read_parquet(self.config.data_path)
        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series]:
        logger.info("Engineering features...")

        if self.config.target_column not in df.columns:
            df["target_return_5"] = df["close"].pct_change(5).shift(-5)
            df = df.dropna()

        exclude_cols = [self.config.target_column, "open_time", "close_time", "symbol"]
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in exclude_cols and not c.startswith("target")]

        self.feature_names = feature_cols
        target = df[self.config.target_column].values
        features = df[feature_cols].values
        times = df["open_time"] if "open_time" in df.columns else pd.Series(range(len(df)))

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Features: {len(feature_cols)}, Samples: {len(features)}")
        return features, target, feature_cols, times

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
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

    def create_optuna_objective(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        def objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            }

            if self.config.boosting_type == "dart":
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3)
                params["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.8)
            elif self.config.boosting_type == "goss":
                params["top_rate"] = trial.suggest_float("top_rate", 0.1, 0.4)
                params["other_rate"] = trial.suggest_float("other_rate", 0.05, 0.2)

            lgb_params = {
                "boosting_type": self.config.boosting_type,
                "objective": "regression",
                "metric": "mse",
                "random_state": self.config.seed,
                "n_jobs": -1,
                "verbose": -1,
                **params
            }

            if self.config.use_gpu:
                lgb_params["device"] = "gpu"

            dtrain = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
            dval = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)

            model = lgb.train(
                lgb_params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            metrics = self._calculate_metrics(y_val, y_pred)
            return metrics.sharpe_ratio

        return objective

    def run_optuna(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        logger.info(f"Running Optuna ({self.config.optuna_trials} trials, boosting={self.config.boosting_type})...")

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.config.seed))
        objective = self.create_optuna_objective(X_train, y_train, X_val, y_val)
        study.optimize(objective, n_trials=self.config.optuna_trials, timeout=self.config.optuna_timeout, show_progress_bar=True)

        logger.info(f"Best Sharpe: {study.best_value:.4f}")
        self.best_params = study.best_params
        return study.best_params

    def train_with_cpcv(self, X: np.ndarray, y: np.ndarray, times: pd.Series, params: Optional[Dict] = None) -> Tuple[Any, Dict[str, Any]]:
        logger.info(f"Training with CPCV ({self.config.cpcv_splits} splits)...")

        cpcv = CombinatorialPurgedKFold(
            n_splits=self.config.cpcv_splits, n_test_groups=2,
            embargo_pct=self.config.embargo_pct, purge_pct=self.config.purge_pct
        )

        fold_sharpes = []
        best_model, best_sharpe = None, -np.inf

        lgb_params = {
            "boosting_type": self.config.boosting_type,
            "objective": "regression",
            "metric": "mse",
            "random_state": self.config.seed,
            "n_jobs": -1,
            "verbose": -1,
            **(params or {})
        }

        if self.config.use_gpu:
            lgb_params["device"] = "gpu"

        for fold_idx, (train_idx, test_idx) in enumerate(cpcv.split(X, y, times)):
            logger.info(f"  Fold {fold_idx + 1}/{cpcv.get_n_splits()}")

            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            val_split = int(len(X_train_fold) * 0.85)
            X_train, X_val = X_train_fold[:val_split], X_train_fold[val_split:]
            y_train, y_val = y_train_fold[:val_split], y_train_fold[val_split:]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test_fold)

            dtrain = lgb.Dataset(X_train_scaled, label=y_train, feature_name=self.feature_names)
            dval = lgb.Dataset(X_val_scaled, label=y_val, feature_name=self.feature_names)

            model = lgb.train(
                lgb_params, dtrain,
                num_boost_round=self.config.n_estimators,
                valid_sets=[dtrain, dval], valid_names=["train", "val"],
                callbacks=[lgb.early_stopping(self.config.early_stopping_rounds), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(X_test_scaled, num_iteration=model.best_iteration)
            sharpe = calculate_sharpe_ratio(y_test_fold * np.sign(y_pred))
            fold_sharpes.append(sharpe)

            if sharpe > best_sharpe:
                best_sharpe, best_model = sharpe, model

            logger.info(f"    Fold {fold_idx + 1} Sharpe: {sharpe:.4f}")

        results = {
            "mean_sharpe": float(np.mean(fold_sharpes)),
            "std_sharpe": float(np.std(fold_sharpes)),
            "min_sharpe": float(np.min(fold_sharpes)),
            "max_sharpe": float(np.max(fold_sharpes)),
        }

        logger.info(f"CPCV Mean Sharpe: {results['mean_sharpe']:.4f} ± {results['std_sharpe']:.4f}")
        return best_model, results

    def train(self) -> Dict[str, Any]:
        start_time = time.time()

        if self._mlflow:
            self._mlflow.start_run()

        df = self.load_data()
        X, y, feature_names, times = self.prepare_features(df)

        n = len(X)
        train_end, val_end = int(n * 0.7), int(n * 0.85)

        X_train_full, X_val_full, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train_full, y_val_full, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

        X_train_scaled = self.scaler.fit_transform(X_train_full)
        X_val_scaled = self.scaler.transform(X_val_full)
        X_test_scaled = self.scaler.transform(X_test)

        if self.config.enable_optuna:
            self.best_params = self.run_optuna(X_train_scaled, y_train_full, X_val_scaled, y_val_full)

        X_combined = np.vstack([X_train_full, X_val_full])
        y_combined = np.concatenate([y_train_full, y_val_full])
        times_combined = times.iloc[:val_end] if hasattr(times, 'iloc') else times[:val_end]

        self.model, cpcv_results = self.train_with_cpcv(X_combined, y_combined, times_combined, self.best_params)

        y_pred = self.model.predict(X_test_scaled, num_iteration=self.model.best_iteration)
        test_metrics = self._calculate_metrics(y_test, y_pred)

        training_time = time.time() - start_time
        test_metrics.training_time_seconds = training_time
        test_metrics.best_iteration = self.model.best_iteration

        model_path = self.output_dir / "lightgbm_model.txt"
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        try:
            self._export_onnx()
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

        results = {
            "training_time_seconds": training_time,
            "boosting_type": self.config.boosting_type,
            "best_params": self.best_params,
            "cpcv_results": cpcv_results,
            "test_metrics": test_metrics.to_dict(),
            "model_path": str(model_path),
            "feature_count": len(feature_names),
        }

        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if self._mlflow:
            self._mlflow.log_params(self.best_params)
            self._mlflow.log_metrics(test_metrics.to_dict())
            self._mlflow.end_run()

        if self._wandb:
            self._wandb.log(test_metrics.to_dict())

        self._print_report(results, test_metrics)
        return results

    def _export_onnx(self):
        try:
            import onnx
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_types = [("input", FloatTensorType([None, len(self.feature_names)]))]
            onnx_model = convert_lightgbm(self.model, initial_types=initial_types, target_opset=17)

            onnx_path = self.output_dir / "lightgbm_model.onnx"
            onnx.save_model(onnx_model, str(onnx_path))
            logger.info(f"ONNX model saved to {onnx_path}")
        except ImportError:
            logger.warning("ONNX export requires: pip install onnx onnxmltools")

    def _print_report(self, results: Dict, test_metrics: ModelMetrics):
        print("\n" + "=" * 60)
        print(f"LIGHTGBM TRAINING REPORT (boosting={self.config.boosting_type})")
        print("=" * 60)
        print(f"Training Time: {results['training_time_seconds']:.1f}s")
        print(f"\nCPCV Mean Sharpe: {results['cpcv_results']['mean_sharpe']:.4f}")
        print(f"\nTest Metrics:")
        print(f"  Sharpe: {test_metrics.sharpe_ratio:.4f}")
        print(f"  Win Rate: {test_metrics.win_rate:.2%}")
        print(f"  Max DD: {test_metrics.max_drawdown:.4f}")
        print(f"\nModel: {results['model_path']}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LightGBM Training")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--data", type=str, default="data/raw/klines_90d.parquet")
    parser.add_argument("--output", type=str, default="trained/lightgbm")
    parser.add_argument("--target", type=str, default="target_return_5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--boosting", type=str, default="gbdt", choices=["gbdt", "dart", "goss"])
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--no-optuna", action="store_true")
    args = parser.parse_args()

    if args.config:
        config = LightGBMConfig.from_yaml(args.config)
    else:
        config = LightGBMConfig(
            data_path=args.data, output_dir=args.output, target_column=args.target,
            seed=args.seed, use_gpu=args.gpu, boosting_type=args.boosting,
            enable_optuna=not args.no_optuna, optuna_trials=args.optuna_trials
        )

    trainer = LightGBMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
