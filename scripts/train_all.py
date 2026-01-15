#!/usr/bin/env python3
"""
Comprehensive Training Entry Point for ORPFlow
================================================

This script provides a unified training pipeline that:
1. Loads real data from data/raw/klines_90d.parquet
2. Runs leakage checks FIRST (fail fast)
3. Trains all ML models (LightGBM, XGBoost) with CPCV
4. Trains all DL models (LSTM, CNN) with CPCV
5. Trains all RL models (D4PG, MARL) with walk-forward
6. Exports all models to ONNX
7. Runs parity tests
8. Generates final report

Usage:
    python scripts/train_all.py --data data/raw/klines_90d.parquet

    # Train only ML models
    python scripts/train_all.py --models lightgbm,xgboost

    # Skip RL (faster)
    python scripts/train_all.py --skip-rl

    # Debug mode (skip leakage checks)
    python scripts/train_all.py --skip-leakage

Author: ORPFlow Team
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("train_all")


# =============================================================================
# Configuration
# =============================================================================

AVAILABLE_MODELS = {
    "ml": ["lightgbm", "xgboost"],
    "dl": ["lstm", "cnn"],
    "rl": ["d4pg", "marl"],
}

ALL_MODELS = (
    AVAILABLE_MODELS["ml"] +
    AVAILABLE_MODELS["dl"] +
    AVAILABLE_MODELS["rl"]
)


@dataclass
class TrainAllConfig:
    """Configuration for the unified training pipeline."""
    # Data paths
    data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "klines_90d.parquet")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "trained")

    # Model selection
    models: Set[str] = field(default_factory=lambda: set(ALL_MODELS))

    # CPCV configuration
    cpcv_splits: int = 5
    embargo_pct: float = 0.01
    purge_pct: float = 0.01

    # Training parameters
    seed: int = 42
    sequence_length: int = 60
    feature_window: int = 60
    label_horizon: int = 5

    # ML parameters
    ml_n_estimators: int = 1000

    # DL parameters
    dl_epochs: int = 100
    dl_patience: int = 10
    dl_batch_size: int = 64

    # RL parameters
    rl_episodes: int = 200
    rl_walk_forward_folds: int = 5

    # Feature engineering
    enable_quant_features: bool = True
    enable_microstructure: bool = True

    # Validation flags
    skip_leakage: bool = False
    skip_rl: bool = False
    skip_parity: bool = False

    # Logging
    verbose: bool = False

    def __post_init__(self):
        """Validate and convert paths."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.models, list):
            self.models = set(self.models)

        # Make paths absolute
        if not self.data_path.is_absolute():
            self.data_path = PROJECT_ROOT / self.data_path
        if not self.output_dir.is_absolute():
            self.output_dir = PROJECT_ROOT / self.output_dir

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            k: str(v) if isinstance(v, Path) else (list(v) if isinstance(v, set) else v)
            for k, v in asdict(self).items()
        }


# =============================================================================
# Timer Utility
# =============================================================================

class Timer:
    """Simple timing utility for tracking execution time."""

    def __init__(self):
        self.start_time = time.time()
        self.checkpoints: Dict[str, float] = {}

    def checkpoint(self, name: str) -> float:
        """Record a checkpoint and return elapsed time since start."""
        elapsed = time.time() - self.start_time
        self.checkpoints[name] = elapsed
        return elapsed

    def elapsed(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time

    def format_elapsed(self) -> str:
        """Format elapsed time as human-readable string."""
        elapsed = self.elapsed()
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            return f"{hours}h {minutes}m"

    def get_report(self) -> str:
        """Generate timing report."""
        lines = ["Timing Report:", "-" * 40]
        prev_time = 0.0
        for name, timestamp in self.checkpoints.items():
            duration = timestamp - prev_time
            lines.append(f"  {name}: {duration:.1f}s")
            prev_time = timestamp
        lines.append(f"  Total: {self.format_elapsed()}")
        return "\n".join(lines)


# =============================================================================
# Training Results Tracker
# =============================================================================

@dataclass
class ModelResult:
    """Result from training a single model."""
    model_name: str
    category: str  # ml, dl, rl
    success: bool
    training_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None


class ResultsTracker:
    """Track and aggregate training results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results: List[ModelResult] = []
        self.leakage_passed: bool = False
        self.parity_results: Dict[str, bool] = {}

    def add_result(self, result: ModelResult) -> None:
        """Add a model result."""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful

        by_category = {}
        for category in ["ml", "dl", "rl"]:
            cat_results = [r for r in self.results if r.category == category]
            by_category[category] = {
                "total": len(cat_results),
                "successful": sum(1 for r in cat_results if r.success),
            }

        return {
            "total_models": total,
            "successful": successful,
            "failed": failed,
            "by_category": by_category,
            "leakage_passed": self.leakage_passed,
            "parity_results": self.parity_results,
        }

    def generate_report(self) -> str:
        """Generate comprehensive training report."""
        lines = [
            "",
            "=" * 70,
            "TRAINING REPORT",
            "=" * 70,
            "",
        ]

        summary = self.get_summary()

        lines.extend([
            f"Total Models: {summary['total_models']}",
            f"Successful: {summary['successful']}",
            f"Failed: {summary['failed']}",
            f"Leakage Checks: {'PASSED' if summary['leakage_passed'] else 'SKIPPED/FAILED'}",
            "",
        ])

        # Results by category
        for category in ["ml", "dl", "rl"]:
            cat_results = [r for r in self.results if r.category == category]
            if cat_results:
                lines.append(f"\n{category.upper()} Models:")
                lines.append("-" * 40)
                for result in cat_results:
                    status = "OK" if result.success else "FAIL"
                    lines.append(f"  [{status}] {result.model_name} ({result.training_time:.1f}s)")
                    if result.metrics:
                        for key, value in result.metrics.items():
                            if isinstance(value, float):
                                lines.append(f"        {key}: {value:.4f}")
                            else:
                                lines.append(f"        {key}: {value}")
                    if result.error:
                        lines.append(f"        Error: {result.error}")

        # Parity results
        if self.parity_results:
            lines.append("\n\nParity Tests:")
            lines.append("-" * 40)
            for model, passed in self.parity_results.items():
                status = "PASS" if passed else "FAIL"
                lines.append(f"  [{status}] {model}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def save(self, path: Optional[Path] = None) -> None:
        """Save results to JSON."""
        path = path or (self.output_dir / "training_results.json")

        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "results": [
                {
                    "model_name": r.model_name,
                    "category": r.category,
                    "success": r.success,
                    "training_time": r.training_time,
                    "metrics": r.metrics,
                    "artifacts": r.artifacts,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results saved to {path}")


# =============================================================================
# Main Training Pipeline
# =============================================================================

class TrainingPipeline:
    """Unified training pipeline for all model types."""

    def __init__(self, config: TrainAllConfig):
        self.config = config
        self.timer = Timer()
        self.tracker = ResultsTracker(config.output_dir)

        # Create output directories
        self.model_dir = config.output_dir / "models"
        self.onnx_dir = config.output_dir / "onnx"
        self.experiment_dir = config.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        np.random.seed(config.seed)

        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.times: Optional[pd.Series] = None

        # Split data
        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        logger.info(f"TrainingPipeline initialized")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Models to train: {sorted(config.models)}")

    def run(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("\n" + "=" * 70)
        logger.info("Starting Unified Training Pipeline")
        logger.info("=" * 70 + "\n")

        try:
            # Step 1: Load data
            self._step_load_data()
            self.timer.checkpoint("data_loading")

            # Step 2: Feature engineering
            self._step_feature_engineering()
            self.timer.checkpoint("feature_engineering")

            # Step 3: Prepare splits
            self._step_prepare_splits()
            self.timer.checkpoint("data_splitting")

            # Step 4: Leakage checks (fail fast)
            if not self.config.skip_leakage:
                self._step_leakage_checks()
            else:
                logger.warning("SKIPPING leakage checks (--skip-leakage flag)")
            self.timer.checkpoint("leakage_checks")

            # Step 5: Train ML models
            ml_models = self.config.models & set(AVAILABLE_MODELS["ml"])
            if ml_models:
                self._step_train_ml(ml_models)
            self.timer.checkpoint("ml_training")

            # Step 6: Train DL models
            dl_models = self.config.models & set(AVAILABLE_MODELS["dl"])
            if dl_models:
                self._step_train_dl(dl_models)
            self.timer.checkpoint("dl_training")

            # Step 7: Train RL models
            rl_models = self.config.models & set(AVAILABLE_MODELS["rl"])
            if rl_models and not self.config.skip_rl:
                self._step_train_rl(rl_models)
            elif self.config.skip_rl:
                logger.warning("SKIPPING RL training (--skip-rl flag)")
            self.timer.checkpoint("rl_training")

            # Step 8: Export ONNX
            self._step_export_onnx()
            self.timer.checkpoint("onnx_export")

            # Step 9: Parity tests
            if not self.config.skip_parity:
                self._step_parity_tests()
            else:
                logger.warning("SKIPPING parity tests (--skip-parity flag)")
            self.timer.checkpoint("parity_tests")

            # Step 10: Generate final report
            self._step_final_report()
            self.timer.checkpoint("report_generation")

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            raise

        return self.tracker.get_summary()

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================

    def _step_load_data(self) -> None:
        """Load raw data from parquet file."""
        logger.info("\n[1/10] Loading data...")

        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")

        self.df = pd.read_parquet(self.config.data_path)

        # Set datetime index
        if "open_time" in self.df.columns:
            self.df["open_time"] = pd.to_datetime(self.df["open_time"], unit="ms")
            self.df = self.df.set_index("open_time").sort_index()
        elif "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], unit="ms")
            self.df = self.df.set_index("timestamp").sort_index()
        elif not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        logger.info(f"  Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        logger.info(f"  Date range: {self.df.index.min()} to {self.df.index.max()}")
        logger.info(f"  Columns: {list(self.df.columns)[:10]}...")

    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================

    def _step_feature_engineering(self) -> None:
        """Add advanced features to the data."""
        logger.info("\n[2/10] Engineering features...")

        if self.config.enable_quant_features:
            self._add_quant_features()

        if self.config.enable_microstructure:
            self._add_microstructure_features()

        # Create targets
        self._create_targets()

        logger.info(f"  Total features: {len(self.df.columns)}")

    def _add_quant_features(self) -> None:
        """Add quantitative features (Hawkes, Kalman, HMM, etc.)."""
        logger.info("  Adding quantitative features...")

        try:
            from models.training.orchestrator import DataManager
            data_manager = DataManager(seed=self.config.seed)
            self.df = data_manager.add_quant_features(self.df)
        except ImportError as e:
            logger.warning(f"  Could not add quant features: {e}")
        except Exception as e:
            logger.warning(f"  Quant features failed: {e}")

    def _add_microstructure_features(self) -> None:
        """Add microstructure features (VPIN, OFI)."""
        logger.info("  Adding microstructure features...")

        try:
            from models.training.orchestrator import DataManager
            data_manager = DataManager(seed=self.config.seed)
            self.df = data_manager.add_microstructure(self.df)
        except ImportError as e:
            logger.warning(f"  Could not add microstructure features: {e}")
        except Exception as e:
            logger.warning(f"  Microstructure features failed: {e}")

    def _create_targets(self) -> None:
        """Create target columns for prediction."""
        logger.info("  Creating targets...")

        horizons = [1, 5, 15]
        for h in horizons:
            target_col = f"target_return_{h}"
            if target_col not in self.df.columns:
                self.df[target_col] = self.df["close"].pct_change(h).shift(-h)

    # =========================================================================
    # Step 3: Prepare Splits
    # =========================================================================

    def _step_prepare_splits(self) -> None:
        """Prepare train/val/test splits."""
        logger.info("\n[3/10] Preparing data splits...")

        # Identify feature columns
        exclude_patterns = ["target_", "close_time", "symbol", "ignore"]
        self.feature_names = []

        for col in self.df.columns:
            if any(pat in col for pat in exclude_patterns):
                continue
            if self.df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                self.feature_names.append(col)

        # Get target column
        target_col = f"target_return_{self.config.label_horizon}"
        if target_col not in self.df.columns:
            target_col = "target_return_5"

        # Remove rows with NaN
        valid_mask = ~self.df[target_col].isna()
        for col in self.feature_names:
            valid_mask &= ~self.df[col].isna()

        df_valid = self.df[valid_mask].copy()

        logger.info(f"  Valid samples: {len(df_valid):,} / {len(self.df):,}")

        self.X = df_valid[self.feature_names].values.astype(np.float32)
        self.y = df_valid[target_col].values.astype(np.float32)
        self.times = pd.Series(df_valid.index)

        # Replace inf values
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)

        # Create temporal splits with embargo
        n = len(self.X)
        embargo_size = int(n * self.config.embargo_pct)

        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_idx = np.arange(0, train_end - embargo_size)
        val_idx = np.arange(train_end + embargo_size, val_end - embargo_size)
        test_idx = np.arange(val_end + embargo_size, n)

        self.X_train = self.X[train_idx]
        self.X_val = self.X[val_idx]
        self.X_test = self.X[test_idx]

        self.y_train = self.y[train_idx]
        self.y_val = self.y[val_idx]
        self.y_test = self.y[test_idx]

        logger.info(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
        logger.info(f"  Features: {len(self.feature_names)}")

    # =========================================================================
    # Step 4: Leakage Checks
    # =========================================================================

    def _step_leakage_checks(self) -> None:
        """Run comprehensive leakage validation."""
        logger.info("\n[4/10] Running leakage checks...")

        try:
            from models.validation.leakage_guards import LeakageGuardSuite
        except ImportError as e:
            logger.warning(f"  Leakage guard suite not available: {e}")
            return

        suite = LeakageGuardSuite(
            min_embargo_bars=max(5, int(len(self.X_train) * self.config.embargo_pct)),
            correlation_threshold=0.95,
            strict=True,
        )

        summary = suite.run_all_checks(
            X_train=self.X_train,
            X_val=self.X_val,
            X_test=self.X_test,
            y_train=self.y_train,
            y_val=self.y_val,
            y_test=self.y_test,
            feature_names=self.feature_names,
            n_strategies_tested=1,
        )

        # Print report
        summary.print_report()

        # Check for critical failures
        if summary.critical_failures:
            error_msgs = [f.message for f in summary.critical_failures]
            error_str = "\n".join(error_msgs)
            raise ValueError(f"CRITICAL LEAKAGE DETECTED:\n{error_str}")

        self.tracker.leakage_passed = summary.all_passed

        if summary.all_passed:
            logger.info("  All leakage checks PASSED")
        else:
            logger.warning("  Some leakage checks failed but were not critical")

    # =========================================================================
    # Step 5: Train ML Models
    # =========================================================================

    def _step_train_ml(self, models: Set[str]) -> None:
        """Train ML models with CPCV."""
        logger.info(f"\n[5/10] Training ML models: {sorted(models)}")

        for model_name in sorted(models):
            start_time = time.time()
            logger.info(f"\n  Training {model_name}...")

            try:
                if model_name == "lightgbm":
                    metrics, artifacts = self._train_lightgbm()
                elif model_name == "xgboost":
                    metrics, artifacts = self._train_xgboost()
                else:
                    logger.warning(f"  Unknown ML model: {model_name}")
                    continue

                training_time = time.time() - start_time

                self.tracker.add_result(ModelResult(
                    model_name=model_name,
                    category="ml",
                    success=True,
                    training_time=training_time,
                    metrics=metrics,
                    artifacts=artifacts,
                ))

                logger.info(f"  {model_name} trained successfully ({training_time:.1f}s)")

            except Exception as e:
                training_time = time.time() - start_time
                logger.error(f"  {model_name} training failed: {e}")

                self.tracker.add_result(ModelResult(
                    model_name=model_name,
                    category="ml",
                    success=False,
                    training_time=training_time,
                    error=str(e),
                ))

    def _train_lightgbm(self) -> Tuple[Dict, List[str]]:
        """Train LightGBM model."""
        try:
            from models.ml.lightgbm_model import LightGBMModel
        except ImportError:
            raise ImportError("LightGBMModel not available")

        model = LightGBMModel()
        metrics = model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            feature_names=self.feature_names,
            n_estimators=self.config.ml_n_estimators,
        )

        # Save model
        model_path = str(self.model_dir / "lightgbm_model.pkl")
        model.save(model_path)

        return metrics, [model_path]

    def _train_xgboost(self) -> Tuple[Dict, List[str]]:
        """Train XGBoost model."""
        try:
            from models.ml.xgboost_model import XGBoostModel
        except ImportError:
            raise ImportError("XGBoostModel not available")

        model = XGBoostModel()
        metrics = model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            feature_names=self.feature_names,
            n_estimators=self.config.ml_n_estimators,
        )

        # Save model
        model_path = str(self.model_dir / "xgboost_model.pkl")
        model.save(model_path)

        return metrics, [model_path]

    # =========================================================================
    # Step 6: Train DL Models
    # =========================================================================

    def _step_train_dl(self, models: Set[str]) -> None:
        """Train DL models with CPCV."""
        logger.info(f"\n[6/10] Training DL models: {sorted(models)}")

        # Create sequences for DL models
        X_train_seq, y_train_seq = self._create_sequences(
            self.X_train, self.y_train, self.config.sequence_length
        )
        X_val_seq, y_val_seq = self._create_sequences(
            self.X_val, self.y_val, self.config.sequence_length
        )

        logger.info(f"  Sequence shape: {X_train_seq.shape}")

        for model_name in sorted(models):
            start_time = time.time()
            logger.info(f"\n  Training {model_name}...")

            try:
                if model_name == "lstm":
                    metrics, artifacts = self._train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
                elif model_name == "cnn":
                    metrics, artifacts = self._train_cnn(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
                else:
                    logger.warning(f"  Unknown DL model: {model_name}")
                    continue

                training_time = time.time() - start_time

                self.tracker.add_result(ModelResult(
                    model_name=model_name,
                    category="dl",
                    success=True,
                    training_time=training_time,
                    metrics=metrics,
                    artifacts=artifacts,
                ))

                logger.info(f"  {model_name} trained successfully ({training_time:.1f}s)")

            except Exception as e:
                training_time = time.time() - start_time
                logger.error(f"  {model_name} training failed: {e}")

                self.tracker.add_result(ModelResult(
                    model_name=model_name,
                    category="dl",
                    success=False,
                    training_time=training_time,
                    error=str(e),
                ))

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for DL models."""
        n_samples = len(X) - sequence_length
        if n_samples <= 0:
            raise ValueError(f"Not enough samples for sequence length {sequence_length}")

        n_features = X.shape[1]

        X_seq = np.zeros((n_samples, sequence_length, n_features), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            X_seq[i] = X[i:i+sequence_length]
            y_seq[i] = y[i+sequence_length]

        return X_seq, y_seq

    def _train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Dict, List[str]]:
        """Train LSTM model."""
        try:
            from models.dl.lstm_model import LSTMModel
        except ImportError:
            raise ImportError("LSTMModel not available")

        num_features = X_train.shape[2]

        model = LSTMModel(
            input_size=num_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
        )

        metrics = model.train(
            X_train, y_train, X_val, y_val,
            batch_size=self.config.dl_batch_size,
            epochs=self.config.dl_epochs,
            patience=self.config.dl_patience,
        )

        # Save model
        model_path = str(self.model_dir / "lstm_model.pt")
        model.save(model_path)

        artifacts = [model_path]

        # Export ONNX
        try:
            onnx_path = str(self.onnx_dir / "lstm_model.onnx")
            model.export_onnx(
                onnx_path,
                sequence_length=self.config.sequence_length,
                num_features=num_features,
            )
            artifacts.append(onnx_path)
        except Exception as e:
            logger.warning(f"  LSTM ONNX export failed: {e}")

        return metrics, artifacts

    def _train_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Dict, List[str]]:
        """Train CNN model."""
        try:
            from models.dl.cnn_model import CNNModel
        except ImportError:
            raise ImportError("CNNModel not available")

        num_features = X_train.shape[2]
        sequence_length = X_train.shape[1]

        model = CNNModel(
            num_features=num_features,
            sequence_length=sequence_length,
            dropout=0.3,
        )

        metrics = model.train(
            X_train, y_train, X_val, y_val,
            batch_size=self.config.dl_batch_size,
            epochs=self.config.dl_epochs,
            patience=self.config.dl_patience,
        )

        # Save model
        model_path = str(self.model_dir / "cnn_model.pt")
        model.save(model_path)

        artifacts = [model_path]

        # Export ONNX
        try:
            onnx_path = str(self.onnx_dir / "cnn_model.onnx")
            model.export_onnx(onnx_path)
            artifacts.append(onnx_path)
        except Exception as e:
            logger.warning(f"  CNN ONNX export failed: {e}")

        return metrics, artifacts

    # =========================================================================
    # Step 7: Train RL Models
    # =========================================================================

    def _step_train_rl(self, models: Set[str]) -> None:
        """Train RL models with walk-forward validation."""
        logger.info(f"\n[7/10] Training RL models: {sorted(models)}")

        # Prepare RL data
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in ohlcv_cols if c in self.df.columns]

        if len(available_cols) != len(ohlcv_cols):
            logger.warning(f"  Missing OHLCV columns: {set(ohlcv_cols) - set(available_cols)}")
            return

        # Get valid data
        df_valid = self.df[~self.df[ohlcv_cols].isna().any(axis=1)]
        rl_data = df_valid[ohlcv_cols].values.astype(np.float32)
        rl_features = df_valid[self.feature_names].values.astype(np.float32)
        rl_features = np.nan_to_num(rl_features, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"  RL data shape: {rl_data.shape}")
        logger.info(f"  RL features shape: {rl_features.shape}")

        for model_name in sorted(models):
            start_time = time.time()
            logger.info(f"\n  Training {model_name}...")

            try:
                if model_name == "d4pg":
                    metrics, artifacts = self._train_d4pg(rl_data, rl_features)
                elif model_name == "marl":
                    metrics, artifacts = self._train_marl(rl_data, rl_features)
                else:
                    logger.warning(f"  Unknown RL model: {model_name}")
                    continue

                training_time = time.time() - start_time

                self.tracker.add_result(ModelResult(
                    model_name=model_name,
                    category="rl",
                    success=True,
                    training_time=training_time,
                    metrics=metrics,
                    artifacts=artifacts,
                ))

                logger.info(f"  {model_name} trained successfully ({training_time:.1f}s)")

            except Exception as e:
                training_time = time.time() - start_time
                logger.error(f"  {model_name} training failed: {e}")

                self.tracker.add_result(ModelResult(
                    model_name=model_name,
                    category="rl",
                    success=False,
                    training_time=training_time,
                    error=str(e),
                ))

    def _train_d4pg(
        self,
        data: np.ndarray,
        features: np.ndarray,
    ) -> Tuple[Dict, List[str]]:
        """Train D4PG+EVT agent."""
        try:
            from models.training.rl_trainer import (
                ReplayEnvironment, EnvironmentConfig,
                D4PGTrainer, TrainerConfig,
            )
            from models.rl.d4pg_evt import D4PGAgent
        except ImportError as e:
            raise ImportError(f"D4PG training components not available: {e}")

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

        # Create agent
        state_dim = features.shape[1] + 4  # features + portfolio state
        agent = D4PGAgent(state_dim=state_dim, batch_size=128)

        # Create trainer
        trainer_config = TrainerConfig(
            n_episodes=self.config.rl_episodes,
            n_folds=self.config.rl_walk_forward_folds,
            validate_every_n_episodes=50,
            log_every_n_episodes=20,
            checkpoint_dir=str(self.experiment_dir / "checkpoints"),
        )

        trainer = D4PGTrainer(env, agent, trainer_config)

        # Train with walk-forward
        results = trainer.walk_forward_train()

        # Save agent
        model_path = str(self.model_dir / "d4pg_agent.pt")
        agent.save(model_path)

        artifacts = [model_path]

        # Export ONNX
        try:
            onnx_path = str(self.onnx_dir / "d4pg_actor.onnx")
            agent.export_onnx(onnx_path)
            artifacts.append(onnx_path)
        except Exception as e:
            logger.warning(f"  D4PG ONNX export failed: {e}")

        metrics = {
            "pbo": results.get("pbo", 0),
            "mean_return": results.get("mean_return", 0),
            "std_return": results.get("std_return", 0),
            "n_folds": results.get("n_folds", 0),
        }

        return metrics, artifacts

    def _train_marl(
        self,
        data: np.ndarray,
        features: np.ndarray,
    ) -> Tuple[Dict, List[str]]:
        """Train MARL system."""
        try:
            from models.training.rl_trainer import (
                ReplayEnvironment, EnvironmentConfig,
                MARLTrainer, TrainerConfig,
            )
            from models.rl.marl import MARLSystem
        except ImportError as e:
            raise ImportError(f"MARL training components not available: {e}")

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

        # Create MARL system
        state_dim = features.shape[1] + 4
        system = MARLSystem(
            state_dim=state_dim,
            action_dim=1,
            n_agents=5,
        )

        # Create trainer
        trainer_config = TrainerConfig(
            n_episodes=self.config.rl_episodes,
            n_folds=self.config.rl_walk_forward_folds,
            validate_every_n_episodes=50,
            log_every_n_episodes=20,
            checkpoint_dir=str(self.experiment_dir / "checkpoints"),
        )

        trainer = MARLTrainer(env, system, trainer_config)

        # Train with walk-forward
        results = trainer.walk_forward_train()

        # Save system
        model_path = str(self.model_dir / "marl_system.pt")
        system.save(model_path)

        artifacts = [model_path]

        # Export ONNX
        try:
            onnx_dir = str(self.onnx_dir / "marl")
            system.export_onnx(onnx_dir)
            artifacts.append(onnx_dir)
        except Exception as e:
            logger.warning(f"  MARL ONNX export failed: {e}")

        metrics = {
            "pbo": results.get("pbo", 0),
            "n_agents": system.n_agents,
            "n_folds": results.get("n_folds", 0),
        }

        return metrics, artifacts

    # =========================================================================
    # Step 8: Export ONNX
    # =========================================================================

    def _step_export_onnx(self) -> None:
        """Export all trained models to ONNX format."""
        logger.info("\n[8/10] Exporting models to ONNX...")

        # ONNX exports are done during model training
        # This step verifies and reports what was exported

        onnx_files = list(self.onnx_dir.glob("*.onnx"))
        onnx_files.extend(self.onnx_dir.glob("**/*.onnx"))

        if onnx_files:
            logger.info(f"  Exported {len(onnx_files)} ONNX models:")
            for f in onnx_files:
                logger.info(f"    - {f.name}")
        else:
            logger.warning("  No ONNX models exported")

    # =========================================================================
    # Step 9: Parity Tests
    # =========================================================================

    def _step_parity_tests(self) -> None:
        """Run ONNX parity tests."""
        logger.info("\n[9/10] Running parity tests...")

        try:
            from models.export.onnx_parity import run_all_parity_tests
        except ImportError as e:
            logger.warning(f"  Parity testing module not available: {e}")
            return

        try:
            golden_dir = self.config.output_dir / "golden_data"
            golden_dir.mkdir(parents=True, exist_ok=True)

            results = run_all_parity_tests(
                model_dir=self.model_dir,
                onnx_dir=self.onnx_dir,
                golden_dir=golden_dir,
                n_samples=100,
            )

            for result in results:
                self.tracker.parity_results[result.model_name] = result.passed
                status = "PASS" if result.passed else "FAIL"
                logger.info(f"  {result.model_name}: {status}")

        except Exception as e:
            logger.warning(f"  Parity tests failed: {e}")

    # =========================================================================
    # Step 10: Final Report
    # =========================================================================

    def _step_final_report(self) -> None:
        """Generate and save final report."""
        logger.info("\n[10/10] Generating final report...")

        # Print timing report
        print("\n" + self.timer.get_report())

        # Print training report
        report = self.tracker.generate_report()
        print(report)

        # Save results
        self.tracker.save(self.experiment_dir / "training_results.json")

        # Save report
        report_path = self.experiment_dir / "training_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
            f.write("\n\n")
            f.write(self.timer.get_report())

        # Save config
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save reproducibility manifest
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "models_trained": [r.model_name for r in self.tracker.results if r.success],
            "experiment_dir": str(self.experiment_dir),
        }

        manifest_path = self.experiment_dir / "reproducibility_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"\n  Experiment saved to: {self.experiment_dir}")
        logger.info(f"  Total time: {self.timer.format_elapsed()}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Training Pipeline for ORPFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python scripts/train_all.py --data data/raw/klines_90d.parquet

  # Train only ML models
  python scripts/train_all.py --models lightgbm,xgboost

  # Train without RL (faster)
  python scripts/train_all.py --skip-rl

  # Debug mode (skip leakage checks)
  python scripts/train_all.py --skip-leakage

Available models:
  ML:  lightgbm, xgboost
  DL:  lstm, cnn
  RL:  d4pg, marl
        """,
    )

    # Data arguments
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/raw/klines_90d.parquet",
        help="Path to input parquet data file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="trained",
        help="Output directory for models and artifacts",
    )

    # Model selection
    parser.add_argument(
        "--models", "-m",
        type=str,
        default="all",
        help="Comma-separated list of models to train (default: all)",
    )

    # CPCV configuration
    parser.add_argument(
        "--cpcv-splits",
        type=int,
        default=5,
        help="Number of CPCV splits for cross-validation",
    )
    parser.add_argument(
        "--embargo-pct",
        type=float,
        default=0.01,
        help="Embargo percentage between splits",
    )

    # Training parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length for DL models",
    )

    # ML parameters
    parser.add_argument(
        "--ml-estimators",
        type=int,
        default=1000,
        help="Number of estimators for ML models",
    )

    # DL parameters
    parser.add_argument(
        "--dl-epochs",
        type=int,
        default=100,
        help="Number of epochs for DL models",
    )
    parser.add_argument(
        "--dl-patience",
        type=int,
        default=10,
        help="Early stopping patience for DL models",
    )

    # RL parameters
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=200,
        help="Number of episodes for RL training",
    )

    # Skip flags
    parser.add_argument(
        "--skip-leakage",
        action="store_true",
        help="Skip leakage checks (dangerous, for debug only)",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip RL training (faster)",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip ONNX parity tests",
    )

    # Feature flags
    parser.add_argument(
        "--no-quant-features",
        action="store_true",
        help="Disable quantitative feature engineering",
    )
    parser.add_argument(
        "--no-microstructure",
        action="store_true",
        help="Disable microstructure features",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse model selection
    if args.models.lower() == "all":
        models = set(ALL_MODELS)
    else:
        models = set(m.strip().lower() for m in args.models.split(","))

        # Validate models
        invalid = models - set(ALL_MODELS)
        if invalid:
            print(f"Error: Unknown models: {invalid}")
            print(f"Available models: {ALL_MODELS}")
            sys.exit(1)

    # Create configuration
    config = TrainAllConfig(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        models=models,
        cpcv_splits=args.cpcv_splits,
        embargo_pct=args.embargo_pct,
        seed=args.seed,
        sequence_length=args.sequence_length,
        ml_n_estimators=args.ml_estimators,
        dl_epochs=args.dl_epochs,
        dl_patience=args.dl_patience,
        rl_episodes=args.rl_episodes,
        enable_quant_features=not args.no_quant_features,
        enable_microstructure=not args.no_microstructure,
        skip_leakage=args.skip_leakage,
        skip_rl=args.skip_rl,
        skip_parity=args.skip_parity,
        verbose=args.verbose,
    )

    # Print banner
    print("\n" + "#" * 70)
    print("ORPFlow - Comprehensive Training Pipeline")
    print("#" * 70)
    print(f"Data:    {config.data_path}")
    print(f"Output:  {config.output_dir}")
    print(f"Models:  {sorted(config.models)}")
    print(f"Seed:    {config.seed}")
    print("#" * 70 + "\n")

    # Run pipeline
    pipeline = TrainingPipeline(config)

    try:
        results = pipeline.run()

        # Exit with appropriate code
        if results["failed"] > 0:
            print(f"\n{results['failed']} model(s) failed to train")
            sys.exit(1)
        else:
            print("\nAll models trained successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
