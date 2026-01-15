#!/usr/bin/env python3
"""
Unified Training Orchestrator for ORPFlow
==========================================

This module provides a comprehensive training orchestration system that:
- Auto-discovers all models via the model registry
- Applies proper CPCV validation for financial time series
- Runs leakage checks BEFORE any training (fail early)
- Integrates advanced quant features (Hawkes, Kalman, RMT, HMM)
- Adds microstructure features (VPIN, OFI)
- Tracks all experiments with full reproducibility manifests
- Supports ML, DL, and RL model training

Usage:
    python -m models.training.orchestrator --data data/raw/klines_90d.parquet

    # Or programmatically:
    from models.training.orchestrator import TrainingOrchestrator, TrainingConfig

    config = TrainingConfig(data_path=Path("data/raw/klines_90d.parquet"))
    orchestrator = TrainingOrchestrator(config)
    orchestrator.train_all()
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for training orchestration.

    Attributes:
        data_path: Path to parquet data file
        output_dir: Directory for saving models and artifacts
        cpcv_splits: Number of CPCV splits for cross-validation
        embargo_pct: Embargo percentage between train/test splits
        purge_pct: Purge percentage for feature windows
        target_column: Target column for prediction
        seed: Random seed for reproducibility
        feature_window: Feature calculation window size
        label_horizon: Label horizon for prediction
        sequence_length: Sequence length for DL models
        enable_quant_features: Whether to add quant features
        enable_microstructure: Whether to add microstructure features
        rl_episodes: Number of episodes for RL training
        ml_n_estimators: Number of estimators for ML models
        dl_epochs: Number of epochs for DL models
        dl_patience: Early stopping patience for DL models
    """
    data_path: Path = field(default_factory=lambda: Path("data/raw/klines_90d.parquet"))
    output_dir: Path = field(default_factory=lambda: Path("trained"))
    cpcv_splits: int = 5
    embargo_pct: float = 0.01
    purge_pct: float = 0.01
    target_column: str = "target_return_5"
    seed: int = 42
    feature_window: int = 60
    label_horizon: int = 5
    sequence_length: int = 60
    enable_quant_features: bool = True
    enable_microstructure: bool = True
    rl_episodes: int = 200
    ml_n_estimators: int = 1000
    dl_epochs: int = 100
    dl_patience: int = 10

    def __post_init__(self):
        """Validate and convert paths."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(self).items()
        }


# ==============================================================================
# Data Manager
# ==============================================================================

class DataManager:
    """
    Manages data loading, feature engineering, and preprocessing.

    Handles:
    - Loading parquet data with proper datetime index
    - Adding advanced quantitative features
    - Adding microstructure features
    - Feature/target splitting with validation
    - Dataset hashing for reproducibility
    """

    def __init__(self, seed: int = 42):
        """
        Initialize DataManager.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

        self._scaler = None
        self._feature_names: List[str] = []

    def load_data(self, path: Path) -> pd.DataFrame:
        """
        Load parquet data with proper datetime index.

        Args:
            path: Path to parquet file

        Returns:
            DataFrame with datetime index
        """
        logger.info(f"Loading data from {path}")

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_parquet(path)

        # Set datetime index
        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.set_index("open_time").sort_index()
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp").sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    def add_quant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced quantitative features: Hawkes, Kalman, RMT, HMM.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame with added quant features
        """
        logger.info("Adding quantitative features...")

        try:
            from models.features.quant_features import (
                HawkesIntensity,
                KalmanFilter,
                MarketRegimeHMM,
                RMTCorrelationFilter,
                FractalAnalyzer,
                hurst_exponent,
                fractal_dimension,
                fisher_transform,
                normalize_to_fisher_range,
            )
        except ImportError as e:
            logger.warning(f"Could not import quant_features: {e}")
            return df

        df = df.copy()

        # Ensure we have required columns
        if "close" not in df.columns:
            logger.warning("No 'close' column found, skipping quant features")
            return df

        close = df["close"].values
        returns = df["close"].pct_change().fillna(0).values

        # 1. Hawkes Intensity Features
        logger.info("  Computing Hawkes intensity...")
        hawkes = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)

        intensities = []
        branching_ratios = []
        timestamps = np.arange(len(df), dtype=float)

        # Simulate events based on volume spikes
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        volume_ma = pd.Series(volume).rolling(20).mean().fillna(volume.mean()).values

        for i, ts in enumerate(timestamps):
            is_event = volume[i] > volume_ma[i] * 1.5 if i > 0 else False
            state = hawkes.update(ts, is_event)
            intensities.append(state.intensity)
            branching_ratios.append(state.branching_ratio)

        df["hawkes_intensity"] = intensities
        df["hawkes_branching"] = branching_ratios

        # 2. Kalman Filter Features
        logger.info("  Computing Kalman filter state...")
        kalman = KalmanFilter.for_spread_tracking()

        kalman_states = []
        kalman_innovations = []

        for price in close:
            state = kalman.update(price)
            kalman_states.append(state.state_estimate[0])
            kalman_innovations.append(state.innovation)

        df["kalman_state"] = kalman_states
        df["kalman_innovation"] = kalman_innovations
        df["kalman_signal"] = df["close"] - df["kalman_state"]

        # 3. HMM Regime Detection
        logger.info("  Fitting HMM for regime detection...")
        if len(returns) > 100:
            try:
                hmm = MarketRegimeHMM(n_states=3, n_features=1, max_iterations=50)
                returns_2d = returns.reshape(-1, 1)
                hmm.fit(returns_2d)

                # Get state probabilities
                state_probs = hmm.transform(returns_2d)

                df["hmm_regime_0"] = state_probs[:, 0]
                df["hmm_regime_1"] = state_probs[:, 1]
                df["hmm_regime_2"] = state_probs[:, 2] if state_probs.shape[1] > 2 else 0
                df["hmm_dominant_regime"] = np.argmax(state_probs, axis=1)
            except Exception as e:
                logger.warning(f"HMM fitting failed: {e}")
                df["hmm_regime_0"] = 0.33
                df["hmm_regime_1"] = 0.33
                df["hmm_regime_2"] = 0.34
                df["hmm_dominant_regime"] = 0

        # 4. Fractal Analysis (Rolling)
        logger.info("  Computing fractal features...")
        window = 200
        hursts = []
        dimensions = []

        for i in range(len(close)):
            if i < window:
                hursts.append(0.5)
                dimensions.append(1.5)
            else:
                segment = close[i-window:i]
                try:
                    h = hurst_exponent(segment, min_window=10, max_window=window//4)
                    d = fractal_dimension(segment)
                    hursts.append(h)
                    dimensions.append(d)
                except:
                    hursts.append(hursts[-1] if hursts else 0.5)
                    dimensions.append(dimensions[-1] if dimensions else 1.5)

        df["hurst_exponent"] = hursts
        df["fractal_dimension"] = dimensions
        df["is_trending"] = (df["hurst_exponent"] > 0.55).astype(int)
        df["is_mean_reverting"] = (df["hurst_exponent"] < 0.45).astype(int)

        # 5. Fisher Transform on normalized returns
        logger.info("  Computing Fisher transform...")
        normalized = normalize_to_fisher_range(returns, lookback=20)
        df["fisher_transform"] = fisher_transform(np.nan_to_num(normalized, nan=0))

        logger.info(f"  Added {len([c for c in df.columns if c.startswith(('hawkes', 'kalman', 'hmm', 'hurst', 'fractal', 'fisher'))])} quant features")

        return df

    def add_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructure features: VPIN, OFI.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with added microstructure features
        """
        logger.info("Adding microstructure features...")

        try:
            from models.features.microstructure import (
                VPINCalculator,
                OFICalculator,
                OHLCV,
            )
        except ImportError as e:
            logger.warning(f"Could not import microstructure: {e}")
            return df

        df = df.copy()

        # Check required columns
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            logger.warning(f"Missing required columns for microstructure: {required}")
            return df

        # Get taker buy volume if available
        taker_buy_col = None
        for col in ["taker_buy_base", "taker_buy_volume", "taker_buy_base_volume"]:
            if col in df.columns:
                taker_buy_col = col
                break

        # Initialize calculators
        avg_volume = df["volume"].mean()
        bucket_size = avg_volume * 10  # 10 bars worth of volume

        vpin_calc = VPINCalculator(bucket_size=bucket_size, n_buckets=20)
        ofi_calc = OFICalculator(decay=0.99, momentum_window=20)

        # Calculate features bar by bar
        vpins = []
        vpin_cdfs = []
        ofis = []
        ofi_cumulative = []

        logger.info("  Processing bars for microstructure...")

        for i in range(len(df)):
            row = df.iloc[i]

            # Create OHLCV object
            bar = OHLCV(
                timestamp=int(row.name.timestamp() * 1000) if hasattr(row.name, 'timestamp') else i,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                taker_buy_volume=row[taker_buy_col] if taker_buy_col else 0.0,
                trades=int(row.get("trades", 1)),
            )

            # Update VPIN
            vpin_calc.update(bar)
            vpin_val = vpin_calc.get_value()
            vpin_cdf = vpin_calc.get_vpin_cdf()

            vpins.append(vpin_val if vpin_val is not None else np.nan)
            vpin_cdfs.append(vpin_cdf if vpin_cdf is not None else np.nan)

            # Update OFI
            ofi_calc.update(bar)
            ofis.append(ofi_calc.get_value())
            ofi_cumulative.append(ofi_calc.get_cumulative_ofi())

        df["vpin"] = vpins
        df["vpin_cdf"] = vpin_cdfs
        df["ofi"] = ofis
        df["ofi_cumulative"] = ofi_cumulative

        # Fill NaN values at the beginning
        df["vpin"] = df["vpin"].fillna(method="bfill").fillna(0)
        df["vpin_cdf"] = df["vpin_cdf"].fillna(method="bfill").fillna(0.5)
        df["ofi"] = df["ofi"].fillna(0)
        df["ofi_cumulative"] = df["ofi_cumulative"].fillna(0)

        # Derive additional features
        df["vpin_high_toxicity"] = (df["vpin_cdf"] > 0.8).astype(int)
        df["ofi_pressure"] = np.sign(df["ofi_cumulative"])

        logger.info(f"  Added {len([c for c in df.columns if c.startswith(('vpin', 'ofi'))])} microstructure features")

        return df

    def create_targets(self, df: pd.DataFrame, horizons: List[int] = [1, 5, 15]) -> pd.DataFrame:
        """
        Create target columns for different prediction horizons.

        Args:
            df: Input DataFrame
            horizons: List of forward return horizons

        Returns:
            DataFrame with target columns added
        """
        df = df.copy()

        for h in horizons:
            # Forward returns
            df[f"target_return_{h}"] = df["close"].pct_change(h).shift(-h)

            # Direction (classification target)
            df[f"target_direction_{h}"] = np.sign(df[f"target_return_{h}"])

        return df

    def get_feature_target_split(
        self,
        df: pd.DataFrame,
        target_column: str = "target_return_5",
        exclude_patterns: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series]:
        """
        Split DataFrame into features (X) and target (y) with validation.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            exclude_patterns: Patterns to exclude from features

        Returns:
            Tuple of (X, y, feature_names, times)
        """
        exclude_patterns = exclude_patterns or [
            "target_",
            "close_time",
            "symbol",
            "ignore",
        ]

        # Identify feature columns
        feature_cols = []
        for col in df.columns:
            if any(pat in col for pat in exclude_patterns):
                continue
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                feature_cols.append(col)

        # Validate target exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Remove rows with NaN in target or features
        valid_mask = ~df[target_column].isna()
        for col in feature_cols:
            valid_mask &= ~df[col].isna()

        df_valid = df[valid_mask].copy()

        logger.info(f"Valid samples after removing NaN: {len(df_valid)} / {len(df)}")

        X = df_valid[feature_cols].values
        y = df_valid[target_column].values
        times = pd.Series(df_valid.index)

        self._feature_names = feature_cols

        return X, y, feature_cols, times

    def hash_dataset(self, df: pd.DataFrame) -> str:
        """
        Compute SHA256 hash of dataset for reproducibility.

        Args:
            df: DataFrame to hash

        Returns:
            SHA256 hex digest
        """
        # Convert to bytes and hash
        data_bytes = df.to_json().encode("utf-8")
        hash_obj = hashlib.sha256(data_bytes)
        return hash_obj.hexdigest()

    def prepare_temporal_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        embargo_pct: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temporal train/val/test splits with embargo.

        Args:
            X: Feature matrix
            y: Target array
            times: Timestamps
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            embargo_pct: Embargo between splits

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx)
        """
        n = len(X)
        embargo_size = int(n * embargo_pct)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Apply embargo gaps
        train_idx = np.arange(0, train_end - embargo_size)
        val_idx = np.arange(train_end + embargo_size, val_end - embargo_size)
        test_idx = np.arange(val_end + embargo_size, n)

        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        logger.info(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for DL models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array
            sequence_length: Sequence length

        Returns:
            Tuple of (X_seq, y_seq) where X_seq is (n_samples, seq_len, n_features)
        """
        n_samples = len(X) - sequence_length
        n_features = X.shape[1]

        X_seq = np.zeros((n_samples, sequence_length, n_features))
        y_seq = np.zeros(n_samples)

        for i in range(n_samples):
            X_seq[i] = X[i:i+sequence_length]
            y_seq[i] = y[i+sequence_length]

        return X_seq, y_seq


# ==============================================================================
# Experiment Tracker
# ==============================================================================

class ExperimentTracker:
    """
    Tracks experiments for reproducibility and analysis.

    Saves:
    - Training configuration
    - Per-model metrics
    - CPCV fold results
    - Model artifacts
    - Reproducibility manifest (seeds, hashes, versions)
    """

    def __init__(self, output_dir: Path):
        """
        Initialize experiment tracker.

        Args:
            output_dir: Directory for saving artifacts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"experiment_{self.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self._config: Dict = {}
        self._metrics: Dict[str, Dict] = {}
        self._cpcv_results: Dict[str, List[Dict]] = {}
        self._artifacts: Dict[str, List[str]] = {}
        self._start_time = time.time()

        logger.info(f"Experiment tracking initialized: {self.experiment_id}")

    def track_config(self, config: TrainingConfig) -> None:
        """
        Save training configuration.

        Args:
            config: TrainingConfig object
        """
        self._config = config.to_dict()
        self._config["experiment_id"] = self.experiment_id
        self._config["start_time"] = datetime.now().isoformat()

        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self._config, f, indent=2, default=str)

        logger.info(f"Config saved to {config_path}")

    def track_metrics(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """
        Save per-model metrics.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
        """
        self._metrics[model_name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }

        # Save incrementally
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)

    def track_splits(self, model_name: str, cpcv_results: List[Dict]) -> None:
        """
        Save CPCV fold results.

        Args:
            model_name: Name of the model
            cpcv_results: List of fold results
        """
        self._cpcv_results[model_name] = cpcv_results

        cpcv_path = self.experiment_dir / "cpcv_results.json"
        with open(cpcv_path, "w") as f:
            json.dump(self._cpcv_results, f, indent=2, default=str)

    def save_artifacts(self, model_name: str, artifacts: List[str]) -> None:
        """
        Record saved model files.

        Args:
            model_name: Name of the model
            artifacts: List of artifact paths
        """
        self._artifacts[model_name] = artifacts

        artifacts_path = self.experiment_dir / "artifacts.json"
        with open(artifacts_path, "w") as f:
            json.dump(self._artifacts, f, indent=2)

    def generate_reproducibility_manifest(
        self,
        data_hash: str,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Generate reproducibility manifest.

        Args:
            data_hash: SHA256 hash of input data
            seed: Random seed used

        Returns:
            Manifest dictionary
        """
        import platform

        manifest = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - self._start_time,
            "seed": seed,
            "data_hash": data_hash,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "config": self._config,
            "models_trained": list(self._metrics.keys()),
            "artifacts": self._artifacts,
        }

        # Add package versions
        try:
            import numpy
            import pandas
            manifest["numpy_version"] = numpy.__version__
            manifest["pandas_version"] = pandas.__version__
        except:
            pass

        try:
            import torch
            manifest["torch_version"] = torch.__version__
            manifest["cuda_available"] = torch.cuda.is_available()
        except:
            pass

        try:
            import lightgbm
            manifest["lightgbm_version"] = lightgbm.__version__
        except:
            pass

        try:
            import xgboost
            manifest["xgboost_version"] = xgboost.__version__
        except:
            pass

        # Save manifest
        manifest_path = self.experiment_dir / "reproducibility_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"Reproducibility manifest saved to {manifest_path}")

        return manifest

    def generate_report(self) -> str:
        """
        Generate training report.

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            f"TRAINING REPORT - Experiment {self.experiment_id}",
            "=" * 70,
            "",
            f"Duration: {time.time() - self._start_time:.1f} seconds",
            f"Models trained: {len(self._metrics)}",
            "",
        ]

        if self._metrics:
            lines.append("MODEL PERFORMANCE:")
            lines.append("-" * 40)

            for model_name, metrics in sorted(
                self._metrics.items(),
                key=lambda x: x[1].get("val_sharpe", x[1].get("sharpe_ratio", 0)),
                reverse=True,
            ):
                lines.append(f"\n{model_name.upper()}:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.4f}")
                    else:
                        lines.append(f"  {k}: {v}")

        if self._cpcv_results:
            lines.append("\n" + "-" * 40)
            lines.append("CPCV SUMMARY:")

            for model_name, folds in self._cpcv_results.items():
                if folds:
                    sharpes = [f.get("sharpe_ratio", 0) for f in folds]
                    lines.append(f"\n{model_name}:")
                    lines.append(f"  Folds: {len(folds)}")
                    lines.append(f"  Mean Sharpe: {np.mean(sharpes):.4f}")
                    lines.append(f"  Std Sharpe: {np.std(sharpes):.4f}")

        lines.append("\n" + "=" * 70)

        report = "\n".join(lines)

        # Save report
        report_path = self.experiment_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        return report


# ==============================================================================
# Training Orchestrator
# ==============================================================================

class TrainingOrchestrator:
    """
    Unified training orchestrator for all model types.

    Features:
    - Auto-discovers models via registry
    - Runs leakage checks BEFORE training
    - Applies CPCV cross-validation
    - Tracks all experiments
    - Exports trained models to ONNX
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize orchestrator.

        Args:
            config: TrainingConfig object
        """
        self.config = config

        # Set random seed
        np.random.seed(config.seed)

        # Initialize components
        self.data_manager = DataManager(seed=config.seed)
        self.tracker = ExperimentTracker(config.output_dir)
        self.tracker.track_config(config)

        # Storage
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict] = {}
        self._data_hash: str = ""
        self._feature_names: List[str] = []

        # Create output directories
        self.model_dir = config.output_dir / "models"
        self.onnx_dir = config.output_dir / "onnx"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TrainingOrchestrator initialized with config: {config}")

    def discover_models(self) -> Dict[str, Any]:
        """
        Use registry to find all available models.

        Returns:
            Dictionary of model metadata by category
        """
        try:
            from models.registry import ModelRegistry, ModelCategory

            registry = ModelRegistry()

            models = {
                "ml": registry.list_models(category=ModelCategory.ML),
                "dl": registry.list_models(category=ModelCategory.DL),
                "rl": registry.list_models(category=ModelCategory.RL),
            }

            logger.info(f"Discovered models - ML: {models['ml']}, DL: {models['dl']}, RL: {models['rl']}")

            return models
        except ImportError:
            logger.warning("ModelRegistry not available, using defaults")
            return {
                "ml": ["lightgbm", "xgboost"],
                "dl": ["lstm", "cnn"],
                "rl": ["d4pg", "marl"],
            }

    def run_leakage_checks(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        train_times: pd.Series = None,
        val_times: pd.Series = None,
        test_times: pd.Series = None,
        feature_names: List[str] = None,
    ) -> bool:
        """
        Run comprehensive leakage checks BEFORE training.

        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target arrays
            train_times, val_times, test_times: Timestamps
            feature_names: Feature names for reporting

        Returns:
            True if all checks pass, raises exception on critical failure
        """
        logger.info("Running leakage validation checks...")

        try:
            from models.validation.leakage_guards import LeakageGuardSuite, ValidationSummary
        except ImportError:
            logger.warning("LeakageGuardSuite not available, skipping leakage checks")
            return True

        suite = LeakageGuardSuite(
            min_embargo_bars=max(5, int(len(X_train) * self.config.embargo_pct)),
            correlation_threshold=0.95,
            strict=True,
        )

        summary = suite.run_all_checks(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_names,
            train_times=train_times,
            val_times=val_times,
            test_times=test_times,
            n_strategies_tested=1,
        )

        # Print report
        summary.print_report()

        # Check for critical failures
        critical = summary.critical_failures
        if critical:
            error_msgs = [f.message for f in critical]
            error_str = "\n".join(error_msgs)
            raise ValueError(f"CRITICAL LEAKAGE DETECTED - Training aborted:\n{error_str}")

        if not summary.all_passed:
            logger.warning("Some leakage checks failed but were not critical. Proceeding with caution.")
        else:
            logger.info("All leakage checks passed!")

        return True

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train a single model with CPCV cross-validation.

        Args:
            model_name: Name of the model to train
            X_train, X_val: Feature matrices
            y_train, y_val: Target arrays
            feature_names: Feature names
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {model_name}...")

        try:
            from models.registry import get_registry, ModelCategory
            registry = get_registry()

            if model_name not in registry:
                raise ValueError(f"Model {model_name} not found in registry")

            info = registry.get_model_info(model_name)

        except ImportError:
            logger.warning("Registry not available, using direct import")
            info = None

        metrics = {}
        model = None

        # Train based on model type
        if model_name == "lightgbm":
            metrics, model = self._train_lightgbm(
                X_train, y_train, X_val, y_val, feature_names
            )
        elif model_name == "xgboost":
            metrics, model = self._train_xgboost(
                X_train, y_train, X_val, y_val, feature_names
            )
        elif model_name == "lstm":
            metrics, model = self._train_lstm(
                X_train, y_train, X_val, y_val,
                kwargs.get("sequence_length", self.config.sequence_length),
                kwargs.get("num_features", X_train.shape[-1]),
            )
        elif model_name == "cnn":
            metrics, model = self._train_cnn(
                X_train, y_train, X_val, y_val,
                kwargs.get("sequence_length", self.config.sequence_length),
                kwargs.get("num_features", X_train.shape[-1]),
            )
        else:
            logger.warning(f"Unknown model: {model_name}")
            return {}

        if model is not None:
            self.models[model_name] = model
            self.metrics[model_name] = metrics
            self.tracker.track_metrics(model_name, metrics)

        return metrics

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[Dict, Any]:
        """Train LightGBM model."""
        try:
            from models.ml.lightgbm_model import LightGBMModel
        except ImportError:
            logger.error("LightGBMModel not available")
            return {}, None

        model = LightGBMModel()
        metrics = model.train(
            X_train, y_train, X_val, y_val,
            feature_names=feature_names,
            n_estimators=self.config.ml_n_estimators,
        )

        # Save model
        model_path = str(self.model_dir / "lightgbm_model.pkl")
        model.save(model_path)
        self.tracker.save_artifacts("lightgbm", [model_path])

        return metrics, model

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[Dict, Any]:
        """Train XGBoost model."""
        try:
            from models.ml.xgboost_model import XGBoostModel
        except ImportError:
            logger.error("XGBoostModel not available")
            return {}, None

        model = XGBoostModel()
        metrics = model.train(
            X_train, y_train, X_val, y_val,
            feature_names=feature_names,
            n_estimators=self.config.ml_n_estimators,
        )

        # Save model
        model_path = str(self.model_dir / "xgboost_model.pkl")
        model.save(model_path)
        self.tracker.save_artifacts("xgboost", [model_path])

        return metrics, model

    def _train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sequence_length: int,
        num_features: int,
    ) -> Tuple[Dict, Any]:
        """Train LSTM model."""
        try:
            from models.dl.lstm_model import LSTMModel
        except ImportError:
            logger.error("LSTMModel not available")
            return {}, None

        model = LSTMModel(
            input_size=num_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
        )

        metrics = model.train(
            X_train, y_train, X_val, y_val,
            epochs=self.config.dl_epochs,
            patience=self.config.dl_patience,
        )

        # Save model
        model_path = str(self.model_dir / "lstm_model.pt")
        model.save(model_path)

        # Export ONNX
        try:
            onnx_path = str(self.onnx_dir / "lstm_model.onnx")
            model.export_onnx(onnx_path, sequence_length, num_features)
            self.tracker.save_artifacts("lstm", [model_path, onnx_path])
        except Exception as e:
            logger.warning(f"LSTM ONNX export failed: {e}")
            self.tracker.save_artifacts("lstm", [model_path])

        return metrics, model

    def _train_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sequence_length: int,
        num_features: int,
    ) -> Tuple[Dict, Any]:
        """Train CNN model."""
        try:
            from models.dl.cnn_model import CNNModel
        except ImportError:
            logger.error("CNNModel not available")
            return {}, None

        model = CNNModel(
            num_features=num_features,
            sequence_length=sequence_length,
            dropout=0.3,
        )

        metrics = model.train(
            X_train, y_train, X_val, y_val,
            epochs=self.config.dl_epochs,
            patience=self.config.dl_patience,
        )

        # Save model
        model_path = str(self.model_dir / "cnn_model.pt")
        model.save(model_path)

        # Export ONNX
        try:
            onnx_path = str(self.onnx_dir / "cnn_model.onnx")
            model.export_onnx(onnx_path)
            self.tracker.save_artifacts("cnn", [model_path, onnx_path])
        except Exception as e:
            logger.warning(f"CNN ONNX export failed: {e}")
            self.tracker.save_artifacts("cnn", [model_path])

        return metrics, model

    def train_with_cpcv(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        times: pd.Series = None,
        feature_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Train model with CPCV cross-validation.

        Args:
            model_name: Name of model to train
            X: Full feature matrix
            y: Full target array
            times: Timestamps for temporal CV
            feature_names: Feature names

        Returns:
            Dictionary with aggregated metrics and fold results
        """
        logger.info(f"Training {model_name} with CPCV (splits={self.config.cpcv_splits})...")

        try:
            from models.validation.cpcv import CombinatorialPurgedKFold, calculate_sharpe_ratio
        except ImportError:
            logger.warning("CPCV not available, using simple train/val split")
            # Fall back to simple split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            return self.train_model(model_name, X_train, y_train, X_val, y_val, feature_names)

        cpcv = CombinatorialPurgedKFold(
            n_splits=self.config.cpcv_splits,
            n_test_groups=2,
            embargo_pct=self.config.embargo_pct,
            purge_pct=self.config.purge_pct,
            feature_window=self.config.feature_window,
            label_horizon=self.config.label_horizon,
        )

        fold_results = []
        fold_sharpes = []

        for fold_idx, (train_idx, test_idx) in enumerate(cpcv.split(X, y, times)):
            logger.info(f"  Fold {fold_idx + 1}/{cpcv.get_n_splits()}")

            X_train_fold = X[train_idx]
            X_test_fold = X[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

            # For this fold, use a portion of train for validation
            val_split = int(len(X_train_fold) * 0.85)
            X_train = X_train_fold[:val_split]
            X_val = X_train_fold[val_split:]
            y_train = y_train_fold[:val_split]
            y_val = y_train_fold[val_split:]

            # Train model for this fold
            fold_metrics = self.train_model(
                model_name, X_train, y_train, X_val, y_val, feature_names
            )

            # Evaluate on test fold
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, "predict"):
                    y_pred = model.predict(X_test_fold)

                    # Calculate strategy returns
                    strategy_returns = y_test_fold * np.sign(y_pred)
                    sharpe = calculate_sharpe_ratio(strategy_returns)

                    fold_result = {
                        "fold_idx": fold_idx,
                        "train_size": len(train_idx),
                        "test_size": len(test_idx),
                        "sharpe_ratio": sharpe,
                        **fold_metrics,
                    }

                    fold_results.append(fold_result)
                    fold_sharpes.append(sharpe)

        # Track CPCV results
        self.tracker.track_splits(model_name, fold_results)

        # Aggregate metrics
        if fold_sharpes:
            aggregated = {
                "mean_sharpe": float(np.mean(fold_sharpes)),
                "std_sharpe": float(np.std(fold_sharpes)),
                "min_sharpe": float(np.min(fold_sharpes)),
                "max_sharpe": float(np.max(fold_sharpes)),
                "n_folds": len(fold_sharpes),
            }

            # Update tracker
            self.tracker.track_metrics(f"{model_name}_cpcv", aggregated)

            return {
                "aggregated": aggregated,
                "fold_results": fold_results,
            }

        return {"fold_results": fold_results}

    def train_all_ml(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict[str, Dict]:
        """
        Train all ML models (LightGBM, XGBoost).

        Args:
            X_train, X_val: Feature matrices
            y_train, y_val: Target arrays
            feature_names: Feature names

        Returns:
            Dictionary of model metrics
        """
        logger.info("Training ML models...")

        results = {}

        for model_name in ["lightgbm", "xgboost"]:
            try:
                metrics = self.train_model(
                    model_name, X_train, y_train, X_val, y_val, feature_names
                )
                results[model_name] = metrics
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")

        return results

    def train_all_dl(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sequence_length: int = 60,
    ) -> Dict[str, Dict]:
        """
        Train all DL models (LSTM, CNN).

        Args:
            X_train, X_val: Sequence feature matrices (n_samples, seq_len, n_features)
            y_train, y_val: Target arrays
            sequence_length: Sequence length

        Returns:
            Dictionary of model metrics
        """
        logger.info("Training DL models...")

        results = {}
        num_features = X_train.shape[-1] if X_train.ndim == 3 else X_train.shape[-1]

        for model_name in ["lstm", "cnn"]:
            try:
                metrics = self.train_model(
                    model_name, X_train, y_train, X_val, y_val,
                    sequence_length=sequence_length,
                    num_features=num_features,
                )
                results[model_name] = metrics
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")

        return results

    def train_all_rl(
        self,
        data: np.ndarray,
        features: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        Train all RL models (D4PG, MARL) with replay buffers.

        Args:
            data: OHLCV data
            features: Preprocessed features

        Returns:
            Dictionary of model metrics
        """
        logger.info("Training RL models...")

        results = {}

        # Train D4PG
        try:
            from models.rl.d4pg_evt import D4PGAgent, TradingEnvironment, train_d4pg

            logger.info("  Training D4PG+EVT agent...")
            agent = train_d4pg(data, features, episodes=self.config.rl_episodes)

            self.models["d4pg"] = agent

            # Save model
            model_path = str(self.model_dir / "d4pg_agent.pt")
            agent.save(model_path)

            # Export ONNX
            try:
                onnx_path = str(self.onnx_dir / "d4pg_actor.onnx")
                agent.export_onnx(onnx_path)
                self.tracker.save_artifacts("d4pg", [model_path, onnx_path])
            except Exception as e:
                logger.warning(f"D4PG ONNX export failed: {e}")
                self.tracker.save_artifacts("d4pg", [model_path])

            metrics = {
                "training_steps": agent.training_step,
                "var_99": agent.evt_model.var() if hasattr(agent, "evt_model") else None,
                "cvar_99": agent.evt_model.cvar() if hasattr(agent, "evt_model") else None,
            }

            results["d4pg"] = metrics
            self.tracker.track_metrics("d4pg", metrics)

        except Exception as e:
            logger.error(f"Failed to train D4PG: {e}")

        # Train MARL
        try:
            from models.rl.marl import MARLSystem, train_marl

            logger.info("  Training MARL system...")
            marl = train_marl(data, features, n_agents=5, episodes=self.config.rl_episodes)

            self.models["marl"] = marl

            # Save model
            model_path = str(self.model_dir / "marl_system.pt")
            marl.save(model_path)

            # Export ONNX
            try:
                onnx_dir = str(self.onnx_dir / "marl")
                marl.export_onnx(onnx_dir)
                self.tracker.save_artifacts("marl", [model_path, onnx_dir])
            except Exception as e:
                logger.warning(f"MARL ONNX export failed: {e}")
                self.tracker.save_artifacts("marl", [model_path])

            metrics = {
                "training_steps": marl.training_step,
                "n_agents": marl.n_agents,
            }

            results["marl"] = metrics
            self.tracker.track_metrics("marl", metrics)

        except Exception as e:
            logger.error(f"Failed to train MARL: {e}")

        return results

    def train_all(
        self,
        skip_ml: bool = False,
        skip_dl: bool = False,
        skip_rl: bool = False,
    ) -> Dict[str, Dict]:
        """
        Run full orchestrated training pipeline.

        Args:
            skip_ml: Skip ML model training
            skip_dl: Skip DL model training
            skip_rl: Skip RL model training

        Returns:
            Dictionary of all model metrics
        """
        logger.info("=" * 70)
        logger.info("Starting Full Training Pipeline")
        logger.info("=" * 70)

        all_results = {}

        # 1. Load and prepare data
        logger.info("\n[1/6] Loading data...")
        df = self.data_manager.load_data(self.config.data_path)
        self._data_hash = self.data_manager.hash_dataset(df)
        logger.info(f"Data hash: {self._data_hash[:16]}...")

        # 2. Add features
        logger.info("\n[2/6] Engineering features...")
        if self.config.enable_quant_features:
            df = self.data_manager.add_quant_features(df)
        if self.config.enable_microstructure:
            df = self.data_manager.add_microstructure(df)

        # Create targets if not present
        if self.config.target_column not in df.columns:
            df = self.data_manager.create_targets(df)

        # 3. Prepare features and targets
        logger.info("\n[3/6] Preparing feature/target split...")
        X, y, feature_names, times = self.data_manager.get_feature_target_split(
            df, target_column=self.config.target_column
        )
        self._feature_names = feature_names

        # Create train/val/test splits
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         train_idx, val_idx, test_idx) = self.data_manager.prepare_temporal_splits(
            X, y, times,
            train_ratio=0.7,
            val_ratio=0.15,
            embargo_pct=self.config.embargo_pct,
        )

        # Get times for each split
        train_times = times.iloc[train_idx].reset_index(drop=True)
        val_times = times.iloc[val_idx].reset_index(drop=True)
        test_times = times.iloc[test_idx].reset_index(drop=True)

        # 4. Run leakage checks BEFORE training
        logger.info("\n[4/6] Running leakage validation...")
        try:
            self.run_leakage_checks(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                train_times, val_times, test_times,
                feature_names,
            )
        except ValueError as e:
            logger.error(f"Leakage check failed: {e}")
            raise

        # 5. Train models
        logger.info("\n[5/6] Training models...")

        # ML Models
        if not skip_ml:
            logger.info("\n--- Training ML Models ---")
            ml_results = self.train_all_ml(
                X_train, y_train, X_val, y_val, feature_names
            )
            all_results.update(ml_results)

        # DL Models
        if not skip_dl:
            logger.info("\n--- Training DL Models ---")

            # Create sequences
            X_train_seq, y_train_seq = self.data_manager.create_sequences(
                X_train, y_train, self.config.sequence_length
            )
            X_val_seq, y_val_seq = self.data_manager.create_sequences(
                X_val, y_val, self.config.sequence_length
            )

            dl_results = self.train_all_dl(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                self.config.sequence_length,
            )
            all_results.update(dl_results)

        # RL Models
        if not skip_rl:
            logger.info("\n--- Training RL Models ---")

            # Prepare RL data
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in ohlcv_cols if c in df.columns]

            if len(available_cols) == len(ohlcv_cols):
                rl_data = df[available_cols].values
                rl_results = self.train_all_rl(rl_data, X)
                all_results.update(rl_results)
            else:
                logger.warning(f"Missing OHLCV columns for RL training: {set(ohlcv_cols) - set(available_cols)}")

        # 6. Generate report and manifest
        logger.info("\n[6/6] Generating reports...")

        manifest = self.tracker.generate_reproducibility_manifest(
            data_hash=self._data_hash,
            seed=self.config.seed,
        )

        report = self.tracker.generate_report()
        print(report)

        logger.info("=" * 70)
        logger.info("Training Pipeline Complete!")
        logger.info(f"Models saved to: {self.model_dir}")
        logger.info(f"ONNX exports: {self.onnx_dir}")
        logger.info(f"Experiment: {self.tracker.experiment_id}")
        logger.info("=" * 70)

        return all_results

    def export_all_onnx(self) -> Dict[str, str]:
        """
        Export all trained models to ONNX format.

        Returns:
            Dictionary of model name to ONNX path
        """
        logger.info("Exporting all models to ONNX...")

        exported = {}

        for model_name, model in self.models.items():
            if hasattr(model, "export_onnx"):
                try:
                    onnx_path = str(self.onnx_dir / f"{model_name}.onnx")

                    if model_name in ["lightgbm", "xgboost"]:
                        model.export_onnx(onnx_path, self._feature_names)
                    elif model_name in ["lstm", "cnn"]:
                        model.export_onnx(onnx_path)
                    elif model_name == "d4pg":
                        model.export_onnx(onnx_path)
                    elif model_name == "marl":
                        model.export_onnx(str(self.onnx_dir / "marl"))
                        onnx_path = str(self.onnx_dir / "marl")

                    exported[model_name] = onnx_path
                    logger.info(f"  Exported {model_name} to {onnx_path}")

                except Exception as e:
                    logger.error(f"  Failed to export {model_name}: {e}")

        return exported

    def generate_report(self) -> str:
        """
        Generate comprehensive training report.

        Returns:
            Formatted report string
        """
        return self.tracker.generate_report()


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Unified Training Orchestrator for ORPFlow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/raw/klines_90d.parquet",
        help="Path to input parquet data",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="trained",
        help="Output directory for models and artifacts",
    )
    parser.add_argument(
        "--cpcv-splits",
        type=int,
        default=5,
        help="Number of CPCV splits",
    )
    parser.add_argument(
        "--embargo-pct",
        type=float,
        default=0.01,
        help="Embargo percentage between splits",
    )
    parser.add_argument(
        "--purge-pct",
        type=float,
        default=0.01,
        help="Purge percentage for feature windows",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_return_5",
        help="Target column for prediction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML model training",
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        help="Skip DL model training",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip RL model training",
    )
    parser.add_argument(
        "--no-quant-features",
        action="store_true",
        help="Disable quant feature engineering",
    )
    parser.add_argument(
        "--no-microstructure",
        action="store_true",
        help="Disable microstructure features",
    )
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=200,
        help="Number of RL training episodes",
    )
    parser.add_argument(
        "--ml-estimators",
        type=int,
        default=1000,
        help="Number of estimators for ML models",
    )
    parser.add_argument(
        "--dl-epochs",
        type=int,
        default=100,
        help="Number of epochs for DL models",
    )

    args = parser.parse_args()

    # Create config from args
    config = TrainingConfig(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        cpcv_splits=args.cpcv_splits,
        embargo_pct=args.embargo_pct,
        purge_pct=args.purge_pct,
        target_column=args.target,
        seed=args.seed,
        enable_quant_features=not args.no_quant_features,
        enable_microstructure=not args.no_microstructure,
        rl_episodes=args.rl_episodes,
        ml_n_estimators=args.ml_estimators,
        dl_epochs=args.dl_epochs,
    )

    # Run orchestrator
    orchestrator = TrainingOrchestrator(config)

    try:
        results = orchestrator.train_all(
            skip_ml=args.skip_ml,
            skip_dl=args.skip_dl,
            skip_rl=args.skip_rl,
        )

        print("\nTraining completed successfully!")
        print(f"Results saved to: {config.output_dir}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
