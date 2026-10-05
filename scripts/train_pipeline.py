#!/usr/bin/env python3
"""
ORPFlow Unified Training Pipeline
==================================
Single command to preprocess with ALL advanced features and train ALL models.

Features included:
- Signal Processing: Kalman Filter, EMD/EEMD, Wavelets, Fisher Transform
- Microstructure: 207 features (OFI, VPIN, Kyle Lambda, Amihud, etc.)
- Hawkes Processes: Order flow modeling
- Regime Detection: Volatility, trend, Hurst exponent
- Triple Barrier Labeling: Dynamic barriers, meta-labeling, sample weights

Models trained:
- ML: XGBoost, LightGBM
- DL: LSTM, CNN (TCN)
- RL: D4PG+EVT, MARL

Usage:
    # Full pipeline (all features + all models)
    python scripts/train_pipeline.py

    # Quick mode (basic features + ML models only)
    python scripts/train_pipeline.py --quick

    # Specific model only
    python scripts/train_pipeline.py --model xgboost

    # Skip preprocessing (use cached features)
    python scripts/train_pipeline.py --skip-preprocessing

    # Skip RL models (faster)
    python scripts/train_pipeline.py --skip-rl
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "logs" / "train_pipeline.log"),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    # Input/Output
    raw_data_path: str = "data/raw/klines_90d.parquet"
    features_output_path: str = "data/processed/features_advanced.parquet"
    models_output_dir: str = "trained"
    onnx_output_dir: str = "trained/onnx"

    # Preprocessing options (ALL enabled by default for maximum features)
    use_signal_processing: bool = True
    use_advanced_microstructure: bool = True
    use_hawkes: bool = True  # User wants ALL features
    use_regimes: bool = True
    use_triple_barrier: bool = True

    # Feature engineering
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])

    # Triple Barrier config
    tb_tp_multiplier: float = 2.0
    tb_sl_multiplier: float = 2.0
    tb_vertical_bars: int = 20
    cusum_threshold: float = 0.02

    # Training options
    train_ml: bool = True
    train_dl: bool = True
    train_rl: bool = True
    export_onnx: bool = True

    # Validation
    cpcv_splits: int = 5
    embargo_pct: float = 0.01
    purge_pct: float = 0.01

    # Seed
    seed: int = 42


@dataclass
class PipelineResult:
    """Result of the training pipeline."""
    preprocessing_time: float = 0.0
    training_time: float = 0.0
    total_time: float = 0.0
    num_features: int = 0
    num_samples: int = 0
    models_trained: List[str] = field(default_factory=list)
    model_metrics: Dict[str, Dict] = field(default_factory=dict)
    onnx_exported: List[str] = field(default_factory=list)
    readiness_decisions: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def print_banner():
    """Print pipeline banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║             ORPFlow - Unified Training Pipeline                       ║
║                                                                       ║
║  Advanced Quantitative Feature Engineering + ML/DL/RL Training        ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def preprocess_data(config: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess raw data with ALL advanced features.

    Returns:
        DataFrame with all features and preprocessing metadata
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Advanced Feature Engineering")
    logger.info("=" * 60)

    start_time = time.time()

    # Load raw data
    raw_data_path = project_root / config.raw_data_path
    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Convert timestamps
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # Import feature engineer
    from models.data.preprocessor import FeatureEngineer

    engineer = FeatureEngineer(windows=config.windows)

    # Process with advanced features
    logger.info("Processing with ADVANCED features (maximum feature set)...")
    logger.info(f"  - Signal Processing: {config.use_signal_processing}")
    logger.info(f"  - Advanced Microstructure: {config.use_advanced_microstructure}")
    logger.info(f"  - Hawkes Processes: {config.use_hawkes}")
    logger.info(f"  - Regime Detection: {config.use_regimes}")

    processed_df = engineer.process_symbol_advanced(
        df,
        use_signal_processing=config.use_signal_processing,
        use_advanced_micro=config.use_advanced_microstructure,
        use_hawkes=config.use_hawkes,
        use_regimes=config.use_regimes,
    )

    # Triple Barrier Labeling
    if config.use_triple_barrier:
        logger.info("Applying Triple Barrier labeling...")
        processed_df = apply_triple_barrier(processed_df, config)

    # Get feature columns
    feature_cols = engineer.get_feature_columns(processed_df)

    # Handle NaN/Inf values (only numeric columns)
    logger.info("Handling NaN/Inf values...")
    numeric_cols = processed_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    for col in numeric_cols:
        processed_df[col] = processed_df[col].replace([np.inf, -np.inf], np.nan)
        col_median = processed_df[col].median()
        if pd.isna(col_median):
            col_median = 0.0
        processed_df[col] = processed_df[col].fillna(col_median)

    # Handle categorical columns (fill with mode or 'unknown')
    for col in categorical_cols:
        mode_val = processed_df[col].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'unknown'
        processed_df[col] = processed_df[col].fillna(fill_val)

    # Update feature_cols to only include numeric columns for training
    feature_cols = numeric_cols

    # Save processed features
    output_path = project_root / config.features_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(output_path, index=False)
    logger.info(f"Saved features to {output_path}")

    processing_time = time.time() - start_time

    # Generate feature summary
    signal_features = [c for c in feature_cols if any(x in c for x in ['kalman', 'emd', 'wavelet', 'fisher'])]
    micro_features = [c for c in feature_cols if c.startswith('micro_')]
    hawkes_features = [c for c in feature_cols if 'hawkes' in c]
    regime_features = [c for c in feature_cols if 'regime' in c]

    metadata = {
        "processing_time": processing_time,
        "num_rows": len(processed_df),
        "num_features": len(feature_cols),
        "feature_breakdown": {
            "signal_processing": len(signal_features),
            "microstructure": len(micro_features),
            "hawkes": len(hawkes_features),
            "regime": len(regime_features),
            "other": len(feature_cols) - len(signal_features) - len(micro_features) - len(hawkes_features) - len(regime_features),
        },
        "feature_columns": feature_cols,
    }

    logger.info(f"Feature engineering completed in {processing_time:.1f}s")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  - Signal Processing: {len(signal_features)}")
    logger.info(f"  - Microstructure: {len(micro_features)}")
    logger.info(f"  - Hawkes: {len(hawkes_features)}")
    logger.info(f"  - Regime: {len(regime_features)}")

    return processed_df, metadata


def apply_triple_barrier(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Apply Triple Barrier labeling to the dataset."""
    try:
        from models.data.triple_barrier import (
            TripleBarrierLabeler,
            BarrierConfig,
            cusum_filter,
            get_sample_weights,
        )

        # Configure barriers
        barrier_config = BarrierConfig(
            tp_multiplier=config.tb_tp_multiplier,
            sl_multiplier=config.tb_sl_multiplier,
            vertical_bars=config.tb_vertical_bars,
            min_return=0.0,
        )

        # Detect events using CUSUM
        events = cusum_filter(df['close'], threshold=config.cusum_threshold)
        logger.info(f"CUSUM detected {len(events)} trading events")

        if len(events) < 100:
            logger.warning("Too few events detected, skipping Triple Barrier labeling")
            return df

        # Generate labels
        labeler = TripleBarrierLabeler(barrier_config)
        labeler.fit(
            close=df['close'],
            high=df['high'],
            low=df['low'],
            t_events=events,
        )

        labels = labeler.get_labels()
        weights = labeler.get_sample_weights(df['close'])

        # Merge labels with processed data
        df['tb_label'] = np.nan
        df['tb_return'] = np.nan
        df['sample_weight'] = 1.0

        df.loc[labels.index, 'tb_label'] = labels['bin']
        df.loc[labels.index, 'tb_return'] = labels['ret']
        df.loc[weights.index, 'sample_weight'] = weights

        label_dist = labels['bin'].value_counts().to_dict()
        logger.info(f"Triple Barrier labels: {label_dist}")

    except Exception as e:
        logger.warning(f"Triple Barrier labeling failed: {e}")

    return df


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_return_5",
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Prepare data for training with temporal splits and embargo.
    """
    from sklearn.preprocessing import RobustScaler

    config = config or PipelineConfig()

    # Select features and target
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32) if target_col in df.columns else df['tb_label'].values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Temporal split with embargo
    n = len(X)
    embargo = int(n * config.embargo_pct)

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_idx = np.arange(0, train_end - embargo)
    val_idx = np.arange(train_end + embargo, val_end - embargo)
    test_idx = np.arange(val_end + embargo, n)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Sample weights (if available)
    if 'sample_weight' in df.columns:
        weights = df['sample_weight'].values
        w_train, w_val, w_test = weights[train_idx], weights[val_idx], weights[test_idx]
    else:
        w_train = w_val = w_test = None

    # Normalize features
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "w_train": w_train,
        "w_val": w_val,
        "w_test": w_test,
        "feature_names": feature_cols,
        "scaler": scaler,
        "n_features": len(feature_cols),
    }


def train_ml_models(
    data: Dict[str, Any],
    config: PipelineConfig,
) -> Dict[str, Dict]:
    """Train ML models (XGBoost, LightGBM)."""
    logger.info("=" * 60)
    logger.info("PHASE 2A: Training ML Models")
    logger.info("=" * 60)

    results = {}
    output_dir = project_root / config.models_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LightGBM
    # =========================================================================
    logger.info("[ML-1/2] Training LightGBM...")
    try:
        import lightgbm as lgb

        lgb_train = lgb.Dataset(
            data["X_train"],
            data["y_train"],
            weight=data["w_train"],
            feature_name=data["feature_names"],
        )
        lgb_val = lgb.Dataset(
            data["X_val"],
            data["y_val"],
            weight=data["w_val"],
            reference=lgb_train,
        )

        lgb_params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
            "seed": config.seed,
        }

        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        lgb_pred = lgb_model.predict(data["X_test"])
        lgb_metrics = evaluate_model(data["y_test"], lgb_pred, "lightgbm")

        # Save model
        lgb_model.save_model(str(output_dir / "lightgbm_advanced.txt"))
        results["lightgbm"] = {
            "metrics": lgb_metrics,
            "model_path": str(output_dir / "lightgbm_advanced.txt"),
            "best_iteration": lgb_model.best_iteration,
        }

        logger.info(f"  LightGBM - Sharpe: {lgb_metrics['sharpe_ratio']:.4f}, "
                   f"Direction: {lgb_metrics['direction_accuracy']:.2%}")

    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        results["lightgbm"] = {"error": str(e)}

    # =========================================================================
    # XGBoost
    # =========================================================================
    logger.info("[ML-2/2] Training XGBoost...")
    try:
        import xgboost as xgb

        xgb_train = xgb.DMatrix(
            data["X_train"],
            label=data["y_train"],
            weight=data["w_train"],
            feature_names=data["feature_names"],
        )
        xgb_val = xgb.DMatrix(
            data["X_val"],
            label=data["y_val"],
            weight=data["w_val"],
            feature_names=data["feature_names"],
        )

        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "seed": config.seed,
            "verbosity": 0,
        }

        xgb_model = xgb.train(
            xgb_params,
            xgb_train,
            num_boost_round=500,
            evals=[(xgb_val, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        xgb_test = xgb.DMatrix(data["X_test"], feature_names=data["feature_names"])
        xgb_pred = xgb_model.predict(xgb_test)
        xgb_metrics = evaluate_model(data["y_test"], xgb_pred, "xgboost")

        # Save model
        xgb_model.save_model(str(output_dir / "xgboost_advanced.json"))
        results["xgboost"] = {
            "metrics": xgb_metrics,
            "model_path": str(output_dir / "xgboost_advanced.json"),
            "best_iteration": xgb_model.best_iteration,
        }

        logger.info(f"  XGBoost - Sharpe: {xgb_metrics['sharpe_ratio']:.4f}, "
                   f"Direction: {xgb_metrics['direction_accuracy']:.2%}")

    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        results["xgboost"] = {"error": str(e)}

    return results


def train_dl_models(
    data: Dict[str, Any],
    config: PipelineConfig,
    sequence_length: int = 60,
) -> Dict[str, Dict]:
    """Train DL models (LSTM, CNN)."""
    logger.info("=" * 60)
    logger.info("PHASE 2B: Training DL Models")
    logger.info("=" * 60)

    results = {}
    output_dir = project_root / config.models_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare sequence data
    def create_sequences(X, y, seq_len):
        """Create sequences for LSTM/CNN."""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(data["X_train"], data["y_train"], sequence_length)
    X_val_seq, y_val_seq = create_sequences(data["X_val"], data["y_val"], sequence_length)
    X_test_seq, y_test_seq = create_sequences(data["X_test"], data["y_test"], sequence_length)

    logger.info(f"Sequence data shapes: train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}")

    # =========================================================================
    # LSTM
    # =========================================================================
    logger.info("[DL-1/2] Training LSTM...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  Using device: {device}")

        # Simple LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout, bidirectional=True
                )
                self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                return self.fc(attn_out[:, -1, :])

        model = LSTMModel(data["n_features"]).to(device)

        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15

        for epoch in range(100):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch).squeeze()
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch).squeeze()
                    val_loss += criterion(pred, y_batch).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'input_size': data["n_features"],
                        'hidden_size': 128,
                        'num_layers': 2,
                        'sequence_length': sequence_length,
                    }
                }, output_dir / "lstm_advanced.pt")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
            lstm_pred = model(X_test_tensor).cpu().numpy().squeeze()

        lstm_metrics = evaluate_model(y_test_seq, lstm_pred, "lstm")
        results["lstm"] = {
            "metrics": lstm_metrics,
            "model_path": str(output_dir / "lstm_advanced.pt"),
            "best_val_loss": best_val_loss,
        }

        logger.info(f"  LSTM - Sharpe: {lstm_metrics['sharpe_ratio']:.4f}, "
                   f"Direction: {lstm_metrics['direction_accuracy']:.2%}")

    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        results["lstm"] = {"error": str(e)}

    # =========================================================================
    # CNN (TCN-style)
    # =========================================================================
    logger.info("[DL-2/2] Training CNN...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class CNNModel(nn.Module):
            def __init__(self, input_size, seq_len):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)
                )
                self.fc = nn.Sequential(
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1)
                )

            def forward(self, x):
                # x: (batch, seq, features) -> (batch, features, seq)
                x = x.permute(0, 2, 1)
                x = self.conv_layers(x).squeeze(-1)
                return self.fc(x)

        model = CNNModel(data["n_features"], sequence_length).to(device)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15

        for epoch in range(100):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch).squeeze()
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch).squeeze()
                    val_loss += criterion(pred, y_batch).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'input_size': data["n_features"],
                        'sequence_length': sequence_length,
                    }
                }, output_dir / "cnn_advanced.pt")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
            cnn_pred = model(X_test_tensor).cpu().numpy().squeeze()

        cnn_metrics = evaluate_model(y_test_seq, cnn_pred, "cnn")
        results["cnn"] = {
            "metrics": cnn_metrics,
            "model_path": str(output_dir / "cnn_advanced.pt"),
            "best_val_loss": best_val_loss,
        }

        logger.info(f"  CNN - Sharpe: {cnn_metrics['sharpe_ratio']:.4f}, "
                   f"Direction: {cnn_metrics['direction_accuracy']:.2%}")

    except Exception as e:
        logger.error(f"CNN training failed: {e}")
        results["cnn"] = {"error": str(e)}

    return results


def train_rl_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    config: PipelineConfig,
) -> Dict[str, Dict]:
    """Train RL models (D4PG, MARL)."""
    logger.info("=" * 60)
    logger.info("PHASE 2C: Training RL Models")
    logger.info("=" * 60)

    results = {}
    output_dir = project_root / config.models_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for RL
    from sklearn.preprocessing import RobustScaler

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    data = df[ohlcv_cols].values
    features = df[feature_cols].values

    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # =========================================================================
    # D4PG+EVT
    # =========================================================================
    logger.info("[RL-1/2] Training D4PG+EVT...")
    try:
        from models.rl.d4pg_evt import train_d4pg

        agent = train_d4pg(
            data=data,
            features=features,
            episodes=100,
            max_steps=2000,
        )

        # Save
        agent.save(str(output_dir / "d4pg_evt_advanced.pt"))
        agent.export_onnx(str(output_dir / "onnx" / "d4pg_actor_advanced.onnx"))

        results["d4pg"] = {
            "metrics": {
                "var_99": float(agent.evt_model.var()),
                "cvar_99": float(agent.evt_model.cvar()),
                "training_steps": agent.training_step,
            },
            "model_path": str(output_dir / "d4pg_evt_advanced.pt"),
        }

        logger.info(f"  D4PG - VaR99: {agent.evt_model.var():.4f}, "
                   f"CVaR99: {agent.evt_model.cvar():.4f}")

    except Exception as e:
        logger.error(f"D4PG training failed: {e}")
        results["d4pg"] = {"error": str(e)}

    # =========================================================================
    # MARL
    # =========================================================================
    logger.info("[RL-2/2] Training MARL...")
    try:
        import torch
        from models.rl.marl import train_marl

        marl_system = train_marl(
            data=data,
            features=features,
            n_agents=5,
            episodes=50,
            max_steps=1000,
        )

        # Save
        torch.save({
            'agents': [agent.state_dict() for agent in marl_system.agents],
            'state_dim': marl_system.state_dim,
            'n_agents': marl_system.n_agents,
            'message_dim': 32,
        }, output_dir / "marl_advanced.pt")

        results["marl"] = {
            "metrics": {
                "n_agents": marl_system.n_agents,
                "state_dim": marl_system.state_dim,
            },
            "model_path": str(output_dir / "marl_advanced.pt"),
        }

        logger.info(f"  MARL - Agents: {marl_system.n_agents}, State dim: {marl_system.state_dim}")

    except Exception as e:
        logger.error(f"MARL training failed: {e}")
        results["marl"] = {"error": str(e)}

    return results


def export_to_onnx(config: PipelineConfig, feature_names: List[str]) -> List[str]:
    """Export all trained models to ONNX format."""
    logger.info("=" * 60)
    logger.info("PHASE 3: ONNX Export")
    logger.info("=" * 60)

    from models.export import ONNXExporter

    output_dir = project_root / config.onnx_output_dir
    exporter = ONNXExporter(output_dir=str(output_dir))
    exported = []

    trained_dir = project_root / config.models_output_dir

    # Export LightGBM
    lgb_path = trained_dir / "lightgbm_advanced.txt"
    if lgb_path.exists():
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(lgb_path))
            path = exporter.export_lightgbm(model, feature_names, "lightgbm_advanced")
            exported.append(path)
            logger.info(f"  Exported LightGBM to {path}")
        except Exception as e:
            logger.error(f"Failed to export LightGBM: {e}")

    # Export XGBoost
    xgb_path = trained_dir / "xgboost_advanced.json"
    if xgb_path.exists():
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(xgb_path))
            path = exporter.export_xgboost(model, feature_names, "xgboost_advanced")
            exported.append(path)
            logger.info(f"  Exported XGBoost to {path}")
        except Exception as e:
            logger.error(f"Failed to export XGBoost: {e}")

    # Export LSTM
    lstm_path = trained_dir / "lstm_advanced.pt"
    if lstm_path.exists():
        try:
            import torch

            checkpoint = torch.load(lstm_path, map_location="cpu")
            seq_len = checkpoint['config']['sequence_length']
            n_features = checkpoint['config']['input_size']

            # Export via PyTorch
            logger.info(f"  LSTM ONNX export skipped (requires model class definition)")
        except Exception as e:
            logger.error(f"Failed to export LSTM: {e}")

    # Export CNN
    cnn_path = trained_dir / "cnn_advanced.pt"
    if cnn_path.exists():
        try:
            logger.info(f"  CNN ONNX export skipped (requires model class definition)")
        except Exception as e:
            logger.error(f"Failed to export CNN: {e}")

    return exported


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
    """Evaluate model predictions with trading metrics."""
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Direction accuracy
    direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

    # Strategy returns
    strategy_returns = y_true * np.sign(y_pred)
    sharpe = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)) * np.sqrt(252 * 24 * 12)

    # Win rate
    win_rate = np.mean(strategy_returns > 0)

    # Max drawdown
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Profit factor
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = np.abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gains / (losses + 1e-8)

    return {
        "model": model_name,
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "direction_accuracy": float(direction_accuracy),
        "sharpe_ratio": float(sharpe),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "profit_factor": float(profit_factor),
        "total_return": float(cumulative[-1]) if len(cumulative) > 0 else 0,
    }


def determine_readiness(metrics: Dict[str, float]) -> Tuple[str, Dict[str, bool], str]:
    """Determine GO/NO-GO based on metrics."""
    checks = {
        "direction_accuracy": metrics.get("direction_accuracy", 0) > 0.50,
        "sharpe_positive": metrics.get("sharpe_ratio", 0) > 0.0,
        "win_rate": metrics.get("win_rate", 0) > 0.45,
        "profit_factor": metrics.get("profit_factor", 0) > 0.8,
    }

    passed = sum(checks.values())
    total = len(checks)

    decision = "GO" if passed >= 3 else "NO-GO"
    return decision, checks, f"{passed}/{total} criteria passed"


def generate_report(
    result: PipelineResult,
    config: PipelineConfig,
) -> str:
    """Generate comprehensive readiness report."""
    report_lines = [
        "=" * 70,
        "ORPFlow - Training Pipeline Readiness Report",
        "=" * 70,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Total Processing Time: {result.total_time:.1f}s",
        "",
        "FEATURE ENGINEERING",
        "-" * 40,
        f"  Samples: {result.num_samples:,}",
        f"  Features: {result.num_features}",
        f"  Processing Time: {result.preprocessing_time:.1f}s",
        "",
        "MODELS TRAINED",
        "-" * 40,
    ]

    for model_name in result.models_trained:
        metrics = result.model_metrics.get(model_name, {})
        if "error" in metrics:
            report_lines.append(f"  ✗ {model_name.upper()}: FAILED - {metrics['error']}")
        else:
            decision = result.readiness_decisions.get(model_name, "UNKNOWN")
            status = "✓" if decision == "GO" else "✗"
            m = metrics.get("metrics", metrics)
            report_lines.append(f"  {status} {model_name.upper()}: {decision}")
            if "sharpe_ratio" in m:
                report_lines.append(f"      Sharpe: {m['sharpe_ratio']:.4f}")
            if "direction_accuracy" in m:
                report_lines.append(f"      Direction: {m['direction_accuracy']:.2%}")
            if "win_rate" in m:
                report_lines.append(f"      Win Rate: {m['win_rate']:.2%}")

    # Overall summary
    go_count = sum(1 for d in result.readiness_decisions.values() if d == "GO")
    total_count = len(result.readiness_decisions)

    report_lines.extend([
        "",
        "ONNX EXPORTS",
        "-" * 40,
        f"  Exported: {len(result.onnx_exported)} models",
    ])

    for path in result.onnx_exported:
        report_lines.append(f"    - {Path(path).name}")

    if result.errors:
        report_lines.extend([
            "",
            "ERRORS",
            "-" * 40,
        ])
        for error in result.errors:
            report_lines.append(f"  - {error}")

    report_lines.extend([
        "",
        "=" * 70,
        f"OVERALL: {go_count}/{total_count} models ready for deployment",
        "=" * 70,
    ])

    return "\n".join(report_lines)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="ORPFlow Unified Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=["all", "xgboost", "lightgbm", "lstm", "cnn", "d4pg", "marl"],
        default="all",
        help="Model to train (default: all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: basic features + ML models only"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing, use cached features"
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip RL models (D4PG, MARL)"
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        help="Skip DL models (LSTM, CNN)"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX export"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Create logs directory
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    print_banner()

    # Configure pipeline
    config = PipelineConfig(seed=args.seed)

    if args.quick:
        config.use_hawkes = False
        config.use_triple_barrier = False
        config.train_dl = False
        config.train_rl = False

    if args.skip_rl:
        config.train_rl = False

    if args.skip_dl:
        config.train_dl = False

    if args.no_export:
        config.export_onnx = False

    result = PipelineResult()
    total_start = time.time()

    # =========================================================================
    # Phase 1: Preprocessing
    # =========================================================================
    if args.skip_preprocessing:
        logger.info("Skipping preprocessing, loading cached features...")
        features_path = project_root / config.features_output_path
        if not features_path.exists():
            logger.error(f"Cached features not found at {features_path}")
            sys.exit(1)
        df = pd.read_parquet(features_path)
        from models.data.preprocessor import FeatureEngineer
        engineer = FeatureEngineer()
        feature_cols = engineer.get_feature_columns(df, numeric_only=True)
        preprocessing_meta = {"num_features": len(feature_cols), "num_rows": len(df)}
    else:
        df, preprocessing_meta = preprocess_data(config)
        from models.data.preprocessor import FeatureEngineer
        engineer = FeatureEngineer()
        feature_cols = engineer.get_feature_columns(df, numeric_only=True)

    result.preprocessing_time = preprocessing_meta.get("processing_time", 0)
    result.num_features = preprocessing_meta.get("num_features", len(feature_cols))
    result.num_samples = len(df)

    # Determine target column (preprocessor creates target_return_X columns)
    target_col = None
    for col_name in ["target_return_5", "target_5", "tb_return"]:
        if col_name in df.columns:
            target_col = col_name
            break

    if target_col is None:
        # Create simple target if none found
        df["target_return_5"] = df["close"].pct_change(5).shift(-5)
        target_col = "target_return_5"
        logger.info(f"Created target column: {target_col}")

    # Drop rows with NaN target
    df = df.dropna(subset=[target_col])

    # Prepare training data
    training_data = prepare_training_data(df, feature_cols, target_col, config)

    # =========================================================================
    # Phase 2: Training
    # =========================================================================
    training_start = time.time()
    all_results = {}

    # Train ML models
    if config.train_ml and args.model in ["all", "xgboost", "lightgbm"]:
        ml_results = train_ml_models(training_data, config)
        all_results.update(ml_results)

    # Train DL models
    if config.train_dl and args.model in ["all", "lstm", "cnn"]:
        dl_results = train_dl_models(training_data, config)
        all_results.update(dl_results)

    # Train RL models
    if config.train_rl and args.model in ["all", "d4pg", "marl"]:
        rl_results = train_rl_models(df, feature_cols, config)
        all_results.update(rl_results)

    result.training_time = time.time() - training_start
    result.model_metrics = all_results
    result.models_trained = list(all_results.keys())

    # Determine readiness for each model
    for model_name, model_result in all_results.items():
        if "error" not in model_result:
            metrics = model_result.get("metrics", model_result)
            decision, _, _ = determine_readiness(metrics)
            result.readiness_decisions[model_name] = decision
        else:
            result.errors.append(f"{model_name}: {model_result['error']}")

    # =========================================================================
    # Phase 3: ONNX Export
    # =========================================================================
    if config.export_onnx:
        result.onnx_exported = export_to_onnx(config, feature_cols)

    # =========================================================================
    # Generate Report
    # =========================================================================
    result.total_time = time.time() - total_start

    report = generate_report(result, config)
    print(report)

    # Save report
    report_path = project_root / config.models_output_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "use_signal_processing": config.use_signal_processing,
                "use_advanced_microstructure": config.use_advanced_microstructure,
                "use_hawkes": config.use_hawkes,
                "use_regimes": config.use_regimes,
                "use_triple_barrier": config.use_triple_barrier,
            },
            "preprocessing_time": result.preprocessing_time,
            "training_time": result.training_time,
            "total_time": result.total_time,
            "num_features": result.num_features,
            "num_samples": result.num_samples,
            "models_trained": result.models_trained,
            "model_metrics": result.model_metrics,
            "readiness_decisions": result.readiness_decisions,
            "onnx_exported": result.onnx_exported,
            "errors": result.errors,
        }, f, indent=2)

    logger.info(f"\nReport saved to {report_path}")
    logger.info(f"Total pipeline time: {result.total_time:.1f}s")

    # Exit with appropriate code
    go_count = sum(1 for d in result.readiness_decisions.values() if d == "GO")
    sys.exit(0 if go_count > 0 else 1)


if __name__ == "__main__":
    main()
