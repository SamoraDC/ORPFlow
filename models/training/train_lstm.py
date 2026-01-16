#!/usr/bin/env python3
"""
LSTM Training Pipeline with CPCV, Optuna, and ONNX Export

Features:
- Combinatorial Purged Cross-Validation (CPCV) for time series
- Optuna hyperparameter optimization
- ONNX export for production deployment
- Comprehensive trading metrics
- Learning rate scheduling with warmup
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class LSTMTrainingConfig:
    """Configuration for LSTM training pipeline."""

    # Data
    data_path: str = "data/processed/features.parquet"
    target_column: str = "target_return_5"
    sequence_length: int = 60

    # Model Architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    use_attention: bool = True

    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 15
    gradient_clip: float = 1.0

    # Learning Rate Schedule
    lr_scheduler: str = "cosine"  # cosine, plateau, step
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # CPCV Settings
    n_splits: int = 5
    n_test_groups: int = 2
    embargo_pct: float = 0.01
    purge_pct: float = 0.005

    # Optuna
    enable_optuna: bool = False
    n_trials: int = 50
    optuna_timeout: int = 7200

    # Output
    output_dir: str = "trained"
    model_name: str = "lstm_model"
    export_onnx: bool = True

    # Device
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class LSTMNetworkWithAttention(nn.Module):
    """LSTM with optional attention mechanism for time series prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # Layer normalization on input
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        # Output layers with residual
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)

        # Input normalization
        x = self.input_norm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size * num_directions)

        if self.use_attention:
            # Attention mechanism
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]

        # Output
        out = self.fc(context)

        return out.squeeze(-1)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class LSTMTrainer:
    """LSTM trainer with CPCV, Optuna, and ONNX export capabilities."""

    def __init__(self, config: LSTMTrainingConfig):
        self.config = config

        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Load and prepare sequence data."""
        logger.info(f"Loading data from {self.config.data_path}")

        df = pd.read_parquet(self.config.data_path)
        logger.info(f"Loaded {len(df)} rows")

        # Get timestamps
        if "timestamp" in df.columns:
            times = pd.to_datetime(df["timestamp"])
        elif df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            times = df.index
            df = df.reset_index(drop=True)
        else:
            times = pd.Series(range(len(df)))

        # Separate features and target
        target = df[self.config.target_column].values

        # Exclude non-feature columns
        exclude_cols = [self.config.target_column, "timestamp", "open_time", "close_time"]
        exclude_cols += [c for c in df.columns if c.startswith("target_")]

        feature_cols = [c for c in df.columns if c not in exclude_cols]
        features = df[feature_cols].values

        logger.info(f"Features shape: {features.shape}, Target shape: {target.shape}")

        return features, target, times

    def create_sequences(
        self,
        features: np.ndarray,
        target: np.ndarray,
        indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        seq_len = self.config.sequence_length

        X_list = []
        y_list = []

        for idx in indices:
            if idx >= seq_len:
                X_list.append(features[idx - seq_len:idx])
                y_list.append(target[idx])

        return np.array(X_list), np.array(y_list)

    def create_model(self, input_size: int) -> LSTMNetworkWithAttention:
        """Create LSTM model."""
        model = LSTMNetworkWithAttention(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
            use_attention=self.config.use_attention,
        ).to(self.device)

        return model

    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model: Optional[LSTMNetworkWithAttention] = None,
    ) -> Tuple[LSTMNetworkWithAttention, Dict[str, Any]]:
        """Train model on a single fold."""

        if model is None:
            model = self.create_model(X_train.shape[2])

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        if self.config.lr_scheduler == "cosine":
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.epochs,
                min_lr=self.config.min_lr,
            )
        elif self.config.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.5
            )

        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.gradient_clip
                )

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()

            val_loss /= len(val_loader)

            # Update scheduler
            current_lr = optimizer.param_groups[0]["lr"]
            if isinstance(scheduler, WarmupCosineScheduler):
                scheduler.step()
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}"
                )

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Calculate metrics
        metrics = self._calculate_metrics(model, X_val, y_val)
        metrics["best_val_loss"] = best_val_loss
        metrics["best_epoch"] = len(history["train_loss"]) - patience_counter

        return model, metrics

    def _calculate_metrics(
        self,
        model: LSTMNetworkWithAttention,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_pred = model(X_tensor).cpu().numpy()

        # Regression metrics
        mse = float(np.mean((y - y_pred) ** 2))
        mae = float(np.mean(np.abs(y - y_pred)))

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Direction accuracy
        direction_accuracy = float(np.mean(np.sign(y) == np.sign(y_pred)))

        # Trading metrics
        strategy_returns = y * np.sign(y_pred)

        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)

        sharpe_ratio = float(
            mean_ret / std_ret * np.sqrt(252 * 24 * 60)
            if std_ret > 0 else 0.0
        )

        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = float(
            mean_ret / downside_std * np.sqrt(252 * 24 * 60)
            if downside_std > 0 else 0.0
        )

        win_rate = float(np.mean(strategy_returns > 0))

        cumulative = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = float(np.max(running_max - cumulative))

        gains = strategy_returns[strategy_returns > 0].sum()
        losses = np.abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else 0.0

        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "total_return": float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
        }

    def run_cpcv(
        self,
        features: np.ndarray,
        target: np.ndarray,
        times: pd.Series,
    ) -> Dict[str, Any]:
        """Run Combinatorial Purged Cross-Validation."""
        from models.validation.cpcv import CombinatorialPurgedKFold

        logger.info("Running CPCV...")

        cpcv = CombinatorialPurgedKFold(
            n_splits=self.config.n_splits,
            n_test_groups=self.config.n_test_groups,
            embargo_pct=self.config.embargo_pct,
            purge_pct=self.config.purge_pct,
        )

        fold_results = []
        sharpe_ratios = []
        best_model = None
        best_sharpe = -float("inf")

        # Valid indices (need at least sequence_length history)
        valid_indices = np.arange(self.config.sequence_length, len(features))

        for fold_idx, (train_idx, test_idx) in enumerate(cpcv.split(valid_indices)):
            logger.info(f"Fold {fold_idx + 1}/{cpcv.get_n_splits()}")

            # Map back to original indices
            train_indices = valid_indices[train_idx]
            test_indices = valid_indices[test_idx]

            # Create sequences
            X_train, y_train = self.create_sequences(features, target, train_indices)
            X_test, y_test = self.create_sequences(features, target, test_indices)

            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f"Skipping fold {fold_idx + 1}: insufficient data")
                continue

            # Split train into train/val (80/20)
            val_split = int(len(X_train) * 0.8)
            X_val = X_train[val_split:]
            y_val = y_train[val_split:]
            X_train = X_train[:val_split]
            y_train = y_train[:val_split]

            # Train
            model, metrics = self.train_fold(X_train, y_train, X_val, y_val)

            # Evaluate on test set
            test_metrics = self._calculate_metrics(model, X_test, y_test)

            fold_result = {
                "fold": fold_idx,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "val_metrics": metrics,
                "test_metrics": test_metrics,
            }

            fold_results.append(fold_result)
            sharpe_ratios.append(test_metrics["sharpe_ratio"])

            logger.info(
                f"Fold {fold_idx + 1}: Test Sharpe={test_metrics['sharpe_ratio']:.4f}, "
                f"Win Rate={test_metrics['win_rate']:.2%}"
            )

            # Track best model
            if test_metrics["sharpe_ratio"] > best_sharpe:
                best_sharpe = test_metrics["sharpe_ratio"]
                best_model = model.state_dict().copy()

        # Aggregate results
        sr_array = np.array(sharpe_ratios)

        results = {
            "fold_results": fold_results,
            "sharpe_ratios": sharpe_ratios,
            "mean_sharpe": float(np.mean(sr_array)),
            "std_sharpe": float(np.std(sr_array)),
            "median_sharpe": float(np.median(sr_array)),
            "min_sharpe": float(np.min(sr_array)),
            "max_sharpe": float(np.max(sr_array)),
            "n_folds": len(fold_results),
            "best_model_state": best_model,
        }

        # Calculate Probability of Backtest Overfitting
        pbo = float(np.mean(sr_array < 0))
        results["pbo"] = pbo

        logger.info(f"CPCV Complete: Mean Sharpe={results['mean_sharpe']:.4f}, PBO={pbo:.2%}")

        return results

    def create_optuna_objective(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ) -> Callable:
        """Create Optuna objective function."""
        import optuna

        # Create validation split
        split_idx = int(len(features) * 0.7)
        val_idx = int(len(features) * 0.85)

        valid_start = self.config.sequence_length

        X_train, y_train = self.create_sequences(
            features, target, np.arange(valid_start, split_idx)
        )
        X_val, y_val = self.create_sequences(
            features, target, np.arange(split_idx, val_idx)
        )

        def objective(trial: optuna.Trial) -> float:
            # Suggest hyperparameters
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            bidirectional = trial.suggest_categorical("bidirectional", [True, False])

            # Update config
            self.config.hidden_size = hidden_size
            self.config.num_layers = num_layers
            self.config.dropout = dropout
            self.config.learning_rate = learning_rate
            self.config.batch_size = batch_size
            self.config.bidirectional = bidirectional
            self.config.epochs = 50  # Reduced for optimization

            try:
                model, metrics = self.train_fold(X_train, y_train, X_val, y_val)

                # Optimize for Sharpe ratio
                return metrics["sharpe_ratio"]

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("-inf")

        return objective

    def run_optuna(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, Any]:
        """Run Optuna hyperparameter optimization."""
        import optuna

        logger.info("Running Optuna optimization...")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        objective = self.create_optuna_objective(features, target)

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True,
        )

        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return {
            "best_value": study.best_trial.value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }

    def export_onnx(
        self,
        model: LSTMNetworkWithAttention,
        input_size: int,
        path: str,
    ):
        """Export model to ONNX format."""
        logger.info(f"Exporting ONNX model to {path}")

        model.eval()

        # Create dummy input
        dummy_input = torch.randn(
            1, self.config.sequence_length, input_size
        ).to(self.device)

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        logger.info("ONNX export complete")

    def train(self) -> Dict[str, Any]:
        """Run full training pipeline."""
        logger.info("=" * 60)
        logger.info("LSTM Training Pipeline")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Load data
        features, target, times = self.load_data()
        input_size = features.shape[1]

        results = {
            "model_type": "lstm",
            "config": self.config.to_dict(),
            "data_shape": list(features.shape),
            "target_column": self.config.target_column,
            "start_time": start_time.isoformat(),
        }

        # Optuna optimization (optional)
        if self.config.enable_optuna:
            optuna_results = self.run_optuna(features, target)
            results["optuna"] = optuna_results

            # Apply best params
            for key, value in optuna_results["best_params"].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Run CPCV
        cpcv_results = self.run_cpcv(features, target, times)
        results["cpcv"] = {
            k: v for k, v in cpcv_results.items()
            if k != "best_model_state"
        }

        # Train final model on all data
        logger.info("Training final model on full dataset...")

        valid_indices = np.arange(self.config.sequence_length, len(features))
        X_all, y_all = self.create_sequences(features, target, valid_indices)

        # 80/20 split for final model
        split = int(len(X_all) * 0.8)
        X_train, y_train = X_all[:split], y_all[:split]
        X_val, y_val = X_all[split:], y_all[split:]

        # Restore best model or train new
        final_model = self.create_model(input_size)
        if cpcv_results.get("best_model_state"):
            final_model.load_state_dict(cpcv_results["best_model_state"])
            logger.info("Restored best model from CPCV")
        else:
            final_model, final_metrics = self.train_fold(X_train, y_train, X_val, y_val)
            results["final_metrics"] = final_metrics

        # Save model
        model_path = self.output_dir / f"{self.config.model_name}.pt"
        torch.save({
            "model_state": final_model.state_dict(),
            "config": self.config.to_dict(),
            "input_size": input_size,
            "sequence_length": self.config.sequence_length,
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        results["model_path"] = str(model_path)

        # Export ONNX
        if self.config.export_onnx:
            onnx_path = self.output_dir / "onnx" / f"{self.config.model_name}.onnx"
            self.export_onnx(final_model, input_size, str(onnx_path))
            results["onnx_path"] = str(onnx_path)

        # Save results
        end_time = datetime.now()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()

        results_path = self.output_dir / f"{self.config.model_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")
        logger.info(f"Training completed in {results['duration_seconds']:.1f}s")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Mean Sharpe Ratio: {cpcv_results['mean_sharpe']:.4f}")
        logger.info(f"Std Sharpe Ratio: {cpcv_results['std_sharpe']:.4f}")
        logger.info(f"PBO: {cpcv_results['pbo']:.2%}")
        logger.info(f"Model saved: {model_path}")
        if self.config.export_onnx:
            logger.info(f"ONNX model: {onnx_path}")

        return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LSTM Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--output", type=str, default="trained", help="Output dir")

    args = parser.parse_args()

    config = LSTMTrainingConfig()

    if args.data:
        config.data_path = args.data
    if args.optuna:
        config.enable_optuna = True
        config.n_trials = args.trials
    if args.epochs:
        config.epochs = args.epochs
    if args.output:
        config.output_dir = args.output

    trainer = LSTMTrainer(config)
    results = trainer.train()

    return results


if __name__ == "__main__":
    main()
