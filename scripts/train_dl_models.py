#!/usr/bin/env python3
"""
Train Deep Learning Models (LSTM and CNN)
Standalone script with absolute paths
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path("/home/samoradc/SamoraDC/ORPFlow")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

import logging
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_lstm():
    """Train LSTM model"""
    from models.dl.lstm_model import LSTMModel
    from models.data.preprocessor import FeatureEngineer

    logger.info("=" * 60)
    logger.info("Training LSTM Model")
    logger.info("=" * 60)

    # Load data
    data_path = project_root / "data" / "processed" / "features.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at: {data_path}")
        return None

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")

    # Prepare sequence data
    engineer = FeatureEngineer()
    sequence_length = 60

    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_sequence_data(
        df,
        target_col="target_return_5",
        sequence_length=sequence_length,
    )

    num_features = X_train.shape[2]
    logger.info(f"Input shape: {X_train.shape}, Features: {num_features}")

    # Create and train model
    model = LSTMModel(
        input_size=num_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
    )

    train_metrics = model.train(
        X_train, y_train, X_val, y_val,
        batch_size=64,
        epochs=100,
        patience=10,
    )

    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)

    # Save
    model_dir = project_root / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(model_dir / "lstm_model.pt"))

    # Export ONNX
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model.export_onnx(
        str(onnx_dir / "lstm_model.onnx"),
        sequence_length=sequence_length,
        num_features=num_features,
    )

    # Print results
    print("\n" + "=" * 60)
    print("LSTM Model Training Results")
    print("=" * 60)
    print(f"Validation MSE:    {train_metrics['val_mse']:.6f}")
    print(f"Validation R²:     {train_metrics['val_r2']:.4f}")
    print(f"Test MSE:          {test_metrics['test_mse']:.6f}")
    print(f"Test R²:           {test_metrics['test_r2']:.4f}")
    print(f"Direction Acc:     {test_metrics['direction_accuracy']:.2%}")
    print(f"Sharpe Ratio:      {test_metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio:     {test_metrics['sortino_ratio']:.4f}")
    print(f"Win Rate:          {test_metrics['win_rate']:.2%}")
    print(f"Profit Factor:     {test_metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:      {test_metrics['max_drawdown']:.4f}")
    print(f"Total Return:      {test_metrics['total_return']:.4f}")
    print(f"Best Epoch:        {train_metrics['best_epoch']}")
    print("=" * 60)

    return test_metrics


def train_cnn():
    """Train CNN model"""
    from models.dl.cnn_model import CNNModel
    from models.data.preprocessor import FeatureEngineer

    logger.info("\n" + "=" * 60)
    logger.info("Training CNN Model")
    logger.info("=" * 60)

    # Load data
    data_path = project_root / "data" / "processed" / "features.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at: {data_path}")
        return None

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")

    # Prepare sequence data
    engineer = FeatureEngineer()
    sequence_length = 60

    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_sequence_data(
        df,
        target_col="target_return_5",
        sequence_length=sequence_length,
    )

    num_features = X_train.shape[2]
    logger.info(f"Input shape: {X_train.shape}, Features: {num_features}")

    # Create and train model
    model = CNNModel(
        num_features=num_features,
        sequence_length=sequence_length,
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        fc_units=[256, 128],
        dropout=0.3,
        learning_rate=0.001,
    )

    train_metrics = model.train(
        X_train, y_train, X_val, y_val,
        batch_size=64,
        epochs=100,
        patience=10,
    )

    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)

    # Save
    model_dir = project_root / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(model_dir / "cnn_model.pt"))

    # Export ONNX
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model.export_onnx(str(onnx_dir / "cnn_model.onnx"))

    # Print results
    print("\n" + "=" * 60)
    print("CNN Model Training Results")
    print("=" * 60)
    print(f"Validation MSE:    {train_metrics['val_mse']:.6f}")
    print(f"Validation R²:     {train_metrics['val_r2']:.4f}")
    print(f"Test MSE:          {test_metrics['test_mse']:.6f}")
    print(f"Test R²:           {test_metrics['test_r2']:.4f}")
    print(f"Direction Acc:     {test_metrics['direction_accuracy']:.2%}")
    print(f"Sharpe Ratio:      {test_metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio:     {test_metrics['sortino_ratio']:.4f}")
    print(f"Win Rate:          {test_metrics['win_rate']:.2%}")
    print(f"Profit Factor:     {test_metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:      {test_metrics['max_drawdown']:.4f}")
    print(f"Total Return:      {test_metrics['total_return']:.4f}")
    print(f"Best Epoch:        {train_metrics['best_epoch']}")
    print("=" * 60)

    return test_metrics


def main():
    """Train both models"""
    logger.info("\n" + "#" * 60)
    logger.info("Deep Learning Training Pipeline")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("#" * 60 + "\n")

    # Train LSTM
    lstm_metrics = train_lstm()

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train CNN
    cnn_metrics = train_cnn()

    # Summary
    print("\n" + "#" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("#" * 60)
    if lstm_metrics:
        print(f"\nLSTM - Sharpe: {lstm_metrics['sharpe_ratio']:.4f}, Win Rate: {lstm_metrics['win_rate']:.2%}")
    if cnn_metrics:
        print(f"CNN  - Sharpe: {cnn_metrics['sharpe_ratio']:.4f}, Win Rate: {cnn_metrics['win_rate']:.2%}")
    print("\nModels saved to: /home/samoradc/SamoraDC/ORPFlow/trained/")
    print("ONNX exports at: /home/samoradc/SamoraDC/ORPFlow/trained/onnx/")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
