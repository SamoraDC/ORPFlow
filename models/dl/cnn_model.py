"""
CNN Model for Time Series Trading Prediction
1D Convolutional Neural Network for pattern recognition in market data
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalConvBlock(nn.Module):
    """Temporal Convolutional Block with residual connection"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class CNNNetwork(nn.Module):
    """1D CNN for time series prediction with multi-scale feature extraction"""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        conv_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 3, 3],
        fc_units: List[int] = [256, 128],
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_features = num_features
        self.sequence_length = sequence_length

        # Initial projection
        self.input_proj = nn.Conv1d(num_features, conv_channels[0], kernel_size=1)

        # Temporal convolution blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = conv_channels[0]

        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            self.conv_blocks.append(
                TemporalConvBlock(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,  # Exponentially increasing dilation
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        # Multi-scale pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        fc_input_size = conv_channels[-1] * 2  # avg + max pooling

        fc_layers = []
        for fc_out in fc_units:
            fc_layers.extend([
                nn.Linear(fc_input_size, fc_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            fc_input_size = fc_out

        fc_layers.append(nn.Linear(fc_units[-1], 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, num_features)
        # Transpose to (batch, num_features, seq_len) for Conv1d
        x = x.transpose(1, 2)

        # Initial projection
        x = self.input_proj(x)

        # Temporal convolutions
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Multi-scale pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)

        # Concatenate
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Fully connected
        out = self.fc(x)

        return out.squeeze(-1)


class CNNModel:
    """CNN model wrapper with training and evaluation"""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        conv_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 3, 3],
        fc_units: List[int] = [256, 128],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_features = num_features
        self.sequence_length = sequence_length

        self.model = CNNNetwork(
            num_features=num_features,
            sequence_length=sequence_length,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            fc_units=fc_units,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.criterion = nn.MSELoss()

        self.history = {"train_loss": [], "val_loss": []}
        logger.info(f"Using device: {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
    ) -> Dict:
        """Train the CNN model"""

        logger.info("Training CNN model...")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    val_loss += self.criterion(outputs, batch_y).item()

            val_loss /= len(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        y_pred_train = self.predict(X_train)
        y_pred_val = self.predict(X_val)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
            "val_mse": mean_squared_error(y_val, y_pred_val),
            "val_r2": r2_score(y_val, y_pred_val),
            "best_epoch": len(self.history["train_loss"]) - patience_counter,
        }

        logger.info(f"Training complete. Val MSE: {metrics['val_mse']:.6f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()

        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        predictions = []

        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate on test data"""
        y_pred = self.predict(X_test)

        metrics = {
            "test_mse": mean_squared_error(y_test, y_pred),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

        metrics.update(self._calculate_trading_metrics(y_test, y_pred))

        logger.info(f"Test MSE: {metrics['test_mse']:.6f}, Sharpe: {metrics.get('sharpe_ratio', 0):.4f}")

        return metrics

    def _calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate trading-specific metrics"""

        direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        strategy_returns = y_true * np.sign(y_pred)

        sharpe_ratio = (
            np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
        ) * np.sqrt(252 * 24 * 60)

        downside_returns = strategy_returns[strategy_returns < 0]
        sortino_ratio = (
            np.mean(strategy_returns) / (np.std(downside_returns) + 1e-8)
        ) * np.sqrt(252 * 24 * 60) if len(downside_returns) > 0 else 0

        win_rate = np.mean(strategy_returns > 0)

        gains = strategy_returns[strategy_returns > 0].sum()
        losses = np.abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gains / (losses + 1e-8)

        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        max_drawdown = np.max(running_max - cumulative_returns)

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
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "num_features": self.num_features,
            "sequence_length": self.sequence_length,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint["history"]
        logger.info(f"Model loaded from {path}")

    def export_onnx(self, path: str):
        """Export model to ONNX format"""
        self.model.eval()

        dummy_input = torch.randn(1, self.sequence_length, self.num_features).to(self.device)

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self.model,
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

        logger.info(f"ONNX model exported to {path}")


def main():
    """Train and evaluate CNN model"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.preprocessor import FeatureEngineer

    # Load processed data - data is at project root level
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "features.parquet"

    if not data_path.exists():
        logger.error("Processed features not found. Run preprocessor.py first.")
        return

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")

    engineer = FeatureEngineer()
    sequence_length = 60

    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_sequence_data(
        df,
        target_col="target_return_5",
        sequence_length=sequence_length,
    )

    num_features = X_train.shape[2]
    logger.info(f"Input shape: {X_train.shape}")

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

    test_metrics = model.evaluate(X_test, y_test)

    # Save to project root trained/ directory
    model_dir = Path(__file__).parent.parent.parent / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(model_dir / "cnn_model.pt"))
    model.export_onnx(str(model_dir / "onnx" / "cnn_model.onnx"))

    print("\n" + "=" * 50)
    print("CNN Model Results")
    print("=" * 50)
    print(f"Validation MSE: {train_metrics['val_mse']:.6f}")
    print(f"Test MSE: {test_metrics['test_mse']:.6f}")
    print(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
    print(f"Win Rate: {test_metrics['win_rate']:.2%}")
    print(f"Profit Factor: {test_metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {test_metrics['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()
