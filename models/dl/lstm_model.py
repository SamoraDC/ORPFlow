"""
LSTM Model for Time Series Trading Prediction
Deep Learning approach for sequential market data
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM neural network for time series prediction"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)

        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size * num_directions)

        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Output
        out = self.fc(context)

        return out.squeeze(-1)


class LSTMModel:
    """LSTM model wrapper with training and evaluation"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
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
        """Train the LSTM model"""

        logger.info("Training LSTM model...")

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

                # Gradient clipping
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

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping
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

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
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

        # Trading metrics
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
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint["history"]
        logger.info(f"Model loaded from {path}")

    def export_onnx(self, path: str, sequence_length: int, num_features: int):
        """Export model to ONNX format"""
        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, sequence_length, num_features).to(self.device)

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
    """Train and evaluate LSTM model"""
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

    # Prepare sequence data
    engineer = FeatureEngineer()
    sequence_length = 60

    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_sequence_data(
        df,
        target_col="target_return_5",
        sequence_length=sequence_length,
    )

    num_features = X_train.shape[2]
    logger.info(f"Input shape: {X_train.shape}")

    # Train model
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

    # Save - save to project root trained/ directory
    model_dir = Path(__file__).parent.parent.parent / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(model_dir / "lstm_model.pt"))
    model.export_onnx(
        str(model_dir / "onnx" / "lstm_model.onnx"),
        sequence_length=sequence_length,
        num_features=num_features,
    )

    # Print results
    print("\n" + "=" * 50)
    print("LSTM Model Results")
    print("=" * 50)
    print(f"Validation MSE: {train_metrics['val_mse']:.6f}")
    print(f"Test MSE: {test_metrics['test_mse']:.6f}")
    print(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
    print(f"Win Rate: {test_metrics['win_rate']:.2%}")
    print(f"Profit Factor: {test_metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {test_metrics['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()
