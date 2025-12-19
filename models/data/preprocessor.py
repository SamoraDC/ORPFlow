"""
Feature Engineering and Data Preprocessing
Transforms raw market data into ML-ready features
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate trading features from raw market data"""

    def __init__(self, windows: List[int] = [5, 10, 20, 50, 100]):
        self.windows = windows
        self.scaler = RobustScaler()
        self.feature_names = []

    def calculate_returns(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Calculate various return metrics"""
        df = df.copy()

        # Simple returns
        df["return_1"] = df[price_col].pct_change()

        # Log returns
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

        # Multi-period returns
        for w in self.windows:
            df[f"return_{w}"] = df[price_col].pct_change(w)
            df[f"log_return_{w}"] = np.log(df[price_col] / df[price_col].shift(w))

        return df

    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        df = df.copy()

        for w in self.windows:
            # Rolling standard deviation of returns
            df[f"volatility_{w}"] = df["log_return"].rolling(window=w).std() * np.sqrt(252 * 24 * 60)

            # Parkinson volatility (high-low based)
            df[f"parkinson_vol_{w}"] = np.sqrt(
                (1 / (4 * np.log(2))) *
                ((np.log(df["high"] / df["low"]) ** 2).rolling(window=w).mean())
            ) * np.sqrt(252 * 24 * 60)

            # Garman-Klass volatility
            log_hl = np.log(df["high"] / df["low"]) ** 2
            log_co = np.log(df["close"] / df["open"]) ** 2
            df[f"gk_vol_{w}"] = np.sqrt(
                (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=w).mean()
            ) * np.sqrt(252 * 24 * 60)

        return df

    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        df = df.copy()

        for w in self.windows:
            # Price momentum
            df[f"momentum_{w}"] = df["close"] / df["close"].shift(w) - 1

            # Rate of change
            df[f"roc_{w}"] = (df["close"] - df["close"].shift(w)) / df["close"].shift(w) * 100

            # Moving average crossover
            df[f"ma_{w}"] = df["close"].rolling(window=w).mean()
            df[f"ma_cross_{w}"] = (df["close"] - df[f"ma_{w}"]) / df[f"ma_{w}"]

        # RSI
        for w in [14, 21]:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            df[f"rsi_{w}"] = 100 - (100 / (1 + rs))

        return df

    def calculate_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate orderbook-derived features"""
        df = df.copy()

        # Bid-ask spread proxy (using high-low)
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"] * 10000  # in bps

        # Volume imbalance proxy
        df["volume_imbalance"] = (
            (df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])) /
            df["volume"]
        )

        # Order flow imbalance
        df["ofi"] = df["taker_buy_base"] / df["volume"]

        for w in self.windows:
            # Rolling imbalance
            df[f"ofi_ma_{w}"] = df["ofi"].rolling(window=w).mean()
            df[f"ofi_std_{w}"] = df["ofi"].rolling(window=w).std()
            df[f"ofi_z_{w}"] = (df["ofi"] - df[f"ofi_ma_{w}"]) / df[f"ofi_std_{w}"]

            # Volume features
            df[f"volume_ma_{w}"] = df["volume"].rolling(window=w).mean()
            df[f"volume_std_{w}"] = df["volume"].rolling(window=w).std()
            df[f"volume_z_{w}"] = (df["volume"] - df[f"volume_ma_{w}"]) / df[f"volume_std_{w}"]

            # Trade count features
            df[f"trades_ma_{w}"] = df["trades"].rolling(window=w).mean()

        return df

    def calculate_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate microstructure features"""
        df = df.copy()

        # Amihud illiquidity
        df["amihud"] = np.abs(df["log_return"]) / df["quote_volume"]

        for w in self.windows:
            df[f"amihud_ma_{w}"] = df["amihud"].rolling(window=w).mean()

        # Kyle's Lambda proxy (price impact)
        for w in self.windows:
            df[f"kyle_lambda_{w}"] = (
                df["log_return"].rolling(window=w).std() /
                df["volume"].rolling(window=w).mean()
            )

        # VPIN (Volume-Synchronized Probability of Informed Trading) proxy
        df["abs_ofi"] = np.abs(df["ofi"] - 0.5)
        for w in [50, 100]:
            df[f"vpin_{w}"] = df["abs_ofi"].rolling(window=w).mean()

        return df

    def calculate_targets(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 5, 15, 30],
    ) -> pd.DataFrame:
        """Calculate prediction targets"""
        df = df.copy()

        for h in horizons:
            # Future returns
            df[f"target_return_{h}"] = df["close"].shift(-h) / df["close"] - 1

            # Direction (classification target)
            df[f"target_direction_{h}"] = (df[f"target_return_{h}"] > 0).astype(int)

            # Volatility target
            df[f"target_vol_{h}"] = df["log_return"].shift(-1).rolling(window=h).std() * np.sqrt(252 * 24 * 60)

        return df

    def add_time_features(self, df: pd.DataFrame, time_col: str = "open_time") -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()

        df["hour"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    def process_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a single symbol's data"""
        logger.info(f"Processing {len(df)} rows...")

        df = self.calculate_returns(df)
        df = self.calculate_volatility(df)
        df = self.calculate_momentum(df)
        df = self.calculate_orderbook_features(df)
        df = self.calculate_microstructure(df)
        df = self.calculate_targets(df)
        df = self.add_time_features(df)

        # Drop NaN rows
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding targets and metadata)"""
        exclude_prefixes = ["target_", "open_time", "close_time", "symbol", "ignore"]
        exclude_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                        "trades", "taker_buy_base", "taker_buy_quote"]

        feature_cols = []
        for col in df.columns:
            if any(col.startswith(p) for p in exclude_prefixes):
                continue
            if col in exclude_cols:
                continue
            feature_cols.append(col)

        return feature_cols

    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target_return_5",
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for ML training"""

        feature_cols = self.get_feature_columns(df)
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df[target_col].values

        # Time-based split (no shuffle for time series)
        n = len(X)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Features: {len(feature_cols)}")

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def prepare_sequence_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target_return_5",
        sequence_length: int = 60,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM/CNN"""

        feature_cols = self.get_feature_columns(df)
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df[target_col].values

        # Scale first
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i - sequence_length:i])
            y_seq.append(y[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Time-based split
        n = len(X_seq)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        X_train = X_seq[:train_end]
        y_train = y_seq[:train_end]
        X_val = X_seq[train_end:val_end]
        y_val = y_seq[train_end:val_end]
        X_test = X_seq[val_end:]
        y_test = y_seq[val_end:]

        logger.info(f"Sequence shape: {X_seq.shape}")
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Process data and save features"""
    from collector import BinanceDataCollector

    collector = BinanceDataCollector()
    engineer = FeatureEngineer()

    # Load raw data
    klines = collector.load_data("klines_90d.parquet")

    if klines.empty:
        logger.error("No data found. Run collector.py first.")
        return

    # Process each symbol
    processed_data = []

    for symbol in klines["symbol"].unique():
        symbol_df = klines[klines["symbol"] == symbol].copy()
        symbol_df = symbol_df.sort_values("open_time")
        processed = engineer.process_symbol(symbol_df)
        processed_data.append(processed)

    # Combine
    all_processed = pd.concat(processed_data, ignore_index=True)

    # Save
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    all_processed.to_parquet(output_path / "features.parquet", index=False)
    logger.info(f"Saved processed features: {len(all_processed)} rows")


if __name__ == "__main__":
    main()
