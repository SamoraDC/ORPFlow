"""
Feature Engineering Module
Calculates trading features using ONLY past data (no leakage)

CRITICAL: All rolling windows look BACKWARD only!
"""

import numpy as np
import pandas as pd
from typing import List


class FeatureEngineer:
    """
    Generate trading features from raw market data.

    ANTI-LEAKAGE GUARANTEES:
    1. All rolling windows use past data only (no future data)
    2. No bfill() or future-looking interpolation
    3. Returns are calculated using shift(1) which looks at past
    4. All momentum indicators use historical data only
    """

    def __init__(self, windows: List[int] = [5, 10, 20, 50, 100]):
        self.windows = windows
        self.warmup_period = max(windows)

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate return metrics using PAST data only.
        shift(n) looks at n periods in the PAST.
        """
        df = df.copy()

        # Simple returns (current vs previous)
        df["return_1"] = df["close"].pct_change()

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Multi-period returns (all look at PAST)
        for w in self.windows:
            df[f"return_{w}"] = df["close"].pct_change(w)
            df[f"log_return_{w}"] = np.log(df["close"] / df["close"].shift(w))

        return df

    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility using PAST data only.
        rolling(window=w) uses the last w observations.
        """
        df = df.copy()

        for w in self.windows:
            # Rolling standard deviation of log returns
            df[f"volatility_{w}"] = (
                df["log_return"].rolling(window=w).std() * np.sqrt(252 * 24 * 60)
            )

            # Parkinson volatility (high-low based)
            log_hl_sq = np.log(df["high"] / df["low"]) ** 2
            df[f"parkinson_vol_{w}"] = np.sqrt(
                (1 / (4 * np.log(2))) * log_hl_sq.rolling(window=w).mean()
            ) * np.sqrt(252 * 24 * 60)

            # Garman-Klass volatility
            log_hl = np.log(df["high"] / df["low"]) ** 2
            log_co = np.log(df["close"] / df["open"]) ** 2
            df[f"gk_vol_{w}"] = np.sqrt(
                (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=w).mean()
            ) * np.sqrt(252 * 24 * 60)

        return df

    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators using PAST data only."""
        df = df.copy()

        for w in self.windows:
            # Price momentum (current vs w periods ago)
            df[f"momentum_{w}"] = df["close"] / df["close"].shift(w) - 1

            # Rate of change
            df[f"roc_{w}"] = (
                (df["close"] - df["close"].shift(w)) / df["close"].shift(w) * 100
            )

            # Moving average (uses last w observations)
            df[f"ma_{w}"] = df["close"].rolling(window=w).mean()
            df[f"ma_cross_{w}"] = (df["close"] - df[f"ma_{w}"]) / df[f"ma_{w}"]

        # RSI (uses past data via rolling)
        for w in [14, 21]:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            df[f"rsi_{w}"] = 100 - (100 / (1 + rs))

        return df

    def calculate_orderflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate orderflow features using PAST data only."""
        df = df.copy()

        # Spread proxy
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"] * 10000

        # Volume imbalance
        df["volume_imbalance"] = (
            (df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])) /
            df["volume"]
        )

        # Order flow imbalance
        df["ofi"] = df["taker_buy_base"] / df["volume"]

        for w in self.windows:
            df[f"ofi_ma_{w}"] = df["ofi"].rolling(window=w).mean()
            df[f"ofi_std_{w}"] = df["ofi"].rolling(window=w).std()
            df[f"volume_ma_{w}"] = df["volume"].rolling(window=w).mean()
            df[f"volume_std_{w}"] = df["volume"].rolling(window=w).std()
            df[f"trades_ma_{w}"] = df["trades"].rolling(window=w).mean()

        return df

    def calculate_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate microstructure features using PAST data only."""
        df = df.copy()

        # Amihud illiquidity
        df["amihud"] = np.abs(df["log_return"]) / df["quote_volume"]

        for w in self.windows:
            df[f"amihud_ma_{w}"] = df["amihud"].rolling(window=w).mean()

            # Kyle's Lambda proxy
            df[f"kyle_lambda_{w}"] = (
                df["log_return"].rolling(window=w).std() /
                df["volume"].rolling(window=w).mean()
            )

        return df

    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features (no leakage - current time only)."""
        df = df.copy()

        df["hour"] = df["open_time"].dt.hour
        df["day_of_week"] = df["open_time"].dt.dayofweek

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    def calculate_targets(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 5, 15, 30]
    ) -> pd.DataFrame:
        """
        Calculate prediction targets (FUTURE values).
        shift(-h) looks at h periods in the FUTURE.
        These are what we want to PREDICT, not features!
        """
        df = df.copy()

        for h in horizons:
            # Future return (what we want to predict)
            df[f"target_return_{h}"] = df["close"].shift(-h) / df["close"] - 1

            # Future direction
            df[f"target_direction_{h}"] = (df[f"target_return_{h}"] > 0).astype(int)

        return df

    def process(self, df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
        """
        Full feature engineering pipeline.

        Args:
            df: Raw OHLCV DataFrame (must be sorted by time!)
            include_targets: Whether to calculate targets

        Returns:
            DataFrame with all features (and optionally targets)
        """
        # Ensure sorted by time
        df = df.sort_values("open_time").reset_index(drop=True)

        # Calculate all features (PAST data only)
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df)
        df = self.calculate_momentum(df)
        df = self.calculate_orderflow(df)
        df = self.calculate_microstructure(df)
        df = self.calculate_time_features(df)

        # Calculate targets if requested (FUTURE data)
        if include_targets:
            df = self.calculate_targets(df)

        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding targets and metadata).

    CRITICAL: This ensures we only use features, never targets!
    """
    exclude_prefixes = ["target_", "open_time", "close_time", "symbol", "ignore"]
    exclude_cols = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote", "hour", "day_of_week"
    ]

    feature_cols = []
    for col in df.columns:
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if col in exclude_cols:
            continue
        feature_cols.append(col)

    return feature_cols
