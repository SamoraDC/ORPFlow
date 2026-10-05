"""
Feature Engineering and Data Preprocessing
Transforms raw market data into ML-ready features

Enhanced with advanced quantitative methods:
- Signal Processing (Kalman, EMD, HHT, Wavelets, RMT, Fisher)
- Advanced Microstructure (207 features)
- Triple Barrier Labeling & Meta-Labeling
- Hawkes Processes for order flow modeling
- Feature Selection (SHAP, RFE, PBO, VIF)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Advanced modules
from .signal_processing import (
    KalmanFilter,
    EMD,
    HilbertHuangTransform,
    WaveletTransform,
    RandomMatrixTheory,
    FisherTransform,
    kalman_smooth_prices,
    emd_decompose,
    wavelet_denoise,
)
from .microstructure import (
    MicrostructureAnalyzer,
    calculate_all_microstructure_features,
)
from .hawkes_processes import (
    HawkesTradingAnalyzer,
    BidAskHawkes,
)
from .triple_barrier import (
    TripleBarrierLabeler,
    BarrierConfig,
    cusum_filter,
    get_daily_vol,
    detect_combined_regime,
)

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

    def calculate_signal_processing_features(
        self,
        df: pd.DataFrame,
        use_kalman: bool = True,
        use_emd: bool = True,
        use_wavelets: bool = True,
        use_fisher: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate advanced signal processing features.

        Includes:
        - Kalman Filter: Smoothed prices, trend extraction
        - EMD: Intrinsic Mode Functions decomposition
        - Wavelets: Multi-scale volatility and denoising
        - Fisher Transform: Normalized indicators
        """
        df = df.copy()

        # ==== Kalman Filter Features ====
        if use_kalman:
            try:
                # Smooth prices with Kalman filter
                kalman_smoothed = kalman_smooth_prices(
                    df["close"].values,
                    process_var=1e-5,
                    obs_var=1e-3
                )
                df["kalman_price"] = kalman_smoothed
                df["kalman_deviation"] = (df["close"] - df["kalman_price"]) / df["kalman_price"]

                # Kalman trend (diff of smoothed)
                df["kalman_trend"] = df["kalman_price"].pct_change()

                # Kalman on volume
                kalman_vol = kalman_smooth_prices(
                    df["volume"].values,
                    process_var=1e-4,
                    obs_var=1e-2
                )
                df["kalman_volume"] = kalman_vol
                df["volume_kalman_ratio"] = df["volume"] / (df["kalman_volume"] + 1e-10)

                logger.debug("Kalman features calculated")
            except Exception as e:
                logger.warning(f"Kalman filter failed: {e}")

        # ==== EMD Features ====
        if use_emd:
            try:
                # Apply EMD to log prices
                log_prices = np.log(df["close"].values + 1e-10)
                imfs, residue = emd_decompose(log_prices, max_imfs=5)

                # Store IMFs (first 3 + residue)
                for i, imf in enumerate(imfs[:3]):
                    df[f"emd_imf_{i+1}"] = imf

                df["emd_residue"] = residue

                # Energy of each IMF
                for i, imf in enumerate(imfs[:3]):
                    df[f"emd_energy_{i+1}"] = imf ** 2

                logger.debug("EMD features calculated")
            except Exception as e:
                logger.warning(f"EMD decomposition failed: {e}")

        # ==== Wavelet Features ====
        if use_wavelets:
            try:
                wt = WaveletTransform(wavelet='db4')

                # Multi-scale volatility using volatility_regime_detection
                returns = df["close"].pct_change().fillna(0).values
                vol_analysis = wt.volatility_regime_detection(returns, level=4)

                for name, values in vol_analysis.items():
                    # Ensure matching length
                    if len(values) < len(df):
                        values = np.pad(values, (0, len(df) - len(values)), mode='edge')
                    elif len(values) > len(df):
                        values = values[:len(df)]
                    df[f"wavelet_{name}"] = values

                # Denoised price
                denoised = wavelet_denoise(df["close"].values, wavelet='db4', level=3)
                df["wavelet_price"] = denoised
                df["wavelet_deviation"] = (df["close"] - df["wavelet_price"]) / df["wavelet_price"]

                logger.debug("Wavelet features calculated")
            except Exception as e:
                logger.warning(f"Wavelet transform failed: {e}")

        # ==== Fisher Transform Features ====
        if use_fisher:
            try:
                ft = FisherTransform(clip_value=0.999)

                # Fisher transform of RSI
                if "rsi_14" in df.columns:
                    rsi_normalized = (df["rsi_14"] - 50) / 50  # Scale to [-1, 1]
                    rsi_normalized = rsi_normalized.clip(-0.999, 0.999)
                    df["fisher_rsi"] = ft.transform(rsi_normalized.values)

                # Fisher transform of price position
                for w in [10, 20]:
                    high_w = df["close"].rolling(window=w).max()
                    low_w = df["close"].rolling(window=w).min()
                    mid = (high_w + low_w) / 2
                    price_pos = (df["close"] - mid) / ((high_w - low_w) / 2 + 1e-10)
                    price_pos = price_pos.clip(-0.999, 0.999).fillna(0)
                    df[f"fisher_price_{w}"] = ft.transform(price_pos.values)

                logger.debug("Fisher transform features calculated")
            except Exception as e:
                logger.warning(f"Fisher transform failed: {e}")

        return df

    def calculate_advanced_microstructure(
        self,
        df: pd.DataFrame,
        bid: Optional[pd.Series] = None,
        ask: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Calculate advanced microstructure features.

        Uses the MicrostructureAnalyzer for comprehensive order flow,
        liquidity, and market quality metrics.
        """
        df = df.copy()

        try:
            # Calculate comprehensive microstructure features using the convenience function
            # It expects a DataFrame with OHLCV columns
            micro_df = calculate_all_microstructure_features(df)

            # Merge microstructure features with original dataframe
            for col in micro_df.columns:
                if col not in df.columns:
                    df[f"micro_{col}"] = micro_df[col].values

            logger.info(f"Added microstructure features")

        except Exception as e:
            logger.warning(f"Advanced microstructure calculation failed: {e}")

        return df

    def calculate_hawkes_features(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> pd.DataFrame:
        """
        Calculate Hawkes process features for order flow modeling.

        Hawkes processes model self-exciting point processes,
        useful for detecting order clustering and market toxicity.
        """
        df = df.copy()

        try:
            # Get trade times and signs
            if "open_time" in df.columns:
                times = (df["open_time"] - df["open_time"].min()).dt.total_seconds().values
            else:
                times = np.arange(len(df)).astype(float)

            # Calculate trade direction
            trade_signs = np.sign(df["close"].diff()).fillna(1).values

            # Rolling Hawkes intensity estimation
            analyzer = HawkesTradingAnalyzer()

            # Calculate rolling intensity features
            intensities = []
            branching_ratios = []

            for i in range(window, len(df)):
                window_times = times[i-window:i]
                window_times = window_times - window_times[0]  # Normalize to start at 0

                try:
                    # Fit Hawkes model to window
                    analyzer.fit_from_trades(
                        trade_times=window_times,
                        trade_signs=trade_signs[i-window:i].astype(float)
                    )

                    # Get current intensity and branching ratio
                    intensity = analyzer.hawkes.get_intensity(window_times[-1])
                    branching = analyzer.hawkes.get_branching_ratio()

                    intensities.append(intensity)
                    branching_ratios.append(branching)

                except Exception:
                    # Use previous value or default
                    intensities.append(intensities[-1] if intensities else 0.1)
                    branching_ratios.append(branching_ratios[-1] if branching_ratios else 0.5)

            # Pad beginning with first valid value
            intensities = [intensities[0]] * window + intensities
            branching_ratios = [branching_ratios[0]] * window + branching_ratios

            df["hawkes_intensity"] = intensities
            df["hawkes_branching"] = branching_ratios

            # Derived features
            df["hawkes_intensity_z"] = (
                (df["hawkes_intensity"] - df["hawkes_intensity"].rolling(50).mean()) /
                (df["hawkes_intensity"].rolling(50).std() + 1e-10)
            )

            # High branching ratio indicates clustering
            df["hawkes_clustering"] = df["hawkes_branching"] > 0.7

            logger.debug("Hawkes features calculated")

        except Exception as e:
            logger.warning(f"Hawkes features calculation failed: {e}")
            # Add placeholder columns
            df["hawkes_intensity"] = 0.1
            df["hawkes_branching"] = 0.5
            df["hawkes_intensity_z"] = 0.0
            df["hawkes_clustering"] = False

        return df

    def calculate_regime_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate market regime features.

        Detects volatility, trend, and Hurst-based regimes
        for adaptive model behavior.
        """
        df = df.copy()

        try:
            # Calculate combined regime
            regimes = detect_combined_regime(
                close=df["close"],
                high=df.get("high"),
                low=df.get("low"),
            )

            # Add regime columns
            for col in regimes.columns:
                df[f"regime_{col}"] = regimes[col].values

            # One-hot encode combined regime
            regime_dummies = pd.get_dummies(
                regimes["combined_regime"],
                prefix="regime"
            )
            for col in regime_dummies.columns:
                df[col] = regime_dummies[col].values

            # Calculate daily volatility target
            df["daily_vol"] = get_daily_vol(df["close"])

            logger.debug("Regime features calculated")

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            df["daily_vol"] = df["close"].pct_change().rolling(20).std()

        return df

    def process_symbol_advanced(
        self,
        df: pd.DataFrame,
        use_signal_processing: bool = True,
        use_advanced_micro: bool = True,
        use_hawkes: bool = False,  # Disabled by default (slow)
        use_regimes: bool = True,
    ) -> pd.DataFrame:
        """
        Process a single symbol's data with ALL advanced features.

        This is the enhanced version that includes:
        - Basic features (returns, volatility, momentum)
        - Signal processing (Kalman, EMD, Wavelets, Fisher)
        - Advanced microstructure (207 features)
        - Hawkes processes (order clustering)
        - Regime detection (volatility, trend, Hurst)

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data
        use_signal_processing : bool
            Include Kalman, EMD, Wavelets, Fisher features
        use_advanced_micro : bool
            Include 207 microstructure features
        use_hawkes : bool
            Include Hawkes process features (slow, default False)
        use_regimes : bool
            Include regime detection features

        Returns
        -------
        pd.DataFrame
            Processed dataframe with all features
        """
        logger.info(f"Processing {len(df)} rows with advanced features...")

        # Basic features
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df)
        df = self.calculate_momentum(df)
        df = self.calculate_orderbook_features(df)
        df = self.calculate_microstructure(df)  # Basic microstructure

        # Advanced features
        if use_signal_processing:
            logger.info("Calculating signal processing features...")
            df = self.calculate_signal_processing_features(df)

        if use_advanced_micro:
            logger.info("Calculating advanced microstructure features...")
            df = self.calculate_advanced_microstructure(df)

        if use_hawkes:
            logger.info("Calculating Hawkes features (this may take a while)...")
            df = self.calculate_hawkes_features(df)

        if use_regimes:
            logger.info("Calculating regime features...")
            df = self.calculate_regime_features(df)

        # Targets
        df = self.calculate_targets(df)
        df = self.add_time_features(df)

        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get feature and target columns
        feature_cols = self.get_feature_columns(df)
        target_cols = [c for c in df.columns if c.startswith("target_")]

        # Separate numeric and categorical columns
        numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_feature_cols = [c for c in feature_cols if c not in numeric_feature_cols]

        logger.info(f"Numeric features: {len(numeric_feature_cols)}, Categorical: {len(categorical_feature_cols)}")

        # Fill NaN in numeric features with forward/backward fill, then median
        for col in numeric_feature_cols:
            df[col] = df[col].ffill().bfill()
            # If still has NaN (all values were NaN), fill with 0
            if df[col].isna().any():
                col_median = df[col].median()
                fill_val = col_median if pd.notna(col_median) else 0.0
                df[col] = df[col].fillna(fill_val)

        # Fill NaN in categorical features with mode or 'unknown'
        for col in categorical_feature_cols:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'unknown'
            df[col] = df[col].fillna(fill_val)

        # Drop rows with NaN targets only (these are at the end due to forward shift)
        initial_len = len(df)
        df = df.dropna(subset=target_cols)
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN targets (end of series)")

        # Check for remaining NaN (should be minimal now)
        remaining_nan = df[numeric_feature_cols + target_cols].isna().any(axis=1).sum()
        if remaining_nan > 0:
            logger.warning(f"Found {remaining_nan} rows with remaining NaN, filling with 0")
            # Fill remaining NaN instead of dropping
            for col in numeric_feature_cols:
                df[col] = df[col].fillna(0.0)

        # Update feature_cols to only include numeric for ML (exclude categorical for now)
        # Categorical columns will be encoded separately if needed
        self._numeric_feature_cols = numeric_feature_cols
        self._categorical_feature_cols = categorical_feature_cols

        logger.info(f"Final shape: {df.shape}, Numeric Features: {len(numeric_feature_cols)}")

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

            # Volatility target - realized volatility over next h periods
            # For h=1, use absolute return as proxy (no rolling possible)
            if h == 1:
                df[f"target_vol_{h}"] = np.abs(df["log_return"].shift(-1)) * np.sqrt(252 * 24 * 60)
            else:
                # Forward-looking realized volatility: std of next h log returns
                df[f"target_vol_{h}"] = df["log_return"].shift(-h).rolling(window=h).std() * np.sqrt(252 * 24 * 60)

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

        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get feature and target columns
        feature_cols = self.get_feature_columns(df)
        target_cols = [c for c in df.columns if c.startswith("target_")]

        # For features: forward fill the warmup period NaN, then backfill any remaining
        df[feature_cols] = df[feature_cols].ffill().bfill()

        # Drop rows where ANY target is NaN (end of data due to shift)
        initial_len = len(df)
        df = df.dropna(subset=target_cols)
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN targets (end of series)")

        # Final safety: drop any remaining NaN rows
        remaining_nan = df[feature_cols + target_cols].isna().any(axis=1).sum()
        if remaining_nan > 0:
            logger.warning(f"Found {remaining_nan} remaining NaN rows, dropping...")
            df = df.dropna(subset=feature_cols + target_cols)

        return df

    def get_feature_columns(self, df: pd.DataFrame, numeric_only: bool = False) -> List[str]:
        """Get list of feature columns (excluding targets and metadata)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to get columns from
        numeric_only : bool
            If True, return only numeric columns suitable for ML training
        """
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

        if numeric_only:
            # Return only numeric columns
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            return numeric_cols

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
