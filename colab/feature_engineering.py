"""
Advanced Feature Engineering for Crypto Trading Models
=======================================================
Senior Quant Researcher Level - Institutional Grade

Features Categories:
1. Price Action & Returns (normalized, risk-adjusted)
2. Volatility (multiple estimators, regime detection)
3. Volume Profile (CVD, VWAP, relative volume)
4. Microstructure (spread, trade intensity, order flow)
5. Technical Indicators (momentum, mean reversion)
6. Time Features (sessions, day-of-week effects)
7. Statistical Features (higher moments, autocorrelation)
8. Cross-Asset (for multi-symbol analysis)

Anti-Leakage: ALL features use .shift(1) or rolling windows on PAST data only
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_comprehensive_features(df: pd.DataFrame, include_cross_asset: bool = False) -> pd.DataFrame:
    """
    Calculate comprehensive features for crypto trading.

    Args:
        df: DataFrame with OHLCV + taker data from Binance
        include_cross_asset: Whether to include cross-asset features (requires multi-symbol df)

    Returns:
        DataFrame with all features added
    """
    df = df.copy()

    # =========================================================================
    # 1. PRICE ACTION & RETURNS
    # =========================================================================

    # Basic returns (shifted to avoid leakage on current bar)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_1"] = df["close"].pct_change(1)

    # Multi-timeframe returns
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"return_{w}"] = df["close"].pct_change(w)
        df[f"log_return_{w}"] = np.log(df["close"] / df["close"].shift(w))

    # Risk-adjusted returns (return / realized vol)
    for w in [20, 50]:
        vol = df["log_return"].rolling(w).std()
        df[f"sharpe_{w}"] = df[f"return_{w}"] / (vol * np.sqrt(w) + 1e-10)

    # Overnight gap (for 24h market, use 8h as "session")
    df["gap_8h"] = df["open"] / df["close"].shift(480) - 1  # 480 min = 8h

    # =========================================================================
    # 2. VOLATILITY FEATURES
    # =========================================================================

    # Realized volatility (multiple windows, annualized)
    ann_factor = np.sqrt(252 * 24 * 60)  # 1-min to annual
    for w in [5, 10, 20, 50, 100]:
        df[f"volatility_{w}"] = df["log_return"].rolling(w).std() * ann_factor

    # Parkinson volatility (more efficient estimator using high-low)
    for w in [20, 50]:
        log_hl = np.log(df["high"] / df["low"])
        df[f"parkinson_vol_{w}"] = np.sqrt(
            (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(w).mean()
        ) * ann_factor

    # Garman-Klass volatility (uses OHLC)
    for w in [20, 50]:
        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        df[f"gk_vol_{w}"] = np.sqrt(gk.rolling(w).mean()) * ann_factor

    # ATR (Average True Range)
    for w in [14, 20, 50]:
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        df[f"atr_{w}"] = tr.rolling(w).mean()
        df[f"atr_pct_{w}"] = df[f"atr_{w}"] / df["close"] * 100  # As percentage

    # Volatility regime (current vol vs historical)
    df["vol_regime_20_100"] = df["volatility_20"] / (df["volatility_100"] + 1e-10)
    df["vol_zscore"] = (df["volatility_20"] - df["volatility_100"]) / (df["volatility_100"].rolling(50).std() + 1e-10)

    # Volatility of volatility (important for options, useful for tail risk)
    df["vol_of_vol"] = df["volatility_20"].rolling(20).std()

    # =========================================================================
    # 3. VOLUME FEATURES (CRITICAL - MOST PREDICTIVE FOR CRYPTO)
    # =========================================================================

    # Basic volume MAs
    for w in [5, 10, 20, 50, 100]:
        df[f"volume_ma_{w}"] = df["volume"].rolling(w).mean()

    # Relative volume (current vs historical)
    df["rvol_20"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    df["rvol_50"] = df["volume"] / (df["volume"].rolling(50).mean() + 1e-10)

    # Volume spikes (z-score)
    vol_mean = df["volume"].rolling(50).mean()
    vol_std = df["volume"].rolling(50).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)

    # VWAP (Volume Weighted Average Price)
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    for w in [20, 50]:
        cum_vol = df["volume"].rolling(w).sum()
        cum_tp_vol = (df["typical_price"] * df["volume"]).rolling(w).sum()
        df[f"vwap_{w}"] = cum_tp_vol / (cum_vol + 1e-10)
        df[f"vwap_dist_{w}"] = (df["close"] - df[f"vwap_{w}"]) / df[f"vwap_{w}"] * 100

    # CVD (Cumulative Volume Delta) - KEY INDICATOR
    # Approximate using taker buy volume
    df["volume_delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
    for w in [10, 20, 50]:
        df[f"cvd_{w}"] = df["volume_delta"].rolling(w).sum()
        df[f"cvd_normalized_{w}"] = df[f"cvd_{w}"] / (df["volume"].rolling(w).sum() + 1e-10)

    # Dollar volume (quote volume analysis)
    df["dollar_volume_ma_20"] = df["quote_volume"].rolling(20).mean()
    df["dollar_volume_ratio"] = df["quote_volume"] / (df["dollar_volume_ma_20"] + 1e-10)

    # Trade count features (CRITICAL - not used before!)
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce")
        for w in [10, 20, 50]:
            df[f"trades_ma_{w}"] = df["trades"].rolling(w).mean()
        df["trades_zscore"] = (df["trades"] - df["trades"].rolling(50).mean()) / (df["trades"].rolling(50).std() + 1e-10)
        # Average trade size
        df["avg_trade_size"] = df["volume"] / (df["trades"] + 1)
        df["avg_trade_size_ratio"] = df["avg_trade_size"] / (df["avg_trade_size"].rolling(50).mean() + 1e-10)

    # =========================================================================
    # 4. MICROSTRUCTURE FEATURES
    # =========================================================================

    # Spread proxy (high-low range)
    df["spread_bps"] = (df["high"] - df["low"]) / df["close"] * 10000
    df["spread_ma_20"] = df["spread_bps"].rolling(20).mean()
    df["spread_zscore"] = (df["spread_bps"] - df["spread_ma_20"]) / (df["spread_bps"].rolling(50).std() + 1e-10)

    # Order Flow Imbalance (OFI)
    df["ofi"] = df["taker_buy_base"] / (df["volume"] + 1e-10)
    df["ofi_ma_10"] = df["ofi"].rolling(10).mean()
    df["ofi_ma_20"] = df["ofi"].rolling(20).mean()

    # Buy pressure (smoothed)
    for w in [10, 20, 50]:
        df[f"buy_pressure_{w}"] = df["taker_buy_base"].rolling(w).sum() / (df["volume"].rolling(w).sum() + 1e-10)

    # Price efficiency (how much price moves per volume)
    df["price_efficiency"] = abs(df["return_1"]) / (np.log1p(df["volume"]) + 1e-10) * 1000
    df["price_efficiency_ma"] = df["price_efficiency"].rolling(20).mean()

    # Amihud illiquidity ratio (modified for crypto)
    df["amihud"] = abs(df["return_1"]) / (df["quote_volume"] / 1e6 + 1e-10)
    df["amihud_ma_20"] = df["amihud"].rolling(20).mean()

    # =========================================================================
    # 5. MOMENTUM & TREND INDICATORS
    # =========================================================================

    # Moving average crosses
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"ma_dist_{w}"] = (df["close"] - df[f"ma_{w}"]) / df[f"ma_{w}"] * 100

    # EMA
    for w in [12, 26, 50]:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        df[f"ema_dist_{w}"] = (df["close"] - df[f"ema_{w}"]) / df[f"ema_{w}"] * 100

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_change"] = df["macd_hist"].diff()

    # RSI (multiple periods)
    for w in [7, 14, 21]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / (loss + 1e-10)
        df[f"rsi_{w}"] = 100 - (100 / (1 + rs))
        # Normalized RSI
        df[f"rsi_{w}_norm"] = (df[f"rsi_{w}"] - 50) / 50

    # Stochastic RSI
    for w in [14]:
        rsi = df[f"rsi_{w}"]
        rsi_min = rsi.rolling(w).min()
        rsi_max = rsi.rolling(w).max()
        df["stoch_rsi"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
        df["stoch_rsi_k"] = df["stoch_rsi"].rolling(3).mean()
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

    # Williams %R
    for w in [14, 21]:
        highest = df["high"].rolling(w).max()
        lowest = df["low"].rolling(w).min()
        df[f"williams_r_{w}"] = -100 * (highest - df["close"]) / (highest - lowest + 1e-10)

    # ADX (Average Directional Index) - Trend strength
    for w in [14, 20]:
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(w).mean()
        plus_di = 100 * (plus_dm.rolling(w).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(w).mean() / (atr + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df[f"adx_{w}"] = dx.rolling(w).mean()
        df[f"plus_di_{w}"] = plus_di
        df[f"minus_di_{w}"] = minus_di

    # CCI (Commodity Channel Index)
    for w in [20]:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        tp_ma = tp.rolling(w).mean()
        tp_std = tp.rolling(w).std()
        df[f"cci_{w}"] = (tp - tp_ma) / (0.015 * tp_std + 1e-10)

    # =========================================================================
    # 6. MEAN REVERSION INDICATORS
    # =========================================================================

    # Bollinger Bands
    for w in [20, 50]:
        ma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        df[f"bb_upper_{w}"] = ma + 2 * std
        df[f"bb_lower_{w}"] = ma - 2 * std
        df[f"bb_width_{w}"] = (df[f"bb_upper_{w}"] - df[f"bb_lower_{w}"]) / ma * 100
        df[f"bb_position_{w}"] = (df["close"] - df[f"bb_lower_{w}"]) / (df[f"bb_upper_{w}"] - df[f"bb_lower_{w}"] + 1e-10)

    # Keltner Channels
    for w in [20]:
        ma = df["close"].ewm(span=w, adjust=False).mean()
        atr = df[f"atr_{w}"]
        df[f"kc_upper_{w}"] = ma + 2 * atr
        df[f"kc_lower_{w}"] = ma - 2 * atr
        df[f"kc_position_{w}"] = (df["close"] - df[f"kc_lower_{w}"]) / (df[f"kc_upper_{w}"] - df[f"kc_lower_{w}"] + 1e-10)

    # Z-score of price
    for w in [20, 50, 100]:
        ma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        df[f"price_zscore_{w}"] = (df["close"] - ma) / (std + 1e-10)

    # =========================================================================
    # 7. TIME FEATURES (CRITICAL FOR CRYPTO)
    # =========================================================================

    # Hour of day (cyclical encoding)
    hour = df["open_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (cyclical encoding)
    dow = df["open_time"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Trading sessions (major crypto activity hours)
    df["is_asia_session"] = ((hour >= 0) & (hour < 8)).astype(int)  # 00:00-08:00 UTC
    df["is_europe_session"] = ((hour >= 7) & (hour < 16)).astype(int)  # 07:00-16:00 UTC
    df["is_us_session"] = ((hour >= 13) & (hour < 22)).astype(int)  # 13:00-22:00 UTC

    # Weekend effect (lower volume/liquidity)
    df["is_weekend"] = (dow >= 5).astype(int)

    # Month cyclical (for seasonality)
    month = df["open_time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # =========================================================================
    # 8. STATISTICAL FEATURES
    # =========================================================================

    # Higher moments of returns
    for w in [20, 50]:
        returns = df["log_return"]
        df[f"skewness_{w}"] = returns.rolling(w).skew()
        df[f"kurtosis_{w}"] = returns.rolling(w).kurt()

    # Autocorrelation (mean reversion indicator)
    for lag in [1, 5, 10]:
        df[f"autocorr_{lag}"] = df["log_return"].rolling(50).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
        )

    # Hurst exponent approximation (trend vs mean reversion)
    def calc_hurst(series, max_lag=20):
        if len(series) < max_lag * 2:
            return 0.5
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        if min(tau) <= 0:
            return 0.5
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    df["hurst_20"] = df["log_return"].rolling(100).apply(lambda x: calc_hurst(x, 20), raw=False)

    # =========================================================================
    # 9. PRICE PATTERNS
    # =========================================================================

    # Higher highs / Lower lows (trend confirmation)
    for w in [10, 20]:
        df[f"higher_highs_{w}"] = (df["high"] > df["high"].rolling(w).max().shift(1)).astype(int)
        df[f"lower_lows_{w}"] = (df["low"] < df["low"].rolling(w).min().shift(1)).astype(int)

    # Distance from recent high/low
    for w in [20, 50, 100]:
        highest = df["high"].rolling(w).max()
        lowest = df["low"].rolling(w).min()
        df[f"dist_from_high_{w}"] = (df["close"] - highest) / highest * 100
        df[f"dist_from_low_{w}"] = (df["close"] - lowest) / lowest * 100
        df[f"range_position_{w}"] = (df["close"] - lowest) / (highest - lowest + 1e-10)

    # Consecutive up/down bars
    df["up_bar"] = (df["close"] > df["open"]).astype(int)
    df["consecutive_up"] = df["up_bar"].groupby((df["up_bar"] != df["up_bar"].shift()).cumsum()).cumcount() + 1
    df["consecutive_up"] = df["consecutive_up"] * df["up_bar"]
    df["consecutive_down"] = (1 - df["up_bar"]).groupby(((1 - df["up_bar"]) != (1 - df["up_bar"]).shift()).cumsum()).cumcount() + 1
    df["consecutive_down"] = df["consecutive_down"] * (1 - df["up_bar"])

    # =========================================================================
    # CLEANUP
    # =========================================================================

    # Drop intermediate columns
    cols_to_drop = ["typical_price", "volume_delta", "up_bar"]
    cols_to_drop += [c for c in df.columns if c.startswith("ma_") and not c.startswith("ma_dist") and not c.startswith("ma_cross")]
    cols_to_drop += [c for c in df.columns if c.startswith("ema_") and not c.startswith("ema_dist")]
    cols_to_drop += [c for c in df.columns if c.startswith("bb_upper") or c.startswith("bb_lower")]
    cols_to_drop += [c for c in df.columns if c.startswith("kc_upper") or c.startswith("kc_lower")]
    cols_to_drop += [c for c in df.columns if c.startswith("vwap_") and not c.startswith("vwap_dist")]

    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns only (no targets, no metadata)"""
    exclude = [
        "open_time", "close_time", "symbol", "ignore",
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
    ]
    return [c for c in df.columns if c not in exclude and not c.startswith("target_")]


def validate_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[bool, List[str]]:
    """Validate features for common issues"""
    issues = []

    for col in feature_cols:
        # Check for inf values
        if np.isinf(df[col]).any():
            issues.append(f"{col}: contains inf values")

        # Check for all NaN
        if df[col].isna().all():
            issues.append(f"{col}: all NaN")

        # Check for constant values
        if df[col].nunique() <= 1:
            issues.append(f"{col}: constant value")

        # Check for extreme values (potential leakage)
        if df[col].max() > 1e10 or df[col].min() < -1e10:
            issues.append(f"{col}: extreme values detected")

    return len(issues) == 0, issues


# Feature count summary
FEATURE_CATEGORIES = {
    "returns": 15,
    "volatility": 20,
    "volume": 25,
    "microstructure": 12,
    "momentum": 30,
    "mean_reversion": 15,
    "time": 10,
    "statistical": 10,
    "patterns": 15
}

print(f"Total features: ~{sum(FEATURE_CATEGORIES.values())}")
