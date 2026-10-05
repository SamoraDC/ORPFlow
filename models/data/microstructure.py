"""
Advanced Market Microstructure Module.

This module provides comprehensive market microstructure analysis for quantitative
trading, implementing advanced metrics for order flow, micro-price, book dynamics,
spreads, liquidity, and realized volatility.

All functions are designed to work with pandas DataFrames and support vectorized
operations for high performance.

Author: ORPFlow Team
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# ORDER FLOW ANALYSIS
# ==============================================================================


def order_flow_imbalance(
    df: pd.DataFrame,
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    decay: float = 0.0,
    window: int = 20,
) -> pd.DataFrame:
    """
    Calculate advanced Order Flow Imbalance (OFI) with optional exponential decay.

    OFI measures the net pressure from aggressive buyers vs sellers.
    With decay, recent order flow is weighted more heavily.

    Args:
        df: DataFrame with OHLCV data
        volume_col: Column name for total volume
        taker_buy_col: Column name for taker buy volume
        decay: Exponential decay factor (0 = no decay, higher = faster decay)
        window: Window size for rolling calculations

    Returns:
        DataFrame with OFI features added

    Reference:
        Cont, R., Kukanov, A., & Stoikov, S. (2014).
        "The Price Impact of Order Book Events"
    """
    result = df.copy()

    # Calculate buy and sell volumes
    buy_vol = result[taker_buy_col]
    sell_vol = result[volume_col] - result[taker_buy_col]

    # Basic OFI (normalized)
    total_vol = buy_vol + sell_vol
    result["ofi_basic"] = np.where(
        total_vol > 0,
        (buy_vol - sell_vol) / total_vol,
        0.0
    )

    # OFI with exponential decay (EMA-style)
    if decay > 0:
        alpha = 1 - np.exp(-decay)
        result["ofi_decayed"] = result["ofi_basic"].ewm(alpha=alpha, adjust=False).mean()
    else:
        result["ofi_decayed"] = result["ofi_basic"]

    # Rolling OFI statistics
    result[f"ofi_ma_{window}"] = result["ofi_basic"].rolling(window=window).mean()
    result[f"ofi_std_{window}"] = result["ofi_basic"].rolling(window=window).std()
    result[f"ofi_z_{window}"] = (
        (result["ofi_basic"] - result[f"ofi_ma_{window}"]) /
        result[f"ofi_std_{window}"].replace(0, np.nan)
    )

    # Cumulative OFI with decay
    result["ofi_cumulative"] = (buy_vol - sell_vol).cumsum()
    if decay > 0:
        weights = np.exp(-decay * np.arange(len(df))[::-1])
        weights = weights / weights.sum()
        result["ofi_weighted_cum"] = np.convolve(
            (buy_vol - sell_vol).values,
            weights,
            mode="full"
        )[:len(df)]

    # OFI momentum (rate of change)
    result["ofi_momentum"] = result["ofi_basic"].diff(window)

    # OFI acceleration
    result["ofi_acceleration"] = result["ofi_momentum"].diff()

    return result


def vpin(
    df: pd.DataFrame,
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    close_col: str = "close",
    bucket_size: Optional[float] = None,
    n_buckets: int = 50,
    sigma_window: int = 100,
) -> pd.DataFrame:
    """
    Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN estimates order flow toxicity using volume-synchronized sampling.
    Higher VPIN indicates higher probability of informed trading.

    Args:
        df: DataFrame with OHLCV data
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume
        close_col: Column for close price
        bucket_size: Volume per bucket (default: mean volume * 10)
        n_buckets: Number of buckets for rolling VPIN
        sigma_window: Window for volatility estimation in BVC

    Returns:
        DataFrame with VPIN features added

    Reference:
        Easley, D., Lopez de Prado, M., & O'Hara, M. (2012).
        "Flow Toxicity and Liquidity in a High Frequency World"
    """
    result = df.copy()

    # Use taker data for classification if available
    buy_vol = result[taker_buy_col]
    sell_vol = result[volume_col] - result[taker_buy_col]

    # If no taker data, use Bulk Volume Classification (BVC)
    if buy_vol.isna().all() or (buy_vol == 0).all():
        returns = np.log(result[close_col] / result[close_col].shift(1))
        sigma = returns.rolling(window=sigma_window).std()

        # Normalized price change
        z = returns / sigma.replace(0, np.nan)

        # Probability of buy using normal CDF
        p_buy = pd.Series(ndtr(z.fillna(0).values), index=result.index)

        buy_vol = result[volume_col] * p_buy
        sell_vol = result[volume_col] * (1 - p_buy)

    # Determine bucket size
    if bucket_size is None:
        bucket_size = result[volume_col].mean() * 10

    # Create volume buckets
    cumulative_vol = result[volume_col].cumsum()
    bucket_ids = (cumulative_vol / bucket_size).astype(int)

    # Calculate order imbalance per bucket
    result["_bucket_id"] = bucket_ids

    bucket_stats = result.groupby("_bucket_id").agg({
        volume_col: "sum",
        taker_buy_col: "sum"
    }).reset_index()

    bucket_stats["bucket_sell"] = bucket_stats[volume_col] - bucket_stats[taker_buy_col]
    bucket_stats["bucket_imbalance"] = np.abs(
        bucket_stats[taker_buy_col] - bucket_stats["bucket_sell"]
    ) / bucket_stats[volume_col]

    # Rolling VPIN over n_buckets
    bucket_stats["vpin"] = bucket_stats["bucket_imbalance"].rolling(
        window=n_buckets, min_periods=1
    ).mean()

    # Map back to original index
    vpin_map = bucket_stats.set_index("_bucket_id")["vpin"]
    result["vpin"] = result["_bucket_id"].map(vpin_map)

    # VPIN CDF (percentile rank)
    result["vpin_cdf"] = result["vpin"].rank(pct=True)

    # VPIN z-score
    vpin_mean = result["vpin"].rolling(window=200, min_periods=50).mean()
    vpin_std = result["vpin"].rolling(window=200, min_periods=50).std()
    result["vpin_z"] = (result["vpin"] - vpin_mean) / vpin_std.replace(0, np.nan)

    # Cleanup
    result.drop("_bucket_id", axis=1, inplace=True)

    return result


def order_flow_persistence(
    df: pd.DataFrame,
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    windows: List[int] = [5, 10, 20, 50],
) -> pd.DataFrame:
    """
    Calculate Order Flow Persistence metrics.

    Measures how consistent order flow direction is over time.
    Higher persistence indicates trending/informed activity.

    Args:
        df: DataFrame with OHLCV data
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume
        windows: List of windows for persistence calculation

    Returns:
        DataFrame with persistence features added
    """
    result = df.copy()

    # Calculate signed flow direction
    buy_vol = result[taker_buy_col]
    sell_vol = result[volume_col] - result[taker_buy_col]
    flow_sign = np.sign(buy_vol - sell_vol)

    result["flow_sign"] = flow_sign

    for w in windows:
        # Persistence = mean of sign over window
        result[f"flow_persistence_{w}"] = flow_sign.rolling(window=w).mean()

        # Absolute persistence (ignoring direction)
        result[f"flow_persistence_abs_{w}"] = np.abs(result[f"flow_persistence_{w}"])

        # Run length (consecutive same-sign periods)
        sign_change = flow_sign.diff().fillna(0) != 0
        run_id = sign_change.cumsum()
        run_lengths = flow_sign.groupby(run_id).transform("count")
        result[f"flow_run_length_{w}"] = run_lengths.rolling(window=w).mean()

        # Autocorrelation of flow
        result[f"flow_autocorr_{w}"] = flow_sign.rolling(window=w).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        )

    return result


def signed_trade_volume(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Calculate signed trade volume metrics.

    Combines volume with direction to create directional volume indicators.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume
        windows: Windows for rolling calculations

    Returns:
        DataFrame with signed volume features added
    """
    result = df.copy()

    # Net signed volume (buy - sell)
    buy_vol = result[taker_buy_col]
    sell_vol = result[volume_col] - result[taker_buy_col]
    result["signed_volume"] = buy_vol - sell_vol

    # Price-weighted signed volume
    result["signed_volume_pw"] = result["signed_volume"] * result[close_col]

    # Dollar volume (signed)
    result["signed_dollar_volume"] = result["signed_volume"] * result[close_col]

    for w in windows:
        # Rolling sum of signed volume
        result[f"signed_volume_sum_{w}"] = result["signed_volume"].rolling(window=w).sum()

        # Rolling mean
        result[f"signed_volume_ma_{w}"] = result["signed_volume"].rolling(window=w).mean()

        # Normalized signed volume (z-score)
        vol_std = result["signed_volume"].rolling(window=w).std()
        result[f"signed_volume_z_{w}"] = (
            result["signed_volume"] / vol_std.replace(0, np.nan)
        )

        # Cumulative signed volume over window
        result[f"cumulative_signed_vol_{w}"] = result["signed_volume"].rolling(
            window=w
        ).sum() / result[volume_col].rolling(window=w).sum()

    # Signed volume ratio
    result["buy_volume_ratio"] = buy_vol / result[volume_col]
    result["sell_volume_ratio"] = sell_vol / result[volume_col]

    return result


# ==============================================================================
# MICRO-PRICE CALCULATIONS
# ==============================================================================


def microprice(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
) -> pd.DataFrame:
    """
    Calculate various micro-price estimators.

    Micro-price provides a better estimate of fair value than simple mid-price
    by incorporating order book imbalance information.

    Args:
        df: DataFrame with OHLCV data
        high_col: Column for high price
        low_col: Column for low price
        close_col: Column for close price
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume

    Returns:
        DataFrame with micro-price features added
    """
    result = df.copy()

    # Proxy bid/ask from high-low
    mid_price = (result[high_col] + result[low_col]) / 2
    half_spread = (result[high_col] - result[low_col]) / 2
    bid_price = mid_price - half_spread
    ask_price = mid_price + half_spread

    # Proxy sizes from volume
    buy_vol = result[taker_buy_col]
    sell_vol = result[volume_col] - result[taker_buy_col]

    # 1. Simple mid price
    result["mid_price"] = mid_price

    # 2. Volume-weighted micro-price
    total_vol = buy_vol + sell_vol
    result["microprice_vw"] = np.where(
        total_vol > 0,
        (bid_price * sell_vol + ask_price * buy_vol) / total_vol,
        mid_price
    )

    # 3. Depth-weighted micro-price (using volume as depth proxy)
    result["microprice_depth"] = np.where(
        total_vol > 0,
        (bid_price * buy_vol + ask_price * sell_vol) / total_vol,
        mid_price
    )

    # 4. Imbalance-adjusted micro-price
    imbalance = np.where(total_vol > 0, (buy_vol - sell_vol) / total_vol, 0)
    result["microprice_imb"] = mid_price + imbalance * half_spread

    # 5. VWAP as fair value estimate
    result["vwap"] = (
        result[close_col] * result[volume_col]
    ).cumsum() / result[volume_col].cumsum()

    # Rolling VWAP
    for w in [5, 10, 20]:
        vol_sum = result[volume_col].rolling(window=w).sum()
        price_vol = (result[close_col] * result[volume_col]).rolling(window=w).sum()
        result[f"vwap_{w}"] = price_vol / vol_sum

    # 6. Fair value estimation (combining micro-price with momentum)
    momentum_5 = result[close_col].pct_change(5)
    result["fair_value_est"] = result["microprice_imb"] * (1 + momentum_5 * 0.5)

    # Micro-price deviation from close
    result["microprice_deviation"] = (
        (result["microprice_imb"] - result[close_col]) / result[close_col]
    )

    return result


# ==============================================================================
# BOOK METRICS
# ==============================================================================


def book_imbalance(
    df: pd.DataFrame,
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    n_levels: int = 5,
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Calculate book imbalance metrics across multiple levels.

    Book imbalance measures the asymmetry between bid and ask sides,
    which can predict short-term price movements.

    Args:
        df: DataFrame with OHLCV data
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume
        high_col: Column for high price
        low_col: Column for low price
        close_col: Column for close price
        n_levels: Number of levels to simulate
        windows: Windows for rolling calculations

    Returns:
        DataFrame with book imbalance features added
    """
    result = df.copy()

    # Basic imbalance using taker volumes as proxy
    buy_vol = result[taker_buy_col]
    sell_vol = result[volume_col] - result[taker_buy_col]

    # Level 1 imbalance
    total_vol = buy_vol + sell_vol
    result["book_imb_l1"] = np.where(
        total_vol > 0,
        (buy_vol - sell_vol) / total_vol,
        0.0
    )

    # Multi-level imbalance simulation (using different lookbacks)
    for level in range(1, n_levels + 1):
        # Simulate deeper levels with lagged volumes
        buy_lag = buy_vol.rolling(window=level).mean()
        sell_lag = sell_vol.rolling(window=level).mean()
        total_lag = buy_lag + sell_lag

        result[f"book_imb_l{level}"] = np.where(
            total_lag > 0,
            (buy_lag - sell_lag) / total_lag,
            0.0
        )

    # Weighted average imbalance across levels
    weights = np.array([1 / (i + 1) for i in range(n_levels)])
    weights = weights / weights.sum()

    imb_cols = [f"book_imb_l{i+1}" for i in range(n_levels)]
    result["book_imb_weighted"] = sum(
        result[col] * w for col, w in zip(imb_cols, weights)
    )

    # Imbalance statistics over windows
    for w in windows:
        result[f"book_imb_ma_{w}"] = result["book_imb_l1"].rolling(window=w).mean()
        result[f"book_imb_std_{w}"] = result["book_imb_l1"].rolling(window=w).std()
        result[f"book_imb_z_{w}"] = (
            (result["book_imb_l1"] - result[f"book_imb_ma_{w}"]) /
            result[f"book_imb_std_{w}"].replace(0, np.nan)
        )

        # Imbalance momentum
        result[f"book_imb_momentum_{w}"] = result["book_imb_l1"].diff(w)

    return result


def book_pressure(
    df: pd.DataFrame,
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    high_col: str = "high",
    low_col: str = "low",
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Calculate book pressure indicators.

    Book pressure measures the relative strength of bid vs ask sides,
    accounting for both volume and price levels.

    Args:
        df: DataFrame with OHLCV data
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume
        high_col: Column for high price
        low_col: Column for low price
        windows: Windows for rolling calculations

    Returns:
        DataFrame with book pressure features added
    """
    result = df.copy()

    buy_vol = result[taker_buy_col]
    sell_vol = result[volume_col] - result[taker_buy_col]

    # Basic pressure ratio
    result["bid_pressure"] = buy_vol / result[volume_col]
    result["ask_pressure"] = sell_vol / result[volume_col]

    # Pressure differential
    result["pressure_diff"] = result["bid_pressure"] - result["ask_pressure"]

    # Price-weighted pressure (high = sell pressure, low = buy pressure proxy)
    mid = (result[high_col] + result[low_col]) / 2
    result["bid_pressure_pw"] = buy_vol * (mid - result[low_col])
    result["ask_pressure_pw"] = sell_vol * (result[high_col] - mid)

    total_pw_pressure = result["bid_pressure_pw"] + result["ask_pressure_pw"]
    result["net_pressure_pw"] = np.where(
        total_pw_pressure > 0,
        (result["bid_pressure_pw"] - result["ask_pressure_pw"]) / total_pw_pressure,
        0.0
    )

    for w in windows:
        # Rolling pressure
        result[f"bid_pressure_{w}"] = result["bid_pressure"].rolling(window=w).mean()
        result[f"ask_pressure_{w}"] = result["ask_pressure"].rolling(window=w).mean()
        result[f"pressure_ratio_{w}"] = (
            result[f"bid_pressure_{w}"] /
            result[f"ask_pressure_{w}"].replace(0, np.nan)
        )

        # Pressure momentum
        result[f"pressure_momentum_{w}"] = result["pressure_diff"].diff(w)

        # Cumulative pressure
        result[f"cumulative_pressure_{w}"] = result["pressure_diff"].rolling(
            window=w
        ).sum()

    return result


def queue_position_estimation(
    df: pd.DataFrame,
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    trades_col: str = "trades",
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Estimate queue position dynamics.

    Approximates how orders might be positioned in the queue based on
    volume and trade patterns.

    Args:
        df: DataFrame with OHLCV data
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume
        trades_col: Column for number of trades
        windows: Windows for rolling calculations

    Returns:
        DataFrame with queue estimation features added
    """
    result = df.copy()

    # Average trade size (proxy for typical order size)
    result["avg_trade_size"] = result[volume_col] / result[trades_col].replace(0, np.nan)

    # Queue consumption rate (volume / trades indicates aggression)
    result["queue_consumption_rate"] = result[volume_col] / result[trades_col].replace(0, np.nan)

    # Estimated queue depth (rolling volume as proxy)
    for w in windows:
        result[f"est_queue_depth_{w}"] = result[volume_col].rolling(window=w).sum()

        # Queue turnover rate
        result[f"queue_turnover_{w}"] = (
            result[volume_col] / result[f"est_queue_depth_{w}"].replace(0, np.nan)
        )

        # Time to fill proxy (based on historical fill rates)
        result[f"est_fill_time_{w}"] = (
            result["avg_trade_size"].rolling(window=w).mean() /
            (result[volume_col] / w).replace(0, np.nan)
        )

    # Priority indicator (how aggressive is current volume)
    overall_avg_size = result["avg_trade_size"].expanding().mean()
    result["queue_priority"] = result["avg_trade_size"] / overall_avg_size

    return result


def depth_profile_analysis(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Analyze depth profile characteristics.

    Creates a volume profile to understand where liquidity is concentrated.

    Args:
        df: DataFrame with OHLCV data
        high_col: Column for high price
        low_col: Column for low price
        close_col: Column for close price
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume
        n_bins: Number of price bins for profile

    Returns:
        DataFrame with depth profile features added
    """
    result = df.copy()

    # Calculate rolling price range
    window = 20
    rolling_high = result[high_col].rolling(window=window).max()
    rolling_low = result[low_col].rolling(window=window).min()
    price_range = rolling_high - rolling_low

    # Position within range (0 = at low, 1 = at high)
    result["price_position"] = np.where(
        price_range > 0,
        (result[close_col] - rolling_low) / price_range,
        0.5
    )

    # Volume concentration (how much volume near current price)
    result["volume_concentration"] = result[volume_col] / result[volume_col].rolling(
        window=window
    ).sum().replace(0, np.nan)

    # Depth asymmetry (comparing volume above vs below current price)
    close_vs_mid = result[close_col] - (result[high_col] + result[low_col]) / 2
    result["depth_asymmetry"] = np.where(
        close_vs_mid > 0,
        result[taker_buy_col] / result[volume_col],
        1 - (result[taker_buy_col] / result[volume_col])
    )

    # Volume-at-price proxy (using high-low as spread)
    result["volume_per_tick"] = result[volume_col] / (
        (result[high_col] - result[low_col]).replace(0, np.nan)
    )

    # Rolling depth profile statistics
    result["depth_skew"] = result["price_position"].rolling(window=window).apply(
        lambda x: stats.skew(x) if len(x) > 2 else 0,
        raw=True
    )

    result["depth_kurtosis"] = result["price_position"].rolling(window=window).apply(
        lambda x: stats.kurtosis(x) if len(x) > 3 else 0,
        raw=True
    )

    return result


# ==============================================================================
# SPREAD CALCULATIONS
# ==============================================================================


def spread_metrics(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    open_col: str = "open",
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
) -> pd.DataFrame:
    """
    Calculate comprehensive spread metrics.

    Includes bid-ask spread proxies, effective spread, and realized spread.

    Args:
        df: DataFrame with OHLCV data
        high_col: Column for high price
        low_col: Column for low price
        close_col: Column for close price
        open_col: Column for open price
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume

    Returns:
        DataFrame with spread features added
    """
    result = df.copy()

    # Proxy mid price
    mid_price = (result[high_col] + result[low_col]) / 2

    # 1. Bid-Ask Spread (absolute) - using high-low as proxy
    result["spread_abs"] = result[high_col] - result[low_col]

    # 2. Bid-Ask Spread (relative, in basis points)
    result["spread_bps"] = (result["spread_abs"] / mid_price) * 10000

    # 3. Effective Spread
    # Approximated as 2 * |trade_price - mid_price|
    # Using close as trade price proxy
    result["effective_spread"] = 2 * np.abs(result[close_col] - mid_price)
    result["effective_spread_bps"] = (result["effective_spread"] / mid_price) * 10000

    # 4. Realized Spread (market maker profit after price movement)
    # Using future mid-price change
    future_mid = mid_price.shift(-1)
    trade_direction = np.sign(result[taker_buy_col] - (result[volume_col] / 2))

    result["realized_spread"] = 2 * trade_direction * (result[close_col] - future_mid)
    result["realized_spread_bps"] = (
        result["realized_spread"].fillna(0) / mid_price.replace(0, np.nan)
    ) * 10000

    # 5. Price Impact (adverse selection component)
    result["price_impact"] = result["effective_spread"] - result["realized_spread"].fillna(0)
    result["price_impact_bps"] = (result["price_impact"] / mid_price) * 10000

    # Rolling spread statistics
    for w in [5, 10, 20]:
        result[f"spread_ma_{w}"] = result["spread_bps"].rolling(window=w).mean()
        result[f"spread_std_{w}"] = result["spread_bps"].rolling(window=w).std()
        result[f"spread_z_{w}"] = (
            (result["spread_bps"] - result[f"spread_ma_{w}"]) /
            result[f"spread_std_{w}"].replace(0, np.nan)
        )

    # Spread/volatility ratio
    returns = np.log(result[close_col] / result[close_col].shift(1))
    volatility = returns.rolling(window=20).std() * np.sqrt(252 * 24 * 60)
    result["spread_vol_ratio"] = result["spread_bps"] / (volatility * 10000).replace(0, np.nan)

    return result


# ==============================================================================
# LIQUIDITY MEASURES
# ==============================================================================


def kyle_lambda(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    taker_buy_col: str = "taker_buy_base",
    windows: List[int] = [20, 50, 100],
) -> pd.DataFrame:
    """
    Calculate Kyle's Lambda (market impact coefficient).

    Lambda measures how much price moves per unit of order flow.
    Higher lambda indicates lower liquidity.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        volume_col: Column for total volume
        taker_buy_col: Column for taker buy volume
        windows: Windows for rolling calculations

    Returns:
        DataFrame with Kyle's Lambda features added

    Reference:
        Kyle, A.S. (1985). "Continuous Auctions and Insider Trading"
    """
    result = df.copy()

    # Calculate returns
    returns = np.log(result[close_col] / result[close_col].shift(1))

    # Signed order flow (buy - sell)
    signed_flow = result[taker_buy_col] - (result[volume_col] - result[taker_buy_col])

    for w in windows:
        # Rolling covariance and variance
        rolling_cov = returns.rolling(window=w).cov(signed_flow)
        rolling_var = signed_flow.rolling(window=w).var()

        # Kyle's Lambda = Cov(return, flow) / Var(flow)
        result[f"kyle_lambda_{w}"] = rolling_cov / rolling_var.replace(0, np.nan)

        # Normalized lambda (z-score)
        lambda_mean = result[f"kyle_lambda_{w}"].rolling(window=w*2).mean()
        lambda_std = result[f"kyle_lambda_{w}"].rolling(window=w*2).std()
        result[f"kyle_lambda_z_{w}"] = (
            (result[f"kyle_lambda_{w}"] - lambda_mean) / lambda_std.replace(0, np.nan)
        )

        # Lambda trend (increasing = deteriorating liquidity)
        result[f"kyle_lambda_trend_{w}"] = result[f"kyle_lambda_{w}"].diff(w)

    return result


def amihud_illiquidity(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    quote_volume_col: str = "quote_volume",
    windows: List[int] = [5, 10, 20, 50],
) -> pd.DataFrame:
    """
    Calculate Amihud Illiquidity Ratio.

    Measures price impact per unit of trading volume.
    Higher values indicate lower liquidity.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        volume_col: Column for total volume
        quote_volume_col: Column for quote volume (dollar volume)
        windows: Windows for rolling calculations

    Returns:
        DataFrame with Amihud ratio features added

    Reference:
        Amihud, Y. (2002). "Illiquidity and Stock Returns"
    """
    result = df.copy()

    # Calculate absolute returns
    abs_returns = np.abs(np.log(result[close_col] / result[close_col].shift(1)))

    # Amihud ratio = |return| / dollar_volume
    result["amihud"] = abs_returns / result[quote_volume_col].replace(0, np.nan)

    # Scale for interpretability (multiply by 10^6)
    result["amihud_scaled"] = result["amihud"] * 1e6

    for w in windows:
        # Rolling Amihud
        result[f"amihud_ma_{w}"] = result["amihud_scaled"].rolling(window=w).mean()

        # Amihud z-score
        amihud_std = result["amihud_scaled"].rolling(window=w).std()
        result[f"amihud_z_{w}"] = (
            (result["amihud_scaled"] - result[f"amihud_ma_{w}"]) /
            amihud_std.replace(0, np.nan)
        )

        # Amihud trend
        result[f"amihud_trend_{w}"] = result[f"amihud_ma_{w}"].diff(w)

    # Log Amihud for better distribution
    result["amihud_log"] = np.log1p(result["amihud_scaled"])

    return result


def roll_measure(
    df: pd.DataFrame,
    close_col: str = "close",
    windows: List[int] = [20, 50, 100],
) -> pd.DataFrame:
    """
    Calculate Roll's Measure of the effective spread.

    Uses the autocovariance of price changes to estimate spread.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        windows: Windows for rolling calculations

    Returns:
        DataFrame with Roll measure features added

    Reference:
        Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread"
    """
    result = df.copy()

    # Price changes
    delta_p = result[close_col].diff()

    for w in windows:
        # Autocovariance of price changes
        autocov = delta_p.rolling(window=w).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0,
            raw=True
        )

        # Roll measure = 2 * sqrt(-autocovariance) if autocov < 0, else 0
        result[f"roll_measure_{w}"] = np.where(
            autocov < 0,
            2 * np.sqrt(np.abs(autocov)),
            0.0
        )

        # As percentage of price
        result[f"roll_measure_pct_{w}"] = (
            result[f"roll_measure_{w}"] / result[close_col]
        ) * 100

    return result


def pastor_stambaugh_liquidity(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    quote_volume_col: str = "quote_volume",
    window: int = 21,
) -> pd.DataFrame:
    """
    Calculate Pastor-Stambaugh liquidity measure.

    Measures the price reversal associated with trading volume.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        volume_col: Column for total volume
        quote_volume_col: Column for quote volume
        window: Window for regression

    Returns:
        DataFrame with Pastor-Stambaugh features added

    Reference:
        Pastor, L., & Stambaugh, R.F. (2003).
        "Liquidity Risk and Expected Stock Returns"
    """
    result = df.copy()

    # Calculate returns
    returns = np.log(result[close_col] / result[close_col].shift(1))
    excess_returns = returns - returns.rolling(window=window).mean()

    # Signed volume (using sign of return)
    signed_volume = np.sign(returns.shift(1)) * result[quote_volume_col].shift(1)

    def calc_gamma(window_data: pd.DataFrame) -> float:
        """Calculate gamma coefficient from regression."""
        y = window_data["ret"].values
        x = window_data["signed_vol"].values

        if len(y) < 5 or np.std(x) == 0:
            return np.nan

        # Simple OLS
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        coef = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

        return coef

    # Create temporary DataFrame for rolling regression
    temp_df = pd.DataFrame({
        "ret": excess_returns,
        "signed_vol": signed_volume
    })

    # Rolling gamma coefficient
    result["ps_gamma"] = temp_df.rolling(window=window).apply(
        lambda x: calc_gamma(pd.DataFrame({"ret": x[:window//2], "signed_vol": x[window//2:]})),
        raw=False
    )["ret"]

    # Alternative: use correlation as proxy
    result["ps_liquidity"] = excess_returns.rolling(window=window).corr(signed_volume)

    # Scaled measure
    result["ps_liquidity_scaled"] = result["ps_liquidity"] * 1e6

    return result


def corwin_schultz_spread(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    window: int = 2,
) -> pd.DataFrame:
    """
    Calculate Corwin-Schultz high-low spread estimator.

    Uses the ratio of high and low prices to estimate spread.

    Args:
        df: DataFrame with OHLCV data
        high_col: Column for high price
        low_col: Column for low price
        window: Window for estimation (typically 2)

    Returns:
        DataFrame with Corwin-Schultz spread features added

    Reference:
        Corwin, S.A., & Schultz, P. (2012).
        "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices"
    """
    result = df.copy()

    # Calculate beta and gamma
    log_hl = np.log(result[high_col] / result[low_col])
    log_hl_sq = log_hl ** 2

    # Rolling max high and min low
    h_bar = result[high_col].rolling(window=window).max()
    l_bar = result[low_col].rolling(window=window).min()
    gamma = np.log(h_bar / l_bar) ** 2

    # Beta
    beta = log_hl_sq.rolling(window=window).sum()

    # Alpha
    sqrt_2 = np.sqrt(2)
    term1 = (sqrt_2 - 1) * np.sqrt(beta)
    term2 = np.sqrt(gamma)
    alpha = (term1 - term2) / (3 - 2 * sqrt_2)

    # Spread estimate
    result["cs_spread"] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Ensure non-negative
    result["cs_spread"] = result["cs_spread"].clip(lower=0)

    # Convert to basis points
    mid_price = (result[high_col] + result[low_col]) / 2
    result["cs_spread_bps"] = result["cs_spread"] * 10000

    # Rolling statistics
    for w in [5, 10, 20]:
        result[f"cs_spread_ma_{w}"] = result["cs_spread_bps"].rolling(window=w).mean()
        result[f"cs_spread_z_{w}"] = (
            (result["cs_spread_bps"] - result[f"cs_spread_ma_{w}"]) /
            result["cs_spread_bps"].rolling(window=w).std().replace(0, np.nan)
        )

    return result


# ==============================================================================
# REALIZED VOLATILITY
# ==============================================================================


def realized_variance(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    sampling_intervals: List[int] = [1, 5],
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Calculate realized variance with different sampling frequencies.

    Implements various realized volatility estimators including
    Parkinson, Garman-Klass, and Rogers-Satchell.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        high_col: Column for high price
        low_col: Column for low price
        open_col: Column for open price
        sampling_intervals: Intervals for sampling (1-min, 5-min equivalent)
        annualize: Whether to annualize volatility

    Returns:
        DataFrame with realized variance features added
    """
    result = df.copy()

    # Annualization factor (assuming minute data, 252 days, 24 hours)
    ann_factor = np.sqrt(252 * 24 * 60) if annualize else 1.0

    # Log returns
    log_returns = np.log(result[close_col] / result[close_col].shift(1))

    for interval in sampling_intervals:
        # 1. Close-to-close realized variance
        if interval == 1:
            rv = log_returns ** 2
        else:
            # Subsampled returns
            rv = log_returns.rolling(window=interval).sum() ** 2

        result[f"rv_cc_{interval}"] = rv.rolling(window=20).sum() ** 0.5 * ann_factor

    # 2. Parkinson estimator (high-low based)
    log_hl = np.log(result[high_col] / result[low_col])
    result["rv_parkinson"] = np.sqrt(
        (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(window=20).mean()
    ) * ann_factor

    # 3. Garman-Klass estimator
    log_hl_sq = np.log(result[high_col] / result[low_col]) ** 2
    log_co_sq = np.log(result[close_col] / result[open_col]) ** 2

    result["rv_garman_klass"] = np.sqrt(
        (0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq).rolling(window=20).mean()
    ) * ann_factor

    # 4. Rogers-Satchell estimator
    log_hc = np.log(result[high_col] / result[close_col])
    log_ho = np.log(result[high_col] / result[open_col])
    log_lc = np.log(result[low_col] / result[close_col])
    log_lo = np.log(result[low_col] / result[open_col])

    rs_var = log_ho * log_hc + log_lo * log_lc
    result["rv_rogers_satchell"] = np.sqrt(rs_var.rolling(window=20).mean()) * ann_factor

    # 5. Yang-Zhang estimator (combines overnight and intraday)
    log_oc = np.log(result[open_col] / result[close_col].shift(1))  # Overnight
    log_co = np.log(result[close_col] / result[open_col])  # Intraday close-open

    overnight_var = log_oc ** 2
    open_var = (log_co - log_co.rolling(window=20).mean()) ** 2
    close_var = (log_returns - log_returns.rolling(window=20).mean()) ** 2

    k = 0.34 / (1.34 + (21) / (20 - 1))
    result["rv_yang_zhang"] = np.sqrt(
        (overnight_var + k * open_var + (1 - k) * result["rv_rogers_satchell"] ** 2 / ann_factor ** 2).rolling(window=20).mean()
    ) * ann_factor

    return result


def bipower_variation(
    df: pd.DataFrame,
    close_col: str = "close",
    windows: List[int] = [20, 50],
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Calculate Bipower Variation for robust volatility estimation.

    Bipower variation is robust to jumps and provides a consistent
    estimator of integrated variance.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        windows: Windows for rolling calculations
        annualize: Whether to annualize

    Returns:
        DataFrame with bipower variation features added

    Reference:
        Barndorff-Nielsen, O.E., & Shephard, N. (2004).
        "Power and Bipower Variation"
    """
    result = df.copy()

    # Annualization factor
    ann_factor = np.sqrt(252 * 24 * 60) if annualize else 1.0

    # Constant for bipower variation
    mu1 = np.sqrt(2 / np.pi)

    # Absolute returns
    log_returns = np.log(result[close_col] / result[close_col].shift(1))
    abs_returns = np.abs(log_returns)

    # Bipower variation: sum of |r_t| * |r_{t-1}|
    bipower_product = abs_returns * abs_returns.shift(1)

    for w in windows:
        # Standard realized variance
        result[f"rv_{w}"] = np.sqrt(
            (log_returns ** 2).rolling(window=w).sum()
        ) * ann_factor

        # Bipower variation
        bv = (np.pi / 2) * bipower_product.rolling(window=w).sum()
        result[f"bv_{w}"] = np.sqrt(bv) * ann_factor

        # Jump component = RV - BV (if positive)
        result[f"jump_var_{w}"] = np.maximum(
            result[f"rv_{w}"] ** 2 - result[f"bv_{w}"] ** 2,
            0
        )
        result[f"jump_component_{w}"] = np.sqrt(result[f"jump_var_{w}"])

        # Continuous component (bipower as proxy)
        result[f"continuous_var_{w}"] = result[f"bv_{w}"]

        # Jump ratio
        result[f"jump_ratio_{w}"] = (
            result[f"jump_component_{w}"] / result[f"rv_{w}"].replace(0, np.nan)
        )

    return result


def realized_kernel(
    df: pd.DataFrame,
    close_col: str = "close",
    kernel_type: str = "parzen",
    bandwidth: int = 10,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Calculate Realized Kernel estimator for microstructure noise-robust volatility.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        kernel_type: Type of kernel ('parzen', 'tukey_hanning', 'quadratic')
        bandwidth: Kernel bandwidth
        window: Window for rolling calculations
        annualize: Whether to annualize

    Returns:
        DataFrame with realized kernel features added

    Reference:
        Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., & Shephard, N. (2008).
        "Designing Realized Kernels to Measure the Ex Post Variation"
    """
    result = df.copy()

    ann_factor = np.sqrt(252 * 24 * 60) if annualize else 1.0

    log_returns = np.log(result[close_col] / result[close_col].shift(1))

    def parzen_kernel(x: float) -> float:
        """Parzen kernel function."""
        if abs(x) <= 0.5:
            return 1 - 6 * x ** 2 + 6 * abs(x) ** 3
        elif abs(x) <= 1:
            return 2 * (1 - abs(x)) ** 3
        return 0.0

    def tukey_hanning_kernel(x: float) -> float:
        """Tukey-Hanning kernel function."""
        if abs(x) <= 1:
            return (1 + np.cos(np.pi * x)) / 2
        return 0.0

    def quadratic_kernel(x: float) -> float:
        """Quadratic spectral kernel."""
        if x == 0:
            return 1.0
        term = 6 * np.pi * x / 5
        return (3 / (term ** 2)) * (np.sin(term) / term - np.cos(term))

    # Select kernel
    if kernel_type == "parzen":
        kernel_func = parzen_kernel
    elif kernel_type == "tukey_hanning":
        kernel_func = tukey_hanning_kernel
    else:
        kernel_func = quadratic_kernel

    def calc_realized_kernel(returns: np.ndarray) -> float:
        """Calculate realized kernel for a window."""
        n = len(returns)
        if n < 2:
            return np.nan

        # Gamma_0 (realized variance)
        gamma_0 = np.sum(returns ** 2)

        # Add autocovariances with kernel weights
        rk = gamma_0
        for h in range(1, min(bandwidth + 1, n)):
            weight = kernel_func(h / (bandwidth + 1))
            gamma_h = np.sum(returns[h:] * returns[:-h])
            rk += 2 * weight * gamma_h

        return max(rk, 0)

    # Rolling realized kernel
    result[f"rk_{kernel_type}"] = log_returns.rolling(window=window).apply(
        calc_realized_kernel, raw=True
    ).apply(lambda x: np.sqrt(max(x, 0))) * ann_factor

    # Standard RV for comparison
    result["rv_standard"] = np.sqrt(
        (log_returns ** 2).rolling(window=window).sum()
    ) * ann_factor

    # Noise ratio (RV / RK - 1)
    result["noise_ratio"] = (
        result["rv_standard"] / result[f"rk_{kernel_type}"].replace(0, np.nan)
    ) - 1

    return result


def jump_detection_lee_mykland(
    df: pd.DataFrame,
    close_col: str = "close",
    window: int = 20,
    significance_level: float = 0.01,
    bipower_window: int = 20,
) -> pd.DataFrame:
    """
    Detect jumps using the Lee-Mykland test.

    Uses a ratio test comparing returns to local volatility.

    Args:
        df: DataFrame with OHLCV data
        close_col: Column for close price
        window: Window for local volatility estimation
        significance_level: Significance level for jump detection
        bipower_window: Window for bipower variation

    Returns:
        DataFrame with jump detection features added

    Reference:
        Lee, S.S., & Mykland, P.A. (2008).
        "Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics"
    """
    result = df.copy()

    # Log returns
    log_returns = np.log(result[close_col] / result[close_col].shift(1))
    abs_returns = np.abs(log_returns)

    # Bipower variation for local volatility
    mu1 = np.sqrt(2 / np.pi)
    bipower_product = abs_returns * abs_returns.shift(1)
    local_bv = (np.pi / 2) * bipower_product.rolling(window=bipower_window).mean()
    local_sigma = np.sqrt(local_bv)

    # L statistic
    c_n = np.sqrt(2 * np.log(window)) / mu1
    s_n = (c_n ** 2 - np.log(np.pi) - np.log(np.log(window))) / (c_n * np.sqrt(2 * np.log(window)))

    # Jump statistic
    result["lm_statistic"] = (
        abs_returns / local_sigma.replace(0, np.nan) - c_n
    ) / s_n

    # Critical value for Gumbel distribution
    critical_value = -np.log(-np.log(1 - significance_level))

    # Jump indicator
    result["jump_detected"] = (result["lm_statistic"] > critical_value).astype(int)

    # Jump magnitude (if detected)
    result["jump_magnitude"] = np.where(
        result["jump_detected"] == 1,
        log_returns,
        0.0
    )

    # Jump direction
    result["jump_direction"] = np.where(
        result["jump_detected"] == 1,
        np.sign(log_returns),
        0
    )

    # Rolling jump statistics
    for w in [20, 50, 100]:
        result[f"jump_count_{w}"] = result["jump_detected"].rolling(window=w).sum()
        result[f"jump_frequency_{w}"] = result[f"jump_count_{w}"] / w
        result[f"avg_jump_magnitude_{w}"] = (
            result["jump_magnitude"].abs().rolling(window=w).mean()
        )

    # Jump contribution to variance
    result["jump_variance_contrib"] = (
        result["jump_magnitude"] ** 2
    ).rolling(window=bipower_window).sum() / (
        log_returns ** 2
    ).rolling(window=bipower_window).sum()

    return result


# ==============================================================================
# COMPREHENSIVE MICROSTRUCTURE ENGINE
# ==============================================================================


class MicrostructureAnalyzer:
    """
    Comprehensive market microstructure analyzer.

    Combines all microstructure metrics into a unified interface
    for easy integration with trading strategies.
    """

    def __init__(
        self,
        ofi_decay: float = 0.1,
        ofi_window: int = 20,
        vpin_n_buckets: int = 50,
        kyle_windows: List[int] = [20, 50, 100],
        book_levels: int = 5,
    ) -> None:
        """
        Initialize Microstructure Analyzer.

        Args:
            ofi_decay: Decay factor for OFI
            ofi_window: Window for OFI calculations
            vpin_n_buckets: Number of buckets for VPIN
            kyle_windows: Windows for Kyle's Lambda
            book_levels: Number of book levels to analyze
        """
        self.ofi_decay = ofi_decay
        self.ofi_window = ofi_window
        self.vpin_n_buckets = vpin_n_buckets
        self.kyle_windows = kyle_windows
        self.book_levels = book_levels

    def analyze(
        self,
        df: pd.DataFrame,
        volume_col: str = "volume",
        taker_buy_col: str = "taker_buy_base",
        close_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
        open_col: str = "open",
        quote_volume_col: str = "quote_volume",
        trades_col: str = "trades",
    ) -> pd.DataFrame:
        """
        Run complete microstructure analysis.

        Args:
            df: DataFrame with OHLCV data
            volume_col: Column for total volume
            taker_buy_col: Column for taker buy volume
            close_col: Column for close price
            high_col: Column for high price
            low_col: Column for low price
            open_col: Column for open price
            quote_volume_col: Column for quote volume
            trades_col: Column for number of trades

        Returns:
            DataFrame with all microstructure features added
        """
        logger.info(f"Starting microstructure analysis on {len(df)} rows...")

        result = df.copy()

        # 1. Order Flow Analysis
        logger.info("  Computing order flow metrics...")
        result = order_flow_imbalance(
            result, volume_col, taker_buy_col,
            decay=self.ofi_decay, window=self.ofi_window
        )
        result = vpin(
            result, volume_col, taker_buy_col, close_col,
            n_buckets=self.vpin_n_buckets
        )
        result = order_flow_persistence(result, volume_col, taker_buy_col)
        result = signed_trade_volume(result, close_col, volume_col, taker_buy_col)

        # 2. Micro-price
        logger.info("  Computing micro-price estimates...")
        result = microprice(
            result, high_col, low_col, close_col, volume_col, taker_buy_col
        )

        # 3. Book Metrics
        logger.info("  Computing book metrics...")
        result = book_imbalance(
            result, volume_col, taker_buy_col, high_col, low_col, close_col,
            n_levels=self.book_levels
        )
        result = book_pressure(result, volume_col, taker_buy_col, high_col, low_col)
        result = queue_position_estimation(result, volume_col, taker_buy_col, trades_col)
        result = depth_profile_analysis(
            result, high_col, low_col, close_col, volume_col, taker_buy_col
        )

        # 4. Spreads
        logger.info("  Computing spread metrics...")
        result = spread_metrics(
            result, high_col, low_col, close_col, open_col,
            volume_col, taker_buy_col
        )

        # 5. Liquidity
        logger.info("  Computing liquidity metrics...")
        result = kyle_lambda(
            result, close_col, volume_col, taker_buy_col,
            windows=self.kyle_windows
        )
        result = amihud_illiquidity(
            result, close_col, volume_col, quote_volume_col
        )
        result = roll_measure(result, close_col)
        result = pastor_stambaugh_liquidity(
            result, close_col, volume_col, quote_volume_col
        )
        result = corwin_schultz_spread(result, high_col, low_col)

        # 6. Realized Volatility
        logger.info("  Computing realized volatility metrics...")
        result = realized_variance(result, close_col, high_col, low_col, open_col)
        result = bipower_variation(result, close_col)
        result = realized_kernel(result, close_col)
        result = jump_detection_lee_mykland(result, close_col)

        # Replace inf with NaN
        result = result.replace([np.inf, -np.inf], np.nan)

        logger.info(f"  Analysis complete. Added {len(result.columns) - len(df.columns)} features.")

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all microstructure feature names."""
        # Run on dummy data to get column names
        dummy_df = pd.DataFrame({
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.5] * 100,
            "volume": [1000.0] * 100,
            "quote_volume": [100500.0] * 100,
            "trades": [100] * 100,
            "taker_buy_base": [500.0] * 100,
        })

        result = self.analyze(dummy_df)
        original_cols = set(dummy_df.columns)

        return [col for col in result.columns if col not in original_cols]


def calculate_all_microstructure_features(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to calculate all microstructure features.

    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional arguments passed to MicrostructureAnalyzer

    Returns:
        DataFrame with all microstructure features added
    """
    analyzer = MicrostructureAnalyzer(**kwargs)
    return analyzer.analyze(df)


# ==============================================================================
# TESTING
# ==============================================================================


def _generate_test_data(n_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    price = 100.0
    data = []

    for i in range(n_rows):
        returns = np.random.normal(0.0001, 0.002)
        price *= (1 + returns)

        volatility = 0.005 * price
        high = price + abs(np.random.normal(0, volatility))
        low = price - abs(np.random.normal(0, volatility))
        open_price = np.random.uniform(low, high)

        volume = np.random.exponential(1000)
        taker_buy = volume * np.random.uniform(0.3, 0.7)
        trades = int(volume / 10) + 1

        data.append({
            "open_time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
            "open": open_price,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
            "quote_volume": volume * price,
            "trades": trades,
            "taker_buy_base": taker_buy,
            "taker_buy_quote": taker_buy * price,
        })

    return pd.DataFrame(data)


def test_microstructure_module():
    """Test the microstructure module."""
    print("=" * 70)
    print("Testing Microstructure Module")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    df = _generate_test_data(500)
    print(f"Generated {len(df)} rows")

    # Run analysis
    print("\nRunning microstructure analysis...")
    analyzer = MicrostructureAnalyzer()
    result = analyzer.analyze(df)

    # Summary
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    print(f"Input columns: {len(df.columns)}")
    print(f"Output columns: {len(result.columns)}")
    print(f"Features added: {len(result.columns) - len(df.columns)}")

    # Feature statistics
    feature_names = analyzer.get_feature_names()
    print(f"\nFeature categories:")
    categories = {
        "Order Flow": [f for f in feature_names if any(x in f for x in ["ofi", "vpin", "flow", "signed"])],
        "Micro-price": [f for f in feature_names if any(x in f for x in ["microprice", "vwap", "fair_value", "mid_price"])],
        "Book Metrics": [f for f in feature_names if any(x in f for x in ["book_imb", "pressure", "queue", "depth"])],
        "Spreads": [f for f in feature_names if any(x in f for x in ["spread", "cs_spread"])],
        "Liquidity": [f for f in feature_names if any(x in f for x in ["kyle", "amihud", "roll", "ps_"])],
        "Volatility": [f for f in feature_names if any(x in f for x in ["rv_", "bv_", "rk_", "jump"])],
    }

    for cat, features in categories.items():
        print(f"  {cat}: {len(features)} features")

    # Sample values
    print("\n" + "=" * 70)
    print("Sample Feature Values (last row)")
    print("=" * 70)
    sample_features = [
        "ofi_basic", "vpin", "microprice_imb", "book_imb_l1",
        "spread_bps", "kyle_lambda_20", "amihud_scaled",
        "rv_garman_klass", "jump_detected"
    ]

    for feat in sample_features:
        if feat in result.columns:
            val = result[feat].iloc[-1]
            print(f"  {feat:25s}: {val:.6f}" if pd.notna(val) else f"  {feat:25s}: NaN")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

    return result


if __name__ == "__main__":
    test_microstructure_module()
