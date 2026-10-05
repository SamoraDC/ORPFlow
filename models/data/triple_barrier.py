"""
Triple Barrier Labeling and Meta-Labeling Module
Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado

This module implements:
1. Triple Barrier Method for labeling financial time series
2. Dynamic barriers (ATR-based, asymmetric, adaptive by regime)
3. Meta-labeling for bet sizing
4. Sample weights (concurrent events, uniqueness, temporal decay)
5. Regime detection (volatility, trend, Hurst exponent)
6. Event-based sampling (CUSUM filter, fractional differentiation)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit, prange

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

class BarrierType(Enum):
    """Barrier type enumeration"""
    TAKE_PROFIT = 1
    STOP_LOSS = -1
    VERTICAL = 0


class RegimeType(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    RANDOM_WALK = "random_walk"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"


@dataclass
class BarrierConfig:
    """Configuration for barrier parameters"""
    tp_multiplier: float = 2.0  # Take-profit multiplier (in volatility units)
    sl_multiplier: float = 2.0  # Stop-loss multiplier (in volatility units)
    vertical_bars: int = 10      # Maximum holding period in bars
    min_return: float = 0.0      # Minimum return threshold for labeling
    use_asymmetric: bool = False # Use asymmetric barriers
    tp_sl_ratio: float = 1.5     # TP/SL ratio for asymmetric barriers


# =============================================================================
# Core Triple Barrier Functions
# =============================================================================

def get_daily_vol(
    close: pd.Series,
    span: int = 100,
    min_periods: int = 20
) -> pd.Series:
    """
    Calculate daily volatility using exponential weighted moving average.

    This is the standard volatility measure used in AFML for barrier sizing.
    Uses returns-based volatility with EWMA smoothing.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by datetime
    span : int
        Span for EWMA calculation (default: 100)
    min_periods : int
        Minimum periods required for calculation

    Returns
    -------
    pd.Series
        Daily volatility estimates
    """
    # Calculate returns
    returns = close.pct_change()

    # EWMA of absolute returns (more robust than squared returns)
    daily_vol = returns.ewm(span=span, min_periods=min_periods).std()

    return daily_vol


def get_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_bars: int
) -> pd.Series:
    """
    Calculate vertical barrier timestamps.

    The vertical barrier is the maximum holding period for a position.
    It limits how long we wait for TP/SL to be hit.

    Parameters
    ----------
    t_events : pd.DatetimeIndex
        Event timestamps (entry points)
    close : pd.Series
        Close prices indexed by datetime
    num_bars : int
        Number of bars for the vertical barrier

    Returns
    -------
    pd.Series
        Vertical barrier timestamps indexed by event times
    """
    t_events = pd.DatetimeIndex(t_events)

    # Get the index of close prices
    close_idx = close.index.searchsorted(t_events)

    # Calculate end indices (vertical barrier)
    end_idx = np.minimum(close_idx + num_bars, len(close) - 1)

    # Map back to timestamps
    vertical_barriers = pd.Series(
        close.index[end_idx],
        index=t_events
    )

    return vertical_barriers


def get_horizontal_barriers(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float],
    target: pd.Series,
    molecule: Optional[List] = None
) -> pd.DataFrame:
    """
    Calculate horizontal barriers (take-profit and stop-loss levels).

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by datetime
    t_events : pd.DatetimeIndex
        Event timestamps
    pt_sl : Tuple[float, float]
        (profit-taking, stop-loss) multipliers
    target : pd.Series
        Volatility target (barrier width in price units)
    molecule : List, optional
        Subset of event indices to process (for parallel processing)

    Returns
    -------
    pd.DataFrame
        DataFrame with 'pt' (take-profit) and 'sl' (stop-loss) levels
    """
    if molecule is None:
        molecule = t_events

    barriers = pd.DataFrame(index=molecule, columns=['pt', 'sl'])

    for t in molecule:
        if t not in target.index:
            continue

        vol = target.loc[t]
        entry_price = close.loc[t]

        # Take-profit barrier
        if pt_sl[0] > 0:
            barriers.loc[t, 'pt'] = entry_price * (1 + pt_sl[0] * vol)
        else:
            barriers.loc[t, 'pt'] = np.nan

        # Stop-loss barrier
        if pt_sl[1] > 0:
            barriers.loc[t, 'sl'] = entry_price * (1 - pt_sl[1] * vol)
        else:
            barriers.loc[t, 'sl'] = np.nan

    return barriers


@jit(nopython=True, cache=True)
def _find_first_barrier_touch(
    prices: np.ndarray,
    timestamps: np.ndarray,
    entry_idx: int,
    end_idx: int,
    pt_level: float,
    sl_level: float,
    entry_price: float
) -> Tuple[int, int, float]:
    """
    Numba-optimized function to find first barrier touch.

    Returns
    -------
    Tuple[int, int, float]
        (barrier_type, touch_idx, return_value)
        barrier_type: 1=TP, -1=SL, 0=vertical
    """
    for i in range(entry_idx + 1, end_idx + 1):
        price = prices[i]
        ret = (price - entry_price) / entry_price

        # Check take-profit (upper barrier)
        if not np.isnan(pt_level) and price >= pt_level:
            return 1, i, ret

        # Check stop-loss (lower barrier)
        if not np.isnan(sl_level) and price <= sl_level:
            return -1, i, ret

    # Vertical barrier hit
    final_ret = (prices[end_idx] - entry_price) / entry_price
    return 0, end_idx, final_ret


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float],
    target: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 1,
    vertical_barrier_times: Optional[pd.Series] = None,
    side: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Get trading events with barrier touches.

    This is the core function that implements the triple barrier method.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by datetime
    t_events : pd.DatetimeIndex
        Timestamps of events (potential entry points)
    pt_sl : Tuple[float, float]
        Profit-taking and stop-loss multipliers
    target : pd.Series
        Volatility target for barrier sizing
    min_ret : float
        Minimum return threshold
    num_threads : int
        Number of parallel threads
    vertical_barrier_times : pd.Series, optional
        Pre-computed vertical barrier times
    side : pd.Series, optional
        Side predictions (1=long, -1=short) for meta-labeling

    Returns
    -------
    pd.DataFrame
        Events with columns: t1 (exit time), trgt (target), side, ret, label
    """
    # Filter events within close index
    t_events = pd.DatetimeIndex(t_events)
    t_events = t_events[t_events >= close.index[0]]
    t_events = t_events[t_events <= close.index[-1]]

    # Get target values aligned with events
    target = target.reindex(t_events, method='ffill')
    target = target[target > min_ret]
    t_events = target.index

    # Initialize vertical barriers
    if vertical_barrier_times is None:
        vertical_barrier_times = pd.Series(
            pd.NaT,
            index=t_events
        )
    else:
        vertical_barrier_times = vertical_barrier_times.reindex(t_events)

    # Initialize side (default: long)
    if side is None:
        side = pd.Series(1, index=t_events)
    else:
        side = side.reindex(t_events, fill_value=1)

    # Create events DataFrame
    events = pd.DataFrame({
        't1': vertical_barrier_times,
        'trgt': target,
        'side': side
    })

    # Remove events with NaN targets
    events = events.dropna(subset=['trgt'])

    return events


def get_labels(
    events: pd.DataFrame,
    close: pd.Series
) -> pd.DataFrame:
    """
    Generate labels from events using triple barrier method.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame from get_events()
    close : pd.Series
        Close prices

    Returns
    -------
    pd.DataFrame
        Labels with columns: ret (return), bin (label), t1 (exit time)
    """
    out = pd.DataFrame(index=events.index)

    # Convert to numpy for numba optimization
    prices = close.values
    price_idx = close.index

    results = []

    for t0 in events.index:
        event = events.loc[t0]

        # Get entry price and index
        entry_price = close.loc[t0]
        entry_idx = price_idx.get_loc(t0)

        # Get vertical barrier
        if pd.notna(event['t1']):
            end_time = event['t1']
            # Use searchsorted for ffill-like behavior (pandas 2.0+ compatible)
            end_idx = price_idx.searchsorted(end_time, side='right') - 1
            end_idx = max(0, min(end_idx, len(close) - 1))
        else:
            end_idx = len(close) - 1
            end_time = price_idx[end_idx]

        # Calculate barrier levels
        target = event['trgt']
        side = event.get('side', 1)

        # Adjust barriers based on side
        if side == 1:  # Long position
            pt_level = entry_price * (1 + target)
            sl_level = entry_price * (1 - target)
        else:  # Short position
            pt_level = entry_price * (1 - target)  # TP is below for short
            sl_level = entry_price * (1 + target)  # SL is above for short

        # Find first barrier touch
        barrier_type, touch_idx, ret = _find_first_barrier_touch(
            prices,
            np.arange(len(prices)),
            entry_idx,
            end_idx,
            pt_level,
            sl_level,
            entry_price
        )

        # Adjust return and label for side
        ret = ret * side

        # Determine label
        if barrier_type == 1:  # TP hit
            label = 1
        elif barrier_type == -1:  # SL hit
            label = -1
        else:  # Vertical barrier
            label = np.sign(ret)
            if label == 0:
                label = 0  # No movement

        results.append({
            't0': t0,
            'ret': ret,
            'bin': label,
            't1': price_idx[touch_idx],
            'barrier_type': barrier_type
        })

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.set_index('t0')

    return out


# =============================================================================
# Dynamic Barriers
# =============================================================================

def get_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR) for dynamic barrier sizing.

    ATR is often better than volatility for barrier sizing because it
    captures intraday price movements.

    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int
        ATR period

    Returns
    -------
    pd.Series
        ATR values
    """
    # True Range components
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))

    # True Range is max of components
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is smoothed True Range
    atr = true_range.ewm(span=period, min_periods=period).mean()

    return atr


def get_atr_barriers(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    t_events: pd.DatetimeIndex,
    atr_period: int = 14,
    atr_multiplier_tp: float = 2.0,
    atr_multiplier_sl: float = 2.0
) -> pd.DataFrame:
    """
    Calculate ATR-based barriers for dynamic sizing.

    ATR barriers adapt to current market conditions better than
    fixed volatility barriers.

    Parameters
    ----------
    close : pd.Series
        Close prices
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    t_events : pd.DatetimeIndex
        Event timestamps
    atr_period : int
        ATR calculation period
    atr_multiplier_tp : float
        ATR multiplier for take-profit
    atr_multiplier_sl : float
        ATR multiplier for stop-loss

    Returns
    -------
    pd.DataFrame
        Barriers with 'pt', 'sl', 'atr' columns
    """
    atr = get_atr(high, low, close, atr_period)

    barriers = pd.DataFrame(index=t_events)

    for t in t_events:
        if t not in close.index:
            continue

        entry_price = close.loc[t]
        atr_value = atr.loc[t]

        barriers.loc[t, 'pt'] = entry_price + atr_multiplier_tp * atr_value
        barriers.loc[t, 'sl'] = entry_price - atr_multiplier_sl * atr_value
        barriers.loc[t, 'atr'] = atr_value

    return barriers


def get_asymmetric_barriers(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    target: pd.Series,
    tp_sl_ratio: float = 1.5,
    base_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Calculate asymmetric barriers where TP != SL.

    Asymmetric barriers allow for different risk/reward profiles.
    A tp_sl_ratio > 1 means TP is further than SL (risk-averse).

    Parameters
    ----------
    close : pd.Series
        Close prices
    t_events : pd.DatetimeIndex
        Event timestamps
    target : pd.Series
        Volatility target
    tp_sl_ratio : float
        Ratio of TP to SL distance (default: 1.5)
    base_multiplier : float
        Base volatility multiplier

    Returns
    -------
    pd.DataFrame
        Asymmetric barriers
    """
    barriers = pd.DataFrame(index=t_events, columns=['pt', 'sl'])

    for t in t_events:
        if t not in close.index or t not in target.index:
            continue

        entry_price = close.loc[t]
        vol = target.loc[t]

        # SL uses base multiplier
        sl_distance = base_multiplier * vol
        # TP is tp_sl_ratio times SL distance
        tp_distance = sl_distance * tp_sl_ratio

        barriers.loc[t, 'pt'] = entry_price * (1 + tp_distance)
        barriers.loc[t, 'sl'] = entry_price * (1 - sl_distance)

    return barriers


def get_adaptive_barriers(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    target: pd.Series,
    regime: pd.Series,
    regime_multipliers: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Calculate regime-adaptive barriers.

    Barriers are adjusted based on detected market regime.

    Parameters
    ----------
    close : pd.Series
        Close prices
    t_events : pd.DatetimeIndex
        Event timestamps
    target : pd.Series
        Volatility target
    regime : pd.Series
        Detected market regime
    regime_multipliers : Dict
        Multipliers for each regime type {regime: (tp_mult, sl_mult)}

    Returns
    -------
    pd.DataFrame
        Regime-adaptive barriers
    """
    if regime_multipliers is None:
        regime_multipliers = {
            RegimeType.TRENDING_UP.value: (3.0, 1.5),      # Wide TP, tight SL
            RegimeType.TRENDING_DOWN.value: (3.0, 1.5),    # Wide TP, tight SL
            RegimeType.MEAN_REVERTING.value: (1.5, 2.0),   # Tight TP, wider SL
            RegimeType.RANDOM_WALK.value: (2.0, 2.0),      # Balanced
            RegimeType.HIGH_VOLATILITY.value: (3.0, 3.0),  # Wider barriers
            RegimeType.LOW_VOLATILITY.value: (1.5, 1.5),   # Tighter barriers
        }

    barriers = pd.DataFrame(index=t_events, columns=['pt', 'sl', 'regime'])

    for t in t_events:
        if t not in close.index:
            continue

        entry_price = close.loc[t]
        vol = target.get(t, np.nan)
        current_regime = regime.get(t, RegimeType.RANDOM_WALK.value)

        if pd.isna(vol):
            continue

        tp_mult, sl_mult = regime_multipliers.get(
            current_regime,
            (2.0, 2.0)
        )

        barriers.loc[t, 'pt'] = entry_price * (1 + tp_mult * vol)
        barriers.loc[t, 'sl'] = entry_price * (1 - sl_mult * vol)
        barriers.loc[t, 'regime'] = current_regime

    return barriers


# =============================================================================
# Meta-Labeling
# =============================================================================

def get_meta_labels(
    primary_model_pred: pd.Series,
    events: pd.DataFrame,
    close: pd.Series
) -> pd.DataFrame:
    """
    Generate meta-labels for bet sizing.

    Meta-labeling works as follows:
    1. Primary model predicts direction (side)
    2. Secondary model (meta-model) predicts whether to take the bet

    Parameters
    ----------
    primary_model_pred : pd.Series
        Primary model predictions (1=long, -1=short, 0=no position)
    events : pd.DataFrame
        Events DataFrame
    close : pd.Series
        Close prices

    Returns
    -------
    pd.DataFrame
        Meta-labels with 'primary_pred', 'meta_label', 'ret'
    """
    # Filter to events where primary model has a prediction
    active_events = events.loc[primary_model_pred != 0].copy()

    if active_events.empty:
        return pd.DataFrame()

    # Update side based on primary model prediction
    active_events['side'] = primary_model_pred.loc[active_events.index]

    # Get labels using triple barrier
    labels = get_labels(active_events, close)

    if labels.empty:
        return pd.DataFrame()

    # Meta-label: 1 if primary model's direction was correct, 0 otherwise
    meta_labels = pd.DataFrame(index=labels.index)
    meta_labels['primary_pred'] = primary_model_pred.loc[labels.index]
    meta_labels['ret'] = labels['ret']

    # If return sign matches prediction sign, meta-label is 1
    meta_labels['meta_label'] = (
        (meta_labels['ret'] * meta_labels['primary_pred']) > 0
    ).astype(int)

    return meta_labels


def combine_predictions(
    primary_pred: pd.Series,
    meta_pred: pd.Series,
    meta_probability: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Combine primary and meta-model predictions.

    Parameters
    ----------
    primary_pred : pd.Series
        Primary model direction predictions
    meta_pred : pd.Series
        Meta-model predictions (take bet or not)
    meta_probability : pd.Series, optional
        Meta-model probability scores for bet sizing

    Returns
    -------
    pd.DataFrame
        Combined predictions with position sizing
    """
    combined = pd.DataFrame(index=primary_pred.index)

    combined['direction'] = primary_pred
    combined['take_bet'] = meta_pred.reindex(primary_pred.index, fill_value=0)

    # Final position = direction * take_bet
    combined['position'] = combined['direction'] * combined['take_bet']

    # Bet size based on meta-model probability
    if meta_probability is not None:
        meta_prob = meta_probability.reindex(primary_pred.index, fill_value=0.5)
        # Size is 2 * (prob - 0.5), ranging from 0 to 1
        combined['bet_size'] = (2 * (meta_prob - 0.5)).clip(0, 1)
        combined['sized_position'] = combined['position'] * combined['bet_size']
    else:
        combined['bet_size'] = 1.0
        combined['sized_position'] = combined['position']

    return combined


# =============================================================================
# Sample Weights
# =============================================================================

@jit(nopython=True, cache=True, parallel=True)
def _count_concurrent_events(
    t_starts: np.ndarray,
    t_ends: np.ndarray,
    timestamps: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized concurrent event counting.

    Count how many events overlap at each timestamp.
    """
    n_events = len(t_starts)
    n_times = len(timestamps)

    counts = np.zeros(n_times, dtype=np.float64)

    for i in prange(n_times):
        t = timestamps[i]
        count = 0
        for j in range(n_events):
            if t_starts[j] <= t <= t_ends[j]:
                count += 1
        counts[i] = count

    return counts


def get_num_co_events(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,
    molecule: Optional[List] = None
) -> pd.Series:
    """
    Count number of concurrent events at each timestamp.

    Concurrent events cause label overlap, which inflates the
    importance of overlapping samples.

    Parameters
    ----------
    close_idx : pd.DatetimeIndex
        Index of close prices
    t1 : pd.Series
        End times of events (indexed by start times)
    molecule : List, optional
        Subset of events to process

    Returns
    -------
    pd.Series
        Number of concurrent events at each timestamp
    """
    if molecule is None:
        molecule = t1.index

    # Filter to valid events
    t1 = t1.dropna()

    # Convert to numpy for numba
    t_starts = t1.index.values.astype('datetime64[ns]').astype(np.int64)
    t_ends = t1.values.astype('datetime64[ns]').astype(np.int64)
    timestamps = close_idx.values.astype('datetime64[ns]').astype(np.int64)

    # Count concurrent events
    counts = _count_concurrent_events(t_starts, t_ends, timestamps)

    return pd.Series(counts, index=close_idx)


def get_sample_uniqueness(
    t1: pd.Series,
    num_co_events: pd.Series,
    molecule: Optional[List] = None
) -> pd.Series:
    """
    Calculate sample uniqueness for each event.

    Uniqueness measures how much information is unique to each sample.
    Samples that overlap with many others have lower uniqueness.

    Parameters
    ----------
    t1 : pd.Series
        End times of events
    num_co_events : pd.Series
        Number of concurrent events at each timestamp
    molecule : List, optional
        Subset of events

    Returns
    -------
    pd.Series
        Sample uniqueness (0 to 1)
    """
    if molecule is None:
        molecule = t1.index

    uniqueness = pd.Series(index=molecule, dtype=float)

    for t0 in molecule:
        if t0 not in t1.index or pd.isna(t1[t0]):
            uniqueness[t0] = np.nan
            continue

        t_end = t1[t0]

        # Get concurrent events during this sample's lifetime
        mask = (num_co_events.index >= t0) & (num_co_events.index <= t_end)
        concurrent = num_co_events[mask]

        if len(concurrent) > 0 and concurrent.sum() > 0:
            # Average uniqueness = 1 / average number of concurrent events
            uniqueness[t0] = (1.0 / concurrent).mean()
        else:
            uniqueness[t0] = 1.0

    return uniqueness


def get_sample_tw(
    t1: pd.Series,
    num_co_events: pd.Series,
    molecule: Optional[List] = None
) -> pd.Series:
    """
    Calculate time-weighted sample weights.

    This combines uniqueness with the duration of each sample.
    Longer samples that don't overlap much get higher weights.

    Parameters
    ----------
    t1 : pd.Series
        End times of events
    num_co_events : pd.Series
        Number of concurrent events
    molecule : List, optional
        Subset of events

    Returns
    -------
    pd.Series
        Time-weighted sample weights
    """
    if molecule is None:
        molecule = t1.index

    weights = pd.Series(index=molecule, dtype=float)

    for t0 in molecule:
        if t0 not in t1.index or pd.isna(t1[t0]):
            weights[t0] = np.nan
            continue

        t_end = t1[t0]

        # Get concurrent events during sample lifetime
        mask = (num_co_events.index >= t0) & (num_co_events.index <= t_end)
        concurrent = num_co_events[mask]

        if len(concurrent) > 0 and concurrent.sum() > 0:
            # Weight = sum of (1 / concurrent_at_t) for each t
            weights[t0] = (1.0 / concurrent).sum()
        else:
            weights[t0] = 1.0

    return weights


def get_decay_weights(
    t1: pd.Series,
    decay_type: str = 'linear',
    decay_factor: float = 1.0
) -> pd.Series:
    """
    Calculate temporal decay weights.

    More recent samples get higher weights.

    Parameters
    ----------
    t1 : pd.Series
        Event end times (used to order events)
    decay_type : str
        'linear' or 'exponential'
    decay_factor : float
        Decay rate (higher = faster decay)

    Returns
    -------
    pd.Series
        Decay weights
    """
    n = len(t1)

    if decay_type == 'linear':
        # Linear decay from decay_factor to 1
        weights = np.linspace(1.0 / decay_factor, 1.0, n)

    elif decay_type == 'exponential':
        # Exponential decay
        x = np.linspace(0, 1, n)
        weights = np.exp(decay_factor * (x - 1))

    else:
        raise ValueError(f"Unknown decay type: {decay_type}")

    return pd.Series(weights, index=t1.index)


def get_sample_weights(
    t1: pd.Series,
    close: pd.Series,
    events: Optional[pd.DataFrame] = None,
    num_co_events: Optional[pd.Series] = None,
    use_uniqueness: bool = True,
    use_decay: bool = True,
    decay_type: str = 'linear',
    decay_factor: float = 1.0
) -> pd.Series:
    """
    Calculate combined sample weights.

    Combines uniqueness weights and temporal decay.

    Parameters
    ----------
    t1 : pd.Series
        Event end times
    close : pd.Series
        Close prices
    events : pd.DataFrame, optional
        Events DataFrame (for returns-based weighting)
    num_co_events : pd.Series, optional
        Pre-computed concurrent events
    use_uniqueness : bool
        Whether to use uniqueness weighting
    use_decay : bool
        Whether to use temporal decay
    decay_type : str
        Type of decay
    decay_factor : float
        Decay rate

    Returns
    -------
    pd.Series
        Final sample weights
    """
    weights = pd.Series(1.0, index=t1.index)

    if use_uniqueness:
        if num_co_events is None:
            num_co_events = get_num_co_events(close.index, t1)
        uniqueness = get_sample_uniqueness(t1, num_co_events)
        weights = weights * uniqueness

    if use_decay:
        decay = get_decay_weights(t1, decay_type, decay_factor)
        weights = weights * decay

    # Normalize weights
    weights = weights / weights.sum() * len(weights)

    return weights


# =============================================================================
# Regime Detection
# =============================================================================

def detect_volatility_regime(
    close: pd.Series,
    lookback: int = 20,
    threshold_low: float = 0.5,
    threshold_high: float = 2.0
) -> pd.Series:
    """
    Detect volatility regime.

    Compares current volatility to historical average.

    Parameters
    ----------
    close : pd.Series
        Close prices
    lookback : int
        Lookback period for volatility calculation
    threshold_low : float
        Threshold for low volatility (relative to average)
    threshold_high : float
        Threshold for high volatility

    Returns
    -------
    pd.Series
        Volatility regime labels
    """
    returns = close.pct_change()
    vol = returns.rolling(window=lookback).std()

    # Long-term average volatility
    vol_avg = vol.rolling(window=lookback * 5).mean()
    vol_ratio = vol / vol_avg

    regime = pd.Series(index=close.index, dtype=str)
    regime[:] = RegimeType.RANDOM_WALK.value

    regime[vol_ratio < threshold_low] = RegimeType.LOW_VOLATILITY.value
    regime[vol_ratio > threshold_high] = RegimeType.HIGH_VOLATILITY.value

    return regime


def detect_trend_regime(
    close: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
    threshold: float = 0.02
) -> pd.Series:
    """
    Detect trend regime using moving average crossover.

    Parameters
    ----------
    close : pd.Series
        Close prices
    short_window : int
        Short MA window
    long_window : int
        Long MA window
    threshold : float
        Minimum separation for trend detection

    Returns
    -------
    pd.Series
        Trend regime labels
    """
    ma_short = close.rolling(window=short_window).mean()
    ma_long = close.rolling(window=long_window).mean()

    # Calculate relative difference
    diff = (ma_short - ma_long) / ma_long

    regime = pd.Series(index=close.index, dtype=str)
    regime[:] = RegimeType.RANDOM_WALK.value

    regime[diff > threshold] = RegimeType.TRENDING_UP.value
    regime[diff < -threshold] = RegimeType.TRENDING_DOWN.value

    return regime


@jit(nopython=True, cache=True)
def _calculate_hurst_rs(prices: np.ndarray, min_lag: int, max_lag: int) -> float:
    """
    Numba-optimized R/S analysis for Hurst exponent.
    """
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    n = len(returns)

    lags = []
    rs_values = []

    # Calculate R/S for different lags
    for lag in range(min_lag, min(max_lag, n // 2)):
        n_chunks = n // lag
        if n_chunks < 1:
            continue

        rs_sum = 0.0
        valid_chunks = 0

        for i in range(n_chunks):
            chunk = returns[i * lag:(i + 1) * lag]
            if len(chunk) < lag:
                continue

            mean = np.mean(chunk)
            std = np.std(chunk)

            if std == 0:
                continue

            # Cumulative deviations
            cumsum = np.zeros(lag)
            cumsum[0] = chunk[0] - mean
            for j in range(1, lag):
                cumsum[j] = cumsum[j-1] + chunk[j] - mean

            # Range
            r = np.max(cumsum) - np.min(cumsum)

            # R/S ratio
            rs_sum += r / std
            valid_chunks += 1

        if valid_chunks > 0:
            lags.append(float(lag))
            rs_values.append(rs_sum / valid_chunks)

    if len(lags) < 2:
        return 0.5  # Return neutral value if insufficient data

    # Linear regression in log-log space
    log_lags = np.log(np.array(lags))
    log_rs = np.log(np.array(rs_values))

    # Simple linear regression for slope (Hurst exponent)
    n_pts = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_lags * log_rs)
    sum_x2 = np.sum(log_lags * log_lags)

    hurst = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x * sum_x)

    return hurst


def calculate_hurst_exponent(
    prices: pd.Series,
    min_lag: int = 2,
    max_lag: int = 100
) -> float:
    """
    Calculate Hurst exponent for persistence detection.

    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending/persistent

    Parameters
    ----------
    prices : pd.Series
        Price series
    min_lag : int
        Minimum lag for R/S analysis
    max_lag : int
        Maximum lag

    Returns
    -------
    float
        Hurst exponent
    """
    return _calculate_hurst_rs(prices.values, min_lag, max_lag)


def detect_hurst_regime(
    close: pd.Series,
    window: int = 100,
    mean_revert_threshold: float = 0.4,
    trend_threshold: float = 0.6
) -> pd.Series:
    """
    Detect regime using rolling Hurst exponent.

    Parameters
    ----------
    close : pd.Series
        Close prices
    window : int
        Rolling window size
    mean_revert_threshold : float
        Below this = mean reverting
    trend_threshold : float
        Above this = trending

    Returns
    -------
    pd.Series
        Hurst-based regime labels
    """
    regime = pd.Series(index=close.index, dtype=str)
    regime[:] = RegimeType.RANDOM_WALK.value

    for i in range(window, len(close)):
        window_prices = close.iloc[i-window:i]
        hurst = calculate_hurst_exponent(window_prices)

        if hurst < mean_revert_threshold:
            regime.iloc[i] = RegimeType.MEAN_REVERTING.value
        elif hurst > trend_threshold:
            # Determine trend direction from recent returns
            recent_return = window_prices.iloc[-1] / window_prices.iloc[0] - 1
            if recent_return > 0:
                regime.iloc[i] = RegimeType.TRENDING_UP.value
            else:
                regime.iloc[i] = RegimeType.TRENDING_DOWN.value

    return regime


def detect_combined_regime(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    vol_lookback: int = 20,
    trend_short: int = 20,
    trend_long: int = 50,
    hurst_window: int = 100
) -> pd.DataFrame:
    """
    Detect regime using multiple methods combined.

    Parameters
    ----------
    close : pd.Series
        Close prices
    high : pd.Series, optional
        High prices (for ATR-based volatility)
    low : pd.Series, optional
        Low prices
    vol_lookback : int
        Volatility lookback period
    trend_short : int
        Short MA for trend
    trend_long : int
        Long MA for trend
    hurst_window : int
        Window for Hurst calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with individual regimes and combined regime
    """
    regimes = pd.DataFrame(index=close.index)

    # Individual regime detections
    regimes['vol_regime'] = detect_volatility_regime(close, vol_lookback)
    regimes['trend_regime'] = detect_trend_regime(close, trend_short, trend_long)

    # Hurst regime (computed less frequently due to cost)
    regimes['hurst_regime'] = RegimeType.RANDOM_WALK.value

    for i in range(hurst_window, len(close), hurst_window // 2):
        window_regime = detect_hurst_regime(
            close.iloc[max(0, i-hurst_window):i],
            window=min(hurst_window, i)
        )
        if len(window_regime) > 0:
            regimes.loc[close.index[max(0, i-hurst_window):i], 'hurst_regime'] = window_regime.values

    # Combined regime logic
    def combine_regimes(row):
        vol = row['vol_regime']
        trend = row['trend_regime']
        hurst = row['hurst_regime']

        # Priority: High volatility > Trending > Mean reverting > Random walk
        if vol == RegimeType.HIGH_VOLATILITY.value:
            return RegimeType.HIGH_VOLATILITY.value
        elif trend in [RegimeType.TRENDING_UP.value, RegimeType.TRENDING_DOWN.value]:
            return trend
        elif hurst == RegimeType.MEAN_REVERTING.value:
            return RegimeType.MEAN_REVERTING.value
        elif vol == RegimeType.LOW_VOLATILITY.value:
            return RegimeType.LOW_VOLATILITY.value
        else:
            return RegimeType.RANDOM_WALK.value

    regimes['combined_regime'] = regimes.apply(combine_regimes, axis=1)

    return regimes


# =============================================================================
# Event-Based Sampling
# =============================================================================

@jit(nopython=True, cache=True)
def _cusum_filter_core(
    returns: np.ndarray,
    threshold: float
) -> List:
    """
    Numba-optimized CUSUM filter core.
    """
    events = []
    s_pos = 0.0
    s_neg = 0.0

    for i in range(len(returns)):
        ret = returns[i]

        # Positive CUSUM
        s_pos = max(0, s_pos + ret)

        # Negative CUSUM
        s_neg = min(0, s_neg + ret)

        # Check thresholds
        if s_pos > threshold:
            events.append(i)
            s_pos = 0.0
        elif s_neg < -threshold:
            events.append(i)
            s_neg = 0.0

    return events


def cusum_filter(
    close: pd.Series,
    threshold: float
) -> pd.DatetimeIndex:
    """
    CUSUM filter for event detection.

    Detects structural breaks in the price series when cumulative
    returns exceed a threshold.

    Parameters
    ----------
    close : pd.Series
        Close prices
    threshold : float
        CUSUM threshold (in return units)

    Returns
    -------
    pd.DatetimeIndex
        Timestamps of detected events
    """
    returns = close.pct_change().fillna(0).values

    event_indices = _cusum_filter_core(returns, threshold)

    return close.index[event_indices]


def cusum_filter_symmetric(
    close: pd.Series,
    threshold: float
) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM filter.

    Unlike standard CUSUM, this resets both positive and negative
    cumulative sums when either crosses threshold.

    Parameters
    ----------
    close : pd.Series
        Close prices
    threshold : float
        CUSUM threshold

    Returns
    -------
    pd.DatetimeIndex
        Event timestamps
    """
    returns = close.pct_change().fillna(0)

    events = []
    s_pos = 0.0
    s_neg = 0.0

    for i, ret in enumerate(returns):
        s_pos = max(0, s_pos + ret)
        s_neg = min(0, s_neg + ret)

        if s_pos > threshold or s_neg < -threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0

    return close.index[events]


def fractional_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
    max_weights: int = 100
) -> pd.Series:
    """
    Fractional differentiation for stationarity while preserving memory.

    Standard differentiation (d=1) loses information about long-term memory.
    Fractional differentiation with d < 1 achieves stationarity while
    preserving some memory.

    Parameters
    ----------
    series : pd.Series
        Input series
    d : float
        Differentiation order (0 < d < 1)
    threshold : float
        Weight cutoff threshold
    max_weights : int
        Maximum number of weights to compute

    Returns
    -------
    pd.Series
        Fractionally differentiated series
    """
    if d == 0:
        return series.copy()

    # Calculate weights using the binomial expansion
    # w_k = (-1)^k * C(d, k) where C(d, k) = d * (d-1) * ... * (d-k+1) / k!
    weights = [1.0]
    k = 1

    while k < max_weights:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights = np.array(weights)
    width = len(weights)

    # Apply weights using convolution (forward-looking weights)
    result = pd.Series(index=series.index, dtype=float)
    values = series.values

    for i in range(width - 1, len(values)):
        # Apply weights to past values including current
        window = values[i - width + 1:i + 1]
        result.iloc[i] = np.dot(weights[::-1], window)

    return result


def find_min_ffd(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.0, 1.0),
    d_step: float = 0.01,
    pvalue_threshold: float = 0.05
) -> Tuple[float, pd.Series]:
    """
    Find minimum d for fractional differentiation to achieve stationarity.

    Uses ADF test to find the minimum d that makes the series stationary.

    Parameters
    ----------
    series : pd.Series
        Input series
    d_range : Tuple[float, float]
        Range of d values to test
    d_step : float
        Step size for d
    pvalue_threshold : float
        P-value threshold for stationarity

    Returns
    -------
    Tuple[float, pd.Series]
        (minimum d, fractionally differentiated series)
    """
    from statsmodels.tsa.stattools import adfuller

    d_values = np.arange(d_range[0], d_range[1] + d_step, d_step)

    for d in d_values:
        if d == 0:
            ffd = series
        else:
            ffd = fractional_diff(series, d)

        ffd_clean = ffd.dropna()

        if len(ffd_clean) < 20:
            continue

        try:
            adf_stat, pvalue, _, _, _, _ = adfuller(ffd_clean)

            if pvalue < pvalue_threshold:
                logger.info(f"Found stationary series at d={d:.2f} (p-value={pvalue:.4f})")
                return d, ffd

        except Exception:
            continue

    logger.warning("Could not find stationary d, returning d=1")
    return 1.0, series.diff()


def event_driven_bars(
    close: pd.Series,
    volume: pd.Series,
    dollar_volume: pd.Series,
    bar_type: str = 'dollar',
    threshold: float = 1e6
) -> pd.DatetimeIndex:
    """
    Generate event-driven bars (dollar bars, volume bars, tick bars).

    Event-driven bars sample the market based on activity rather than time.

    Parameters
    ----------
    close : pd.Series
        Close prices
    volume : pd.Series
        Volume
    dollar_volume : pd.Series
        Dollar volume (price * volume)
    bar_type : str
        'dollar', 'volume', or 'tick'
    threshold : float
        Threshold for bar generation

    Returns
    -------
    pd.DatetimeIndex
        Bar timestamps
    """
    if bar_type == 'dollar':
        values = dollar_volume
    elif bar_type == 'volume':
        values = volume
    elif bar_type == 'tick':
        values = pd.Series(1, index=close.index)
    else:
        raise ValueError(f"Unknown bar type: {bar_type}")

    cumsum = 0.0
    bars = []

    for i, val in enumerate(values):
        cumsum += val
        if cumsum >= threshold:
            bars.append(i)
            cumsum = 0.0

    return close.index[bars]


# =============================================================================
# Triple Barrier Labeler (High-Level Interface)
# =============================================================================

class TripleBarrierLabeler:
    """
    High-level interface for triple barrier labeling.

    This class provides a convenient interface for the complete
    triple barrier labeling pipeline.

    Parameters
    ----------
    config : BarrierConfig
        Barrier configuration
    """

    def __init__(self, config: Optional[BarrierConfig] = None):
        self.config = config or BarrierConfig()
        self._vol = None
        self._events = None
        self._labels = None
        self._regime = None

    def fit(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        t_events: Optional[pd.DatetimeIndex] = None,
        side: Optional[pd.Series] = None
    ) -> 'TripleBarrierLabeler':
        """
        Fit the labeler to data and generate labels.

        Parameters
        ----------
        close : pd.Series
            Close prices
        high : pd.Series, optional
            High prices (for ATR barriers)
        low : pd.Series, optional
            Low prices (for ATR barriers)
        t_events : pd.DatetimeIndex, optional
            Event timestamps (if None, uses all timestamps)
        side : pd.Series, optional
            Side predictions for meta-labeling

        Returns
        -------
        self
        """
        # Calculate volatility
        self._vol = get_daily_vol(close)

        # Default events: all timestamps
        if t_events is None:
            t_events = close.index

        # Calculate vertical barriers
        vertical_barriers = get_vertical_barrier(
            t_events, close, self.config.vertical_bars
        )

        # Prepare barrier multipliers
        if self.config.use_asymmetric:
            pt_mult = self.config.sl_multiplier * self.config.tp_sl_ratio
            sl_mult = self.config.sl_multiplier
        else:
            pt_mult = self.config.tp_multiplier
            sl_mult = self.config.sl_multiplier

        # Get events
        self._events = get_events(
            close=close,
            t_events=t_events,
            pt_sl=(pt_mult, sl_mult),
            target=self._vol,
            min_ret=self.config.min_return,
            vertical_barrier_times=vertical_barriers,
            side=side
        )

        # Generate labels
        self._labels = get_labels(self._events, close)

        return self

    def fit_with_regime(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        t_events: Optional[pd.DatetimeIndex] = None,
        side: Optional[pd.Series] = None
    ) -> 'TripleBarrierLabeler':
        """
        Fit with regime-adaptive barriers.

        Parameters
        ----------
        close : pd.Series
            Close prices
        high : pd.Series, optional
            High prices
        low : pd.Series, optional
            Low prices
        t_events : pd.DatetimeIndex, optional
            Event timestamps
        side : pd.Series, optional
            Side predictions

        Returns
        -------
        self
        """
        # Detect regime
        self._regime = detect_combined_regime(close, high, low)

        # Calculate volatility
        self._vol = get_daily_vol(close)

        if t_events is None:
            t_events = close.index

        # Get regime-adaptive barriers
        barriers = get_adaptive_barriers(
            close=close,
            t_events=t_events,
            target=self._vol,
            regime=self._regime['combined_regime']
        )

        # Vertical barriers
        vertical_barriers = get_vertical_barrier(
            t_events, close, self.config.vertical_bars
        )

        # Create events with adaptive targets
        self._events = pd.DataFrame({
            't1': vertical_barriers,
            'trgt': self._vol,
            'side': side if side is not None else 1
        })
        self._events = self._events.dropna(subset=['trgt'])

        # Generate labels
        self._labels = get_labels(self._events, close)

        return self

    def get_labels(self) -> pd.DataFrame:
        """Get generated labels."""
        if self._labels is None:
            raise ValueError("Must call fit() first")
        return self._labels

    def get_events(self) -> pd.DataFrame:
        """Get events DataFrame."""
        if self._events is None:
            raise ValueError("Must call fit() first")
        return self._events

    def get_sample_weights(
        self,
        close: pd.Series,
        use_uniqueness: bool = True,
        use_decay: bool = True,
        decay_type: str = 'linear',
        decay_factor: float = 1.0
    ) -> pd.Series:
        """
        Get sample weights for the generated labels.

        Parameters
        ----------
        close : pd.Series
            Close prices
        use_uniqueness : bool
            Whether to use uniqueness weighting
        use_decay : bool
            Whether to use temporal decay
        decay_type : str
            Type of decay
        decay_factor : float
            Decay rate

        Returns
        -------
        pd.Series
            Sample weights
        """
        if self._labels is None:
            raise ValueError("Must call fit() first")

        # Get t1 from labels
        t1 = self._labels['t1']

        return get_sample_weights(
            t1=t1,
            close=close,
            events=self._events,
            use_uniqueness=use_uniqueness,
            use_decay=use_decay,
            decay_type=decay_type,
            decay_factor=decay_factor
        )

    def get_regime(self) -> pd.DataFrame:
        """Get detected regime."""
        return self._regime


class MetaLabeler:
    """
    Meta-labeling implementation for bet sizing.

    Meta-labeling separates direction prediction from bet sizing.
    The primary model predicts direction, the meta-model predicts
    whether to take the bet.
    """

    def __init__(self, primary_model=None, meta_model=None):
        """
        Initialize MetaLabeler.

        Parameters
        ----------
        primary_model : sklearn estimator
            Model for direction prediction (optional, can be set later)
        meta_model : sklearn estimator
            Model for meta-labeling (optional, can be set later)
        """
        self.primary_model = primary_model
        self.meta_model = meta_model
        self._primary_labels = None
        self._meta_labels = None

    def generate_meta_labels(
        self,
        primary_pred: pd.Series,
        events: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Generate meta-labels from primary model predictions.

        Parameters
        ----------
        primary_pred : pd.Series
            Primary model predictions
        events : pd.DataFrame
            Events DataFrame
        close : pd.Series
            Close prices

        Returns
        -------
        pd.DataFrame
            Meta-labels
        """
        self._meta_labels = get_meta_labels(primary_pred, events, close)
        return self._meta_labels

    def train_meta_model(
        self,
        X: pd.DataFrame,
        meta_labels: pd.Series
    ) -> 'MetaLabeler':
        """
        Train the meta-model.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        meta_labels : pd.Series
            Meta-labels (binary: take bet or not)

        Returns
        -------
        self
        """
        if self.meta_model is None:
            raise ValueError("meta_model must be set before training")

        # Align data
        common_idx = X.index.intersection(meta_labels.index)
        X_train = X.loc[common_idx]
        y_train = meta_labels.loc[common_idx]

        self.meta_model.fit(X_train, y_train)
        return self

    def predict_with_sizing(
        self,
        X: pd.DataFrame,
        primary_pred: pd.Series
    ) -> pd.DataFrame:
        """
        Predict with bet sizing.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        primary_pred : pd.Series
            Primary model predictions

        Returns
        -------
        pd.DataFrame
            Combined predictions with sizing
        """
        if self.meta_model is None:
            raise ValueError("meta_model must be trained first")

        # Get meta-model predictions
        common_idx = X.index.intersection(primary_pred.index)
        X_pred = X.loc[common_idx]

        meta_pred = pd.Series(
            self.meta_model.predict(X_pred),
            index=common_idx
        )

        # Get probabilities if available
        if hasattr(self.meta_model, 'predict_proba'):
            meta_prob = pd.Series(
                self.meta_model.predict_proba(X_pred)[:, 1],
                index=common_idx
            )
        else:
            meta_prob = None

        return combine_predictions(primary_pred, meta_pred, meta_prob)


# =============================================================================
# Utility Functions
# =============================================================================

def create_labels_pipeline(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
    config: Optional[BarrierConfig] = None,
    use_cusum: bool = True,
    cusum_threshold: float = 0.02,
    use_regime: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete pipeline for generating triple barrier labels.

    This function combines all steps:
    1. Event detection (CUSUM)
    2. Regime detection
    3. Triple barrier labeling
    4. Sample weight calculation

    Parameters
    ----------
    close : pd.Series
        Close prices
    high : pd.Series, optional
        High prices
    low : pd.Series, optional
        Low prices
    volume : pd.Series, optional
        Volume
    config : BarrierConfig, optional
        Barrier configuration
    use_cusum : bool
        Whether to use CUSUM for event detection
    cusum_threshold : float
        CUSUM threshold
    use_regime : bool
        Whether to use regime-adaptive barriers

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (labels DataFrame, sample weights)
    """
    config = config or BarrierConfig()

    # Step 1: Event detection
    if use_cusum:
        t_events = cusum_filter(close, cusum_threshold)
        logger.info(f"CUSUM detected {len(t_events)} events")
    else:
        t_events = close.index

    # Step 2: Create labeler and fit
    labeler = TripleBarrierLabeler(config)

    if use_regime and high is not None and low is not None:
        labeler.fit_with_regime(close, high, low, t_events)
    else:
        labeler.fit(close, high, low, t_events)

    # Step 3: Get labels and weights
    labels = labeler.get_labels()
    weights = labeler.get_sample_weights(close)

    logger.info(f"Generated {len(labels)} labels")
    logger.info(f"Label distribution: {labels['bin'].value_counts().to_dict()}")

    return labels, weights


def validate_labels(labels: pd.DataFrame) -> Dict:
    """
    Validate label quality and statistics.

    Parameters
    ----------
    labels : pd.DataFrame
        Labels from triple barrier method

    Returns
    -------
    Dict
        Validation statistics
    """
    stats = {
        'total_labels': len(labels),
        'label_distribution': labels['bin'].value_counts().to_dict(),
        'avg_return': labels['ret'].mean(),
        'std_return': labels['ret'].std(),
        'avg_holding_period': None,
        'win_rate': None,
        'profit_factor': None
    }

    # Calculate holding period
    if 't1' in labels.columns:
        holding_periods = (labels['t1'] - labels.index).dt.total_seconds() / 3600  # hours
        stats['avg_holding_period'] = holding_periods.mean()
        stats['median_holding_period'] = holding_periods.median()

    # Win rate
    stats['win_rate'] = (labels['ret'] > 0).mean()

    # Profit factor
    profits = labels.loc[labels['ret'] > 0, 'ret'].sum()
    losses = abs(labels.loc[labels['ret'] < 0, 'ret'].sum())
    stats['profit_factor'] = profits / losses if losses > 0 else np.inf

    # Check for class imbalance
    label_counts = labels['bin'].value_counts()
    if len(label_counts) > 1:
        max_count = label_counts.max()
        min_count = label_counts.min()
        stats['class_imbalance_ratio'] = max_count / min_count if min_count > 0 else np.inf

    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    """Example usage of triple barrier labeling."""

    # Create sample data
    np.random.seed(42)
    n = 1000

    dates = pd.date_range('2023-01-01', periods=n, freq='1h')

    # Simulate price with trend + noise
    trend = np.cumsum(np.random.randn(n) * 0.001)
    noise = np.random.randn(n) * 0.01
    close = 100 * np.exp(trend + noise)

    close = pd.Series(close, index=dates)
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))

    high = pd.Series(high, index=dates)
    low = pd.Series(low, index=dates)

    logger.info("Creating labels with default config...")

    # Use pipeline
    config = BarrierConfig(
        tp_multiplier=2.0,
        sl_multiplier=2.0,
        vertical_bars=20
    )

    labels, weights = create_labels_pipeline(
        close=close,
        high=high,
        low=low,
        config=config,
        use_cusum=True,
        cusum_threshold=0.02,
        use_regime=True
    )

    # Validate
    stats = validate_labels(labels)

    logger.info("Label Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    return labels, weights, stats


if __name__ == "__main__":
    labels, weights, stats = main()
