"""
Advanced Market Microstructure Features Module.

This module implements proper VPIN, OFI, spread decomposition, queue metrics,
and toxicity indicators for quantitative trading strategies.

All calculators support streaming updates for real-time computation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Protocol,
    TypeVar,
    Generic,
)
import warnings

import numpy as np
from scipy import stats
from scipy.special import ndtr  # Normal CDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Type Definitions and Protocols
# ==============================================================================

T = TypeVar("T")


@dataclass
class Trade:
    """Single trade record."""
    timestamp: int  # Unix timestamp in ms
    price: float
    volume: float
    side: int  # 1 = buy, -1 = sell, 0 = unknown


@dataclass
class Quote:
    """Quote update record."""
    timestamp: int
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


@dataclass
class OHLCV:
    """OHLCV bar data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float = 0.0
    trades: int = 0
    quote_volume: float = 0.0


@dataclass
class VolumeBucket:
    """Volume bucket for VPIN calculation."""
    start_time: int
    end_time: int
    volume: float
    buy_volume: float
    sell_volume: float
    vwap: float
    trades: int

    @property
    def order_imbalance(self) -> float:
        """Calculate order imbalance for this bucket."""
        if self.volume == 0:
            return 0.0
        return abs(self.buy_volume - self.sell_volume) / self.volume


# ==============================================================================
# Base Calculator Class
# ==============================================================================

class StreamingCalculator(ABC):
    """Abstract base class for streaming calculators."""

    @abstractmethod
    def update(self, data: Union[Trade, Quote, OHLCV]) -> None:
        """Update calculator with new data."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset calculator state."""
        pass

    @abstractmethod
    def get_value(self) -> Optional[float]:
        """Get current calculated value."""
        pass


# ==============================================================================
# VPIN Calculator
# ==============================================================================

class VPINCalculator(StreamingCalculator):
    """
    Volume-Synchronized Probability of Informed Trading (VPIN) Calculator.

    VPIN estimates the probability of informed trading by measuring order
    flow toxicity. It uses volume-synchronized sampling to create buckets
    of equal volume and measures order imbalance within each bucket.

    Reference: Easley, D., Lopez de Prado, M., & O'Hara, M. (2012).
    "Flow Toxicity and Liquidity in a High Frequency World"

    Attributes:
        bucket_size: Target volume per bucket
        n_buckets: Number of buckets for rolling VPIN calculation
        use_bulk_classification: Whether to use bulk volume classification
    """

    def __init__(
        self,
        bucket_size: float = 1000.0,
        n_buckets: int = 50,
        use_bulk_classification: bool = True,
        sigma_window: int = 100,
    ) -> None:
        """
        Initialize VPIN Calculator.

        Args:
            bucket_size: Volume per bucket (e.g., 1000 BTC)
            n_buckets: Number of buckets for rolling VPIN (default 50)
            use_bulk_classification: Use BVC algorithm for trade classification
            sigma_window: Window for price volatility estimation
        """
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self.use_bulk_classification = use_bulk_classification
        self.sigma_window = sigma_window

        # Current bucket accumulator
        self._current_bucket_volume: float = 0.0
        self._current_bucket_buy: float = 0.0
        self._current_bucket_sell: float = 0.0
        self._current_bucket_vwap_num: float = 0.0
        self._current_bucket_start_time: int = 0
        self._current_bucket_trades: int = 0

        # Completed buckets
        self._buckets: Deque[VolumeBucket] = deque(maxlen=n_buckets)

        # Price history for BVC
        self._prices: Deque[float] = deque(maxlen=sigma_window)
        self._returns: Deque[float] = deque(maxlen=sigma_window)

        # VPIN history for CDF estimation
        self._vpin_history: Deque[float] = deque(maxlen=1000)

    def update(self, data: Union[Trade, OHLCV]) -> Optional[VolumeBucket]:
        """
        Update VPIN with new trade or OHLCV data.

        Args:
            data: Trade or OHLCV data

        Returns:
            Completed VolumeBucket if one was filled, None otherwise
        """
        if isinstance(data, Trade):
            return self._update_from_trade(data)
        elif isinstance(data, OHLCV):
            return self._update_from_ohlcv(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _update_from_trade(self, trade: Trade) -> Optional[VolumeBucket]:
        """Process a single trade."""
        # Update price history
        if len(self._prices) > 0:
            ret = np.log(trade.price / self._prices[-1])
            self._returns.append(ret)
        self._prices.append(trade.price)

        # Classify trade direction
        if trade.side != 0:
            buy_vol = trade.volume if trade.side > 0 else 0.0
            sell_vol = trade.volume if trade.side < 0 else 0.0
        elif self.use_bulk_classification:
            buy_vol, sell_vol = self._bulk_classify(trade)
        else:
            # Simple tick rule
            buy_vol, sell_vol = self._tick_classify(trade)

        # Initialize bucket start time
        if self._current_bucket_start_time == 0:
            self._current_bucket_start_time = trade.timestamp

        # Update current bucket
        self._current_bucket_volume += trade.volume
        self._current_bucket_buy += buy_vol
        self._current_bucket_sell += sell_vol
        self._current_bucket_vwap_num += trade.price * trade.volume
        self._current_bucket_trades += 1

        # Check if bucket is complete
        completed_bucket = None
        if self._current_bucket_volume >= self.bucket_size:
            completed_bucket = self._complete_bucket(trade.timestamp)

        return completed_bucket

    def _update_from_ohlcv(self, bar: OHLCV) -> Optional[VolumeBucket]:
        """
        Process OHLCV bar data.

        Uses bulk volume classification (BVC) to estimate buy/sell volumes
        when explicit trade-level classification is not available.
        """
        # Update price history
        if len(self._prices) > 0:
            ret = np.log(bar.close / self._prices[-1])
            self._returns.append(ret)
        self._prices.append(bar.close)

        # Estimate buy/sell volume using BVC or taker data
        if bar.taker_buy_volume > 0:
            # Direct taker buy information available
            buy_vol = bar.taker_buy_volume
            sell_vol = bar.volume - bar.taker_buy_volume
        else:
            # Use BVC approximation
            buy_vol, sell_vol = self._bulk_classify_bar(bar)

        # Initialize bucket start time
        if self._current_bucket_start_time == 0:
            self._current_bucket_start_time = bar.timestamp

        # Update current bucket
        self._current_bucket_volume += bar.volume
        self._current_bucket_buy += buy_vol
        self._current_bucket_sell += sell_vol
        self._current_bucket_vwap_num += bar.close * bar.volume  # Approximate
        self._current_bucket_trades += bar.trades if bar.trades > 0 else 1

        # Check if bucket(s) complete - bar may span multiple buckets
        completed_bucket = None
        while self._current_bucket_volume >= self.bucket_size:
            overflow = self._current_bucket_volume - self.bucket_size

            # Proportionally split overflow
            if self._current_bucket_volume > 0:
                overflow_ratio = overflow / self._current_bucket_volume
                overflow_buy = self._current_bucket_buy * overflow_ratio
                overflow_sell = self._current_bucket_sell * overflow_ratio
            else:
                overflow_buy = overflow_sell = 0

            # Complete current bucket
            self._current_bucket_volume = self.bucket_size
            completed_bucket = self._complete_bucket(bar.timestamp)

            # Start new bucket with overflow
            self._current_bucket_volume = overflow
            self._current_bucket_buy = overflow_buy
            self._current_bucket_sell = overflow_sell

        return completed_bucket

    def _bulk_classify(self, trade: Trade) -> Tuple[float, float]:
        """
        Bulk Volume Classification for single trade.

        Uses the standard normal CDF based on normalized price change.
        """
        if len(self._returns) < 10:
            # Not enough history, assume 50/50 split
            return trade.volume / 2, trade.volume / 2

        sigma = np.std(self._returns) if len(self._returns) > 0 else 1.0
        if sigma == 0:
            sigma = 1e-8

        # Calculate normalized price change
        if len(self._prices) >= 2:
            delta_p = self._prices[-1] - self._prices[-2]
            z = delta_p / (sigma * self._prices[-2])

            # Probability of buy
            p_buy = float(ndtr(z))
        else:
            p_buy = 0.5

        buy_vol = trade.volume * p_buy
        sell_vol = trade.volume * (1 - p_buy)

        return buy_vol, sell_vol

    def _bulk_classify_bar(self, bar: OHLCV) -> Tuple[float, float]:
        """
        Bulk Volume Classification for OHLCV bar.

        Estimates buy/sell split based on close position within bar range.
        """
        if len(self._returns) < 10:
            return bar.volume / 2, bar.volume / 2

        sigma = np.std(self._returns) if len(self._returns) > 0 else 1.0
        if sigma == 0:
            sigma = 1e-8

        # Calculate normalized return for the bar
        if bar.open > 0:
            bar_return = (bar.close - bar.open) / bar.open
            z = bar_return / sigma
            p_buy = float(ndtr(z))
        else:
            # Fallback: use position within high-low range
            if bar.high != bar.low:
                p_buy = (bar.close - bar.low) / (bar.high - bar.low)
            else:
                p_buy = 0.5

        buy_vol = bar.volume * p_buy
        sell_vol = bar.volume * (1 - p_buy)

        return buy_vol, sell_vol

    def _tick_classify(self, trade: Trade) -> Tuple[float, float]:
        """Simple tick rule classification."""
        if len(self._prices) < 2:
            return trade.volume / 2, trade.volume / 2

        if trade.price > self._prices[-2]:
            return trade.volume, 0.0
        elif trade.price < self._prices[-2]:
            return 0.0, trade.volume
        else:
            return trade.volume / 2, trade.volume / 2

    def _complete_bucket(self, end_time: int) -> VolumeBucket:
        """Complete current bucket and reset accumulator."""
        vwap = (
            self._current_bucket_vwap_num / self._current_bucket_volume
            if self._current_bucket_volume > 0 else 0.0
        )

        bucket = VolumeBucket(
            start_time=self._current_bucket_start_time,
            end_time=end_time,
            volume=self._current_bucket_volume,
            buy_volume=self._current_bucket_buy,
            sell_volume=self._current_bucket_sell,
            vwap=vwap,
            trades=self._current_bucket_trades,
        )

        self._buckets.append(bucket)

        # Reset accumulator
        self._current_bucket_volume = 0.0
        self._current_bucket_buy = 0.0
        self._current_bucket_sell = 0.0
        self._current_bucket_vwap_num = 0.0
        self._current_bucket_start_time = end_time
        self._current_bucket_trades = 0

        # Update VPIN history
        current_vpin = self.get_value()
        if current_vpin is not None:
            self._vpin_history.append(current_vpin)

        return bucket

    def get_value(self) -> Optional[float]:
        """
        Calculate current VPIN value.

        VPIN = sum(|V_buy - V_sell|) / (n * V_bucket)

        Returns:
            VPIN value between 0 and 1, or None if insufficient data
        """
        if len(self._buckets) < self.n_buckets:
            return None

        total_imbalance = sum(b.order_imbalance for b in self._buckets)
        vpin = total_imbalance / len(self._buckets)

        return vpin

    def get_vpin_cdf(self, vpin_value: Optional[float] = None) -> Optional[float]:
        """
        Calculate CDF of VPIN for toxicity assessment.

        A CDF > 0.9 indicates high probability of informed trading.

        Args:
            vpin_value: VPIN value to evaluate (default: current VPIN)

        Returns:
            CDF value between 0 and 1
        """
        if vpin_value is None:
            vpin_value = self.get_value()

        if vpin_value is None or len(self._vpin_history) < 50:
            return None

        # Use empirical CDF
        vpin_array = np.array(self._vpin_history)
        cdf = np.mean(vpin_array <= vpin_value)

        return float(cdf)

    def get_vpin_z_score(self) -> Optional[float]:
        """
        Calculate z-score of current VPIN.

        Returns:
            Z-score (standard deviations from mean)
        """
        vpin = self.get_value()
        if vpin is None or len(self._vpin_history) < 50:
            return None

        vpin_array = np.array(self._vpin_history)
        mean = np.mean(vpin_array)
        std = np.std(vpin_array)

        if std == 0:
            return 0.0

        return (vpin - mean) / std

    def get_buckets(self) -> List[VolumeBucket]:
        """Get list of completed buckets."""
        return list(self._buckets)

    def reset(self) -> None:
        """Reset calculator state."""
        self._current_bucket_volume = 0.0
        self._current_bucket_buy = 0.0
        self._current_bucket_sell = 0.0
        self._current_bucket_vwap_num = 0.0
        self._current_bucket_start_time = 0
        self._current_bucket_trades = 0
        self._buckets.clear()
        self._prices.clear()
        self._returns.clear()
        self._vpin_history.clear()


# ==============================================================================
# OFI Calculator
# ==============================================================================

class OFICalculator(StreamingCalculator):
    """
    Order Flow Imbalance (OFI) Calculator.

    OFI measures the net order flow pressure based on changes in bid/ask
    quotes. It captures the imbalance between aggressive buying and selling.

    Reference: Cont, R., Kukanov, A., & Stoikov, S. (2014).
    "The Price Impact of Order Book Events"

    Attributes:
        decay: Exponential decay factor for cumulative OFI
        momentum_window: Window for OFI momentum calculation
    """

    def __init__(
        self,
        decay: float = 0.99,
        momentum_window: int = 20,
        normalize_window: int = 100,
    ) -> None:
        """
        Initialize OFI Calculator.

        Args:
            decay: Decay factor for cumulative OFI (0 < decay <= 1)
            momentum_window: Window for momentum calculation
            normalize_window: Window for z-score normalization
        """
        self.decay = decay
        self.momentum_window = momentum_window
        self.normalize_window = normalize_window

        # Previous quote state
        self._prev_quote: Optional[Quote] = None

        # OFI values
        self._ofi: float = 0.0
        self._cumulative_ofi: float = 0.0

        # History for analysis
        self._ofi_history: Deque[float] = deque(maxlen=normalize_window)
        self._momentum_window_values: Deque[float] = deque(maxlen=momentum_window)

    def update(self, data: Union[Quote, OHLCV]) -> float:
        """
        Update OFI with new quote data.

        Args:
            data: Quote update or OHLCV bar

        Returns:
            Current OFI value
        """
        if isinstance(data, Quote):
            return self._update_from_quote(data)
        elif isinstance(data, OHLCV):
            return self._update_from_ohlcv(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _update_from_quote(self, quote: Quote) -> float:
        """
        Calculate OFI from quote change.

        OFI = dBid_size * I(Bid_up) - dAsk_size * I(Ask_down)
              + Bid_size_new * I(Bid_up) - Ask_size_new * I(Ask_down)
              - Bid_size_old * I(Bid_down) + Ask_size_old * I(Ask_up)
        """
        if self._prev_quote is None:
            self._prev_quote = quote
            return 0.0

        prev = self._prev_quote

        # Calculate OFI components
        ofi = 0.0

        # Bid side changes
        if quote.bid_price > prev.bid_price:
            # Bid price increased - aggressive buy
            ofi += quote.bid_size
        elif quote.bid_price < prev.bid_price:
            # Bid price decreased - bid canceled/filled
            ofi -= prev.bid_size
        else:
            # Same price - size change
            ofi += quote.bid_size - prev.bid_size

        # Ask side changes
        if quote.ask_price < prev.ask_price:
            # Ask price decreased - aggressive sell
            ofi -= quote.ask_size
        elif quote.ask_price > prev.ask_price:
            # Ask price increased - ask canceled/filled
            ofi += prev.ask_size
        else:
            # Same price - size change
            ofi -= quote.ask_size - prev.ask_size

        self._ofi = ofi
        self._cumulative_ofi = self._cumulative_ofi * self.decay + ofi

        # Update history
        self._ofi_history.append(ofi)
        self._momentum_window_values.append(self._cumulative_ofi)

        self._prev_quote = quote

        return ofi

    def _update_from_ohlcv(self, bar: OHLCV) -> float:
        """
        Approximate OFI from OHLCV bar.

        Uses taker buy/sell imbalance as OFI proxy.
        """
        if bar.taker_buy_volume > 0 and bar.volume > 0:
            taker_sell_volume = bar.volume - bar.taker_buy_volume
            ofi = bar.taker_buy_volume - taker_sell_volume
        else:
            # Use price direction as proxy
            price_change = bar.close - bar.open
            ofi = np.sign(price_change) * bar.volume * abs(price_change) / bar.open

        self._ofi = ofi
        self._cumulative_ofi = self._cumulative_ofi * self.decay + ofi

        self._ofi_history.append(ofi)
        self._momentum_window_values.append(self._cumulative_ofi)

        return ofi

    def get_value(self) -> float:
        """Get current instantaneous OFI."""
        return self._ofi

    def get_cumulative_ofi(self) -> float:
        """Get cumulative OFI with decay."""
        return self._cumulative_ofi

    def get_ofi_momentum(self) -> Optional[float]:
        """
        Calculate OFI momentum (rate of change).

        Returns:
            Linear regression slope of cumulative OFI
        """
        if len(self._momentum_window_values) < self.momentum_window:
            return None

        values = np.array(self._momentum_window_values)
        x = np.arange(len(values))

        # Simple linear regression
        x_mean = x.mean()
        y_mean = values.mean()

        numerator = np.sum((x - x_mean) * (values - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    def get_ofi_z_score(self) -> Optional[float]:
        """Get z-score of current OFI."""
        if len(self._ofi_history) < 20:
            return None

        ofi_array = np.array(self._ofi_history)
        mean = np.mean(ofi_array)
        std = np.std(ofi_array)

        if std == 0:
            return 0.0

        return (self._ofi - mean) / std

    def reset(self) -> None:
        """Reset calculator state."""
        self._prev_quote = None
        self._ofi = 0.0
        self._cumulative_ofi = 0.0
        self._ofi_history.clear()
        self._momentum_window_values.clear()


# ==============================================================================
# Spread Decomposition
# ==============================================================================

class SpreadDecomposition(StreamingCalculator):
    """
    Spread Decomposition Calculator.

    Decomposes the bid-ask spread into:
    - Effective spread: Actual cost of trading
    - Realized spread: Market maker's profit after price movement
    - Price impact: Permanent component of spread
    - Kyle's Lambda: Price impact coefficient

    Reference: Huang, R. D., & Stoll, H. R. (1997).
    "The Components of the Bid-Ask Spread"
    """

    def __init__(
        self,
        impact_horizon: int = 5,
        lambda_window: int = 50,
    ) -> None:
        """
        Initialize Spread Decomposition Calculator.

        Args:
            impact_horizon: Number of bars for realized spread calculation
            lambda_window: Window for Kyle's Lambda estimation
        """
        self.impact_horizon = impact_horizon
        self.lambda_window = lambda_window

        # Quote history
        self._quotes: Deque[Quote] = deque(maxlen=impact_horizon + 1)

        # Trade history for price impact
        self._trades: Deque[Trade] = deque(maxlen=lambda_window)
        self._returns_after_trades: Deque[Tuple[float, float]] = deque(
            maxlen=lambda_window
        )

        # Current values
        self._effective_spread: float = 0.0
        self._realized_spread: Optional[float] = None
        self._price_impact: Optional[float] = None

        # Kyle's Lambda history
        self._lambda_history: Deque[float] = deque(maxlen=100)

    def update(self, data: Union[Quote, Trade, OHLCV]) -> None:
        """Update with new data."""
        if isinstance(data, Quote):
            self._update_quote(data)
        elif isinstance(data, Trade):
            self._update_trade(data)
        elif isinstance(data, OHLCV):
            self._update_ohlcv(data)

    def _update_quote(self, quote: Quote) -> None:
        """Update with new quote."""
        self._quotes.append(quote)

        # Calculate effective spread
        if quote.ask_price > 0 and quote.bid_price > 0:
            mid = (quote.bid_price + quote.ask_price) / 2
            self._effective_spread = (quote.ask_price - quote.bid_price) / mid

        # Calculate realized spread if enough history
        if len(self._quotes) > self.impact_horizon:
            self._calculate_realized_spread()

    def _update_trade(self, trade: Trade) -> None:
        """Update with new trade for price impact calculation."""
        self._trades.append(trade)

        # Store signed trade volume for regression
        if len(self._trades) >= 2:
            prev_trade = self._trades[-2]
            price_change = (trade.price - prev_trade.price) / prev_trade.price
            signed_volume = trade.volume * trade.side
            self._returns_after_trades.append((signed_volume, price_change))

    def _update_ohlcv(self, bar: OHLCV) -> None:
        """Update from OHLCV bar."""
        # Create synthetic quote from OHLCV
        spread_proxy = (bar.high - bar.low) / bar.close if bar.close > 0 else 0
        mid = (bar.high + bar.low) / 2

        quote = Quote(
            timestamp=bar.timestamp,
            bid_price=mid - spread_proxy * mid / 2,
            bid_size=bar.volume / 2,
            ask_price=mid + spread_proxy * mid / 2,
            ask_size=bar.volume / 2,
        )
        self._update_quote(quote)

        # Synthetic trade for price impact
        if bar.taker_buy_volume > 0:
            side = 1 if bar.taker_buy_volume > bar.volume / 2 else -1
        else:
            side = 1 if bar.close > bar.open else -1

        trade = Trade(
            timestamp=bar.timestamp,
            price=bar.close,
            volume=bar.volume,
            side=side,
        )
        self._update_trade(trade)

    def _calculate_realized_spread(self) -> None:
        """
        Calculate realized spread.

        Realized spread = 2 * d * (P_trade - M_t+n)
        where d is trade direction, M_t+n is midpoint after n periods.
        """
        if len(self._quotes) <= self.impact_horizon:
            return

        old_quote = self._quotes[0]
        new_quote = self._quotes[-1]

        old_mid = (old_quote.bid_price + old_quote.ask_price) / 2
        new_mid = (new_quote.bid_price + new_quote.ask_price) / 2

        if old_mid == 0:
            return

        # Realized spread as fraction of initial mid
        mid_change = (new_mid - old_mid) / old_mid

        # Price impact is the permanent component
        self._price_impact = abs(mid_change)

        # Realized spread is effective spread minus price impact
        self._realized_spread = max(0, self._effective_spread - 2 * self._price_impact)

    def get_value(self) -> float:
        """Get effective spread."""
        return self._effective_spread

    def get_effective_spread(self, in_bps: bool = True) -> float:
        """
        Get effective spread.

        Args:
            in_bps: Return in basis points (default True)
        """
        spread = self._effective_spread
        return spread * 10000 if in_bps else spread

    def get_realized_spread(self, in_bps: bool = True) -> Optional[float]:
        """
        Get realized spread (market maker profit).

        Args:
            in_bps: Return in basis points (default True)
        """
        if self._realized_spread is None:
            return None
        spread = self._realized_spread
        return spread * 10000 if in_bps else spread

    def get_price_impact(self, in_bps: bool = True) -> Optional[float]:
        """
        Get price impact (adverse selection component).

        Args:
            in_bps: Return in basis points (default True)
        """
        if self._price_impact is None:
            return None
        impact = self._price_impact
        return impact * 10000 if in_bps else impact

    def get_kyle_lambda(self) -> Optional[float]:
        """
        Calculate Kyle's Lambda (market impact coefficient).

        Lambda = Cov(dP, Q) / Var(Q)
        where dP is price change and Q is signed order flow.

        Returns:
            Kyle's Lambda coefficient
        """
        if len(self._returns_after_trades) < 20:
            return None

        data = np.array(list(self._returns_after_trades))
        volumes = data[:, 0]
        returns = data[:, 1]

        # Remove outliers
        vol_std = np.std(volumes)
        if vol_std > 0:
            mask = np.abs(volumes) < 3 * vol_std
            volumes = volumes[mask]
            returns = returns[mask]

        if len(volumes) < 10:
            return None

        # Calculate lambda via regression
        vol_var = np.var(volumes)
        if vol_var == 0:
            return None

        cov = np.cov(volumes, returns)[0, 1]
        kyle_lambda = cov / vol_var

        self._lambda_history.append(kyle_lambda)

        return kyle_lambda

    def get_lambda_trend(self) -> Optional[float]:
        """
        Get trend in Kyle's Lambda.

        Increasing lambda indicates deteriorating liquidity.
        """
        if len(self._lambda_history) < 20:
            return None

        lambdas = np.array(self._lambda_history)
        x = np.arange(len(lambdas))

        # Linear regression
        x_mean = x.mean()
        y_mean = lambdas.mean()

        slope = np.sum((x - x_mean) * (lambdas - y_mean)) / np.sum((x - x_mean) ** 2)

        return float(slope)

    def reset(self) -> None:
        """Reset calculator state."""
        self._quotes.clear()
        self._trades.clear()
        self._returns_after_trades.clear()
        self._effective_spread = 0.0
        self._realized_spread = None
        self._price_impact = None
        self._lambda_history.clear()


# ==============================================================================
# Queue Metrics
# ==============================================================================

@dataclass
class LOBLevel:
    """Single level of limit order book."""
    price: float
    size: float


@dataclass
class LOBSnapshot:
    """Limit Order Book snapshot."""
    timestamp: int
    bids: List[LOBLevel]
    asks: List[LOBLevel]


class QueueMetrics(StreamingCalculator):
    """
    Queue Metrics Calculator for Limit Order Book.

    Calculates:
    - Queue imbalance at best levels
    - Depth-weighted mid price
    - Trade-through rate
    - Book pressure indicators
    """

    def __init__(
        self,
        depth_levels: int = 5,
        trade_through_window: int = 100,
    ) -> None:
        """
        Initialize Queue Metrics Calculator.

        Args:
            depth_levels: Number of book levels to consider
            trade_through_window: Window for trade-through rate
        """
        self.depth_levels = depth_levels
        self.trade_through_window = trade_through_window

        # Current LOB state
        self._current_lob: Optional[LOBSnapshot] = None

        # Trade-through tracking
        self._trades_through_best: Deque[bool] = deque(
            maxlen=trade_through_window
        )

        # Historical metrics
        self._imbalance_history: Deque[float] = deque(maxlen=100)

    def update(self, data: Union[LOBSnapshot, OHLCV]) -> None:
        """Update with new LOB snapshot."""
        if isinstance(data, LOBSnapshot):
            self._current_lob = data
            imbalance = self.get_queue_imbalance()
            if imbalance is not None:
                self._imbalance_history.append(imbalance)
        elif isinstance(data, OHLCV):
            # Create synthetic LOB from OHLCV
            self._update_from_ohlcv(data)

    def _update_from_ohlcv(self, bar: OHLCV) -> None:
        """Create synthetic LOB from OHLCV."""
        mid = (bar.high + bar.low) / 2
        spread = (bar.high - bar.low) / 2 if bar.high > bar.low else mid * 0.001

        # Estimate bid/ask sizes
        if bar.taker_buy_volume > 0:
            bid_size = bar.volume - bar.taker_buy_volume
            ask_size = bar.taker_buy_volume
        else:
            bid_size = bar.volume / 2
            ask_size = bar.volume / 2

        self._current_lob = LOBSnapshot(
            timestamp=bar.timestamp,
            bids=[LOBLevel(mid - spread, bid_size)],
            asks=[LOBLevel(mid + spread, ask_size)],
        )

        imbalance = self.get_queue_imbalance()
        if imbalance is not None:
            self._imbalance_history.append(imbalance)

    def get_value(self) -> Optional[float]:
        """Get queue imbalance."""
        return self.get_queue_imbalance()

    def get_queue_imbalance(self, levels: Optional[int] = None) -> Optional[float]:
        """
        Calculate queue imbalance at best levels.

        Imbalance = (Bid_size - Ask_size) / (Bid_size + Ask_size)

        Args:
            levels: Number of levels to include (default: all)

        Returns:
            Imbalance between -1 (all ask) and +1 (all bid)
        """
        if self._current_lob is None:
            return None

        lob = self._current_lob
        n_levels = levels if levels else min(
            self.depth_levels,
            len(lob.bids),
            len(lob.asks)
        )

        if n_levels == 0:
            return None

        bid_size = sum(lob.bids[i].size for i in range(min(n_levels, len(lob.bids))))
        ask_size = sum(lob.asks[i].size for i in range(min(n_levels, len(lob.asks))))

        total = bid_size + ask_size
        if total == 0:
            return 0.0

        return (bid_size - ask_size) / total

    def get_depth_weighted_mid(self) -> Optional[float]:
        """
        Calculate depth-weighted mid price.

        Weights mid price towards the side with more depth.
        """
        if self._current_lob is None:
            return None

        lob = self._current_lob

        if len(lob.bids) == 0 or len(lob.asks) == 0:
            return None

        best_bid = lob.bids[0]
        best_ask = lob.asks[0]

        total_size = best_bid.size + best_ask.size
        if total_size == 0:
            return (best_bid.price + best_ask.price) / 2

        # Weight towards side with more size
        weighted_mid = (
            best_bid.price * best_ask.size +
            best_ask.price * best_bid.size
        ) / total_size

        return weighted_mid

    def get_microprice(self) -> Optional[float]:
        """
        Calculate microprice (imbalance-adjusted mid).

        Microprice = Bid * (Ask_size / Total) + Ask * (Bid_size / Total)
        """
        return self.get_depth_weighted_mid()

    def record_trade_through(self, trade_price: float) -> None:
        """
        Record whether a trade went through the best bid/ask.

        Args:
            trade_price: Executed trade price
        """
        if self._current_lob is None:
            return

        lob = self._current_lob
        if len(lob.bids) == 0 or len(lob.asks) == 0:
            return

        best_bid = lob.bids[0].price
        best_ask = lob.asks[0].price

        # Trade-through if price is worse than best
        trade_through = trade_price > best_ask or trade_price < best_bid
        self._trades_through_best.append(trade_through)

    def get_trade_through_rate(self) -> Optional[float]:
        """
        Get rate of trades that went through best bid/ask.

        Higher rate indicates market impact / aggressive trading.
        """
        if len(self._trades_through_best) < 10:
            return None

        return sum(self._trades_through_best) / len(self._trades_through_best)

    def get_book_pressure(self) -> Optional[float]:
        """
        Calculate book pressure (asymmetry in depth).

        Positive = more bid pressure (bullish)
        Negative = more ask pressure (bearish)
        """
        if self._current_lob is None:
            return None

        lob = self._current_lob

        # Calculate total depth-weighted pressure
        bid_pressure = 0.0
        for i, level in enumerate(lob.bids[:self.depth_levels]):
            weight = 1.0 / (i + 1)  # Closer levels matter more
            bid_pressure += level.size * weight

        ask_pressure = 0.0
        for i, level in enumerate(lob.asks[:self.depth_levels]):
            weight = 1.0 / (i + 1)
            ask_pressure += level.size * weight

        total = bid_pressure + ask_pressure
        if total == 0:
            return 0.0

        return (bid_pressure - ask_pressure) / total

    def get_imbalance_z_score(self) -> Optional[float]:
        """Get z-score of current imbalance."""
        if len(self._imbalance_history) < 20:
            return None

        current = self.get_queue_imbalance()
        if current is None:
            return None

        arr = np.array(self._imbalance_history)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return 0.0

        return (current - mean) / std

    def reset(self) -> None:
        """Reset calculator state."""
        self._current_lob = None
        self._trades_through_best.clear()
        self._imbalance_history.clear()


# ==============================================================================
# Toxicity Indicators
# ==============================================================================

class ToxicityIndicators(StreamingCalculator):
    """
    Market Toxicity Indicators Calculator.

    Combines multiple metrics to assess informed trading probability:
    - VPIN-based toxicity
    - Adverse selection cost
    - PIN (Probability of Informed Trading)
    - Combined toxicity score
    """

    def __init__(
        self,
        vpin_calculator: Optional[VPINCalculator] = None,
        spread_calc: Optional[SpreadDecomposition] = None,
        pin_window: int = 1000,
    ) -> None:
        """
        Initialize Toxicity Indicators.

        Args:
            vpin_calculator: Existing VPIN calculator to use
            spread_calc: Existing spread decomposition calculator
            pin_window: Window for PIN estimation
        """
        self.vpin_calc = vpin_calculator or VPINCalculator()
        self.spread_calc = spread_calc or SpreadDecomposition()
        self.pin_window = pin_window

        # Trade statistics for PIN
        self._buy_trades: Deque[int] = deque(maxlen=pin_window)
        self._sell_trades: Deque[int] = deque(maxlen=pin_window)

        # Adverse selection tracking
        self._adverse_selection_costs: Deque[float] = deque(maxlen=100)

        # Combined score history
        self._toxicity_scores: Deque[float] = deque(maxlen=100)

    def update(self, data: Union[Trade, Quote, OHLCV]) -> None:
        """Update all toxicity indicators."""
        # Update VPIN
        if isinstance(data, (Trade, OHLCV)):
            self.vpin_calc.update(data)

        # Update spread decomposition
        self.spread_calc.update(data)

        # Update trade counts for PIN
        if isinstance(data, Trade):
            if data.side > 0:
                self._buy_trades.append(1)
                self._sell_trades.append(0)
            elif data.side < 0:
                self._buy_trades.append(0)
                self._sell_trades.append(1)
        elif isinstance(data, OHLCV):
            # Approximate from taker volumes
            if data.taker_buy_volume > 0:
                buy_count = int(data.trades * data.taker_buy_volume / data.volume) if data.volume > 0 else 0
                sell_count = data.trades - buy_count
                for _ in range(buy_count):
                    self._buy_trades.append(1)
                    self._sell_trades.append(0)
                for _ in range(sell_count):
                    self._buy_trades.append(0)
                    self._sell_trades.append(1)

        # Calculate adverse selection cost
        price_impact = self.spread_calc.get_price_impact(in_bps=False)
        if price_impact is not None:
            self._adverse_selection_costs.append(price_impact)

        # Update combined score
        score = self.get_combined_toxicity()
        if score is not None:
            self._toxicity_scores.append(score)

    def get_value(self) -> Optional[float]:
        """Get combined toxicity score."""
        return self.get_combined_toxicity()

    def get_vpin_toxicity(self) -> Optional[float]:
        """
        Get VPIN-based toxicity.

        Returns:
            Toxicity level based on VPIN CDF
        """
        return self.vpin_calc.get_vpin_cdf()

    def get_adverse_selection_cost(self) -> Optional[float]:
        """
        Get average adverse selection cost.

        Returns:
            Mean adverse selection cost in decimal
        """
        if len(self._adverse_selection_costs) < 10:
            return None

        return float(np.mean(self._adverse_selection_costs))

    def estimate_pin(self) -> Optional[float]:
        """
        Estimate PIN (Probability of Informed Trading).

        Simplified estimation based on buy/sell trade imbalance.

        PIN = (alpha * mu) / (alpha * mu + 2 * epsilon)

        where:
        - alpha: Probability of information event
        - mu: Arrival rate of informed traders
        - epsilon: Arrival rate of uninformed traders

        Returns:
            Estimated PIN between 0 and 1
        """
        if len(self._buy_trades) < 100:
            return None

        buys = sum(self._buy_trades)
        sells = sum(self._sell_trades)
        total = buys + sells

        if total == 0:
            return None

        # Simple approximation: PIN related to imbalance
        # Higher imbalance suggests more informed trading
        imbalance = abs(buys - sells) / total

        # Scale to reasonable PIN range (typically 0.1 - 0.3)
        # Using logistic function
        pin = 0.5 * (1 + np.tanh(2 * (imbalance - 0.3)))

        return float(np.clip(pin, 0, 1))

    def get_informed_trader_probability(self) -> Optional[float]:
        """
        Get probability that recent trades are from informed traders.

        Combines VPIN and PIN estimates.
        """
        vpin = self.vpin_calc.get_value()
        pin = self.estimate_pin()

        if vpin is None and pin is None:
            return None

        # Weighted combination
        if vpin is not None and pin is not None:
            return 0.6 * vpin + 0.4 * pin
        elif vpin is not None:
            return vpin
        else:
            return pin

    def get_combined_toxicity(self) -> Optional[float]:
        """
        Get combined toxicity score.

        Aggregates multiple indicators into single score (0-1).
        Higher = more toxic / more informed trading.
        """
        components = []
        weights = []

        # VPIN CDF
        vpin_cdf = self.get_vpin_toxicity()
        if vpin_cdf is not None:
            components.append(vpin_cdf)
            weights.append(0.4)

        # Adverse selection (normalized)
        asc = self.get_adverse_selection_cost()
        if asc is not None:
            # Normalize: 10bps = 0.5, 50bps = ~1.0
            asc_normalized = min(1.0, asc * 10000 / 50)
            components.append(asc_normalized)
            weights.append(0.3)

        # PIN estimate
        pin = self.estimate_pin()
        if pin is not None:
            components.append(pin)
            weights.append(0.3)

        if not components:
            return None

        # Weighted average
        total_weight = sum(weights)
        score = sum(c * w for c, w in zip(components, weights)) / total_weight

        return float(score)

    def get_toxicity_regime(self) -> str:
        """
        Get current toxicity regime.

        Returns:
            'LOW', 'MEDIUM', 'HIGH', or 'EXTREME'
        """
        score = self.get_combined_toxicity()

        if score is None:
            return "UNKNOWN"
        elif score < 0.3:
            return "LOW"
        elif score < 0.5:
            return "MEDIUM"
        elif score < 0.7:
            return "HIGH"
        else:
            return "EXTREME"

    def get_toxicity_z_score(self) -> Optional[float]:
        """Get z-score of current toxicity."""
        if len(self._toxicity_scores) < 20:
            return None

        current = self.get_combined_toxicity()
        if current is None:
            return None

        arr = np.array(self._toxicity_scores)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return 0.0

        return (current - mean) / std

    def reset(self) -> None:
        """Reset all calculators."""
        self.vpin_calc.reset()
        self.spread_calc.reset()
        self._buy_trades.clear()
        self._sell_trades.clear()
        self._adverse_selection_costs.clear()
        self._toxicity_scores.clear()


# ==============================================================================
# Microstructure Engine
# ==============================================================================

class MicrostructureEngine:
    """
    Unified Microstructure Analysis Engine.

    Combines all microstructure calculators into a single interface
    for comprehensive market analysis.
    """

    def __init__(
        self,
        vpin_bucket_size: float = 1000.0,
        vpin_n_buckets: int = 50,
        ofi_decay: float = 0.99,
        spread_impact_horizon: int = 5,
        queue_depth_levels: int = 5,
    ) -> None:
        """
        Initialize Microstructure Engine.

        Args:
            vpin_bucket_size: Volume per VPIN bucket
            vpin_n_buckets: Number of buckets for VPIN
            ofi_decay: Decay factor for cumulative OFI
            spread_impact_horizon: Horizon for realized spread
            queue_depth_levels: LOB depth levels for queue metrics
        """
        self.vpin = VPINCalculator(
            bucket_size=vpin_bucket_size,
            n_buckets=vpin_n_buckets,
        )
        self.ofi = OFICalculator(decay=ofi_decay)
        self.spread = SpreadDecomposition(impact_horizon=spread_impact_horizon)
        self.queue = QueueMetrics(depth_levels=queue_depth_levels)
        self.toxicity = ToxicityIndicators(
            vpin_calculator=self.vpin,
            spread_calc=self.spread,
        )

        # Track processing stats
        self._bars_processed: int = 0
        self._trades_processed: int = 0

    def process_ohlcv(self, bar: OHLCV) -> Dict[str, Optional[float]]:
        """
        Process OHLCV bar and return all metrics.

        Args:
            bar: OHLCV bar data

        Returns:
            Dictionary of all microstructure metrics
        """
        # Update all calculators
        self.vpin.update(bar)
        self.ofi.update(bar)
        self.spread.update(bar)
        self.queue.update(bar)
        self.toxicity.update(bar)

        self._bars_processed += 1

        return self.get_all_metrics()

    def process_trade(self, trade: Trade) -> Dict[str, Optional[float]]:
        """
        Process trade and return all metrics.

        Args:
            trade: Trade data

        Returns:
            Dictionary of all microstructure metrics
        """
        self.vpin.update(trade)
        self.spread.update(trade)
        self.toxicity.update(trade)

        self._trades_processed += 1

        return self.get_all_metrics()

    def process_quote(self, quote: Quote) -> Dict[str, Optional[float]]:
        """
        Process quote update and return all metrics.

        Args:
            quote: Quote data

        Returns:
            Dictionary of all microstructure metrics
        """
        self.ofi.update(quote)
        self.spread.update(quote)

        return self.get_all_metrics()

    def process_lob(self, lob: LOBSnapshot) -> Dict[str, Optional[float]]:
        """
        Process LOB snapshot and return all metrics.

        Args:
            lob: Limit order book snapshot

        Returns:
            Dictionary of all microstructure metrics
        """
        self.queue.update(lob)

        return self.get_all_metrics()

    def get_all_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get all current microstructure metrics.

        Returns:
            Dictionary with all metric values
        """
        return {
            # VPIN metrics
            "vpin": self.vpin.get_value(),
            "vpin_cdf": self.vpin.get_vpin_cdf(),
            "vpin_z_score": self.vpin.get_vpin_z_score(),

            # OFI metrics
            "ofi": self.ofi.get_value(),
            "ofi_cumulative": self.ofi.get_cumulative_ofi(),
            "ofi_momentum": self.ofi.get_ofi_momentum(),
            "ofi_z_score": self.ofi.get_ofi_z_score(),

            # Spread metrics
            "effective_spread_bps": self.spread.get_effective_spread(),
            "realized_spread_bps": self.spread.get_realized_spread(),
            "price_impact_bps": self.spread.get_price_impact(),
            "kyle_lambda": self.spread.get_kyle_lambda(),

            # Queue metrics
            "queue_imbalance": self.queue.get_queue_imbalance(),
            "microprice": self.queue.get_microprice(),
            "book_pressure": self.queue.get_book_pressure(),
            "trade_through_rate": self.queue.get_trade_through_rate(),

            # Toxicity metrics
            "toxicity_score": self.toxicity.get_combined_toxicity(),
            "informed_prob": self.toxicity.get_informed_trader_probability(),
            "adverse_selection": self.toxicity.get_adverse_selection_cost(),
            "pin_estimate": self.toxicity.estimate_pin(),
        }

    def get_feature_vector(self) -> np.ndarray:
        """
        Get feature vector for ML models.

        Returns:
            Numpy array of features (NaN replaced with 0)
        """
        metrics = self.get_all_metrics()
        features = np.array([v if v is not None else 0.0 for v in metrics.values()])
        return features

    def get_feature_names(self) -> List[str]:
        """Get names of features in feature vector."""
        return list(self.get_all_metrics().keys())

    def reset(self) -> None:
        """Reset all calculators."""
        self.vpin.reset()
        self.ofi.reset()
        self.spread.reset()
        self.queue.reset()
        self.toxicity.reset()
        self._bars_processed = 0
        self._trades_processed = 0


# ==============================================================================
# Test / Demo
# ==============================================================================

def generate_sample_data(n_bars: int = 500) -> List[OHLCV]:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    bars = []
    price = 100.0
    timestamp = 1700000000000  # Starting timestamp

    for i in range(n_bars):
        # Random walk with drift
        returns = np.random.normal(0.0001, 0.002)
        price *= (1 + returns)

        # Generate OHLCV
        volatility = 0.005 * price
        high = price + abs(np.random.normal(0, volatility))
        low = price - abs(np.random.normal(0, volatility))
        open_price = np.random.uniform(low, high)
        close = price

        volume = np.random.exponential(1000)

        # Simulate informed trading bursts
        if np.random.random() < 0.1:  # 10% chance of informed activity
            volume *= 3
            # Directional imbalance
            if returns > 0:
                taker_buy = volume * np.random.uniform(0.7, 0.9)
            else:
                taker_buy = volume * np.random.uniform(0.1, 0.3)
        else:
            taker_buy = volume * np.random.uniform(0.4, 0.6)

        trades = int(volume / 10) + 1

        bar = OHLCV(
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            taker_buy_volume=taker_buy,
            trades=trades,
            quote_volume=volume * price,
        )
        bars.append(bar)
        timestamp += 60000  # 1 minute bars

    return bars


def test_microstructure_engine():
    """Test the microstructure engine with sample data."""
    print("=" * 70)
    print("Testing Microstructure Engine")
    print("=" * 70)

    # Initialize engine
    engine = MicrostructureEngine(
        vpin_bucket_size=5000.0,  # Smaller buckets for test data
        vpin_n_buckets=20,
    )

    # Generate sample data
    print("\nGenerating sample data...")
    bars = generate_sample_data(500)
    print(f"Generated {len(bars)} OHLCV bars")

    # Process bars
    print("\nProcessing bars...")
    metrics_history = []

    for i, bar in enumerate(bars):
        metrics = engine.process_ohlcv(bar)
        metrics_history.append(metrics)

        # Print progress every 100 bars
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} bars")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Final Metrics Summary")
    print("=" * 70)

    final_metrics = engine.get_all_metrics()

    for name, value in final_metrics.items():
        if value is not None:
            print(f"  {name:25s}: {value:.6f}")
        else:
            print(f"  {name:25s}: N/A (insufficient data)")

    # Toxicity regime
    regime = engine.toxicity.get_toxicity_regime()
    print(f"\n  Current Toxicity Regime: {regime}")

    # Feature vector
    feature_vector = engine.get_feature_vector()
    print(f"\n  Feature vector shape: {feature_vector.shape}")
    print(f"  Non-zero features: {np.count_nonzero(feature_vector)}")

    # VPIN buckets
    buckets = engine.vpin.get_buckets()
    print(f"\n  VPIN buckets completed: {len(buckets)}")
    if buckets:
        avg_imbalance = np.mean([b.order_imbalance for b in buckets])
        print(f"  Average bucket imbalance: {avg_imbalance:.4f}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

    return engine, metrics_history


if __name__ == "__main__":
    test_microstructure_engine()
