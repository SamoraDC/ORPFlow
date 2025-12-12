"""Realistic execution simulation.

Models real-world execution factors:
- Fill probability based on queue position
- Market impact for large orders
- Variable latency based on market conditions
- Partial fills across multiple price levels
"""

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Tuple
from datetime import datetime

import numpy as np
import structlog

from ..models import OrderBookState, PriceLevel, Side

logger = structlog.get_logger()


@dataclass
class ExecutionResult:
    """Result of a simulated execution."""

    filled: bool
    fill_price: Optional[Decimal]
    fill_quantity: Decimal
    slippage_bps: Decimal
    latency_ms: float
    partial_fills: List[Tuple[Decimal, Decimal]]  # (price, quantity) pairs
    market_impact_bps: Decimal
    queue_position_factor: float
    execution_time: datetime


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation."""

    # Base latency parameters (milliseconds)
    base_latency_ms: float = 50.0
    latency_std_ms: float = 20.0
    high_vol_latency_multiplier: float = 3.0

    # Fill probability parameters
    base_fill_probability: float = 0.95
    queue_position_decay: float = 0.1  # Probability decreases with queue position

    # Market impact parameters
    impact_coefficient: float = 0.1  # bps per % of level volume
    permanent_impact_ratio: float = 0.5  # How much impact is permanent

    # Partial fill parameters
    min_fill_ratio: float = 0.5  # Minimum fill ratio for market orders
    level_fill_probability: float = 0.8  # Probability of filling each level


class ExecutionSimulator:
    """Simulates realistic order execution."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize execution simulator."""
        self.config = config or ExecutionConfig()
        self._rng = np.random.default_rng()

    def simulate_market_order(
        self,
        side: Side,
        quantity: Decimal,
        orderbook: OrderBookState,
        volatility: Optional[float] = None,
    ) -> ExecutionResult:
        """Simulate a market order execution.

        Args:
            side: Order side (buy/sell)
            quantity: Order quantity
            orderbook: Current order book state
            volatility: Current market volatility (for latency adjustment)

        Returns:
            ExecutionResult with fill details
        """
        start_time = datetime.utcnow()

        # Calculate latency
        latency = self._calculate_latency(volatility)

        # Get relevant side of the book
        levels = orderbook.asks if side == Side.BUY else orderbook.bids

        if not levels:
            return ExecutionResult(
                filled=False,
                fill_price=None,
                fill_quantity=Decimal("0"),
                slippage_bps=Decimal("0"),
                latency_ms=latency,
                partial_fills=[],
                market_impact_bps=Decimal("0"),
                queue_position_factor=0.0,
                execution_time=start_time,
            )

        # Simulate walking through the book
        partial_fills, total_filled, weighted_price = self._walk_book(
            levels, quantity, side
        )

        if total_filled == Decimal("0"):
            return ExecutionResult(
                filled=False,
                fill_price=None,
                fill_quantity=Decimal("0"),
                slippage_bps=Decimal("0"),
                latency_ms=latency,
                partial_fills=[],
                market_impact_bps=Decimal("0"),
                queue_position_factor=0.0,
                execution_time=start_time,
            )

        # Calculate effective fill price
        fill_price = weighted_price / total_filled if total_filled > 0 else Decimal("0")

        # Calculate slippage from mid price
        mid_price = orderbook.mid_price or levels[0].price
        if mid_price > 0:
            if side == Side.BUY:
                slippage_bps = (fill_price - mid_price) / mid_price * Decimal("10000")
            else:
                slippage_bps = (mid_price - fill_price) / mid_price * Decimal("10000")
        else:
            slippage_bps = Decimal("0")

        # Calculate market impact
        market_impact = self._calculate_market_impact(
            total_filled, levels, side
        )

        logger.debug(
            "execution_simulated",
            side=side.value,
            requested_qty=str(quantity),
            filled_qty=str(total_filled),
            fill_price=str(fill_price),
            slippage_bps=str(slippage_bps),
            market_impact_bps=str(market_impact),
            latency_ms=latency,
            num_levels=len(partial_fills),
        )

        return ExecutionResult(
            filled=total_filled > 0,
            fill_price=fill_price,
            fill_quantity=total_filled,
            slippage_bps=slippage_bps,
            latency_ms=latency,
            partial_fills=partial_fills,
            market_impact_bps=market_impact,
            queue_position_factor=1.0,  # Market orders have priority
            execution_time=start_time,
        )

    def simulate_limit_order(
        self,
        side: Side,
        quantity: Decimal,
        limit_price: Decimal,
        orderbook: OrderBookState,
        time_in_queue_ms: float = 0,
    ) -> ExecutionResult:
        """Simulate a limit order execution.

        Args:
            side: Order side
            quantity: Order quantity
            limit_price: Limit price
            orderbook: Current order book state
            time_in_queue_ms: How long the order has been in queue

        Returns:
            ExecutionResult with fill details
        """
        start_time = datetime.utcnow()

        # Check if order would execute immediately (crosses the spread)
        best_ask = orderbook.asks[0].price if orderbook.asks else None
        best_bid = orderbook.bids[0].price if orderbook.bids else None

        immediate_fill = False
        if side == Side.BUY and best_ask and limit_price >= best_ask:
            immediate_fill = True
        elif side == Side.SELL and best_bid and limit_price <= best_bid:
            immediate_fill = True

        if immediate_fill:
            # Execute as market order at the crossing price
            return self.simulate_market_order(side, quantity, orderbook)

        # Limit order sits in the book - calculate fill probability
        queue_position = self._estimate_queue_position(
            side, limit_price, orderbook, time_in_queue_ms
        )

        fill_probability = self._calculate_fill_probability(
            side, limit_price, orderbook, queue_position
        )

        # Random fill decision
        if self._rng.random() < fill_probability:
            # Partial fill based on queue position
            fill_ratio = min(1.0, (1 - queue_position) + 0.3)
            filled_qty = Decimal(str(float(quantity) * fill_ratio))

            return ExecutionResult(
                filled=True,
                fill_price=limit_price,
                fill_quantity=filled_qty,
                slippage_bps=Decimal("0"),  # No slippage for limit orders
                latency_ms=self._calculate_latency(None),
                partial_fills=[(limit_price, filled_qty)],
                market_impact_bps=Decimal("0"),  # Limit orders don't cause impact
                queue_position_factor=queue_position,
                execution_time=start_time,
            )

        return ExecutionResult(
            filled=False,
            fill_price=None,
            fill_quantity=Decimal("0"),
            slippage_bps=Decimal("0"),
            latency_ms=self._calculate_latency(None),
            partial_fills=[],
            market_impact_bps=Decimal("0"),
            queue_position_factor=queue_position,
            execution_time=start_time,
        )

    def _walk_book(
        self,
        levels: List[PriceLevel],
        quantity: Decimal,
        side: Side,
    ) -> Tuple[List[Tuple[Decimal, Decimal]], Decimal, Decimal]:
        """Walk through order book levels to fill an order.

        Returns:
            Tuple of (partial_fills, total_filled, weighted_price_sum)
        """
        partial_fills = []
        remaining = quantity
        weighted_price_sum = Decimal("0")
        total_filled = Decimal("0")

        for level in levels:
            if remaining <= 0:
                break

            # Probability of getting filled at this level
            if self._rng.random() > self.config.level_fill_probability:
                continue

            # How much can we fill at this level
            available = level.quantity

            # Random fill amount (simulate queue position effects)
            fill_ratio = self._rng.uniform(0.5, 1.0)
            available_to_fill = Decimal(str(float(available) * fill_ratio))

            fill_qty = min(remaining, available_to_fill)

            if fill_qty > 0:
                partial_fills.append((level.price, fill_qty))
                weighted_price_sum += level.price * fill_qty
                total_filled += fill_qty
                remaining -= fill_qty

        return partial_fills, total_filled, weighted_price_sum

    def _calculate_latency(self, volatility: Optional[float]) -> float:
        """Calculate execution latency based on market conditions."""
        base = self.config.base_latency_ms
        std = self.config.latency_std_ms

        # Base latency with random variation
        latency = max(10, self._rng.normal(base, std))

        # Increase latency during high volatility
        if volatility and volatility > 0.5:  # High volatility threshold
            latency *= self.config.high_vol_latency_multiplier

        return latency

    def _calculate_market_impact(
        self,
        quantity: Decimal,
        levels: List[PriceLevel],
        side: Side,
    ) -> Decimal:
        """Calculate market impact of the order in basis points."""
        if not levels:
            return Decimal("0")

        # Total available volume in visible levels
        total_volume = sum(level.quantity for level in levels)

        if total_volume == 0:
            return Decimal("0")

        # Order as percentage of visible volume
        order_pct = float(quantity / total_volume) * 100

        # Impact in bps
        impact = Decimal(str(order_pct * self.config.impact_coefficient))

        return impact

    def _estimate_queue_position(
        self,
        side: Side,
        price: Decimal,
        orderbook: OrderBookState,
        time_in_queue_ms: float,
    ) -> float:
        """Estimate position in the queue (0 = front, 1 = back)."""
        # Get relevant levels
        if side == Side.BUY:
            levels = orderbook.bids
            # For buy orders, being at a higher price is better
            price_levels_ahead = sum(1 for level in levels if level.price > price)
        else:
            levels = orderbook.asks
            # For sell orders, being at a lower price is better
            price_levels_ahead = sum(1 for level in levels if level.price < price)

        if not levels:
            return 0.5

        # Position based on price level
        price_position = price_levels_ahead / len(levels)

        # Time improves queue position
        time_improvement = min(0.5, time_in_queue_ms / 60000)  # Max 0.5 improvement after 1 min

        return max(0, price_position - time_improvement)

    def _calculate_fill_probability(
        self,
        side: Side,
        price: Decimal,
        orderbook: OrderBookState,
        queue_position: float,
    ) -> float:
        """Calculate probability of limit order being filled."""
        base_prob = self.config.base_fill_probability

        # Reduce probability based on queue position
        position_penalty = queue_position * self.config.queue_position_decay

        # Price distance from best affects probability
        if side == Side.BUY:
            best = orderbook.bids[0].price if orderbook.bids else price
            price_distance = float((best - price) / best) if best > 0 else 0
        else:
            best = orderbook.asks[0].price if orderbook.asks else price
            price_distance = float((price - best) / best) if best > 0 else 0

        # Further from best = lower probability
        distance_penalty = min(0.5, price_distance * 10)

        probability = max(0.1, base_prob - position_penalty - distance_penalty)

        return probability


class LatencyModel:
    """Models network and processing latency."""

    def __init__(
        self,
        base_latency_ms: float = 50,
        jitter_ms: float = 20,
        spike_probability: float = 0.01,
        spike_multiplier: float = 10,
    ):
        """Initialize latency model."""
        self.base_latency = base_latency_ms
        self.jitter = jitter_ms
        self.spike_prob = spike_probability
        self.spike_mult = spike_multiplier
        self._rng = np.random.default_rng()

    def sample(self, volatility_factor: float = 1.0) -> float:
        """Sample a latency value.

        Args:
            volatility_factor: Multiplier for high volatility periods

        Returns:
            Latency in milliseconds
        """
        # Base latency with normal jitter
        latency = self.base_latency + self._rng.normal(0, self.jitter)

        # Apply volatility factor
        latency *= volatility_factor

        # Random spikes (simulating network congestion)
        if self._rng.random() < self.spike_prob:
            latency *= self.spike_mult

        return max(10, latency)  # Minimum 10ms
