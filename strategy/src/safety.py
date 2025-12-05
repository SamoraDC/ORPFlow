"""Safety mechanisms and catastrophic bug protection.

Provides multiple layers of protection:
- Hard rate limits
- Sanity checks on all inputs
- Kill switch (manual and automatic)
- Anomaly detection
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Callable, Awaitable, Set
from enum import Enum

import structlog

logger = structlog.get_logger()


class SafetyViolation(Enum):
    """Types of safety violations."""
    RATE_LIMIT = "rate_limit"
    INVALID_PRICE = "invalid_price"
    INVALID_QUANTITY = "invalid_quantity"
    INVALID_SYMBOL = "invalid_symbol"
    POSITION_LIMIT = "position_limit"
    KILL_SWITCH = "kill_switch"
    ANOMALY_DETECTED = "anomaly_detected"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class SafetyConfig:
    """Configuration for safety mechanisms."""

    # Hard rate limits (cannot be bypassed)
    max_orders_per_second: int = 5
    max_orders_per_minute: int = 60
    max_orders_per_hour: int = 500

    # Value limits
    min_price: Decimal = Decimal("0.01")
    max_price: Decimal = Decimal("1000000")
    min_quantity: Decimal = Decimal("0.00001")
    max_quantity: Decimal = Decimal("100")  # Max per order
    max_notional_per_order: Decimal = Decimal("100000")

    # Position limits (absolute)
    max_position_value: Decimal = Decimal("50000")
    max_total_exposure: Decimal = Decimal("100000")

    # Anomaly detection
    price_change_threshold_pct: float = 10.0  # Alert if price changes >10%
    volume_spike_threshold: float = 5.0  # Alert if volume > 5x average

    # Circuit breaker
    max_consecutive_losses: int = 10
    loss_threshold_for_pause: Decimal = Decimal("500")

    # Valid symbols
    valid_symbols: Set[str] = field(default_factory=lambda: {"BTCUSDT", "ETHUSDT"})


@dataclass
class SafetyState:
    """Current state of safety systems."""

    kill_switch_active: bool = False
    kill_switch_reason: Optional[str] = None
    kill_switch_activated_at: Optional[datetime] = None

    circuit_breaker_active: bool = False
    circuit_breaker_reason: Optional[str] = None

    consecutive_losses: int = 0
    cumulative_loss: Decimal = Decimal("0")

    violations_today: int = 0
    last_violation_time: Optional[datetime] = None


class RateLimiter:
    """Token bucket rate limiter with multiple time windows."""

    def __init__(
        self,
        per_second: int = 5,
        per_minute: int = 60,
        per_hour: int = 500,
    ):
        """Initialize rate limiter."""
        self.limits = {
            "second": (per_second, 1),
            "minute": (per_minute, 60),
            "hour": (per_hour, 3600),
        }
        self.timestamps: deque = deque(maxlen=per_hour)

    def check(self) -> bool:
        """Check if action is allowed under rate limits.

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()

        # Clean old timestamps
        while self.timestamps and now - self.timestamps[0] > 3600:
            self.timestamps.popleft()

        # Check each window
        for name, (limit, window) in self.limits.items():
            count = sum(1 for ts in self.timestamps if now - ts <= window)
            if count >= limit:
                logger.warning(
                    "rate_limit_exceeded",
                    window=name,
                    limit=limit,
                    current=count,
                )
                return False

        return True

    def record(self) -> None:
        """Record an action."""
        self.timestamps.append(time.time())


class SafetyGuard:
    """Main safety guard with all protection mechanisms."""

    def __init__(self, config: Optional[SafetyConfig] = None):
        """Initialize safety guard."""
        self.config = config or SafetyConfig()
        self.state = SafetyState()
        self.rate_limiter = RateLimiter(
            per_second=self.config.max_orders_per_second,
            per_minute=self.config.max_orders_per_minute,
            per_hour=self.config.max_orders_per_hour,
        )

        # Last known prices for anomaly detection
        self._last_prices: dict[str, Decimal] = {}

        # Callbacks for kill switch
        self._kill_switch_callbacks: list[Callable[[], Awaitable[None]]] = []

        logger.info(
            "safety_guard_initialized",
            max_orders_per_second=self.config.max_orders_per_second,
            max_position_value=str(self.config.max_position_value),
        )

    def register_kill_switch_callback(
        self, callback: Callable[[], Awaitable[None]]
    ) -> None:
        """Register a callback to be called when kill switch is activated."""
        self._kill_switch_callbacks.append(callback)

    async def activate_kill_switch(self, reason: str) -> None:
        """Activate the kill switch, stopping all trading."""
        self.state.kill_switch_active = True
        self.state.kill_switch_reason = reason
        self.state.kill_switch_activated_at = datetime.utcnow()

        logger.critical(
            "KILL_SWITCH_ACTIVATED",
            reason=reason,
        )

        # Call all registered callbacks
        for callback in self._kill_switch_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error("kill_switch_callback_failed", error=str(e))

    def deactivate_kill_switch(self) -> None:
        """Deactivate the kill switch."""
        if self.state.kill_switch_active:
            logger.info(
                "kill_switch_deactivated",
                was_active_for=(
                    datetime.utcnow() - self.state.kill_switch_activated_at
                ).total_seconds() if self.state.kill_switch_activated_at else 0,
            )
        self.state.kill_switch_active = False
        self.state.kill_switch_reason = None
        self.state.kill_switch_activated_at = None

    def is_trading_allowed(self) -> tuple[bool, Optional[SafetyViolation], Optional[str]]:
        """Check if trading is currently allowed.

        Returns:
            Tuple of (allowed, violation_type, reason)
        """
        if self.state.kill_switch_active:
            return False, SafetyViolation.KILL_SWITCH, self.state.kill_switch_reason

        if self.state.circuit_breaker_active:
            return False, SafetyViolation.CIRCUIT_BREAKER, self.state.circuit_breaker_reason

        if not self.rate_limiter.check():
            return False, SafetyViolation.RATE_LIMIT, "Rate limit exceeded"

        return True, None, None

    def validate_order(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        current_position_value: Decimal = Decimal("0"),
    ) -> tuple[bool, Optional[SafetyViolation], Optional[str]]:
        """Validate an order against safety rules.

        Args:
            symbol: Trading symbol
            price: Order price
            quantity: Order quantity
            current_position_value: Current position value in quote currency

        Returns:
            Tuple of (valid, violation_type, reason)
        """
        # Check kill switch first
        allowed, violation, reason = self.is_trading_allowed()
        if not allowed:
            return False, violation, reason

        # Validate symbol
        if symbol not in self.config.valid_symbols:
            self._record_violation(SafetyViolation.INVALID_SYMBOL)
            return False, SafetyViolation.INVALID_SYMBOL, f"Invalid symbol: {symbol}"

        # Validate price
        if price < self.config.min_price:
            self._record_violation(SafetyViolation.INVALID_PRICE)
            return False, SafetyViolation.INVALID_PRICE, f"Price too low: {price}"

        if price > self.config.max_price:
            self._record_violation(SafetyViolation.INVALID_PRICE)
            return False, SafetyViolation.INVALID_PRICE, f"Price too high: {price}"

        # Check for price anomaly
        if symbol in self._last_prices:
            last_price = self._last_prices[symbol]
            if last_price > 0:
                change_pct = abs(float((price - last_price) / last_price * 100))
                if change_pct > self.config.price_change_threshold_pct:
                    logger.warning(
                        "price_anomaly_detected",
                        symbol=symbol,
                        last_price=str(last_price),
                        current_price=str(price),
                        change_pct=change_pct,
                    )
                    self._record_violation(SafetyViolation.ANOMALY_DETECTED)
                    return False, SafetyViolation.ANOMALY_DETECTED, f"Price change too large: {change_pct:.1f}%"

        # Validate quantity
        if quantity < self.config.min_quantity:
            self._record_violation(SafetyViolation.INVALID_QUANTITY)
            return False, SafetyViolation.INVALID_QUANTITY, f"Quantity too small: {quantity}"

        if quantity > self.config.max_quantity:
            self._record_violation(SafetyViolation.INVALID_QUANTITY)
            return False, SafetyViolation.INVALID_QUANTITY, f"Quantity too large: {quantity}"

        # Validate notional
        notional = price * quantity
        if notional > self.config.max_notional_per_order:
            self._record_violation(SafetyViolation.INVALID_QUANTITY)
            return False, SafetyViolation.INVALID_QUANTITY, f"Notional too large: {notional}"

        # Validate position limits
        new_position_value = current_position_value + notional
        if new_position_value > self.config.max_position_value:
            self._record_violation(SafetyViolation.POSITION_LIMIT)
            return False, SafetyViolation.POSITION_LIMIT, f"Position limit exceeded: {new_position_value}"

        return True, None, None

    def record_order(self, symbol: str, price: Decimal) -> None:
        """Record a successful order for rate limiting and price tracking."""
        self.rate_limiter.record()
        self._last_prices[symbol] = price

    def record_trade_result(self, pnl: Decimal) -> None:
        """Record a trade result for circuit breaker logic."""
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.cumulative_loss += abs(pnl)

            # Check circuit breaker conditions
            if self.state.consecutive_losses >= self.config.max_consecutive_losses:
                self._activate_circuit_breaker(
                    f"Max consecutive losses reached: {self.state.consecutive_losses}"
                )
            elif self.state.cumulative_loss >= self.config.loss_threshold_for_pause:
                self._activate_circuit_breaker(
                    f"Cumulative loss threshold reached: {self.state.cumulative_loss}"
                )
        else:
            # Reset on winning trade
            self.state.consecutive_losses = 0
            self.state.cumulative_loss = Decimal("0")

    def _activate_circuit_breaker(self, reason: str) -> None:
        """Activate the circuit breaker."""
        self.state.circuit_breaker_active = True
        self.state.circuit_breaker_reason = reason
        logger.warning("circuit_breaker_activated", reason=reason)

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        self.state.circuit_breaker_active = False
        self.state.circuit_breaker_reason = None
        self.state.consecutive_losses = 0
        self.state.cumulative_loss = Decimal("0")
        logger.info("circuit_breaker_reset")

    def _record_violation(self, violation: SafetyViolation) -> None:
        """Record a safety violation."""
        self.state.violations_today += 1
        self.state.last_violation_time = datetime.utcnow()

        logger.warning(
            "safety_violation",
            type=violation.value,
            violations_today=self.state.violations_today,
        )

        # Auto kill switch after too many violations
        if self.state.violations_today >= 100:
            asyncio.create_task(
                self.activate_kill_switch("Too many safety violations")
            )

    def get_status(self) -> dict:
        """Get current safety status."""
        return {
            "kill_switch": {
                "active": self.state.kill_switch_active,
                "reason": self.state.kill_switch_reason,
                "activated_at": self.state.kill_switch_activated_at.isoformat() if self.state.kill_switch_activated_at else None,
            },
            "circuit_breaker": {
                "active": self.state.circuit_breaker_active,
                "reason": self.state.circuit_breaker_reason,
                "consecutive_losses": self.state.consecutive_losses,
                "cumulative_loss": str(self.state.cumulative_loss),
            },
            "violations_today": self.state.violations_today,
            "trading_allowed": self.is_trading_allowed()[0],
        }


# Sanity check decorators
def sanity_check_price(min_val: float = 0.01, max_val: float = 1000000):
    """Decorator to sanity check price parameters."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            price = kwargs.get("price") or (args[1] if len(args) > 1 else None)
            if price is not None:
                price_float = float(price)
                if price_float < min_val or price_float > max_val:
                    raise ValueError(f"Price {price} outside valid range [{min_val}, {max_val}]")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def sanity_check_quantity(min_val: float = 0.00001, max_val: float = 100):
    """Decorator to sanity check quantity parameters."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            quantity = kwargs.get("quantity") or (args[2] if len(args) > 2 else None)
            if quantity is not None:
                qty_float = float(quantity)
                if qty_float < min_val or qty_float > max_val:
                    raise ValueError(f"Quantity {quantity} outside valid range [{min_val}, {max_val}]")
            return await func(*args, **kwargs)
        return wrapper
    return decorator
