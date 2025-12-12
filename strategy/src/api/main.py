"""FastAPI application for the strategy engine.

Provides REST API for health checks, account status, and manual controls.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from collections.abc import AsyncIterator
from typing import Any, Optional

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..broker import PaperBroker
from ..config import get_settings
from ..features import MicrostructureFeatures
from ..models import Account, HealthStatus, OrderType, Position, Side, Trade
from ..scheduler import ShabbatScheduler
from ..signals import ImbalanceStrategy
from ..storage import Database

logger = structlog.get_logger()

# Global instances (initialized in lifespan)
db: Optional[Database] = None
broker: Optional[PaperBroker] = None
features: Optional[MicrostructureFeatures] = None
strategy: Optional[ImbalanceStrategy] = None
scheduler: Optional[ShabbatScheduler] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global db, broker, features, strategy, scheduler

    settings = get_settings()

    # Initialize database
    db_path = settings.database_url.replace("sqlite:///", "")
    db = Database(db_path)
    await db.connect()

    # Initialize broker
    broker = PaperBroker(db)
    await broker.initialize()

    # Initialize features and strategy
    features = MicrostructureFeatures()
    strategy = ImbalanceStrategy()

    # Initialize scheduler
    scheduler = ShabbatScheduler(
        latitude=settings.shabbat_latitude,
        longitude=settings.shabbat_longitude,
        timezone=settings.timezone,
    )

    logger.info("application_started", symbols=settings.symbols)

    yield

    # Cleanup
    if broker:
        await broker.close()
    if db:
        await db.close()

    logger.info("application_stopped")


app = FastAPI(
    title="QuantumFlow Strategy Engine",
    description="HFT Paper Trading Strategy Engine API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Health & Status Endpoints
# ===========================================================================


@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Health check endpoint."""
    is_paused = scheduler.is_shabbat() if scheduler else False

    return HealthStatus(
        status="healthy" if not is_paused else "paused",
        component="strategy-engine",
        timestamp=datetime.utcnow(),
        details={
            "shabbat_pause": is_paused,
            "next_resume": scheduler.next_resume_time().isoformat() if scheduler and is_paused else None,
        },
    )


@app.get("/status")
async def get_status() -> dict[str, Any]:
    """Get comprehensive system status."""
    if not broker:
        raise HTTPException(status_code=503, detail="Service not initialized")

    account = broker.get_account()
    is_paused = scheduler.is_shabbat() if scheduler else False

    return {
        "status": "paused" if is_paused else "active",
        "account": {
            "balance": str(account.balance),
            "equity": str(account.equity),
            "total_pnl": str(account.total_pnl),
            "pnl_pct": float((account.equity - account.initial_balance) / account.initial_balance * 100),
            "total_trades": account.total_trades,
            "win_rate": account.win_rate,
        },
        "positions": [
            {
                "symbol": p.symbol,
                "quantity": str(p.quantity),
                "entry_price": str(p.entry_price),
                "unrealized_pnl": str(p.unrealized_pnl),
                "realized_pnl": str(p.realized_pnl),
            }
            for p in account.positions
        ],
        "scheduler": {
            "shabbat_pause": is_paused,
            "next_event": scheduler.next_event().isoformat() if scheduler else None,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# ===========================================================================
# Account Endpoints
# ===========================================================================


@app.get("/account", response_model=Account)
async def get_account() -> Account:
    """Get current account state."""
    if not broker:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return broker.get_account()


@app.get("/positions")
async def get_positions() -> list[Position]:
    """Get all open positions."""
    if not broker:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return list(broker.positions.values())


@app.get("/positions/{symbol}")
async def get_position(symbol: str) -> Position:
    """Get position for a specific symbol."""
    if not broker:
        raise HTTPException(status_code=503, detail="Service not initialized")
    position = broker.get_position(symbol.upper())
    if not position:
        raise HTTPException(status_code=404, detail=f"No position for {symbol}")
    return position


# ===========================================================================
# Trade Endpoints
# ===========================================================================


@app.get("/trades", response_model=list[Trade])
async def get_trades(
    symbol: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[Trade]:
    """Get trade history."""
    if not db:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return await db.get_trades(symbol=symbol.upper() if symbol else None, limit=limit)


class ManualOrderRequest(BaseModel):
    """Request for manual order placement."""

    symbol: str
    side: Side
    quantity: float
    price: Optional[float] = None


@app.post("/orders/manual")
async def place_manual_order(request: ManualOrderRequest) -> dict[str, Any]:
    """Place a manual order (for testing/emergency)."""
    if not broker:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if scheduler and scheduler.is_shabbat():
        raise HTTPException(status_code=403, detail="Trading paused for Shabbat")

    order = await broker.submit_order(
        symbol=request.symbol.upper(),
        side=request.side,
        quantity=Decimal(str(request.quantity)),
        price=Decimal(str(request.price)) if request.price else None,
        order_type=OrderType.MARKET if request.price is None else OrderType.LIMIT,
    )

    if not order:
        raise HTTPException(status_code=400, detail="Order rejected")

    # Execute immediately for market orders
    if order.order_type == OrderType.MARKET:
        # Would need current price from market data
        # For now, use a placeholder
        trade = await broker.execute_market_order(
            order=order,
            current_price=Decimal(str(request.price or 50000)),  # Placeholder
        )
        return {
            "order_id": order.id,
            "trade_id": trade.id if trade else None,
            "status": "filled" if trade else "pending",
        }

    return {"order_id": order.id, "status": "pending"}


# ===========================================================================
# Strategy Control Endpoints
# ===========================================================================


@app.post("/strategy/pause")
async def pause_strategy() -> dict[str, str]:
    """Manually pause the strategy."""
    # In production, this would set a flag that the main loop checks
    logger.info("strategy_paused_manually")
    return {"status": "paused"}


@app.post("/strategy/resume")
async def resume_strategy() -> dict[str, str]:
    """Resume the strategy."""
    logger.info("strategy_resumed_manually")
    return {"status": "resumed"}


@app.post("/strategy/reset")
async def reset_strategy() -> dict[str, str]:
    """Reset strategy state (clear persistence tracking)."""
    if strategy:
        strategy.reset()
    if features:
        features.reset()
    logger.info("strategy_reset")
    return {"status": "reset"}


# ===========================================================================
# Circuit Breaker Endpoints
# ===========================================================================


class CircuitBreakerRequest(BaseModel):
    """Request to control circuit breaker."""

    active: bool
    reason: Optional[str] = None


@app.post("/circuit-breaker")
async def set_circuit_breaker(request: CircuitBreakerRequest) -> dict[str, Any]:
    """Control the circuit breaker."""
    import httpx

    settings = get_settings()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.core_api_url}/circuit-breaker",
                json={"active": request.active, "reason": request.reason},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error("circuit_breaker_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# ===========================================================================
# Metrics Endpoints
# ===========================================================================


@app.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Get performance metrics."""
    if not db or not broker:
        raise HTTPException(status_code=503, detail="Service not initialized")

    trades = await db.get_trades(limit=1000)
    account = broker.get_account()

    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_pnl": "0",
            "sharpe_ratio": None,
            "max_drawdown": "0",
        }

    winning = sum(1 for t in trades if t.pnl and t.pnl > 0)
    total_pnl = sum(float(t.pnl or 0) for t in trades)

    # Calculate simple Sharpe (would need daily returns for proper calculation)
    returns = [float(t.pnl or 0) for t in trades]
    if returns:
        import numpy as np
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe = 0

    return {
        "total_trades": len(trades),
        "winning_trades": winning,
        "losing_trades": len(trades) - winning,
        "win_rate": winning / len(trades) if trades else 0,
        "total_pnl": str(account.total_pnl),
        "pnl_pct": float((account.equity - account.initial_balance) / account.initial_balance * 100),
        "sharpe_ratio": sharpe,
        "avg_trade_pnl": str(Decimal(str(total_pnl / len(trades)))) if trades else "0",
        "largest_win": str(max((t.pnl for t in trades if t.pnl), default=Decimal("0"))),
        "largest_loss": str(min((t.pnl for t in trades if t.pnl), default=Decimal("0"))),
    }
