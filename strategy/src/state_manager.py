"""State management with checkpointing and recovery.

Provides periodic checkpointing of system state and recovery on startup.
Ensures no data loss on crashes or deploys.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Any
import asyncio

import structlog

from .models import Account, Position, Side

logger = structlog.get_logger()


def decimal_serializer(obj: Any) -> Any:
    """JSON serializer for Decimal and datetime."""
    if isinstance(obj, Decimal):
        return {"__decimal__": str(obj)}
    if isinstance(obj, datetime):
        return {"__datetime__": obj.isoformat()}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def decimal_deserializer(obj: dict) -> Any:
    """JSON deserializer for Decimal and datetime."""
    if "__decimal__" in obj:
        return Decimal(obj["__decimal__"])
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["__datetime__"])
    return obj


@dataclass
class CircuitBreakerState:
    """State of the circuit breaker."""
    active: bool = False
    reason: Optional[str] = None
    triggered_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None


@dataclass
class WarmupState:
    """State of the warmup period."""
    is_warming_up: bool = True
    started_at: Optional[datetime] = None
    data_points_collected: int = 0
    required_data_points: int = 100  # Minimum data points before trading
    warmup_duration_seconds: int = 300  # 5 minutes minimum warmup


@dataclass
class SystemCheckpoint:
    """Complete system state checkpoint."""

    # Metadata
    checkpoint_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"

    # Account state
    balance: Decimal = Decimal("10000")
    equity: Decimal = Decimal("10000")
    initial_balance: Decimal = Decimal("10000")
    total_pnl: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0

    # Positions (serialized)
    positions: dict = field(default_factory=dict)

    # Risk state
    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)
    daily_trade_count: int = 0
    last_trade_date: Optional[str] = None

    # Strategy state
    imbalance_streaks: dict = field(default_factory=dict)
    last_imbalance_signs: dict = field(default_factory=dict)

    # Warmup state
    warmup: WarmupState = field(default_factory=WarmupState)

    # Metrics
    max_drawdown: Decimal = Decimal("0")
    peak_equity: Decimal = Decimal("10000")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "balance": str(self.balance),
            "equity": str(self.equity),
            "initial_balance": str(self.initial_balance),
            "total_pnl": str(self.total_pnl),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "positions": {
                k: {
                    "symbol": v["symbol"],
                    "quantity": str(v["quantity"]),
                    "entry_price": str(v["entry_price"]),
                    "unrealized_pnl": str(v["unrealized_pnl"]),
                    "realized_pnl": str(v["realized_pnl"]),
                }
                for k, v in self.positions.items()
            },
            "circuit_breaker": {
                "active": self.circuit_breaker.active,
                "reason": self.circuit_breaker.reason,
                "triggered_at": self.circuit_breaker.triggered_at.isoformat() if self.circuit_breaker.triggered_at else None,
                "cooldown_until": self.circuit_breaker.cooldown_until.isoformat() if self.circuit_breaker.cooldown_until else None,
            },
            "daily_trade_count": self.daily_trade_count,
            "last_trade_date": self.last_trade_date,
            "imbalance_streaks": self.imbalance_streaks,
            "last_imbalance_signs": self.last_imbalance_signs,
            "warmup": {
                "is_warming_up": self.warmup.is_warming_up,
                "started_at": self.warmup.started_at.isoformat() if self.warmup.started_at else None,
                "data_points_collected": self.warmup.data_points_collected,
                "required_data_points": self.warmup.required_data_points,
                "warmup_duration_seconds": self.warmup.warmup_duration_seconds,
            },
            "max_drawdown": str(self.max_drawdown),
            "peak_equity": str(self.peak_equity),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SystemCheckpoint":
        """Create from dictionary."""
        checkpoint = cls()
        checkpoint.checkpoint_id = data.get("checkpoint_id", "")
        checkpoint.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow()
        checkpoint.version = data.get("version", "1.0.0")
        checkpoint.balance = Decimal(data.get("balance", "10000"))
        checkpoint.equity = Decimal(data.get("equity", "10000"))
        checkpoint.initial_balance = Decimal(data.get("initial_balance", "10000"))
        checkpoint.total_pnl = Decimal(data.get("total_pnl", "0"))
        checkpoint.total_trades = data.get("total_trades", 0)
        checkpoint.winning_trades = data.get("winning_trades", 0)

        # Positions
        positions = data.get("positions", {})
        checkpoint.positions = {
            k: {
                "symbol": v["symbol"],
                "quantity": Decimal(v["quantity"]),
                "entry_price": Decimal(v["entry_price"]),
                "unrealized_pnl": Decimal(v.get("unrealized_pnl", "0")),
                "realized_pnl": Decimal(v.get("realized_pnl", "0")),
            }
            for k, v in positions.items()
        }

        # Circuit breaker
        cb_data = data.get("circuit_breaker", {})
        checkpoint.circuit_breaker = CircuitBreakerState(
            active=cb_data.get("active", False),
            reason=cb_data.get("reason"),
            triggered_at=datetime.fromisoformat(cb_data["triggered_at"]) if cb_data.get("triggered_at") else None,
            cooldown_until=datetime.fromisoformat(cb_data["cooldown_until"]) if cb_data.get("cooldown_until") else None,
        )

        checkpoint.daily_trade_count = data.get("daily_trade_count", 0)
        checkpoint.last_trade_date = data.get("last_trade_date")
        checkpoint.imbalance_streaks = data.get("imbalance_streaks", {})
        checkpoint.last_imbalance_signs = data.get("last_imbalance_signs", {})

        # Warmup
        warmup_data = data.get("warmup", {})
        checkpoint.warmup = WarmupState(
            is_warming_up=warmup_data.get("is_warming_up", True),
            started_at=datetime.fromisoformat(warmup_data["started_at"]) if warmup_data.get("started_at") else None,
            data_points_collected=warmup_data.get("data_points_collected", 0),
            required_data_points=warmup_data.get("required_data_points", 100),
            warmup_duration_seconds=warmup_data.get("warmup_duration_seconds", 300),
        )

        checkpoint.max_drawdown = Decimal(data.get("max_drawdown", "0"))
        checkpoint.peak_equity = Decimal(data.get("peak_equity", "10000"))

        return checkpoint


class StateManager:
    """Manages system state with checkpointing and recovery."""

    def __init__(
        self,
        checkpoint_dir: str = "/data/checkpoints",
        checkpoint_interval_seconds: int = 60,
        max_checkpoints: int = 10,
    ):
        """Initialize state manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            checkpoint_interval_seconds: How often to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval_seconds
        self.max_checkpoints = max_checkpoints

        self.current_state = SystemCheckpoint()
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._running = False

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start the checkpoint background task."""
        self._running = True
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        logger.info("state_manager_started", interval=self.checkpoint_interval)

    async def stop(self) -> None:
        """Stop the checkpoint task and save final state."""
        self._running = False
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass

        # Save final checkpoint
        await self.save_checkpoint()
        logger.info("state_manager_stopped")

    async def _checkpoint_loop(self) -> None:
        """Background loop to periodically save checkpoints."""
        while self._running:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                await self.save_checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("checkpoint_failed", error=str(e))

    async def save_checkpoint(self) -> None:
        """Save current state to disk."""
        self.current_state.checkpoint_id = f"ckpt_{int(time.time() * 1000)}"
        self.current_state.created_at = datetime.utcnow()

        filename = f"{self.current_state.checkpoint_id}.json"
        filepath = self.checkpoint_dir / filename

        try:
            data = self.current_state.to_dict()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("checkpoint_saved", path=str(filepath))

            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()

        except Exception as e:
            logger.error("checkpoint_save_failed", error=str(e))
            raise

    async def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_checkpoint in checkpoints[self.max_checkpoints:]:
            try:
                old_checkpoint.unlink()
                logger.debug("old_checkpoint_removed", path=str(old_checkpoint))
            except Exception as e:
                logger.warning("checkpoint_cleanup_failed", path=str(old_checkpoint), error=str(e))

    def load_latest_checkpoint(self) -> Optional[SystemCheckpoint]:
        """Load the most recent valid checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for checkpoint_path in checkpoints:
            try:
                with open(checkpoint_path) as f:
                    data = json.load(f)

                checkpoint = SystemCheckpoint.from_dict(data)
                logger.info(
                    "checkpoint_loaded",
                    path=str(checkpoint_path),
                    created_at=checkpoint.created_at.isoformat(),
                    trades=checkpoint.total_trades,
                )
                return checkpoint

            except Exception as e:
                logger.warning(
                    "checkpoint_load_failed",
                    path=str(checkpoint_path),
                    error=str(e),
                )
                continue

        logger.info("no_valid_checkpoint_found")
        return None

    def recover_state(self) -> bool:
        """Recover state from the latest checkpoint.

        Returns:
            True if state was recovered, False if starting fresh
        """
        checkpoint = self.load_latest_checkpoint()

        if checkpoint:
            self.current_state = checkpoint

            # Reset warmup state for safety
            self.current_state.warmup.is_warming_up = True
            self.current_state.warmup.started_at = datetime.utcnow()
            self.current_state.warmup.data_points_collected = 0

            # Check if daily trade count should reset
            today = datetime.utcnow().date().isoformat()
            if self.current_state.last_trade_date != today:
                self.current_state.daily_trade_count = 0
                self.current_state.last_trade_date = today

            logger.info(
                "state_recovered",
                balance=str(self.current_state.balance),
                positions=len(self.current_state.positions),
                total_trades=self.current_state.total_trades,
            )
            return True

        # Start fresh
        self.current_state = SystemCheckpoint()
        self.current_state.warmup.started_at = datetime.utcnow()
        logger.info("starting_fresh_state")
        return False

    def update_warmup(self, data_points: int = 1) -> bool:
        """Update warmup state and check if ready to trade.

        Args:
            data_points: Number of new data points received

        Returns:
            True if system is ready to trade
        """
        if not self.current_state.warmup.is_warming_up:
            return True

        self.current_state.warmup.data_points_collected += data_points

        # Check if warmup is complete
        has_enough_data = (
            self.current_state.warmup.data_points_collected >=
            self.current_state.warmup.required_data_points
        )

        time_elapsed = datetime.utcnow() - (self.current_state.warmup.started_at or datetime.utcnow())
        has_enough_time = time_elapsed.total_seconds() >= self.current_state.warmup.warmup_duration_seconds

        if has_enough_data and has_enough_time:
            self.current_state.warmup.is_warming_up = False
            logger.info(
                "warmup_complete",
                data_points=self.current_state.warmup.data_points_collected,
                duration_seconds=time_elapsed.total_seconds(),
            )
            return True

        return False

    def is_ready_to_trade(self) -> bool:
        """Check if system is ready to trade."""
        return not self.current_state.warmup.is_warming_up

    def get_warmup_progress(self) -> dict:
        """Get warmup progress information."""
        warmup = self.current_state.warmup
        data_progress = warmup.data_points_collected / warmup.required_data_points

        if warmup.started_at:
            time_elapsed = (datetime.utcnow() - warmup.started_at).total_seconds()
            time_progress = time_elapsed / warmup.warmup_duration_seconds
        else:
            time_progress = 0

        return {
            "is_warming_up": warmup.is_warming_up,
            "data_progress": min(data_progress, 1.0),
            "time_progress": min(time_progress, 1.0),
            "overall_progress": min(min(data_progress, time_progress), 1.0),
            "data_points": warmup.data_points_collected,
            "required_points": warmup.required_data_points,
        }
