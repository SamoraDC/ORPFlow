"""Graceful shutdown handling.

Ensures clean shutdown on SIGTERM/SIGINT:
- Stops accepting new signals
- Waits for pending trades to complete
- Persists state to checkpoint
- Closes connections cleanly
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional, Callable, Awaitable, List
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class ShutdownState:
    """State of the shutdown process."""
    shutdown_requested: bool = False
    shutdown_started_at: Optional[datetime] = None
    pending_operations: int = 0
    phase: str = "running"  # running, stopping, cleanup, terminated


class GracefulShutdown:
    """Manages graceful shutdown of the application."""

    def __init__(
        self,
        shutdown_timeout_seconds: int = 30,
        drain_timeout_seconds: int = 10,
    ):
        """Initialize shutdown manager.

        Args:
            shutdown_timeout_seconds: Maximum time for full shutdown
            drain_timeout_seconds: Maximum time to wait for pending ops
        """
        self.shutdown_timeout = shutdown_timeout_seconds
        self.drain_timeout = drain_timeout_seconds
        self.state = ShutdownState()

        # Callbacks to run during shutdown
        self._pre_shutdown_callbacks: List[Callable[[], Awaitable[None]]] = []
        self._shutdown_callbacks: List[Callable[[], Awaitable[None]]] = []
        self._cleanup_callbacks: List[Callable[[], Awaitable[None]]] = []

        # Event for waiting on shutdown
        self._shutdown_event = asyncio.Event()

    def register_pre_shutdown(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a callback to run before shutdown starts.

        Use for stopping new work from being accepted.
        """
        self._pre_shutdown_callbacks.append(callback)

    def register_shutdown(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a callback to run during shutdown.

        Use for stopping background tasks.
        """
        self._shutdown_callbacks.append(callback)

    def register_cleanup(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a callback to run during cleanup.

        Use for closing connections and saving state.
        """
        self._cleanup_callbacks.append(callback)

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s)),
            )

        logger.info("signal_handlers_registered")

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        sig_name = sig.name
        logger.info("shutdown_signal_received", signal=sig_name)

        if self.state.shutdown_requested:
            logger.warning("forced_shutdown", reason="second signal received")
            sys.exit(1)

        await self.initiate_shutdown(reason=f"Signal: {sig_name}")

    async def initiate_shutdown(self, reason: str = "requested") -> None:
        """Initiate the graceful shutdown process."""
        if self.state.shutdown_requested:
            return

        self.state.shutdown_requested = True
        self.state.shutdown_started_at = datetime.utcnow()
        self.state.phase = "stopping"

        logger.info(
            "shutdown_initiated",
            reason=reason,
            pending_operations=self.state.pending_operations,
        )

        try:
            # Phase 1: Pre-shutdown (stop accepting new work)
            await self._run_phase("pre_shutdown", self._pre_shutdown_callbacks)

            # Phase 2: Drain pending operations
            await self._drain_pending_operations()

            # Phase 3: Shutdown (stop background tasks)
            self.state.phase = "shutdown"
            await self._run_phase("shutdown", self._shutdown_callbacks)

            # Phase 4: Cleanup (close connections, save state)
            self.state.phase = "cleanup"
            await self._run_phase("cleanup", self._cleanup_callbacks)

            self.state.phase = "terminated"
            logger.info(
                "shutdown_complete",
                duration_seconds=(datetime.utcnow() - self.state.shutdown_started_at).total_seconds(),
            )

        except Exception as e:
            logger.error("shutdown_error", error=str(e))
            raise

        finally:
            self._shutdown_event.set()

    async def _run_phase(
        self,
        phase_name: str,
        callbacks: List[Callable[[], Awaitable[None]]],
    ) -> None:
        """Run callbacks for a shutdown phase."""
        logger.debug(f"shutdown_phase_{phase_name}_start", callbacks=len(callbacks))

        for i, callback in enumerate(callbacks):
            try:
                await asyncio.wait_for(
                    callback(),
                    timeout=self.shutdown_timeout / 3,  # Give each phase 1/3 of total time
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"shutdown_{phase_name}_callback_timeout",
                    callback_index=i,
                )
            except Exception as e:
                logger.error(
                    f"shutdown_{phase_name}_callback_error",
                    callback_index=i,
                    error=str(e),
                )

        logger.debug(f"shutdown_phase_{phase_name}_complete")

    async def _drain_pending_operations(self) -> None:
        """Wait for pending operations to complete."""
        logger.debug(
            "draining_pending_operations",
            count=self.state.pending_operations,
        )

        start = datetime.utcnow()
        while self.state.pending_operations > 0:
            elapsed = (datetime.utcnow() - start).total_seconds()
            if elapsed > self.drain_timeout:
                logger.warning(
                    "drain_timeout_exceeded",
                    remaining_operations=self.state.pending_operations,
                )
                break

            await asyncio.sleep(0.1)

        logger.debug("drain_complete")

    def increment_pending(self) -> None:
        """Increment pending operations counter."""
        self.state.pending_operations += 1

    def decrement_pending(self) -> None:
        """Decrement pending operations counter."""
        self.state.pending_operations = max(0, self.state.pending_operations - 1)

    def is_shutting_down(self) -> bool:
        """Check if shutdown has been requested."""
        return self.state.shutdown_requested

    def should_accept_work(self) -> bool:
        """Check if new work should be accepted."""
        return not self.state.shutdown_requested

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self._shutdown_event.wait()

    def get_status(self) -> dict:
        """Get shutdown status."""
        return {
            "shutdown_requested": self.state.shutdown_requested,
            "phase": self.state.phase,
            "pending_operations": self.state.pending_operations,
            "started_at": self.state.shutdown_started_at.isoformat() if self.state.shutdown_started_at else None,
        }


class ShutdownContext:
    """Context manager for tracking pending operations."""

    def __init__(self, shutdown_manager: GracefulShutdown):
        self.shutdown_manager = shutdown_manager

    async def __aenter__(self):
        if not self.shutdown_manager.should_accept_work():
            raise RuntimeError("System is shutting down, not accepting new work")
        self.shutdown_manager.increment_pending()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.shutdown_manager.decrement_pending()
        return False
