"""Replay logging for debugging and analysis.

Records all system events in a structured format that allows:
- Deterministic replay of historical sessions
- Root cause analysis of issues
- Backtesting validation
"""

import gzip
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator
import asyncio

import structlog

logger = structlog.get_logger()


class EventType(Enum):
    """Types of events that can be recorded."""

    # Market data events
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_TICK = "trade_tick"
    FUNDING_UPDATE = "funding_update"

    # Strategy events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    WARMUP_PROGRESS = "warmup_progress"

    # Order events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"

    # Risk events
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CHECKPOINT_SAVED = "checkpoint_saved"
    ERROR = "error"


@dataclass
class ReplayEvent:
    """A single event in the replay log."""

    timestamp: int  # Unix timestamp in microseconds
    event_type: str
    data: Dict[str, Any]

    # Optional metadata
    sequence_number: int = 0
    session_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ts": self.timestamp,
            "type": self.event_type,
            "seq": self.sequence_number,
            "sid": self.session_id,
            "data": self._serialize_data(self.data),
        }

    @staticmethod
    def _serialize_data(data: dict) -> dict:
        """Serialize data, handling Decimal and datetime."""
        result = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                result[key] = {"__decimal__": str(value)}
            elif isinstance(value, datetime):
                result[key] = {"__datetime__": value.isoformat()}
            elif isinstance(value, dict):
                result[key] = ReplayEvent._serialize_data(value)
            elif isinstance(value, list):
                result[key] = [
                    ReplayEvent._serialize_data(v) if isinstance(v, dict)
                    else {"__decimal__": str(v)} if isinstance(v, Decimal)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ReplayEvent":
        """Create from dictionary."""
        return cls(
            timestamp=data["ts"],
            event_type=data["type"],
            sequence_number=data.get("seq", 0),
            session_id=data.get("sid", ""),
            data=cls._deserialize_data(data["data"]),
        )

    @staticmethod
    def _deserialize_data(data: dict) -> dict:
        """Deserialize data, restoring Decimal and datetime."""
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                if "__decimal__" in value:
                    result[key] = Decimal(value["__decimal__"])
                elif "__datetime__" in value:
                    result[key] = datetime.fromisoformat(value["__datetime__"])
                else:
                    result[key] = ReplayEvent._deserialize_data(value)
            elif isinstance(value, list):
                result[key] = [
                    ReplayEvent._deserialize_data(v) if isinstance(v, dict)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result


class ReplayLogger:
    """Records events for replay and debugging."""

    def __init__(
        self,
        log_dir: str = "/data/replay",
        max_file_size_mb: int = 100,
        compress: bool = True,
        buffer_size: int = 1000,
        flush_interval_seconds: int = 5,
    ):
        """Initialize replay logger.

        Args:
            log_dir: Directory to store replay logs
            max_file_size_mb: Max size before rotating
            compress: Whether to gzip log files
            buffer_size: Events to buffer before writing
            flush_interval_seconds: How often to flush buffer
        """
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.compress = compress
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds

        self.session_id = f"session_{int(time.time() * 1000)}"
        self._sequence = 0
        self._buffer: List[ReplayEvent] = []
        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Ensure directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start the replay logger."""
        self._running = True
        self._rotate_file()
        self._flush_task = asyncio.create_task(self._flush_loop())

        # Record session start
        await self.record(
            EventType.SYSTEM_START,
            {
                "session_id": self.session_id,
                "start_time": datetime.utcnow().isoformat(),
            },
        )

        logger.info(
            "replay_logger_started",
            session_id=self.session_id,
            log_dir=str(self.log_dir),
        )

    async def stop(self) -> None:
        """Stop the replay logger."""
        # Record session end
        await self.record(
            EventType.SYSTEM_STOP,
            {"end_time": datetime.utcnow().isoformat()},
        )

        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()

        if self._file_handle:
            self._file_handle.close()

        logger.info("replay_logger_stopped")

    async def record(
        self,
        event_type: EventType,
        data: Dict[str, Any],
    ) -> None:
        """Record an event.

        Args:
            event_type: Type of event
            data: Event data
        """
        self._sequence += 1

        event = ReplayEvent(
            timestamp=int(time.time() * 1_000_000),  # Microseconds
            event_type=event_type.value,
            data=data,
            sequence_number=self._sequence,
            session_id=self.session_id,
        )

        async with self._lock:
            self._buffer.append(event)

            if len(self._buffer) >= self.buffer_size:
                await self._flush()

    async def _flush_loop(self) -> None:
        """Background loop to periodically flush buffer."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("replay_flush_error", error=str(e))

    async def _flush(self) -> None:
        """Flush buffer to disk."""
        async with self._lock:
            if not self._buffer:
                return

            events_to_write = self._buffer
            self._buffer = []

        try:
            # Check file size and rotate if needed
            if self._current_file and self._current_file.exists():
                if self._current_file.stat().st_size > self.max_file_size:
                    self._rotate_file()

            # Write events
            for event in events_to_write:
                line = json.dumps(event.to_dict()) + "\n"
                self._file_handle.write(line)

            self._file_handle.flush()

        except Exception as e:
            logger.error("replay_write_error", error=str(e))
            # Re-add events to buffer on error
            async with self._lock:
                self._buffer = events_to_write + self._buffer

    def _rotate_file(self) -> None:
        """Rotate to a new log file."""
        if self._file_handle:
            self._file_handle.close()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"replay_{self.session_id}_{timestamp}.jsonl"

        if self.compress:
            filename += ".gz"
            self._current_file = self.log_dir / filename
            self._file_handle = gzip.open(self._current_file, "wt", encoding="utf-8")
        else:
            self._current_file = self.log_dir / filename
            self._file_handle = open(self._current_file, "w", encoding="utf-8")

        logger.debug("replay_file_rotated", path=str(self._current_file))


class ReplayReader:
    """Reads and replays events from log files."""

    def __init__(self, log_dir: str = "/data/replay"):
        """Initialize replay reader."""
        self.log_dir = Path(log_dir)

    def list_sessions(self) -> List[dict]:
        """List available replay sessions."""
        sessions = []

        for path in self.log_dir.glob("replay_session_*.jsonl*"):
            parts = path.stem.replace(".jsonl", "").split("_")
            if len(parts) >= 3:
                sessions.append({
                    "session_id": f"session_{parts[1]}",
                    "timestamp": parts[2] if len(parts) > 2 else None,
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "compressed": path.suffix == ".gz",
                })

        return sorted(sessions, key=lambda x: x["timestamp"] or "", reverse=True)

    def read_session(
        self,
        session_id: str,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Iterator[ReplayEvent]:
        """Read events from a session.

        Args:
            session_id: Session ID to read
            event_types: Filter by event types (None = all)
            start_time: Filter events after this time
            end_time: Filter events before this time

        Yields:
            ReplayEvent objects
        """
        type_filter = {e.value for e in event_types} if event_types else None
        start_ts = int(start_time.timestamp() * 1_000_000) if start_time else None
        end_ts = int(end_time.timestamp() * 1_000_000) if end_time else None

        # Find all files for this session
        pattern = f"replay_{session_id}_*.jsonl*"
        files = sorted(self.log_dir.glob(pattern))

        for file_path in files:
            opener = gzip.open if file_path.suffix == ".gz" else open

            with opener(file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        event = ReplayEvent.from_dict(data)

                        # Apply filters
                        if type_filter and event.event_type not in type_filter:
                            continue
                        if start_ts and event.timestamp < start_ts:
                            continue
                        if end_ts and event.timestamp > end_ts:
                            continue

                        yield event

                    except json.JSONDecodeError:
                        continue

    def get_session_summary(self, session_id: str) -> dict:
        """Get summary statistics for a session."""
        event_counts: Dict[str, int] = {}
        first_event: Optional[int] = None
        last_event: Optional[int] = None
        total_events = 0

        for event in self.read_session(session_id):
            total_events += 1
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

            if first_event is None or event.timestamp < first_event:
                first_event = event.timestamp
            if last_event is None or event.timestamp > last_event:
                last_event = event.timestamp

        duration_seconds = (last_event - first_event) / 1_000_000 if first_event and last_event else 0

        return {
            "session_id": session_id,
            "total_events": total_events,
            "event_counts": event_counts,
            "first_event": datetime.fromtimestamp(first_event / 1_000_000).isoformat() if first_event else None,
            "last_event": datetime.fromtimestamp(last_event / 1_000_000).isoformat() if last_event else None,
            "duration_seconds": duration_seconds,
        }
