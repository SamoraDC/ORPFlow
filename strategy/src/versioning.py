"""Strategy versioning and experiment tracking.

Tracks:
- Strategy version with each trade
- Active parameters at execution time
- A/B testing support with feature flags
- Performance comparison by version
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from enum import Enum

import structlog

logger = structlog.get_logger()


class FeatureFlag(Enum):
    """Available feature flags for experimentation."""

    # Strategy variations
    USE_WEIGHTED_IMBALANCE = "use_weighted_imbalance"
    USE_MOMENTUM_CONFIRMATION = "use_momentum_confirmation"
    USE_VOLATILITY_FILTER = "use_volatility_filter"
    USE_FUNDING_RATE_FILTER = "use_funding_rate_filter"

    # Execution variations
    USE_ADAPTIVE_SIZING = "use_adaptive_sizing"
    USE_MARKET_IMPACT_MODEL = "use_market_impact_model"

    # Risk variations
    AGGRESSIVE_STOPS = "aggressive_stops"
    TRAILING_STOPS = "trailing_stops"


@dataclass
class StrategyVersion:
    """Represents a specific version of the strategy."""

    # Version identification
    version_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Strategy parameters
    imbalance_threshold: float = 0.3
    min_confidence: float = 0.6
    persistence_required: int = 3
    position_size_pct: float = 0.1
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0

    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    # Metadata
    description: str = ""
    parent_version: Optional[str] = None

    def __post_init__(self):
        if not self.version_id:
            self.version_id = self._generate_version_id()

    def _generate_version_id(self) -> str:
        """Generate a unique version ID based on parameters."""
        params = {
            "imbalance_threshold": self.imbalance_threshold,
            "min_confidence": self.min_confidence,
            "persistence_required": self.persistence_required,
            "position_size_pct": self.position_size_pct,
            "feature_flags": sorted(self.feature_flags.items()),
        }
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.sha256(param_str.encode()).hexdigest()[:8]
        return f"v{datetime.utcnow().strftime('%Y%m%d')}_{hash_val}"

    def is_flag_enabled(self, flag: FeatureFlag) -> bool:
        """Check if a feature flag is enabled."""
        return self.feature_flags.get(flag.value, False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "parameters": {
                "imbalance_threshold": self.imbalance_threshold,
                "min_confidence": self.min_confidence,
                "persistence_required": self.persistence_required,
                "position_size_pct": self.position_size_pct,
                "stop_loss_atr_mult": self.stop_loss_atr_mult,
                "take_profit_atr_mult": self.take_profit_atr_mult,
            },
            "feature_flags": self.feature_flags,
            "description": self.description,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyVersion":
        """Create from dictionary."""
        return cls(
            version_id=data.get("version_id", ""),
            name=data.get("name", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            imbalance_threshold=data.get("parameters", {}).get("imbalance_threshold", 0.3),
            min_confidence=data.get("parameters", {}).get("min_confidence", 0.6),
            persistence_required=data.get("parameters", {}).get("persistence_required", 3),
            position_size_pct=data.get("parameters", {}).get("position_size_pct", 0.1),
            stop_loss_atr_mult=data.get("parameters", {}).get("stop_loss_atr_mult", 2.0),
            take_profit_atr_mult=data.get("parameters", {}).get("take_profit_atr_mult", 3.0),
            feature_flags=data.get("feature_flags", {}),
            description=data.get("description", ""),
            parent_version=data.get("parent_version"),
        )


@dataclass
class TradeVersionMetadata:
    """Version metadata attached to each trade."""

    version_id: str
    parameters_snapshot: dict
    feature_flags: Dict[str, bool]
    signal_details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "parameters_snapshot": self.parameters_snapshot,
            "feature_flags": self.feature_flags,
            "signal_details": self.signal_details,
        }


@dataclass
class VersionPerformance:
    """Performance metrics for a strategy version."""

    version_id: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Optional[float] = None
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl_per_trade(self) -> Decimal:
        if self.total_trades == 0:
            return Decimal("0")
        return self.total_pnl / self.total_trades


class VersionManager:
    """Manages strategy versions and tracks performance."""

    def __init__(self):
        """Initialize version manager."""
        self._versions: Dict[str, StrategyVersion] = {}
        self._performance: Dict[str, VersionPerformance] = {}
        self._active_version: Optional[str] = None
        self._shadow_versions: List[str] = []  # Versions running in shadow mode

    def register_version(self, version: StrategyVersion) -> None:
        """Register a new strategy version."""
        self._versions[version.version_id] = version
        self._performance[version.version_id] = VersionPerformance(
            version_id=version.version_id
        )

        logger.info(
            "version_registered",
            version_id=version.version_id,
            name=version.name,
        )

    def set_active_version(self, version_id: str) -> None:
        """Set the active (trading) version."""
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not registered")

        old_version = self._active_version
        self._active_version = version_id

        logger.info(
            "active_version_changed",
            old_version=old_version,
            new_version=version_id,
        )

    def add_shadow_version(self, version_id: str) -> None:
        """Add a version to run in shadow mode (generates signals but doesn't trade)."""
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not registered")

        if version_id not in self._shadow_versions:
            self._shadow_versions.append(version_id)

            logger.info("shadow_version_added", version_id=version_id)

    def remove_shadow_version(self, version_id: str) -> None:
        """Remove a version from shadow mode."""
        if version_id in self._shadow_versions:
            self._shadow_versions.remove(version_id)
            logger.info("shadow_version_removed", version_id=version_id)

    def get_active_version(self) -> Optional[StrategyVersion]:
        """Get the currently active version."""
        if self._active_version:
            return self._versions.get(self._active_version)
        return None

    def get_version(self, version_id: str) -> Optional[StrategyVersion]:
        """Get a specific version."""
        return self._versions.get(version_id)

    def get_shadow_versions(self) -> List[StrategyVersion]:
        """Get all shadow versions."""
        return [
            self._versions[vid]
            for vid in self._shadow_versions
            if vid in self._versions
        ]

    def create_trade_metadata(
        self,
        signal_details: Optional[dict] = None,
    ) -> TradeVersionMetadata:
        """Create version metadata for a trade."""
        version = self.get_active_version()
        if not version:
            raise ValueError("No active version set")

        return TradeVersionMetadata(
            version_id=version.version_id,
            parameters_snapshot={
                "imbalance_threshold": version.imbalance_threshold,
                "min_confidence": version.min_confidence,
                "persistence_required": version.persistence_required,
                "position_size_pct": version.position_size_pct,
            },
            feature_flags=version.feature_flags.copy(),
            signal_details=signal_details or {},
        )

    def record_trade(
        self,
        version_id: str,
        pnl: Decimal,
        fee: Decimal,
        timestamp: datetime,
    ) -> None:
        """Record a trade result for a version."""
        if version_id not in self._performance:
            self._performance[version_id] = VersionPerformance(
                version_id=version_id
            )

        perf = self._performance[version_id]
        perf.total_trades += 1
        perf.total_pnl += pnl
        perf.total_fees += fee

        if pnl > 0:
            perf.winning_trades += 1

        if perf.first_trade is None:
            perf.first_trade = timestamp
        perf.last_trade = timestamp

        logger.debug(
            "trade_recorded_for_version",
            version_id=version_id,
            pnl=str(pnl),
            total_trades=perf.total_trades,
        )

    def get_performance(self, version_id: str) -> Optional[VersionPerformance]:
        """Get performance metrics for a version."""
        return self._performance.get(version_id)

    def compare_versions(
        self,
        version_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """Compare performance across versions."""
        if version_ids is None:
            version_ids = list(self._versions.keys())

        comparisons = []
        for vid in version_ids:
            version = self._versions.get(vid)
            perf = self._performance.get(vid)

            if version and perf:
                comparisons.append({
                    "version_id": vid,
                    "name": version.name,
                    "is_active": vid == self._active_version,
                    "is_shadow": vid in self._shadow_versions,
                    "total_trades": perf.total_trades,
                    "win_rate": perf.win_rate,
                    "total_pnl": str(perf.total_pnl),
                    "avg_pnl": str(perf.avg_pnl_per_trade),
                    "created_at": version.created_at.isoformat(),
                })

        # Sort by total P&L
        comparisons.sort(key=lambda x: Decimal(x["total_pnl"]), reverse=True)

        return comparisons

    def get_status(self) -> dict:
        """Get version manager status."""
        return {
            "active_version": self._active_version,
            "shadow_versions": self._shadow_versions,
            "total_versions": len(self._versions),
            "versions": [v.to_dict() for v in self._versions.values()],
        }


# Default version
DEFAULT_VERSION = StrategyVersion(
    version_id="v1.0.0",
    name="Imbalance Strategy v1",
    description="Order flow imbalance strategy with momentum confirmation",
    feature_flags={
        FeatureFlag.USE_WEIGHTED_IMBALANCE.value: True,
        FeatureFlag.USE_MOMENTUM_CONFIRMATION.value: True,
        FeatureFlag.USE_VOLATILITY_FILTER.value: True,
    },
)
