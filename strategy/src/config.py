"""Configuration management for the strategy engine."""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Database
    database_url: str = Field(default="sqlite:///data/trades.db")

    # Core API
    core_api_url: str = Field(default="http://localhost:8081")

    # Trading configuration
    symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT"])
    initial_balance: float = Field(default=10000.0)

    # Strategy parameters
    imbalance_threshold: float = Field(default=0.3)
    min_confidence: float = Field(default=0.6)
    position_size_pct: float = Field(default=0.1)  # 10% of balance per trade
    stop_loss_atr_mult: float = Field(default=2.0)
    take_profit_atr_mult: float = Field(default=3.0)

    # Risk parameters
    max_position_size: float = Field(default=1.0)
    max_drawdown_pct: float = Field(default=0.05)
    max_daily_trades: int = Field(default=50)

    # Scheduler - Shabbat pause
    timezone: str = Field(default="America/Sao_Paulo")
    shabbat_latitude: float = Field(default=-23.5505)  # São Paulo
    shabbat_longitude: float = Field(default=-46.6333)
    shabbat_pause_enabled: bool = Field(default=True)

    # Telegram notifications
    telegram_bot_token: str | None = Field(default=None)
    telegram_chat_id: str | None = Field(default=None)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v: str | List[str]) -> List[str]:
        """Parse comma-separated symbols from environment variable."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
