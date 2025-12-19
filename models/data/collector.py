"""
Historical Data Collector for Binance
Collects orderbook snapshots and trades for model training
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import aiohttp
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataCollector:
    """Collect historical data from Binance API"""

    BASE_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch kline/candlestick data"""

        url = f"{self.BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(data, columns=columns)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_base", "taker_buy_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["symbol"] = symbol
        return df

    async def fetch_orderbook_snapshot(
        self,
        symbol: str,
        limit: int = 100,
    ) -> dict:
        """Fetch current orderbook snapshot"""

        url = f"{self.BASE_URL}/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                return await response.json()

    async def fetch_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical trades"""

        url = f"{self.BASE_URL}/api/v3/aggTrades"
        params = {"symbol": symbol, "limit": limit}

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

        df = pd.DataFrame(data)
        if not df.empty:
            df.columns = ["agg_trade_id", "price", "quantity", "first_trade_id",
                          "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["price"] = pd.to_numeric(df["price"])
            df["quantity"] = pd.to_numeric(df["quantity"])
            df["symbol"] = symbol

        return df

    async def collect_historical_klines(
        self,
        symbols: List[str],
        interval: str = "1m",
        days: int = 90,
    ) -> pd.DataFrame:
        """Collect historical klines for multiple symbols"""

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        all_data = []

        for symbol in symbols:
            logger.info(f"Collecting {symbol} klines for {days} days...")

            current_start = start_time
            symbol_data = []

            with tqdm(total=days, desc=f"{symbol}") as pbar:
                while current_start < end_time:
                    try:
                        df = await self.fetch_klines(
                            symbol=symbol,
                            interval=interval,
                            start_time=current_start,
                            end_time=min(current_start + timedelta(days=1), end_time),
                            limit=1440,  # Max minutes in a day
                        )

                        if not df.empty:
                            symbol_data.append(df)

                        current_start += timedelta(days=1)
                        pbar.update(1)

                        # Rate limiting
                        await asyncio.sleep(0.1)

                    except Exception as e:
                        logger.error(f"Error fetching {symbol}: {e}")
                        await asyncio.sleep(1)

            if symbol_data:
                all_data.append(pd.concat(symbol_data, ignore_index=True))

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates(subset=["symbol", "open_time"])
            result = result.sort_values(["symbol", "open_time"])
            return result

        return pd.DataFrame()

    async def collect_historical_trades(
        self,
        symbols: List[str],
        days: int = 7,
    ) -> pd.DataFrame:
        """Collect historical trades for multiple symbols"""

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        all_data = []

        for symbol in symbols:
            logger.info(f"Collecting {symbol} trades for {days} days...")

            current_start = start_time
            symbol_data = []

            while current_start < end_time:
                try:
                    df = await self.fetch_trades(
                        symbol=symbol,
                        start_time=current_start,
                        end_time=min(current_start + timedelta(hours=1), end_time),
                        limit=1000,
                    )

                    if not df.empty:
                        symbol_data.append(df)

                    current_start += timedelta(hours=1)
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error fetching trades for {symbol}: {e}")
                    await asyncio.sleep(1)

            if symbol_data:
                all_data.append(pd.concat(symbol_data, ignore_index=True))

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result

        return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save dataframe to parquet"""
        filepath = self.data_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load dataframe from parquet"""
        filepath = self.data_dir / filename
        if filepath.exists():
            return pd.read_parquet(filepath)
        return pd.DataFrame()


async def main():
    """Main data collection script"""
    collector = BinanceDataCollector()

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Collect klines
    klines = await collector.collect_historical_klines(
        symbols=symbols,
        interval="1m",
        days=90,
    )

    if not klines.empty:
        collector.save_data(klines, "klines_90d.parquet")
        logger.info(f"Collected {len(klines)} kline records")

    # Collect recent trades
    trades = await collector.collect_historical_trades(
        symbols=symbols,
        days=7,
    )

    if not trades.empty:
        collector.save_data(trades, "trades_7d.parquet")
        logger.info(f"Collected {len(trades)} trade records")


if __name__ == "__main__":
    asyncio.run(main())
