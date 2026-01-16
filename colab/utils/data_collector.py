"""
Data Collection Module
Fetches historical market data from Binance API
"""

import aiohttp
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
from tqdm.notebook import tqdm


class DataCollector:
    """Collects OHLCV data from Binance API"""

    def __init__(self, symbols: List[str], days: int = 90, interval: str = "1m"):
        self.symbols = symbols
        self.days = days
        self.interval = interval
        self.base_url = "https://api.binance.com/api/v3/klines"

    async def _fetch_klines_async(
        self,
        symbol: str,
        session: aiohttp.ClientSession
    ) -> pd.DataFrame:
        """Fetch klines for a single symbol"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.days)

        all_data = []
        current = start_time

        while current < end_time:
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": int(current.timestamp() * 1000),
                "endTime": int(min(current + timedelta(days=1), end_time).timestamp() * 1000),
                "limit": 1440
            }

            async with session.get(self.base_url, params=params) as resp:
                data = await resp.json()
                if isinstance(data, list):
                    all_data.extend(data)

            current += timedelta(days=1)
            await asyncio.sleep(0.1)  # Rate limiting

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(all_data, columns=columns)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        numeric_cols = ["open", "high", "low", "close", "volume",
                        "quote_volume", "taker_buy_base", "taker_buy_quote"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["symbol"] = symbol
        return df.drop_duplicates(subset=["open_time"]).sort_values("open_time")

    async def collect_async(self) -> pd.DataFrame:
        """Collect data for all symbols asynchronously"""
        all_data = []

        async with aiohttp.ClientSession() as session:
            for symbol in tqdm(self.symbols, desc="Collecting data"):
                df = await self._fetch_klines_async(symbol, session)
                all_data.append(df)
                print(f"  {symbol}: {len(df):,} rows")

        raw_data = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal collected: {len(raw_data):,} rows")

        return raw_data

    def collect(self) -> pd.DataFrame:
        """Synchronous wrapper for collect_async"""
        return asyncio.get_event_loop().run_until_complete(self.collect_async())

    def validate_data(self, df: pd.DataFrame) -> dict:
        """Validate collected data quality"""
        validation = {
            "total_rows": len(df),
            "symbols": df["symbol"].unique().tolist(),
            "date_range": {
                "start": str(df["open_time"].min()),
                "end": str(df["open_time"].max())
            },
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated(subset=["symbol", "open_time"]).sum(),
            "issues": []
        }

        # Check for gaps
        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].sort_values("open_time")
            time_diff = symbol_df["open_time"].diff()
            gaps = time_diff[time_diff > pd.Timedelta(minutes=2)]
            if len(gaps) > 0:
                validation["issues"].append(f"{symbol}: {len(gaps)} gaps detected")

        # Check for zero/negative prices
        for col in ["open", "high", "low", "close"]:
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                validation["issues"].append(f"{col}: {invalid} invalid values")

        return validation
