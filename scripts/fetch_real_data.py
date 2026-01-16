#!/usr/bin/env python3
"""
Fetch real market data from Binance for model training.
Downloads 90 days of 1-minute klines for BTCUSDT.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.data.collector import BinanceDataCollector


async def fetch_90d_klines(symbol: str = "BTCUSDT", interval: str = "1m") -> pd.DataFrame:
    """Fetch 90 days of klines data."""
    collector = BinanceDataCollector(data_dir=str(PROJECT_ROOT / "data" / "raw"))

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=90)

    print(f"Fetching {symbol} {interval} data from {start_time} to {end_time}")

    all_data = []
    current_start = start_time
    batch_size = timedelta(hours=12)  # Fetch 12 hours at a time (720 candles)

    while current_start < end_time:
        current_end = min(current_start + batch_size, end_time)

        try:
            df = await collector.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000,
            )

            if len(df) > 0:
                all_data.append(df)
                print(f"  Fetched {len(df)} rows: {df['open_time'].min()} to {df['open_time'].max()}")

        except Exception as e:
            print(f"  Error fetching batch: {e}")

        current_start = current_end
        await asyncio.sleep(0.1)  # Rate limiting

    if not all_data:
        raise ValueError("No data fetched!")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time")

    print(f"\nTotal rows: {len(combined)}")
    print(f"Date range: {combined['open_time'].min()} to {combined['open_time'].max()}")

    return combined


async def main():
    """Main entry point."""
    output_path = PROJECT_ROOT / "data" / "raw" / "klines_90d.parquet"

    df = await fetch_90d_klines("BTCUSDT", "1m")

    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    asyncio.run(main())
