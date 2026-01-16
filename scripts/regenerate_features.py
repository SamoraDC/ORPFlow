#!/usr/bin/env python3
"""
Regenerate features.parquet from raw klines data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Regenerate features from raw data."""

    raw_data_path = project_root / "data" / "raw" / "klines_90d.parquet"
    output_path = project_root / "data" / "processed" / "features.parquet"

    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    logger.info(f"Loaded {len(df)} rows")

    # Import feature engineer
    from models.data.preprocessor import FeatureEngineer

    # Process features
    engineer = FeatureEngineer(windows=[5, 10, 20, 50, 100])

    # Convert timestamps
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    logger.info("Processing features...")
    processed_df = engineer.process_symbol(df)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    logger.info(f"Saving {len(processed_df)} rows to {output_path}")
    processed_df.to_parquet(output_path, index=False)

    # Verify
    verify_df = pd.read_parquet(output_path)
    logger.info(f"Verification: {verify_df.shape}")

    # Show feature summary
    feature_cols = engineer.get_feature_columns(verify_df)
    target_cols = [c for c in verify_df.columns if c.startswith("target_")]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Targets: {target_cols}")

    # Check for NaN
    nan_counts = verify_df[feature_cols].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    if len(nan_features) > 0:
        logger.warning(f"Features with NaN: {nan_features.to_dict()}")
    else:
        logger.info("No NaN values in features - data is clean!")

    print(f"\n✓ Features regenerated successfully!")
    print(f"  Shape: {verify_df.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets: {len(target_cols)}")

    return processed_df


if __name__ == "__main__":
    main()
