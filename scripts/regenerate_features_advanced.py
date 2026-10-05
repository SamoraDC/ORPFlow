#!/usr/bin/env python3
"""
Regenerate features.parquet from raw klines data using ALL advanced methods.

Includes:
- Basic features (returns, volatility, momentum)
- Signal Processing (Kalman, EMD, HHT, Wavelets, Fisher)
- Advanced Microstructure (207 features)
- Hawkes Processes (order flow modeling)
- Regime Detection (volatility, trend, Hurst)
- Triple Barrier Labeling (optional)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Regenerate features from raw data with advanced methods."""

    parser = argparse.ArgumentParser(
        description="Regenerate features with advanced quantitative methods"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/klines_90d.parquet",
        help="Input raw data path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features_advanced.parquet",
        help="Output features path"
    )
    parser.add_argument(
        "--no-signal-processing",
        action="store_true",
        help="Skip signal processing features (Kalman, EMD, Wavelets)"
    )
    parser.add_argument(
        "--no-advanced-micro",
        action="store_true",
        help="Skip advanced microstructure features"
    )
    parser.add_argument(
        "--use-hawkes",
        action="store_true",
        help="Include Hawkes process features (slow)"
    )
    parser.add_argument(
        "--no-regimes",
        action="store_true",
        help="Skip regime detection features"
    )
    parser.add_argument(
        "--use-triple-barrier",
        action="store_true",
        help="Use Triple Barrier labeling instead of simple returns"
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Only generate basic features (fast mode)"
    )

    args = parser.parse_args()

    # Resolve paths
    raw_data_path = project_root / args.input
    output_path = project_root / args.output

    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    logger.info(f"Loaded {len(df)} rows")

    # Convert timestamps
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # Import feature engineer
    from models.data.preprocessor import FeatureEngineer

    # Process features
    engineer = FeatureEngineer(windows=[5, 10, 20, 50, 100])

    start_time = datetime.now()

    if args.basic_only:
        logger.info("Processing with BASIC features only (fast mode)...")
        processed_df = engineer.process_symbol(df)
    else:
        logger.info("Processing with ADVANCED features...")
        processed_df = engineer.process_symbol_advanced(
            df,
            use_signal_processing=not args.no_signal_processing,
            use_advanced_micro=not args.no_advanced_micro,
            use_hawkes=args.use_hawkes,
            use_regimes=not args.no_regimes,
        )

    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Processing completed in {processing_time:.1f} seconds")

    # Triple Barrier Labeling (optional)
    if args.use_triple_barrier:
        logger.info("Applying Triple Barrier labeling...")
        try:
            from models.data.triple_barrier import (
                TripleBarrierLabeler,
                BarrierConfig,
                cusum_filter,
            )

            # Configure barriers
            config = BarrierConfig(
                tp_multiplier=2.0,
                sl_multiplier=2.0,
                vertical_bars=20,
                min_return=0.0,
            )

            # Detect events using CUSUM
            events = cusum_filter(processed_df['close'], threshold=0.02)
            logger.info(f"CUSUM detected {len(events)} events")

            # Generate labels
            labeler = TripleBarrierLabeler(config)
            labeler.fit(
                close=processed_df['close'],
                high=processed_df['high'],
                low=processed_df['low'],
                t_events=events,
            )

            labels = labeler.get_labels()
            weights = labeler.get_sample_weights(processed_df['close'])

            # Merge labels with processed data
            processed_df['tb_label'] = np.nan
            processed_df['tb_return'] = np.nan
            processed_df['sample_weight'] = 1.0

            processed_df.loc[labels.index, 'tb_label'] = labels['bin']
            processed_df.loc[labels.index, 'tb_return'] = labels['ret']
            processed_df.loc[weights.index, 'sample_weight'] = weights

            logger.info(f"Triple Barrier labels: {labels['bin'].value_counts().to_dict()}")

        except Exception as e:
            logger.warning(f"Triple Barrier labeling failed: {e}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    logger.info(f"Saving {len(processed_df)} rows to {output_path}")
    processed_df.to_parquet(output_path, index=False)

    # Verification and summary
    verify_df = pd.read_parquet(output_path)

    # Get feature columns
    feature_cols = engineer.get_feature_columns(verify_df)
    target_cols = [c for c in verify_df.columns if c.startswith("target_")]

    # Check for NaN
    nan_counts = verify_df[feature_cols].isna().sum()
    nan_features = nan_counts[nan_counts > 0]

    print(f"\n{'='*60}")
    print(f"✓ Features regenerated successfully!")
    print(f"{'='*60}")
    print(f"  Output: {output_path}")
    print(f"  Shape: {verify_df.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets: {len(target_cols)}")
    print(f"  Processing time: {processing_time:.1f}s")

    if len(nan_features) > 0:
        print(f"\n  ⚠ Features with NaN: {len(nan_features)}")
    else:
        print(f"\n  ✓ No NaN values in features")

    # Feature categories
    signal_features = [c for c in feature_cols if any(x in c for x in ['kalman', 'emd', 'wavelet', 'fisher'])]
    micro_features = [c for c in feature_cols if c.startswith('micro_')]
    hawkes_features = [c for c in feature_cols if 'hawkes' in c]
    regime_features = [c for c in feature_cols if 'regime' in c]

    print(f"\n  Feature breakdown:")
    print(f"    - Signal Processing: {len(signal_features)}")
    print(f"    - Microstructure: {len(micro_features)}")
    print(f"    - Hawkes: {len(hawkes_features)}")
    print(f"    - Regime: {len(regime_features)}")
    print(f"    - Other: {len(feature_cols) - len(signal_features) - len(micro_features) - len(hawkes_features) - len(regime_features)}")
    print(f"{'='*60}\n")

    return processed_df


if __name__ == "__main__":
    main()
