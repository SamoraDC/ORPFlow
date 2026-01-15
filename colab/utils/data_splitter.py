"""
Per-Symbol Temporal Data Splitter
Splits data maintaining temporal order within each symbol

CRITICAL: This eliminates cross-symbol contamination at split boundaries!
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from .feature_engineer import get_feature_columns


class PerSymbolSplitter:
    """
    Splits data temporally PER SYMBOL to avoid cross-symbol contamination.

    PROBLEM WITH GLOBAL SPLIT:
    - Global sort by time + index split causes different symbols at
      the SAME timestamp to end up in different splits
    - Example: BTCUSDT at 10:00 in TRAIN, ETHUSDT at 10:00 in VAL = LEAKAGE!

    SOLUTION:
    - Split EACH symbol independently (70/15/15)
    - Then concatenate train from all, val from all, test from all
    - Guarantees each symbol's train < val < test temporally
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_by_symbol(
        self,
        processed_by_symbol: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split each symbol independently, then concatenate.

        Args:
            processed_by_symbol: Dict mapping symbol -> processed DataFrame

        Returns:
            (train_df, val_df, test_df) with no cross-symbol contamination
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []

        split_info = {}

        for symbol, symbol_df in processed_by_symbol.items():
            # Ensure sorted by time
            symbol_df = symbol_df.sort_values("open_time").reset_index(drop=True)

            n = len(symbol_df)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))

            symbol_train = symbol_df.iloc[:train_end].copy()
            symbol_val = symbol_df.iloc[train_end:val_end].copy()
            symbol_test = symbol_df.iloc[val_end:].copy()

            train_dfs.append(symbol_train)
            val_dfs.append(symbol_val)
            test_dfs.append(symbol_test)

            split_info[symbol] = {
                "train": {
                    "count": len(symbol_train),
                    "start": str(symbol_train["open_time"].min()),
                    "end": str(symbol_train["open_time"].max())
                },
                "val": {
                    "count": len(symbol_val),
                    "start": str(symbol_val["open_time"].min()),
                    "end": str(symbol_val["open_time"].max())
                },
                "test": {
                    "count": len(symbol_test),
                    "start": str(symbol_test["open_time"].min()),
                    "end": str(symbol_test["open_time"].max())
                }
            }

        # Concatenate (each split has data from same time period!)
        train_df = pd.concat(train_dfs, ignore_index=True).sort_values("open_time").reset_index(drop=True)
        val_df = pd.concat(val_dfs, ignore_index=True).sort_values("open_time").reset_index(drop=True)
        test_df = pd.concat(test_dfs, ignore_index=True).sort_values("open_time").reset_index(drop=True)

        self.split_info = split_info

        return train_df, val_df, test_df

    def validate_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        processed_by_symbol: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Validate that split has no temporal leakage.

        IMPORTANT: Validates PER-SYMBOL, not globally!
        """
        validation = {
            "passed": True,
            "checks": [],
            "per_symbol": {}
        }

        symbols = list(processed_by_symbol.keys())

        for symbol in symbols:
            symbol_train = train_df[train_df["symbol"] == symbol]
            symbol_val = val_df[val_df["symbol"] == symbol]
            symbol_test = test_df[test_df["symbol"] == symbol]

            train_max = symbol_train["open_time"].max()
            val_min = symbol_val["open_time"].min()
            val_max = symbol_val["open_time"].max()
            test_min = symbol_test["open_time"].min()

            # Check temporal ordering
            train_before_val = train_max < val_min
            val_before_test = val_max < test_min

            validation["per_symbol"][symbol] = {
                "train_max": str(train_max),
                "val_min": str(val_min),
                "val_max": str(val_max),
                "test_min": str(test_min),
                "train_before_val": train_before_val,
                "val_before_test": val_before_test,
                "valid": train_before_val and val_before_test
            }

            if not (train_before_val and val_before_test):
                validation["passed"] = False
                validation["checks"].append(
                    f"FAIL: {symbol} has temporal overlap!"
                )
            else:
                validation["checks"].append(
                    f"PASS: {symbol} temporal order correct"
                )

        # Summary stats
        total = len(train_df) + len(val_df) + len(test_df)
        validation["summary"] = {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_pct": len(train_df) / total,
            "val_pct": len(val_df) / total,
            "test_pct": len(test_df) / total
        }

        return validation

    def print_split_summary(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        processed_by_symbol: Dict[str, pd.DataFrame]
    ):
        """Print detailed split summary"""
        print("=" * 70)
        print("TEMPORAL SPLIT SUMMARY (Per-Symbol)")
        print("=" * 70)

        for symbol, info in self.split_info.items():
            print(f"\n{symbol}:")
            print(f"  Train: {info['train']['count']:,} rows")
            print(f"         {info['train']['start']} to {info['train']['end']}")
            print(f"  Val:   {info['val']['count']:,} rows")
            print(f"         {info['val']['start']} to {info['val']['end']}")
            print(f"  Test:  {info['test']['count']:,} rows")
            print(f"         {info['test']['start']} to {info['test']['end']}")

        print("\n" + "=" * 70)
        print("TOTALS:")
        print("=" * 70)

        total = len(train_df) + len(val_df) + len(test_df)
        print(f"Train: {len(train_df):,} ({len(train_df)/total:.1%})")
        print(f"Val:   {len(val_df):,} ({len(val_df)/total:.1%})")
        print(f"Test:  {len(test_df):,} ({len(test_df)/total:.1%})")

        # Validate
        validation = self.validate_split(train_df, val_df, test_df, processed_by_symbol)

        print("\n" + "=" * 70)
        print("VALIDATION:")
        print("=" * 70)

        for check in validation["checks"]:
            emoji = "" if "PASS" in check else ""
            print(f"  {emoji} {check}")

        if validation["passed"]:
            print("\n NO CROSS-SYMBOL CONTAMINATION - SPLIT IS VALID!")
        else:
            print("\n TEMPORAL LEAKAGE DETECTED!")
            raise ValueError("Data split validation failed!")
