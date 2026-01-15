"""
Validation Framework for Financial Machine Learning
====================================================

Provides robust validation tools for quantitative finance:

1. CPCV (Combinatorial Purged Cross-Validation)
   - Proper cross-validation with embargo and purge gaps
   - PBO (Probability of Backtest Overfitting) calculation
   - Deflated Sharpe Ratio for multiple testing correction

2. Leakage Guards
   - Temporal ordering validation
   - Feature-target leakage detection
   - Normalization leakage detection
   - Look-ahead bias detection
   - Data snooping detection

Usage:
------
>>> from models.validation import CombinatorialPurgedKFold, CPCVEvaluator
>>> from models.validation import LeakageGuardSuite, TemporalOrderValidator
>>>
>>> # CPCV for proper cross-validation
>>> cpcv = CombinatorialPurgedKFold(n_splits=5, n_test_groups=2, embargo_pct=0.01)
>>> for train_idx, test_idx in cpcv.split(X, y, times=timestamps):
...     model.fit(X[train_idx], y[train_idx])
...     predictions = model.predict(X[test_idx])
>>>
>>> # Leakage validation
>>> suite = LeakageGuardSuite()
>>> summary = suite.run_all_checks(X_train, X_val, X_test, ...)
>>> summary.print_report()
"""

from .cpcv import (
    CombinatorialPurgedKFold,
    CPCVEvaluator,
    CPCVResult,
    calculate_deflated_sharpe_ratio,
    calculate_pbo,
    calculate_sharpe_ratio,
    validate_no_lookahead,
)

from .leakage_guards import (
    DataSnoopingDetector,
    FeatureTargetLeakageScanner,
    LeakageGuardSuite,
    LeakageReport,
    LeakageSeverity,
    LookAheadBiasDetector,
    NormalizationLeakageDetector,
    TemporalOrderValidator,
    ValidationSummary,
)

__all__ = [
    # CPCV
    "CombinatorialPurgedKFold",
    "CPCVEvaluator",
    "CPCVResult",
    "calculate_deflated_sharpe_ratio",
    "calculate_pbo",
    "calculate_sharpe_ratio",
    "validate_no_lookahead",
    # Leakage Guards
    "DataSnoopingDetector",
    "FeatureTargetLeakageScanner",
    "LeakageGuardSuite",
    "LeakageReport",
    "LeakageSeverity",
    "LookAheadBiasDetector",
    "NormalizationLeakageDetector",
    "TemporalOrderValidator",
    "ValidationSummary",
]
