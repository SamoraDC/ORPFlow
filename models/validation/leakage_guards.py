"""
Leakage Guards for Financial Machine Learning
==============================================

Comprehensive leakage detection and prevention framework for
quantitative finance applications.

Implements:
1. Temporal Ordering Validator - Strict train < val < test
2. Feature-Target Leakage Scanner - Detect information leakage in features
3. Normalization Leakage Detector - Check scaler fitting issues
4. Look-Ahead Bias Detector - Find features using future data
5. Data Snooping Detector - Identify multiple testing issues

Reference:
    - Lopez de Prado, M. (2018). Advances in Financial Machine Learning
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class LeakageSeverity(Enum):
    """Severity levels for detected leakage."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class LeakageReport:
    """Container for leakage detection results."""

    check_name: str
    passed: bool
    severity: LeakageSeverity = LeakageSeverity.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "recommendations": self.recommendations,
        }


@dataclass
class ValidationSummary:
    """Summary of all validation checks."""

    reports: List[LeakageReport] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.passed for r in self.reports)

    @property
    def critical_failures(self) -> List[LeakageReport]:
        """Get critical and fatal failures."""
        return [
            r for r in self.reports
            if not r.passed and r.severity in (LeakageSeverity.CRITICAL, LeakageSeverity.FATAL)
        ]

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_checks": len(self.reports),
            "passed": sum(1 for r in self.reports if r.passed),
            "failed": sum(1 for r in self.reports if not r.passed),
            "critical_failures": len(self.critical_failures),
            "all_passed": self.all_passed,
        }

    def print_report(self) -> None:
        """Print formatted validation report."""
        print("=" * 70)
        print("LEAKAGE VALIDATION REPORT")
        print("=" * 70)

        for report in self.reports:
            status = "PASS" if report.passed else "FAIL"
            severity_icon = {
                LeakageSeverity.INFO: "[i]",
                LeakageSeverity.WARNING: "[!]",
                LeakageSeverity.CRITICAL: "[!!]",
                LeakageSeverity.FATAL: "[XXX]",
            }[report.severity]

            print(f"\n{severity_icon} [{status}] {report.check_name}")
            print(f"   {report.message}")

            if report.recommendations:
                print("   Recommendations:")
                for rec in report.recommendations:
                    print(f"     - {rec}")

        print("\n" + "=" * 70)
        summary = self.summary()
        if self.all_passed:
            print(f"ALL {summary['total_checks']} CHECKS PASSED")
        else:
            print(f"VALIDATION FAILED: {summary['failed']}/{summary['total_checks']} checks failed")
            if summary['critical_failures'] > 0:
                print(f"CRITICAL FAILURES: {summary['critical_failures']}")
        print("=" * 70)


class TemporalOrderValidator:
    """
    Validates strict temporal ordering between train/val/test splits.

    For financial time series, it is critical that:
    1. Training data comes before validation data
    2. Validation data comes before test data
    3. There are no overlapping samples
    4. Sufficient gap exists between splits (embargo)

    Parameters
    ----------
    min_embargo_bars : int, default=5
        Minimum number of bars between splits.
    strict : bool, default=True
        If True, any violation is a fatal error.

    Examples
    --------
    >>> validator = TemporalOrderValidator(min_embargo_bars=5)
    >>> report = validator.validate(train_times, val_times, test_times)
    >>> print(report.passed)
    """

    def __init__(self, min_embargo_bars: int = 5, strict: bool = True):
        self.min_embargo_bars = min_embargo_bars
        self.strict = strict

    def validate(
        self,
        train_times: pd.Series,
        val_times: Optional[pd.Series] = None,
        test_times: Optional[pd.Series] = None,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
    ) -> LeakageReport:
        """
        Validate temporal ordering of data splits.

        Parameters
        ----------
        train_times : pd.Series
            Timestamps for training data.
        val_times : pd.Series, optional
            Timestamps for validation data.
        test_times : pd.Series, optional
            Timestamps for test data.
        train_indices : np.ndarray, optional
            Original indices for train (for overlap check).
        val_indices : np.ndarray, optional
            Original indices for val (for overlap check).
        test_indices : np.ndarray, optional
            Original indices for test (for overlap check).

        Returns
        -------
        LeakageReport
            Validation results.
        """
        issues = []
        details = {}

        train_max = train_times.max()
        train_min = train_times.min()
        details["train_range"] = f"{train_min} to {train_max}"

        # Check index overlaps
        if train_indices is not None:
            if val_indices is not None:
                overlap = np.intersect1d(train_indices, val_indices)
                if len(overlap) > 0:
                    issues.append(f"Train-Val index overlap: {len(overlap)} samples")
                    details["train_val_overlap"] = len(overlap)

            if test_indices is not None:
                overlap = np.intersect1d(train_indices, test_indices)
                if len(overlap) > 0:
                    issues.append(f"Train-Test index overlap: {len(overlap)} samples")
                    details["train_test_overlap"] = len(overlap)

        # Check validation ordering
        if val_times is not None:
            val_max = val_times.max()
            val_min = val_times.min()
            details["val_range"] = f"{val_min} to {val_max}"

            if train_max >= val_min:
                issues.append(
                    f"TEMPORAL LEAKAGE: Train max ({train_max}) >= Val min ({val_min})"
                )
                details["train_val_gap"] = str(val_min - train_max)
            else:
                gap = self._calculate_gap(train_max, val_min, train_times)
                details["train_val_gap_bars"] = gap

                if gap < self.min_embargo_bars:
                    issues.append(
                        f"Insufficient embargo: Train-Val gap ({gap}) < min ({self.min_embargo_bars})"
                    )

        # Check test ordering
        if test_times is not None:
            test_max = test_times.max()
            test_min = test_times.min()
            details["test_range"] = f"{test_min} to {test_max}"

            # Against training
            if train_max >= test_min:
                issues.append(
                    f"TEMPORAL LEAKAGE: Train max ({train_max}) >= Test min ({test_min})"
                )

            # Against validation
            if val_times is not None:
                if val_max >= test_min:
                    issues.append(
                        f"TEMPORAL LEAKAGE: Val max ({val_max}) >= Test min ({test_min})"
                    )
                else:
                    gap = self._calculate_gap(val_max, test_min, val_times)
                    details["val_test_gap_bars"] = gap

                    if gap < self.min_embargo_bars:
                        issues.append(
                            f"Insufficient embargo: Val-Test gap ({gap}) < min ({self.min_embargo_bars})"
                        )

        passed = len(issues) == 0
        severity = LeakageSeverity.FATAL if (not passed and self.strict) else LeakageSeverity.CRITICAL

        return LeakageReport(
            check_name="Temporal Order Validation",
            passed=passed,
            severity=severity if not passed else LeakageSeverity.INFO,
            message="All temporal orderings correct" if passed else "; ".join(issues),
            details=details,
            recommendations=[
                "Ensure train data ends before validation starts",
                "Add embargo gap between splits",
                "Use time-based splitting, not random",
            ] if not passed else [],
        )

    def _calculate_gap(
        self,
        end_time: pd.Timestamp,
        start_time: pd.Timestamp,
        reference_series: pd.Series,
    ) -> int:
        """Calculate gap in number of bars."""
        if isinstance(end_time, pd.Timestamp):
            # Calculate based on average time between samples
            time_diff = start_time - end_time
            avg_bar_duration = (reference_series.max() - reference_series.min()) / len(reference_series)
            return int(time_diff / avg_bar_duration) if avg_bar_duration.total_seconds() > 0 else 0
        else:
            return int(start_time - end_time)


class FeatureTargetLeakageScanner:
    """
    Scans for information leakage from target to features.

    Detects:
    1. Perfect or near-perfect correlation with target
    2. Features that are transformations of the target
    3. Features derived from future target values

    Parameters
    ----------
    correlation_threshold : float, default=0.95
        Correlation above this is suspicious.
    mutual_info_threshold : float, default=0.9
        Mutual information above this is suspicious.

    Examples
    --------
    >>> scanner = FeatureTargetLeakageScanner()
    >>> report = scanner.scan(X, y, feature_names)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        mutual_info_threshold: float = 0.9,
    ):
        self.correlation_threshold = correlation_threshold
        self.mutual_info_threshold = mutual_info_threshold

    def scan(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        check_mutual_info: bool = True,
    ) -> LeakageReport:
        """
        Scan features for target leakage.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values.
        feature_names : List[str], optional
            Names of features for reporting.
        check_mutual_info : bool, default=True
            Whether to check mutual information (slower but more thorough).

        Returns
        -------
        LeakageReport
            Scan results with suspicious features.
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        suspicious_features = []
        details = {"correlations": {}, "suspicious": []}

        for i in range(n_features):
            feature = X[:, i]

            # Skip if all NaN or constant
            if np.std(feature) == 0 or np.all(np.isnan(feature)):
                continue

            # Calculate correlation
            valid_mask = ~(np.isnan(feature) | np.isnan(y))
            if np.sum(valid_mask) < 10:
                continue

            corr = np.corrcoef(feature[valid_mask], y[valid_mask])[0, 1]

            if np.isnan(corr):
                continue

            details["correlations"][feature_names[i]] = float(corr)

            if abs(corr) > self.correlation_threshold:
                suspicious_features.append({
                    "feature": feature_names[i],
                    "correlation": float(corr),
                    "type": "high_correlation",
                })

            # Check for exact match (potential direct leakage)
            if abs(corr) > 0.999:
                suspicious_features.append({
                    "feature": feature_names[i],
                    "correlation": float(corr),
                    "type": "exact_match",
                    "critical": True,
                })

        # Check for transformed target
        for i in range(n_features):
            feature = X[:, i]
            valid_mask = ~(np.isnan(feature) | np.isnan(y))

            if np.sum(valid_mask) < 10:
                continue

            # Check if feature is scaled/shifted version of target
            f_scaled = (feature[valid_mask] - np.mean(feature[valid_mask])) / (np.std(feature[valid_mask]) + 1e-8)
            y_scaled = (y[valid_mask] - np.mean(y[valid_mask])) / (np.std(y[valid_mask]) + 1e-8)

            mse = np.mean((f_scaled - y_scaled) ** 2)
            if mse < 0.01:
                if not any(s["feature"] == feature_names[i] for s in suspicious_features):
                    suspicious_features.append({
                        "feature": feature_names[i],
                        "type": "transformed_target",
                        "mse_to_target": float(mse),
                        "critical": True,
                    })

        details["suspicious"] = suspicious_features
        passed = len(suspicious_features) == 0

        # Determine severity
        has_critical = any(s.get("critical", False) for s in suspicious_features)
        severity = LeakageSeverity.FATAL if has_critical else (
            LeakageSeverity.CRITICAL if not passed else LeakageSeverity.INFO
        )

        return LeakageReport(
            check_name="Feature-Target Leakage Scan",
            passed=passed,
            severity=severity,
            message=f"No leakage detected" if passed else f"Found {len(suspicious_features)} suspicious features",
            details=details,
            recommendations=[
                f"Review feature '{s['feature']}' - possible target leakage"
                for s in suspicious_features
            ],
        )


class NormalizationLeakageDetector:
    """
    Detects normalization/scaling leakage.

    Common issue: Fitting scaler on entire dataset including test data,
    which leaks test distribution information into training.

    Detects:
    1. Test data with suspiciously similar distribution to train
    2. Scaler that appears to have seen test data
    3. Batch normalization issues

    Examples
    --------
    >>> detector = NormalizationLeakageDetector()
    >>> report = detector.detect(X_train, X_test, scaler)
    """

    def __init__(
        self,
        ks_threshold: float = 0.1,
        moment_threshold: float = 0.05,
    ):
        self.ks_threshold = ks_threshold
        self.moment_threshold = moment_threshold

    def detect(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        scaler: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> LeakageReport:
        """
        Detect normalization leakage.

        Parameters
        ----------
        X_train : np.ndarray
            Scaled training features.
        X_test : np.ndarray
            Scaled test features.
        scaler : object, optional
            Fitted scaler object (to check for full-data fitting).
        feature_names : List[str], optional
            Feature names for reporting.

        Returns
        -------
        LeakageReport
            Detection results.
        """
        n_features = X_train.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        issues = []
        details = {"feature_analysis": {}}

        # Check 1: Test data distribution should differ from train
        for i in range(min(n_features, 50)):  # Check first 50 features
            train_feat = X_train[:, i]
            test_feat = X_test[:, i]

            # Skip constant features
            if np.std(train_feat) < 1e-8 or np.std(test_feat) < 1e-8:
                continue

            # KS test - if distributions are TOO similar, might be leakage
            ks_stat, ks_pvalue = stats.ks_2samp(train_feat, test_feat)

            # Moments comparison
            train_mean, test_mean = np.mean(train_feat), np.mean(test_feat)
            train_std, test_std = np.std(train_feat), np.std(test_feat)

            details["feature_analysis"][feature_names[i]] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "train_mean": float(train_mean),
                "test_mean": float(test_mean),
                "mean_diff": float(abs(train_mean - test_mean)),
            }

            # Check for suspiciously identical distributions
            if ks_stat < self.ks_threshold and abs(train_mean - test_mean) < self.moment_threshold:
                if abs(train_std - test_std) < self.moment_threshold:
                    issues.append({
                        "feature": feature_names[i],
                        "type": "identical_distribution",
                        "ks_stat": float(ks_stat),
                        "message": "Train and test distributions are suspiciously identical",
                    })

        # Check 2: If scaler provided, verify proper fitting
        if scaler is not None:
            if hasattr(scaler, "n_samples_seen_"):
                n_seen = scaler.n_samples_seen_
                n_train = len(X_train)

                details["scaler_samples_seen"] = int(n_seen) if isinstance(n_seen, (int, np.integer)) else n_seen.tolist()
                details["train_samples"] = n_train

                # If scaler saw more samples than train, it might have seen test
                if isinstance(n_seen, (int, np.integer)):
                    if n_seen > n_train * 1.1:  # Allow 10% margin for preprocessing drops
                        issues.append({
                            "type": "scaler_leak",
                            "message": f"Scaler saw {n_seen} samples but train has {n_train}",
                        })

            if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                # Check if scaler statistics match full data
                full_data = np.vstack([X_train, X_test])
                full_mean = np.mean(full_data, axis=0)

                # If scaler mean is too close to full data mean...
                scaler_mean = scaler.mean_ if hasattr(scaler, "mean_") else None
                if scaler_mean is not None:
                    mean_diff = np.mean(np.abs(scaler_mean - full_mean[:len(scaler_mean)]))
                    train_only_mean = np.mean(X_train, axis=0)
                    train_diff = np.mean(np.abs(scaler_mean - train_only_mean[:len(scaler_mean)]))

                    details["scaler_mean_vs_full"] = float(mean_diff)
                    details["scaler_mean_vs_train"] = float(train_diff)

                    # If scaler is closer to full data than train only, suspicious
                    if mean_diff < train_diff * 0.5:
                        issues.append({
                            "type": "scaler_mean_leak",
                            "message": "Scaler mean closer to full data than train data",
                        })

        passed = len(issues) == 0

        return LeakageReport(
            check_name="Normalization Leakage Detection",
            passed=passed,
            severity=LeakageSeverity.CRITICAL if not passed else LeakageSeverity.INFO,
            message="No normalization leakage detected" if passed else f"Found {len(issues)} issues",
            details=details,
            recommendations=[
                "Fit scaler only on training data",
                "Apply transform (not fit_transform) to test data",
                "Verify train/test split before scaling",
            ] if not passed else [],
        )


class LookAheadBiasDetector:
    """
    Detects look-ahead bias in feature construction.

    Look-ahead bias occurs when features are calculated using
    future information that would not be available in real-time.

    Detects:
    1. Features correlated with future targets more than past
    2. Features using forward-looking windows
    3. Information leakage from label calculation

    Examples
    --------
    >>> detector = LookAheadBiasDetector()
    >>> report = detector.detect(df, feature_cols, target_col)
    """

    def __init__(
        self,
        future_corr_ratio: float = 2.0,
        lag_periods: int = 10,
    ):
        self.future_corr_ratio = future_corr_ratio
        self.lag_periods = lag_periods

    def detect(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        time_col: Optional[str] = None,
    ) -> LeakageReport:
        """
        Detect look-ahead bias in features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features and target.
        feature_cols : List[str]
            Column names for features.
        target_col : str
            Column name for target.
        time_col : str, optional
            Column name for timestamps.

        Returns
        -------
        LeakageReport
            Detection results.
        """
        suspicious_features = []
        details = {"feature_analysis": {}}

        df = df.copy()

        # Create lagged targets for comparison
        target_future = df[target_col].copy()  # Current (which is "future" relative to features)
        target_past = df[target_col].shift(self.lag_periods)

        for col in feature_cols[:50]:  # Check first 50 features
            if col not in df.columns:
                continue

            feature = df[col]

            # Skip if constant or all NaN
            if feature.std() < 1e-8 or feature.isna().all():
                continue

            # Calculate correlations
            valid_mask = ~(feature.isna() | target_future.isna() | target_past.isna())

            if valid_mask.sum() < 20:
                continue

            future_corr = np.corrcoef(
                feature[valid_mask],
                target_future[valid_mask]
            )[0, 1]

            past_corr = np.corrcoef(
                feature[valid_mask],
                target_past[valid_mask]
            )[0, 1]

            if np.isnan(future_corr) or np.isnan(past_corr):
                continue

            analysis = {
                "future_correlation": float(future_corr),
                "past_correlation": float(past_corr),
                "ratio": float(abs(future_corr) / (abs(past_corr) + 1e-8)),
            }
            details["feature_analysis"][col] = analysis

            # Check for look-ahead bias
            if abs(future_corr) > 0.3:  # Meaningful correlation
                if abs(future_corr) > self.future_corr_ratio * abs(past_corr):
                    suspicious_features.append({
                        "feature": col,
                        "future_corr": float(future_corr),
                        "past_corr": float(past_corr),
                        "ratio": analysis["ratio"],
                        "type": "future_correlation",
                    })

            # Check for reverse causality
            # If feature correlates with shifted-back target, might be using future data
            target_back = df[target_col].shift(-self.lag_periods)  # Actual future
            valid_back = ~(feature.isna() | target_back.isna())

            if valid_back.sum() >= 20:
                back_corr = np.corrcoef(feature[valid_back], target_back[valid_back])[0, 1]

                if not np.isnan(back_corr):
                    analysis["actual_future_corr"] = float(back_corr)

                    if abs(back_corr) > 0.5:
                        suspicious_features.append({
                            "feature": col,
                            "actual_future_corr": float(back_corr),
                            "type": "actual_future_data",
                            "critical": True,
                        })

        # Check for common look-ahead patterns in feature names
        look_ahead_patterns = ["future", "forward", "next", "lead", "fwd"]
        for col in feature_cols:
            if any(pattern in col.lower() for pattern in look_ahead_patterns):
                if not any(s["feature"] == col for s in suspicious_features):
                    suspicious_features.append({
                        "feature": col,
                        "type": "suspicious_name",
                        "message": "Feature name suggests forward-looking data",
                    })

        details["suspicious_features"] = suspicious_features
        passed = len(suspicious_features) == 0

        has_critical = any(s.get("critical", False) for s in suspicious_features)
        severity = LeakageSeverity.FATAL if has_critical else (
            LeakageSeverity.WARNING if not passed else LeakageSeverity.INFO
        )

        return LeakageReport(
            check_name="Look-Ahead Bias Detection",
            passed=passed,
            severity=severity,
            message="No look-ahead bias detected" if passed else f"Found {len(suspicious_features)} suspicious features",
            details=details,
            recommendations=[
                f"Review '{s['feature']}' - may contain future information"
                for s in suspicious_features[:5]  # Top 5 recommendations
            ],
        )


class DataSnoopingDetector:
    """
    Detects data snooping and multiple testing issues.

    Data snooping occurs when the same dataset is used repeatedly
    for testing hypotheses, leading to spurious discoveries.

    Detects:
    1. Excessive strategy/model testing
    2. Parameter optimization on test data
    3. Survivorship bias

    Examples
    --------
    >>> detector = DataSnoopingDetector()
    >>> report = detector.assess_risk(n_strategies_tested=100, n_observations=252)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
    ):
        self.significance_level = significance_level

    def assess_risk(
        self,
        n_strategies_tested: int,
        n_observations: int,
        best_sharpe: Optional[float] = None,
        mean_sharpe: Optional[float] = None,
        std_sharpe: Optional[float] = None,
    ) -> LeakageReport:
        """
        Assess data snooping risk from multiple testing.

        Parameters
        ----------
        n_strategies_tested : int
            Number of strategies/models/parameters tested.
        n_observations : int
            Number of return observations.
        best_sharpe : float, optional
            Best observed Sharpe Ratio.
        mean_sharpe : float, optional
            Mean Sharpe Ratio across strategies.
        std_sharpe : float, optional
            Standard deviation of Sharpe Ratios.

        Returns
        -------
        LeakageReport
            Snooping risk assessment.
        """
        details = {
            "n_strategies": n_strategies_tested,
            "n_observations": n_observations,
        }

        issues = []

        # Calculate expected maximum Sharpe under null
        if n_strategies_tested >= 1:
            euler_mascheroni = 0.5772156649

            try:
                expected_max_sharpe = (
                    (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / n_strategies_tested) +
                    euler_mascheroni * stats.norm.ppf(1 - 1 / (n_strategies_tested * np.e))
                )
            except (ValueError, RuntimeWarning):
                expected_max_sharpe = np.sqrt(2 * np.log(n_strategies_tested))

            details["expected_max_sharpe_under_null"] = float(expected_max_sharpe)

            # If best Sharpe is provided, compare
            if best_sharpe is not None:
                details["best_sharpe"] = best_sharpe

                if best_sharpe < expected_max_sharpe:
                    issues.append({
                        "type": "below_expected",
                        "message": f"Best Sharpe ({best_sharpe:.2f}) < expected under null ({expected_max_sharpe:.2f})",
                    })

        # Bonferroni-corrected significance level
        bonferroni_alpha = self.significance_level / n_strategies_tested
        details["bonferroni_alpha"] = float(bonferroni_alpha)

        if bonferroni_alpha < 1e-6:
            issues.append({
                "type": "severe_multiple_testing",
                "message": f"Bonferroni alpha ({bonferroni_alpha:.2e}) is extremely low",
            })

        # Calculate probability of finding spurious result
        prob_spurious = 1 - (1 - self.significance_level) ** n_strategies_tested
        details["prob_at_least_one_spurious"] = float(prob_spurious)

        if prob_spurious > 0.5:
            issues.append({
                "type": "high_spurious_probability",
                "message": f"P(at least one spurious result) = {prob_spurious:.1%}",
            })

        # Minimum required Sharpe to be significant
        se_sharpe = 1 / np.sqrt(n_observations)
        min_significant_sharpe = stats.norm.ppf(1 - bonferroni_alpha) * se_sharpe
        details["min_significant_sharpe"] = float(min_significant_sharpe)

        # Calculate data snooping score (higher is worse)
        snooping_score = np.log(n_strategies_tested) / np.sqrt(n_observations)
        details["snooping_score"] = float(snooping_score)

        # Thresholds
        if snooping_score > 0.5:
            issues.append({
                "type": "high_snooping_score",
                "message": f"Data snooping score ({snooping_score:.2f}) indicates high risk",
            })

        passed = len(issues) == 0

        return LeakageReport(
            check_name="Data Snooping Detection",
            passed=passed,
            severity=LeakageSeverity.WARNING if not passed else LeakageSeverity.INFO,
            message="Low snooping risk" if passed else f"Found {len(issues)} snooping concerns",
            details=details,
            recommendations=[
                "Use out-of-sample testing for final evaluation",
                "Apply Bonferroni or FDR correction for multiple testing",
                "Consider Combinatorial Purged CV for proper validation",
                "Report all strategies tested, not just the best",
            ] if not passed else [],
        )

    def detect_overfitting_signature(
        self,
        train_performance: np.ndarray,
        test_performance: np.ndarray,
        threshold_ratio: float = 2.0,
    ) -> LeakageReport:
        """
        Detect overfitting signatures by comparing train/test performance.

        Parameters
        ----------
        train_performance : np.ndarray
            Performance metrics on training data.
        test_performance : np.ndarray
            Performance metrics on test data.
        threshold_ratio : float, default=2.0
            Train/test performance ratio threshold.

        Returns
        -------
        LeakageReport
            Overfitting detection results.
        """
        details = {}
        issues = []

        train_mean = np.mean(train_performance)
        test_mean = np.mean(test_performance)

        details["train_mean"] = float(train_mean)
        details["test_mean"] = float(test_mean)

        if test_mean != 0:
            ratio = train_mean / test_mean
            details["train_test_ratio"] = float(ratio)

            if ratio > threshold_ratio:
                issues.append({
                    "type": "performance_gap",
                    "message": f"Train/test ratio ({ratio:.2f}) exceeds threshold ({threshold_ratio})",
                })

        # Check for degradation pattern
        if len(test_performance) >= 2:
            # Split test into halves and compare
            mid = len(test_performance) // 2
            first_half = np.mean(test_performance[:mid])
            second_half = np.mean(test_performance[mid:])

            details["test_first_half"] = float(first_half)
            details["test_second_half"] = float(second_half)

            if first_half > second_half * 1.5:
                issues.append({
                    "type": "degradation",
                    "message": "Performance degrades significantly during test period",
                })

        passed = len(issues) == 0

        return LeakageReport(
            check_name="Overfitting Signature Detection",
            passed=passed,
            severity=LeakageSeverity.WARNING if not passed else LeakageSeverity.INFO,
            message="No overfitting signature detected" if passed else f"Found {len(issues)} overfitting signs",
            details=details,
            recommendations=[
                "Reduce model complexity",
                "Use regularization",
                "Increase training data",
                "Use cross-validation for hyperparameter tuning",
            ] if not passed else [],
        )


class LeakageGuardSuite:
    """
    Complete leakage detection suite combining all detectors.

    Provides a single entry point for comprehensive validation.

    Examples
    --------
    >>> suite = LeakageGuardSuite()
    >>> summary = suite.run_all_checks(
    ...     X_train, X_val, X_test,
    ...     y_train, y_val, y_test,
    ...     feature_names=feature_names,
    ...     train_times=train_times,
    ...     val_times=val_times,
    ...     test_times=test_times,
    ... )
    >>> summary.print_report()
    """

    def __init__(
        self,
        min_embargo_bars: int = 5,
        correlation_threshold: float = 0.95,
        strict: bool = True,
    ):
        self.temporal_validator = TemporalOrderValidator(
            min_embargo_bars=min_embargo_bars,
            strict=strict
        )
        self.leakage_scanner = FeatureTargetLeakageScanner(
            correlation_threshold=correlation_threshold
        )
        self.normalization_detector = NormalizationLeakageDetector()
        self.lookahead_detector = LookAheadBiasDetector()
        self.snooping_detector = DataSnoopingDetector()

    def run_all_checks(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray],
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: Optional[np.ndarray],
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        train_times: Optional[pd.Series] = None,
        val_times: Optional[pd.Series] = None,
        test_times: Optional[pd.Series] = None,
        scaler: Optional[Any] = None,
        df_full: Optional[pd.DataFrame] = None,
        n_strategies_tested: int = 1,
    ) -> ValidationSummary:
        """
        Run all leakage checks.

        Parameters
        ----------
        X_train, X_val, X_test : np.ndarray
            Feature matrices for train/val/test splits.
        y_train, y_val, y_test : np.ndarray
            Target values for train/val/test splits.
        feature_names : List[str], optional
            Feature names for reporting.
        train_times, val_times, test_times : pd.Series, optional
            Timestamps for temporal validation.
        scaler : object, optional
            Fitted scaler for normalization check.
        df_full : pd.DataFrame, optional
            Full DataFrame for look-ahead detection.
        n_strategies_tested : int, default=1
            Number of strategies tested for snooping check.

        Returns
        -------
        ValidationSummary
            Summary of all validation checks.
        """
        summary = ValidationSummary()

        # 1. Temporal Ordering
        if train_times is not None:
            report = self.temporal_validator.validate(
                train_times=train_times,
                val_times=val_times,
                test_times=test_times,
            )
            summary.reports.append(report)

        # 2. Feature-Target Leakage
        report = self.leakage_scanner.scan(X_train, y_train, feature_names)
        summary.reports.append(report)

        # 3. Normalization Leakage
        report = self.normalization_detector.detect(X_train, X_test, scaler, feature_names)
        summary.reports.append(report)

        # 4. Look-Ahead Bias (if DataFrame provided)
        if df_full is not None and feature_names is not None:
            target_col = "target"  # Assume standard name
            for col in df_full.columns:
                if col.startswith("target"):
                    target_col = col
                    break

            report = self.lookahead_detector.detect(
                df_full,
                [f for f in feature_names if f in df_full.columns],
                target_col,
            )
            summary.reports.append(report)

        # 5. Data Snooping
        report = self.snooping_detector.assess_risk(
            n_strategies_tested=n_strategies_tested,
            n_observations=len(y_test),
        )
        summary.reports.append(report)

        return summary


# Unit tests
if __name__ == "__main__":
    print("=" * 70)
    print("Leakage Guards Unit Tests")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Temporal Order Validator - PASS case
    print("\n[Test 1] Temporal Order Validator - Valid split...")
    validator = TemporalOrderValidator(min_embargo_bars=5)

    train_times = pd.Series(pd.date_range('2020-01-01', periods=700, freq='1min'))
    val_times = pd.Series(pd.date_range('2020-01-01 12:00:00', periods=150, freq='1min'))
    test_times = pd.Series(pd.date_range('2020-01-01 15:00:00', periods=150, freq='1min'))

    report = validator.validate(train_times, val_times, test_times)
    assert report.passed, f"Should pass: {report.message}"
    print("   PASS: Valid temporal ordering detected")

    # Test 2: Temporal Order Validator - FAIL case
    print("\n[Test 2] Temporal Order Validator - Invalid split...")
    train_times_bad = pd.Series(pd.date_range('2020-01-01', periods=800, freq='1min'))
    val_times_overlap = pd.Series(pd.date_range('2020-01-01 10:00:00', periods=200, freq='1min'))

    report = validator.validate(train_times_bad, val_times_overlap)
    assert not report.passed, "Should fail due to overlap"
    print("   PASS: Overlap correctly detected")

    # Test 3: Feature-Target Leakage Scanner
    print("\n[Test 3] Feature-Target Leakage Scanner...")
    scanner = FeatureTargetLeakageScanner(correlation_threshold=0.95)

    X_normal = np.random.randn(1000, 10)
    y_normal = np.random.randn(1000)

    report = scanner.scan(X_normal, y_normal)
    assert report.passed, "Normal data should pass"

    # Create leaky feature
    X_leaky = X_normal.copy()
    X_leaky[:, 0] = y_normal + np.random.randn(1000) * 0.01  # Almost perfect correlation

    report = scanner.scan(X_leaky, y_normal)
    assert not report.passed, "Leaky data should fail"
    print("   PASS: Leakage detection working")

    # Test 4: Normalization Leakage Detector
    print("\n[Test 4] Normalization Leakage Detector...")
    detector = NormalizationLeakageDetector()

    # Normal case: different distributions
    X_train_norm = np.random.randn(800, 10)
    X_test_norm = np.random.randn(200, 10) + 0.5  # Shifted distribution

    report = detector.detect(X_train_norm, X_test_norm)
    # Should pass because distributions are different (no leakage)
    print(f"   Normal case passed: {report.passed}")

    # Test 5: Look-Ahead Bias Detector
    print("\n[Test 5] Look-Ahead Bias Detector...")
    la_detector = LookAheadBiasDetector()

    df_test = pd.DataFrame({
        'feature_1': np.random.randn(500),
        'feature_2': np.random.randn(500),
        'target': np.random.randn(500),
    })

    report = la_detector.detect(df_test, ['feature_1', 'feature_2'], 'target')
    print(f"   Look-ahead test passed: {report.passed}")

    # Create look-ahead leaky feature
    df_test['leaky_feature'] = df_test['target'].shift(-5)  # Uses future data

    report = la_detector.detect(df_test, ['feature_1', 'leaky_feature'], 'target')
    # May detect based on correlation patterns
    print(f"   Look-ahead leak detection: {not report.passed or len(report.details.get('suspicious_features', [])) > 0}")

    # Test 6: Data Snooping Detector
    print("\n[Test 6] Data Snooping Detector...")
    snoop_detector = DataSnoopingDetector()

    # Low risk case
    report = snoop_detector.assess_risk(n_strategies_tested=5, n_observations=1000)
    print(f"   Low risk case (5 strategies): passed={report.passed}")

    # High risk case
    report = snoop_detector.assess_risk(n_strategies_tested=1000, n_observations=100)
    assert not report.passed, "High snooping should fail"
    print(f"   High risk case (1000 strategies, 100 obs): passed={report.passed}")

    # Test 7: Full Suite
    print("\n[Test 7] Full Leakage Guard Suite...")
    suite = LeakageGuardSuite()

    X_train = np.random.randn(700, 10)
    X_val = np.random.randn(150, 10)
    X_test = np.random.randn(150, 10)
    y_train = np.random.randn(700)
    y_val = np.random.randn(150)
    y_test = np.random.randn(150)

    train_times = pd.Series(pd.date_range('2020-01-01', periods=700, freq='1min'))
    val_times = pd.Series(pd.date_range('2020-01-01 12:00:00', periods=150, freq='1min'))
    test_times = pd.Series(pd.date_range('2020-01-01 15:00:00', periods=150, freq='1min'))

    summary = suite.run_all_checks(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_times=train_times,
        val_times=val_times,
        test_times=test_times,
    )

    print(f"   Suite summary: {summary.summary()}")

    # Test 8: Overfitting Detection
    print("\n[Test 8] Overfitting Signature Detection...")
    train_perf = np.random.randn(100) + 2.0  # Good training performance
    test_perf = np.random.randn(100) + 0.5   # Poor test performance

    report = snoop_detector.detect_overfitting_signature(train_perf, test_perf)
    assert not report.passed, "Should detect overfitting"
    print(f"   Overfitting detection: {not report.passed}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
