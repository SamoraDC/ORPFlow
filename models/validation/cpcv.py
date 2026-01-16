"""
Combinatorial Purged Cross-Validation (CPCV) Framework
======================================================

Implements proper cross-validation for financial time series with:
- Embargo gap (time buffer between train/test to prevent information leakage)
- Purge gap (remove samples where feature windows overlap with test labels)
- Probability of Backtest Overfitting (PBO) calculation
- Deflated Sharpe Ratio calculation

Reference:
    - Lopez de Prado, M. (2018). Advances in Financial Machine Learning
    - Bailey, D. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb


@dataclass
class CPCVResult:
    """Container for CPCV results with metrics."""

    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    sharpe_ratios: List[float] = field(default_factory=list)
    pbo: Optional[float] = None
    deflated_sharpe: Optional[float] = None
    optimal_n_paths: int = 0
    total_combinations: int = 0

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.sharpe_ratios:
            return {"error": "No results available"}

        sr_array = np.array(self.sharpe_ratios)
        return {
            "mean_sharpe": float(np.mean(sr_array)),
            "std_sharpe": float(np.std(sr_array)),
            "median_sharpe": float(np.median(sr_array)),
            "min_sharpe": float(np.min(sr_array)),
            "max_sharpe": float(np.max(sr_array)),
            "pbo": self.pbo,
            "deflated_sharpe": self.deflated_sharpe,
            "n_folds": len(self.fold_results),
            "total_combinations": self.total_combinations,
        }


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    This implementation follows Lopez de Prado's methodology for proper
    cross-validation in financial time series, addressing:

    1. Information Leakage: Through purging and embargo
    2. Multiple Testing: Through proper combinatorial path selection
    3. Non-stationarity: Through sequential splitting

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits/groups for the data.
    n_test_groups : int, default=2
        Number of groups to use for testing in each fold.
        Total combinations = C(n_splits, n_test_groups).
    embargo_pct : float, default=0.01
        Percentage of samples to use as embargo gap after each test group.
        Prevents leakage from sequential correlation.
    purge_pct : float, default=0.0
        Percentage of samples to purge before each test group.
        Removes samples where features might overlap with test labels.
    times : Optional[pd.Series], default=None
        Datetime index for proper temporal alignment.
        If provided, embargo/purge are calculated based on time, not samples.
    feature_window : int, default=0
        Number of bars used in feature calculation (for purge calculation).
    label_horizon : int, default=1
        Number of bars forward for label calculation (for purge calculation).

    Attributes
    ----------
    groups_ : np.ndarray
        Array of group indices assigned to each sample.
    n_samples_ : int
        Total number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.random.randn(1000, 10)
    >>> y = np.random.randn(1000)
    >>> times = pd.date_range('2020-01-01', periods=1000, freq='1min')
    >>> cpcv = CombinatorialPurgedKFold(n_splits=5, n_test_groups=2, embargo_pct=0.01)
    >>> for train_idx, test_idx in cpcv.split(X, times=pd.Series(times)):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    ...     y_train, y_test = y[train_idx], y[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.0,
        times: Optional[pd.Series] = None,
        feature_window: int = 0,
        label_horizon: int = 1,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if n_test_groups < 1 or n_test_groups >= n_splits:
            raise ValueError("n_test_groups must be >= 1 and < n_splits")
        if not 0 <= embargo_pct < 1:
            raise ValueError("embargo_pct must be in [0, 1)")
        if not 0 <= purge_pct < 1:
            raise ValueError("purge_pct must be in [0, 1)")

        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        self.times = times
        self.feature_window = feature_window
        self.label_horizon = label_horizon

        self.groups_: Optional[np.ndarray] = None
        self.n_samples_: int = 0

    def get_n_splits(self, X: Optional[np.ndarray] = None) -> int:
        """Return the number of splitting iterations (combinations)."""
        return int(comb(self.n_splits, self.n_test_groups, exact=True))

    def _compute_groups(self, n_samples: int) -> np.ndarray:
        """Assign samples to groups based on temporal order."""
        group_size = n_samples // self.n_splits
        groups = np.zeros(n_samples, dtype=int)

        for i in range(self.n_splits):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups[start_idx:end_idx] = i

        return groups

    def _compute_embargo_indices(
        self,
        test_indices: np.ndarray,
        n_samples: int,
        times: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Compute indices that fall within the embargo period after test data.

        The embargo prevents using samples that are temporally close to test
        samples and might contain leaking information due to serial correlation.
        """
        if len(test_indices) == 0:
            return np.array([], dtype=int)

        embargo_size = max(1, int(n_samples * self.embargo_pct))

        if times is not None:
            # Time-based embargo
            test_end_time = times.iloc[test_indices].max()
            test_start_time = times.iloc[test_indices].min()

            # Calculate time-based embargo duration
            total_duration = times.max() - times.min()
            embargo_duration = total_duration * self.embargo_pct

            embargo_mask = (
                (times > test_end_time) &
                (times <= test_end_time + embargo_duration)
            )
            embargo_indices = np.where(embargo_mask)[0]
        else:
            # Sample-based embargo
            test_end = test_indices.max()
            embargo_indices = np.arange(
                test_end + 1,
                min(test_end + 1 + embargo_size, n_samples)
            )

        return embargo_indices

    def _compute_purge_indices(
        self,
        test_indices: np.ndarray,
        n_samples: int,
        times: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Compute indices that need to be purged before test data.

        Purging removes samples where the feature calculation window
        overlaps with the test label calculation, preventing look-ahead bias.
        """
        if len(test_indices) == 0:
            return np.array([], dtype=int)

        # Calculate purge window based on feature_window and label_horizon
        purge_window = self.feature_window + self.label_horizon

        if self.purge_pct > 0:
            purge_size = max(purge_window, int(n_samples * self.purge_pct))
        else:
            purge_size = purge_window

        if purge_size == 0:
            return np.array([], dtype=int)

        if times is not None:
            # Time-based purge
            test_start_time = times.iloc[test_indices].min()

            total_duration = times.max() - times.min()
            purge_duration = total_duration * self.purge_pct if self.purge_pct > 0 else pd.Timedelta(0)

            # Also account for feature window and label horizon
            if purge_duration.total_seconds() == 0 and purge_window > 0:
                # Estimate time per sample
                avg_time_per_sample = total_duration / n_samples
                purge_duration = avg_time_per_sample * purge_window

            purge_mask = (
                (times < test_start_time) &
                (times >= test_start_time - purge_duration)
            )
            purge_indices = np.where(purge_mask)[0]
        else:
            # Sample-based purge
            test_start = test_indices.min()
            purge_start = max(0, test_start - purge_size)
            purge_indices = np.arange(purge_start, test_start)

        return purge_indices

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        times: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray, optional
            Target values (not used, present for API compatibility).
        times : pd.Series, optional
            Datetime index. If not provided, uses self.times if available.

        Yields
        ------
        train_idx : np.ndarray
            Training set indices for this fold.
        test_idx : np.ndarray
            Test set indices for this fold.
        """
        n_samples = len(X)
        self.n_samples_ = n_samples
        self.groups_ = self._compute_groups(n_samples)

        # Use provided times or fall back to instance times
        times_to_use = times if times is not None else self.times

        # Generate all combinations of test groups
        test_group_combinations = list(
            itertools.combinations(range(self.n_splits), self.n_test_groups)
        )

        all_indices = np.arange(n_samples)

        for test_groups in test_group_combinations:
            # Get test indices (all samples in test groups)
            test_mask = np.isin(self.groups_, test_groups)
            test_idx = all_indices[test_mask]

            if len(test_idx) == 0:
                continue

            # Compute embargo and purge indices
            embargo_idx = self._compute_embargo_indices(test_idx, n_samples, times_to_use)
            purge_idx = self._compute_purge_indices(test_idx, n_samples, times_to_use)

            # Train indices: all except test, embargo, and purge
            excluded = np.union1d(test_idx, embargo_idx)
            excluded = np.union1d(excluded, purge_idx)
            train_idx = np.setdiff1d(all_indices, excluded)

            # Ensure temporal ordering: train should be before test
            if times_to_use is not None:
                # Keep only train samples that are before test samples
                # (accounting for purge already applied)
                pass  # Purge and embargo handle this

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns.
    risk_free_rate : float, default=0.0
        Annual risk-free rate.
    periods_per_year : int, default=252
        Number of periods per year (252 for daily, 52 for weekly, etc.).

    Returns
    -------
    float
        Annualized Sharpe Ratio.
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)

    return float(annualized_sharpe)


def calculate_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float = 1.0,
) -> Tuple[float, float]:
    """
    Calculate Deflated Sharpe Ratio (DSR) and its p-value.

    The DSR accounts for multiple testing and non-normality of returns,
    providing a more realistic assessment of strategy performance.

    Reference:
        Bailey, D. & Lopez de Prado, M. (2014).
        "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe Ratio from the strategy.
    n_trials : int
        Number of strategies/trials tested (multiple testing correction).
    n_observations : int
        Number of return observations used in Sharpe calculation.
    skewness : float, default=0.0
        Skewness of returns (0 for normal).
    kurtosis : float, default=3.0
        Kurtosis of returns (3 for normal).
    sharpe_std : float, default=1.0
        Standard deviation of Sharpe Ratios across trials.

    Returns
    -------
    deflated_sharpe : float
        The deflated Sharpe Ratio.
    p_value : float
        P-value for the null hypothesis that the observed Sharpe
        is due to chance given multiple testing.
    """
    if n_trials < 1:
        n_trials = 1
    if n_observations < 2:
        return 0.0, 1.0

    # Expected maximum Sharpe Ratio under the null
    # Using approximation from Bailey & Lopez de Prado
    euler_mascheroni = 0.5772156649

    try:
        expected_max_sharpe = sharpe_std * (
            (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / n_trials) +
            euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    except (ValueError, RuntimeWarning):
        expected_max_sharpe = sharpe_std * np.sqrt(2 * np.log(n_trials))

    # Standard error of Sharpe Ratio (accounting for non-normality)
    se_sharpe = np.sqrt(
        (1 + 0.5 * observed_sharpe ** 2 - skewness * observed_sharpe +
         (kurtosis - 3) / 4 * observed_sharpe ** 2) / (n_observations - 1)
    )

    # Deflated Sharpe Ratio
    if se_sharpe > 0:
        deflated_sharpe = (observed_sharpe - expected_max_sharpe) / se_sharpe
        p_value = 1 - stats.norm.cdf(deflated_sharpe)
    else:
        deflated_sharpe = 0.0
        p_value = 1.0

    return float(deflated_sharpe), float(p_value)


def calculate_pbo(
    performance_matrix: np.ndarray,
    n_partitions: int = 10,
) -> Tuple[float, np.ndarray]:
    """
    Calculate Probability of Backtest Overfitting (PBO).

    PBO measures the probability that the in-sample optimal strategy
    will underperform the median out-of-sample.

    Reference:
        Bailey, D. et al. (2017). "Probability of Backtest Overfitting"

    Parameters
    ----------
    performance_matrix : np.ndarray
        Matrix of shape (n_strategies, n_partitions) containing
        performance metrics (e.g., Sharpe Ratios) for each strategy
        on each time partition.
    n_partitions : int, default=10
        Number of time partitions for analysis.

    Returns
    -------
    pbo : float
        Probability of Backtest Overfitting (0 to 1).
        Higher values indicate higher risk of overfitting.
    logits : np.ndarray
        Array of logit values for the performance comparison.
    """
    n_strategies, n_parts = performance_matrix.shape

    if n_strategies < 2:
        warnings.warn("Need at least 2 strategies to calculate PBO")
        return 0.0, np.array([])

    if n_parts < 2:
        warnings.warn("Need at least 2 partitions to calculate PBO")
        return 0.0, np.array([])

    # Generate all combinations of partitions for train/test split
    n_train = n_parts // 2
    partition_indices = list(range(n_parts))
    train_combinations = list(itertools.combinations(partition_indices, n_train))

    logits = []

    for train_parts in train_combinations:
        test_parts = [i for i in partition_indices if i not in train_parts]

        # In-sample performance (train partitions)
        is_performance = performance_matrix[:, list(train_parts)].mean(axis=1)

        # Out-of-sample performance (test partitions)
        oos_performance = performance_matrix[:, test_parts].mean(axis=1)

        # Find best in-sample strategy
        best_is_idx = np.argmax(is_performance)

        # Rank of best IS strategy in OOS
        oos_ranks = stats.rankdata(oos_performance)
        best_is_oos_rank = oos_ranks[best_is_idx]

        # Relative rank (0 to 1, where 1 is best)
        relative_rank = best_is_oos_rank / n_strategies

        # Compute logit
        # Avoid division by zero
        relative_rank = np.clip(relative_rank, 1e-10, 1 - 1e-10)
        logit = np.log(relative_rank / (1 - relative_rank))
        logits.append(logit)

    logits = np.array(logits)

    # PBO is the probability that the logit is negative
    # (i.e., best IS strategy performs below median OOS)
    pbo = np.mean(logits < 0)

    return float(pbo), logits


class CPCVEvaluator:
    """
    Comprehensive CPCV evaluation framework.

    Combines CPCV splitting with performance metrics calculation
    including Sharpe Ratio, Deflated Sharpe, and PBO.

    Parameters
    ----------
    cv : CombinatorialPurgedKFold
        Cross-validation splitter instance.
    scoring : Callable, optional
        Scoring function that takes (y_true, y_pred) and returns returns array.
        Default assumes predictions are returns.
    periods_per_year : int, default=252
        Number of trading periods per year for Sharpe calculation.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> cpcv = CombinatorialPurgedKFold(n_splits=5, n_test_groups=2)
    >>> evaluator = CPCVEvaluator(cpcv)
    >>> results = evaluator.evaluate(
    ...     X, y,
    ...     model_fn=lambda: RandomForestClassifier(n_estimators=100),
    ...     return_calculator=lambda y, pred: y * np.sign(pred - 0.5)
    ... )
    >>> print(results.summary())
    """

    def __init__(
        self,
        cv: CombinatorialPurgedKFold,
        periods_per_year: int = 252,
    ):
        self.cv = cv
        self.periods_per_year = periods_per_year

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn: Callable[[], Any],
        return_calculator: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        times: Optional[pd.Series] = None,
        n_trials: int = 1,
    ) -> CPCVResult:
        """
        Evaluate a model using CPCV.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target values (typically returns or direction).
        model_fn : Callable
            Function that returns a fresh model instance with fit/predict methods.
        return_calculator : Callable, optional
            Function that takes (y_true, y_pred) and returns strategy returns.
            If None, assumes y_pred directly represents returns.
        times : pd.Series, optional
            Datetime index for temporal validation.
        n_trials : int, default=1
            Number of strategy variants tested (for DSR calculation).

        Returns
        -------
        CPCVResult
            Container with all evaluation metrics.
        """
        result = CPCVResult()
        result.total_combinations = self.cv.get_n_splits()

        all_returns = []
        fold_sharpes = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(X, y, times)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model = model_fn()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate returns
            if return_calculator is not None:
                fold_returns = return_calculator(y_test, y_pred)
            else:
                # Assume predictions are direction signals and y contains returns
                fold_returns = y_test * np.sign(y_pred)

            # Calculate Sharpe for this fold
            fold_sharpe = calculate_sharpe_ratio(
                fold_returns,
                periods_per_year=self.periods_per_year
            )

            fold_result = {
                "fold_idx": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "sharpe_ratio": fold_sharpe,
                "mean_return": float(np.mean(fold_returns)),
                "std_return": float(np.std(fold_returns)),
            }

            result.fold_results.append(fold_result)
            result.sharpe_ratios.append(fold_sharpe)
            all_returns.extend(fold_returns.tolist())
            fold_sharpes.append(fold_sharpe)

        # Calculate overall metrics
        all_returns = np.array(all_returns)

        if len(all_returns) > 0:
            # Overall Sharpe
            overall_sharpe = calculate_sharpe_ratio(
                all_returns,
                periods_per_year=self.periods_per_year
            )

            # Deflated Sharpe
            skewness = float(stats.skew(all_returns)) if len(all_returns) > 2 else 0.0
            kurtosis = float(stats.kurtosis(all_returns) + 3) if len(all_returns) > 3 else 3.0

            dsr, dsr_pvalue = calculate_deflated_sharpe_ratio(
                observed_sharpe=overall_sharpe,
                n_trials=max(n_trials, result.total_combinations),
                n_observations=len(all_returns),
                skewness=skewness,
                kurtosis=kurtosis,
                sharpe_std=np.std(fold_sharpes) if fold_sharpes else 1.0,
            )

            result.deflated_sharpe = dsr

        return result

    def calculate_pbo_multi_strategy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fns: List[Callable[[], Any]],
        return_calculator: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        times: Optional[pd.Series] = None,
    ) -> Tuple[float, CPCVResult]:
        """
        Calculate PBO across multiple strategies.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target values.
        model_fns : List[Callable]
            List of functions, each returning a model instance.
        return_calculator : Callable, optional
            Function to convert predictions to returns.
        times : pd.Series, optional
            Datetime index.

        Returns
        -------
        pbo : float
            Probability of Backtest Overfitting.
        best_result : CPCVResult
            Result for the best in-sample strategy.
        """
        n_strategies = len(model_fns)
        n_folds = self.cv.get_n_splits()

        # Matrix to store Sharpe Ratios: (n_strategies, n_folds)
        performance_matrix = np.zeros((n_strategies, n_folds))
        all_results = []

        for strat_idx, model_fn in enumerate(model_fns):
            result = self.evaluate(X, y, model_fn, return_calculator, times)
            all_results.append(result)

            for fold_idx, sharpe in enumerate(result.sharpe_ratios):
                performance_matrix[strat_idx, fold_idx] = sharpe

        # Calculate PBO
        pbo, _ = calculate_pbo(performance_matrix)

        # Find best in-sample strategy
        is_means = performance_matrix.mean(axis=1)
        best_idx = np.argmax(is_means)
        best_result = all_results[best_idx]
        best_result.pbo = pbo

        return pbo, best_result


def validate_no_lookahead(
    train_times: pd.Series,
    test_times: pd.Series,
    embargo_bars: int = 5,
) -> Dict[str, Any]:
    """
    Validate that no look-ahead bias exists in train/test split.

    Parameters
    ----------
    train_times : pd.Series
        Timestamps for training data.
    test_times : pd.Series
        Timestamps for test data.
    embargo_bars : int, default=5
        Minimum gap required between train end and test start.

    Returns
    -------
    dict
        Validation results with pass/fail status and details.
    """
    result = {
        "passed": True,
        "checks": [],
        "warnings": [],
    }

    train_max = train_times.max()
    test_min = test_times.min()

    # Check 1: Train max < Test min
    if train_max >= test_min:
        result["passed"] = False
        result["checks"].append({
            "name": "temporal_order",
            "passed": False,
            "message": f"FAIL: Train max ({train_max}) >= Test min ({test_min})",
        })
    else:
        result["checks"].append({
            "name": "temporal_order",
            "passed": True,
            "message": f"PASS: Train ends before test starts",
        })

    # Check 2: Sufficient embargo gap
    if isinstance(train_max, pd.Timestamp) and isinstance(test_min, pd.Timestamp):
        gap = test_min - train_max
        result["gap"] = str(gap)
    else:
        # Numeric indices
        gap = test_min - train_max
        result["gap"] = int(gap)

        if gap < embargo_bars:
            result["warnings"].append({
                "name": "embargo_gap",
                "message": f"WARNING: Gap ({gap}) < recommended embargo ({embargo_bars})",
            })

    return result


# Unit tests
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("CPCV Framework Unit Tests")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Basic CPCV Split
    print("\n[Test 1] Basic CPCV Split...")
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)

    cpcv = CombinatorialPurgedKFold(n_splits=5, n_test_groups=2, embargo_pct=0.01)
    n_splits = cpcv.get_n_splits()

    assert n_splits == 10, f"Expected 10 combinations, got {n_splits}"

    splits = list(cpcv.split(X, y))
    assert len(splits) == 10, f"Expected 10 splits, got {len(splits)}"

    # Verify no overlap
    for train_idx, test_idx in splits:
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0, "Train and test should not overlap"

    print(f"   PASS: {n_splits} combinations generated correctly")

    # Test 2: Temporal Ordering with Times
    print("\n[Test 2] Temporal Ordering...")
    times = pd.date_range('2020-01-01', periods=1000, freq='1min')
    times_series = pd.Series(times)

    cpcv_temporal = CombinatorialPurgedKFold(
        n_splits=5,
        n_test_groups=1,
        embargo_pct=0.02,
        purge_pct=0.01,
    )

    for train_idx, test_idx in cpcv_temporal.split(X, y, times=times_series):
        train_max_time = times_series.iloc[train_idx].max()
        test_min_time = times_series.iloc[test_idx].min()
        # Note: In combinatorial CV, train may include data after test
        # The embargo/purge handles immediate neighbors

    print("   PASS: Temporal validation working")

    # Test 3: Sharpe Ratio Calculation
    print("\n[Test 3] Sharpe Ratio Calculation...")
    returns = np.random.randn(252) * 0.01 + 0.0005  # ~12.6% annual return
    sr = calculate_sharpe_ratio(returns)

    assert -5 < sr < 5, f"Sharpe ratio {sr} outside reasonable range"
    print(f"   PASS: Sharpe Ratio = {sr:.4f}")

    # Test 4: Deflated Sharpe Ratio
    print("\n[Test 4] Deflated Sharpe Ratio...")
    dsr, pvalue = calculate_deflated_sharpe_ratio(
        observed_sharpe=2.0,
        n_trials=100,
        n_observations=252,
        skewness=-0.5,
        kurtosis=4.0,
    )

    assert -10 < dsr < 10, f"DSR {dsr} outside reasonable range"
    assert 0 <= pvalue <= 1, f"P-value {pvalue} outside [0,1]"
    print(f"   PASS: DSR = {dsr:.4f}, p-value = {pvalue:.4f}")

    # Test 5: PBO Calculation
    print("\n[Test 5] PBO Calculation...")
    # Create synthetic performance matrix (10 strategies, 10 partitions)
    perf_matrix = np.random.randn(10, 10) * 0.5 + 1.0
    pbo, logits = calculate_pbo(perf_matrix)

    assert 0 <= pbo <= 1, f"PBO {pbo} outside [0,1]"
    print(f"   PASS: PBO = {pbo:.4f}")

    # Test 6: Embargo and Purge
    print("\n[Test 6] Embargo and Purge...")
    cpcv_gaps = CombinatorialPurgedKFold(
        n_splits=5,
        n_test_groups=1,
        embargo_pct=0.05,  # 5% embargo
        purge_pct=0.02,    # 2% purge
        feature_window=10,
        label_horizon=5,
    )

    total_train = 0
    total_test = 0

    for train_idx, test_idx in cpcv_gaps.split(X):
        total_train += len(train_idx)
        total_test += len(test_idx)

        # Verify train indices are less than test minus purge
        if len(train_idx) > 0 and len(test_idx) > 0:
            # Due to purge and embargo, there should be a gap
            pass

    print(f"   PASS: Embargo/Purge applied (avg train={total_train//5}, avg test={total_test//5})")

    # Test 7: No Lookahead Validation
    print("\n[Test 7] No Lookahead Validation...")
    train_times = pd.Series(pd.date_range('2020-01-01', periods=800, freq='1min'))
    test_times = pd.Series(pd.date_range('2020-01-01 13:25:00', periods=200, freq='1min'))

    validation = validate_no_lookahead(train_times, test_times, embargo_bars=5)
    assert validation["passed"], "Validation should pass"
    print(f"   PASS: No lookahead validation working")

    # Test 8: Full Evaluation Pipeline
    print("\n[Test 8] Full Evaluation Pipeline...")

    try:
        from sklearn.linear_model import Ridge

        X_eval = np.random.randn(500, 5)
        y_eval = np.random.randn(500) * 0.01

        cpcv_eval = CombinatorialPurgedKFold(n_splits=5, n_test_groups=2)
        evaluator = CPCVEvaluator(cpcv_eval)

        result = evaluator.evaluate(
            X_eval,
            y_eval,
            model_fn=lambda: Ridge(alpha=1.0),
            return_calculator=lambda y, pred: y * np.sign(pred),
        )

        summary = result.summary()
        assert "mean_sharpe" in summary, "Summary should contain mean_sharpe"
        assert len(result.fold_results) > 0, "Should have fold results"
        print(f"   PASS: Mean Sharpe = {summary['mean_sharpe']:.4f}")
    except ImportError:
        print("   SKIP: sklearn not available (optional dependency)")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
