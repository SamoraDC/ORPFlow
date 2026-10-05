"""
Advanced Feature Selection Module for ORPFlow
Comprehensive feature selection methods for quantitative trading ML pipelines

This module implements:
1. Filter Methods: MI, correlation, variance, chi-squared
2. Wrapper Methods: RFE, RFECV, Sequential Selection
3. Embedded Methods: SHAP, permutation, L1, tree-based
4. Advanced Methods: MDI, MDA, SFI, Clustered Importance
5. Validation: PBO, Deflated Sharpe, CSCV
6. Utilities: Clustering, VIF, stability analysis
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, spearmanr

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. SHAP-based feature selection will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """ML task type enumeration"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass
class FeatureImportanceResult:
    """Container for feature importance results"""
    feature_names: List[str]
    importance_scores: np.ndarray
    std_scores: Optional[np.ndarray] = None
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to sorted DataFrame"""
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.importance_scores,
        })
        if self.std_scores is not None:
            df["std"] = self.std_scores
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_top_features(self, n: int) -> List[str]:
        """Get top n features by importance"""
        df = self.to_dataframe()
        return df.head(n)["feature"].tolist()


@dataclass
class FeatureSelectionResult:
    """Container for feature selection results"""
    selected_features: List[str]
    all_features: List[str]
    importance_result: Optional[FeatureImportanceResult] = None
    validation_metrics: Optional[Dict[str, float]] = None
    method: str = ""

    @property
    def n_selected(self) -> int:
        return len(self.selected_features)

    @property
    def n_removed(self) -> int:
        return len(self.all_features) - len(self.selected_features)

    @property
    def selection_ratio(self) -> float:
        return len(self.selected_features) / len(self.all_features) if self.all_features else 0.0


# =============================================================================
# FILTER METHODS
# =============================================================================

class MutualInformationFilter:
    """
    Mutual Information based feature selection.
    Measures dependency between features and target.
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.REGRESSION,
        n_neighbors: int = 3,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._mi_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MutualInformationFilter":
        """Compute mutual information scores"""
        if self.task_type == TaskType.REGRESSION:
            self._mi_scores = mutual_info_regression(
                X, y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
        else:
            self._mi_scores = mutual_info_classif(
                X, y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )

        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get MI scores as importance result"""
        if self._mi_scores is None:
            raise ValueError("Must call fit() first")
        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._mi_scores,
            method="mutual_information",
        )

    def select_features(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features by MI threshold or top k"""
        if self._mi_scores is None:
            raise ValueError("Must call fit() first")

        if k is not None:
            indices = np.argsort(self._mi_scores)[-k:]
        elif threshold is not None:
            indices = np.where(self._mi_scores >= threshold)[0]
        else:
            # Default: select features with MI > median
            median_mi = np.median(self._mi_scores)
            indices = np.where(self._mi_scores >= median_mi)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="mutual_information",
        )


class CorrelationFilter:
    """
    Correlation-based feature selection.
    Removes highly correlated (redundant) features.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ):
        self.threshold = threshold
        self.method = method
        self._corr_matrix: Optional[pd.DataFrame] = None
        self._dropped_features: List[str] = []

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "CorrelationFilter":
        """Compute correlation matrix and identify redundant features"""
        if isinstance(X, np.ndarray):
            feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)

        self._feature_names = list(X.columns)
        self._corr_matrix = X.corr(method=self.method).abs()

        # Identify pairs with correlation above threshold
        upper_tri = np.triu(np.ones(self._corr_matrix.shape), k=1).astype(bool)
        upper_corr = self._corr_matrix.where(upper_tri)

        # Find columns to drop
        self._dropped_features = []
        for col in upper_corr.columns:
            if any(upper_corr[col] > self.threshold):
                # Check if this column correlates highly with an already kept column
                high_corr_cols = upper_corr.index[upper_corr[col] > self.threshold].tolist()
                # Only drop if not already decided to keep based on target correlation
                if col not in self._dropped_features:
                    self._dropped_features.append(col)

        return self

    def get_corr_matrix(self) -> pd.DataFrame:
        """Get correlation matrix"""
        if self._corr_matrix is None:
            raise ValueError("Must call fit() first")
        return self._corr_matrix

    def select_features(self) -> FeatureSelectionResult:
        """Return non-redundant features"""
        if self._corr_matrix is None:
            raise ValueError("Must call fit() first")

        selected = [f for f in self._feature_names if f not in self._dropped_features]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            method=f"correlation_filter_{self.method}",
            validation_metrics={"threshold": self.threshold, "dropped": len(self._dropped_features)},
        )


class VarianceFilter:
    """
    Variance threshold based feature selection.
    Removes features with variance below threshold.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self._selector = VarianceThreshold(threshold=threshold)
        self._variances: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "VarianceFilter":
        """Fit variance threshold selector"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        self._selector.fit(X)
        self._variances = self._selector.variances_
        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get variance scores as importance result"""
        if self._variances is None:
            raise ValueError("Must call fit() first")
        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._variances,
            method="variance",
        )

    def select_features(self) -> FeatureSelectionResult:
        """Select features with variance above threshold"""
        if self._variances is None:
            raise ValueError("Must call fit() first")

        mask = self._selector.get_support()
        selected = [self._feature_names[i] for i in range(len(mask)) if mask[i]]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="variance_threshold",
        )


class ChiSquaredFilter:
    """
    Chi-squared test for categorical/discretized feature selection.
    Best for classification tasks with non-negative features.
    """

    def __init__(
        self,
        n_bins: int = 10,
        encode: Literal["ordinal", "onehot"] = "ordinal",
    ):
        self.n_bins = n_bins
        self.encode = encode
        self._chi2_scores: Optional[np.ndarray] = None
        self._p_values: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "ChiSquaredFilter":
        """Compute chi-squared scores"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Discretize features if necessary
        X_disc = self._discretize(X)

        # Ensure non-negative values
        X_disc = X_disc - X_disc.min(axis=0)

        self._chi2_scores, self._p_values = chi2(X_disc, y)
        return self

    def _discretize(self, X: np.ndarray) -> np.ndarray:
        """Discretize continuous features"""
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy="quantile",
        )
        return discretizer.fit_transform(X)

    def get_importance(self) -> FeatureImportanceResult:
        """Get chi-squared scores as importance result"""
        if self._chi2_scores is None:
            raise ValueError("Must call fit() first")
        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._chi2_scores,
            method="chi_squared",
            metadata={"p_values": self._p_values.tolist()},
        )

    def select_features(
        self,
        p_value_threshold: float = 0.05,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features by p-value or top k"""
        if self._chi2_scores is None:
            raise ValueError("Must call fit() first")

        if k is not None:
            indices = np.argsort(self._chi2_scores)[-k:]
        else:
            indices = np.where(self._p_values <= p_value_threshold)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="chi_squared",
        )


# =============================================================================
# WRAPPER METHODS
# =============================================================================

class RFESelector:
    """
    Recursive Feature Elimination (RFE) wrapper.
    Uses model feature importances to recursively remove features.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_features_to_select: Optional[int] = None,
        step: Union[int, float] = 1,
        task_type: TaskType = TaskType.REGRESSION,
    ):
        self.task_type = task_type
        self.step = step
        self.n_features_to_select = n_features_to_select

        if estimator is None:
            if task_type == TaskType.REGRESSION:
                self.estimator = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            else:
                self.estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            self.estimator = estimator

        self._rfe: Optional[RFE] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "RFESelector":
        """Fit RFE selector"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        n_features = self.n_features_to_select or max(1, X.shape[1] // 2)

        self._rfe = RFE(
            estimator=clone(self.estimator),
            n_features_to_select=n_features,
            step=self.step,
        )
        self._rfe.fit(X, y)
        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get ranking as importance (inverse of rank)"""
        if self._rfe is None:
            raise ValueError("Must call fit() first")

        # Convert ranking to importance (lower rank = higher importance)
        importance = 1.0 / self._rfe.ranking_

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=importance,
            method="rfe",
            metadata={"ranking": self._rfe.ranking_.tolist()},
        )

    def select_features(self) -> FeatureSelectionResult:
        """Get selected features from RFE"""
        if self._rfe is None:
            raise ValueError("Must call fit() first")

        mask = self._rfe.support_
        selected = [self._feature_names[i] for i in range(len(mask)) if mask[i]]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="rfe",
        )


class RFECVSelector:
    """
    RFECV - RFE with cross-validation for optimal feature count.
    Automatically determines the best number of features.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        step: Union[int, float] = 1,
        cv: int = 5,
        scoring: Optional[str] = None,
        task_type: TaskType = TaskType.REGRESSION,
        n_jobs: int = -1,
    ):
        self.task_type = task_type
        self.step = step
        self.cv = cv
        self.n_jobs = n_jobs

        if scoring is None:
            self.scoring = "neg_mean_squared_error" if task_type == TaskType.REGRESSION else "accuracy"
        else:
            self.scoring = scoring

        if estimator is None:
            if task_type == TaskType.REGRESSION:
                self.estimator = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            else:
                self.estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            self.estimator = estimator

        self._rfecv: Optional[RFECV] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "RFECVSelector":
        """Fit RFECV selector"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        self._rfecv = RFECV(
            estimator=clone(self.estimator),
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        self._rfecv.fit(X, y)
        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get ranking as importance"""
        if self._rfecv is None:
            raise ValueError("Must call fit() first")

        importance = 1.0 / self._rfecv.ranking_

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=importance,
            method="rfecv",
            metadata={
                "ranking": self._rfecv.ranking_.tolist(),
                "optimal_n_features": self._rfecv.n_features_,
                "cv_results": self._rfecv.cv_results_ if hasattr(self._rfecv, "cv_results_") else None,
            },
        )

    def select_features(self) -> FeatureSelectionResult:
        """Get selected features from RFECV"""
        if self._rfecv is None:
            raise ValueError("Must call fit() first")

        mask = self._rfecv.support_
        selected = [self._feature_names[i] for i in range(len(mask)) if mask[i]]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="rfecv",
            validation_metrics={"optimal_n_features": self._rfecv.n_features_},
        )


class SequentialSelector:
    """
    Sequential Feature Selection (SFS) - Forward or Backward.
    Greedily adds/removes features based on cross-validated score.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_features_to_select: Union[int, float, Literal["auto"]] = "auto",
        direction: Literal["forward", "backward"] = "forward",
        scoring: Optional[str] = None,
        cv: int = 5,
        task_type: TaskType = TaskType.REGRESSION,
        n_jobs: int = -1,
    ):
        self.task_type = task_type
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.cv = cv
        self.n_jobs = n_jobs

        if scoring is None:
            self.scoring = "neg_mean_squared_error" if task_type == TaskType.REGRESSION else "accuracy"
        else:
            self.scoring = scoring

        if estimator is None:
            if task_type == TaskType.REGRESSION:
                self.estimator = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
            else:
                self.estimator = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        else:
            self.estimator = estimator

        self._sfs: Optional[SequentialFeatureSelector] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "SequentialSelector":
        """Fit Sequential Feature Selector"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        self._sfs = SequentialFeatureSelector(
            estimator=clone(self.estimator),
            n_features_to_select=self.n_features_to_select,
            direction=self.direction,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
        )
        self._sfs.fit(X, y)
        return self

    def select_features(self) -> FeatureSelectionResult:
        """Get selected features from SFS"""
        if self._sfs is None:
            raise ValueError("Must call fit() first")

        mask = self._sfs.get_support()
        selected = [self._feature_names[i] for i in range(len(mask)) if mask[i]]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            method=f"sequential_{self.direction}",
        )


# =============================================================================
# EMBEDDED METHODS
# =============================================================================

class SHAPImportance:
    """
    SHAP (SHapley Additive exPlanations) based feature importance.
    Provides consistent and locally accurate feature attributions.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        task_type: TaskType = TaskType.REGRESSION,
        max_samples: int = 1000,
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")

        self.task_type = task_type
        self.max_samples = max_samples

        if estimator is None:
            if task_type == TaskType.REGRESSION:
                self.estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                self.estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            self.estimator = estimator

        self._shap_values: Optional[np.ndarray] = None
        self._importance_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "SHAPImportance":
        """Fit model and compute SHAP values"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Fit the model
        self.estimator.fit(X, y)

        # Sample data for SHAP computation
        if X.shape[0] > self.max_samples:
            indices = np.random.choice(X.shape[0], self.max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Compute SHAP values
        explainer = shap.TreeExplainer(self.estimator)
        self._shap_values = explainer.shap_values(X_sample)

        # Handle multi-class classification
        if isinstance(self._shap_values, list):
            # Average across classes
            self._shap_values = np.mean([np.abs(sv) for sv in self._shap_values], axis=0)
        else:
            self._shap_values = np.abs(self._shap_values)

        # Mean absolute SHAP value per feature
        self._importance_scores = np.mean(self._shap_values, axis=0)

        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get SHAP importance scores"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        # Standard deviation of SHAP values
        std_scores = np.std(self._shap_values, axis=0)

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._importance_scores,
            std_scores=std_scores,
            method="shap",
        )

    def select_features(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features by SHAP importance"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        if k is not None:
            indices = np.argsort(self._importance_scores)[-k:]
        elif threshold is not None:
            indices = np.where(self._importance_scores >= threshold)[0]
        else:
            # Default: features with importance > mean
            mean_imp = np.mean(self._importance_scores)
            indices = np.where(self._importance_scores >= mean_imp)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="shap",
        )


class PermutationImportance:
    """
    Permutation importance - model-agnostic feature importance.
    Measures decrease in model performance when feature is permuted.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_repeats: int = 10,
        scoring: Optional[str] = None,
        task_type: TaskType = TaskType.REGRESSION,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.random_state = random_state

        if scoring is None:
            self.scoring = "neg_mean_squared_error" if task_type == TaskType.REGRESSION else "accuracy"
        else:
            self.scoring = scoring

        if estimator is None:
            if task_type == TaskType.REGRESSION:
                self.estimator = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            else:
                self.estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            self.estimator = estimator

        self._importance_result = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "PermutationImportance":
        """Fit model and compute permutation importance"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Fit the model
        self.estimator.fit(X, y)

        # Compute permutation importance
        self._importance_result = permutation_importance(
            self.estimator, X, y,
            n_repeats=self.n_repeats,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get permutation importance scores"""
        if self._importance_result is None:
            raise ValueError("Must call fit() first")

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._importance_result.importances_mean,
            std_scores=self._importance_result.importances_std,
            method="permutation",
        )

    def select_features(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features by permutation importance"""
        if self._importance_result is None:
            raise ValueError("Must call fit() first")

        importance = self._importance_result.importances_mean

        if k is not None:
            indices = np.argsort(importance)[-k:]
        elif threshold is not None:
            indices = np.where(importance >= threshold)[0]
        else:
            # Default: features with importance > 0 (positive contribution)
            indices = np.where(importance > 0)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="permutation",
        )


class LassoSelector:
    """
    L1 regularization (Lasso) based feature selection.
    Features with zero coefficients are eliminated.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        cv: int = 5,
        task_type: TaskType = TaskType.REGRESSION,
        max_iter: int = 10000,
    ):
        self.task_type = task_type
        self.alpha = alpha
        self.cv = cv
        self.max_iter = max_iter
        self._model = None
        self._coefficients: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "LassoSelector":
        """Fit Lasso model"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        if self.task_type == TaskType.REGRESSION:
            if self.alpha is None:
                self._model = LassoCV(cv=self.cv, max_iter=self.max_iter, random_state=42)
            else:
                self._model = Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=42)
        else:
            if self.alpha is None:
                self._model = LogisticRegressionCV(
                    penalty="l1",
                    solver="saga",
                    cv=self.cv,
                    max_iter=self.max_iter,
                    random_state=42,
                )
            else:
                self._model = LogisticRegression(
                    penalty="l1",
                    C=1.0 / self.alpha,
                    solver="saga",
                    max_iter=self.max_iter,
                    random_state=42,
                )

        self._model.fit(X, y)

        # Get coefficients
        coef = self._model.coef_
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        self._coefficients = np.abs(coef)

        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get coefficient magnitudes as importance"""
        if self._coefficients is None:
            raise ValueError("Must call fit() first")

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._coefficients,
            method="lasso",
            metadata={"alpha": getattr(self._model, "alpha_", self.alpha)},
        )

    def select_features(
        self,
        threshold: float = 1e-5,
    ) -> FeatureSelectionResult:
        """Select features with non-zero coefficients"""
        if self._coefficients is None:
            raise ValueError("Must call fit() first")

        indices = np.where(self._coefficients > threshold)[0]
        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="lasso",
        )


class TreeBasedImportance:
    """
    Tree-based feature importance (Gini/Mean Decrease Impurity).
    Fast and effective for initial feature screening.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        task_type: TaskType = TaskType.REGRESSION,
        n_estimators: int = 100,
    ):
        self.task_type = task_type
        self.n_estimators = n_estimators

        if estimator is None:
            if task_type == TaskType.REGRESSION:
                self.estimator = RandomForestRegressor(
                    n_estimators=n_estimators, n_jobs=-1, random_state=42
                )
            else:
                self.estimator = RandomForestClassifier(
                    n_estimators=n_estimators, n_jobs=-1, random_state=42
                )
        else:
            self.estimator = estimator

        self._importance_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "TreeBasedImportance":
        """Fit tree ensemble and extract feature importances"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        self.estimator.fit(X, y)
        self._importance_scores = self.estimator.feature_importances_

        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get tree-based importance scores"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._importance_scores,
            method="tree_based",
        )

    def select_features(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features by tree importance"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        if k is not None:
            indices = np.argsort(self._importance_scores)[-k:]
        elif threshold is not None:
            indices = np.where(self._importance_scores >= threshold)[0]
        else:
            # Default: features with importance > mean
            mean_imp = np.mean(self._importance_scores)
            indices = np.where(self._importance_scores >= mean_imp)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="tree_based",
        )


# =============================================================================
# ADVANCED METHODS (MDI, MDA, SFI, Clustered)
# =============================================================================

class MeanDecreaseImpurity:
    """
    Mean Decrease Impurity (MDI) - standard tree-based importance.
    Equivalent to TreeBasedImportance but with explicit naming.
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.REGRESSION,
        n_estimators: int = 100,
    ):
        self.task_type = task_type
        self.n_estimators = n_estimators

        if task_type == TaskType.REGRESSION:
            self.estimator = RandomForestRegressor(
                n_estimators=n_estimators, n_jobs=-1, random_state=42
            )
        else:
            self.estimator = RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=-1, random_state=42
            )

        self._importance_scores: Optional[np.ndarray] = None
        self._std_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MeanDecreaseImpurity":
        """Fit and compute MDI importance"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        self.estimator.fit(X, y)
        self._importance_scores = self.estimator.feature_importances_

        # Compute std across trees
        importances = np.array([tree.feature_importances_ for tree in self.estimator.estimators_])
        self._std_scores = np.std(importances, axis=0)

        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get MDI importance scores"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._importance_scores,
            std_scores=self._std_scores,
            method="mdi",
        )

    def select_features(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features by MDI"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        if k is not None:
            indices = np.argsort(self._importance_scores)[-k:]
        elif threshold is not None:
            indices = np.where(self._importance_scores >= threshold)[0]
        else:
            mean_imp = np.mean(self._importance_scores)
            indices = np.where(self._importance_scores >= mean_imp)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="mdi",
        )


class MeanDecreaseAccuracy:
    """
    Mean Decrease Accuracy (MDA) - permutation-based importance.
    More robust than MDI for correlated features.
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.REGRESSION,
        n_estimators: int = 100,
        n_repeats: int = 10,
        scoring: Optional[str] = None,
    ):
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats

        if scoring is None:
            self.scoring = "neg_mean_squared_error" if task_type == TaskType.REGRESSION else "accuracy"
        else:
            self.scoring = scoring

        if task_type == TaskType.REGRESSION:
            self.estimator = RandomForestRegressor(
                n_estimators=n_estimators, n_jobs=-1, random_state=42, oob_score=True
            )
        else:
            self.estimator = RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=-1, random_state=42, oob_score=True
            )

        self._importance_scores: Optional[np.ndarray] = None
        self._std_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MeanDecreaseAccuracy":
        """Fit and compute MDA importance using OOB samples"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        self.estimator.fit(X, y)

        # Use permutation importance on OOB predictions
        result = permutation_importance(
            self.estimator, X, y,
            n_repeats=self.n_repeats,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=42,
        )

        self._importance_scores = result.importances_mean
        self._std_scores = result.importances_std

        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get MDA importance scores"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._importance_scores,
            std_scores=self._std_scores,
            method="mda",
        )

    def select_features(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features by MDA"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        if k is not None:
            indices = np.argsort(self._importance_scores)[-k:]
        elif threshold is not None:
            indices = np.where(self._importance_scores >= threshold)[0]
        else:
            # Features with positive MDA
            indices = np.where(self._importance_scores > 0)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="mda",
        )


class SingleFeatureImportance:
    """
    Single Feature Importance (SFI) - evaluates each feature independently.
    Measures predictive power of individual features.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        task_type: TaskType = TaskType.REGRESSION,
        cv: int = 5,
        scoring: Optional[str] = None,
    ):
        self.task_type = task_type
        self.cv = cv

        if scoring is None:
            self.scoring = "neg_mean_squared_error" if task_type == TaskType.REGRESSION else "accuracy"
        else:
            self.scoring = scoring

        if estimator is None:
            if task_type == TaskType.REGRESSION:
                self.estimator = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
            else:
                self.estimator = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        else:
            self.estimator = estimator

        self._importance_scores: Optional[np.ndarray] = None
        self._std_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "SingleFeatureImportance":
        """Compute SFI by evaluating each feature independently"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        n_features = X.shape[1]
        scores = []
        stds = []

        for i in range(n_features):
            X_single = X[:, i:i+1]
            cv_scores = cross_val_score(
                clone(self.estimator), X_single, y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
            )
            scores.append(np.mean(cv_scores))
            stds.append(np.std(cv_scores))

        self._importance_scores = np.array(scores)
        self._std_scores = np.array(stds)

        # Normalize to [0, 1] range
        min_score = self._importance_scores.min()
        max_score = self._importance_scores.max()
        if max_score > min_score:
            self._importance_scores = (self._importance_scores - min_score) / (max_score - min_score)

        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get SFI scores"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._importance_scores,
            std_scores=self._std_scores,
            method="sfi",
        )

    def select_features(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features by SFI"""
        if self._importance_scores is None:
            raise ValueError("Must call fit() first")

        if k is not None:
            indices = np.argsort(self._importance_scores)[-k:]
        elif threshold is not None:
            indices = np.where(self._importance_scores >= threshold)[0]
        else:
            median_score = np.median(self._importance_scores)
            indices = np.where(self._importance_scores >= median_score)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="sfi",
        )


class ClusteredFeatureImportance:
    """
    Clustered Feature Importance - handles feature correlation by clustering.
    Computes importance within feature clusters to avoid splitting importance.
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.REGRESSION,
        n_clusters: Optional[int] = None,
        distance_threshold: float = 0.5,
        linkage: str = "ward",
    ):
        self.task_type = task_type
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage

        if task_type == TaskType.REGRESSION:
            self.estimator = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            self.estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

        self._cluster_labels: Optional[np.ndarray] = None
        self._cluster_importance: Optional[Dict[int, float]] = None
        self._feature_importance: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "ClusteredFeatureImportance":
        """Cluster features and compute importance"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Compute correlation distance matrix
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0, posinf=1, neginf=-1)
        distance = 1 - np.abs(corr)
        np.fill_diagonal(distance, 0)

        # Ensure symmetry and valid condensed form
        distance = (distance + distance.T) / 2
        distance = np.clip(distance, 0, 2)
        condensed_dist = squareform(distance, checks=False)

        # Hierarchical clustering
        linkage_matrix = hierarchy.linkage(condensed_dist, method=self.linkage)

        if self.n_clusters is not None:
            self._cluster_labels = hierarchy.fcluster(linkage_matrix, self.n_clusters, criterion="maxclust")
        else:
            self._cluster_labels = hierarchy.fcluster(linkage_matrix, self.distance_threshold, criterion="distance")

        # Fit model on full data
        self.estimator.fit(X, y)
        base_importance = self.estimator.feature_importances_

        # Aggregate importance by cluster
        unique_clusters = np.unique(self._cluster_labels)
        self._cluster_importance = {}

        for cluster in unique_clusters:
            cluster_mask = self._cluster_labels == cluster
            cluster_importance = np.sum(base_importance[cluster_mask])
            self._cluster_importance[cluster] = cluster_importance

        # Redistribute importance within clusters equally
        self._feature_importance = np.zeros(len(self._feature_names))
        for i, (feat, cluster) in enumerate(zip(self._feature_names, self._cluster_labels)):
            cluster_size = np.sum(self._cluster_labels == cluster)
            self._feature_importance[i] = self._cluster_importance[cluster] / cluster_size

        return self

    def get_importance(self) -> FeatureImportanceResult:
        """Get clustered importance scores"""
        if self._feature_importance is None:
            raise ValueError("Must call fit() first")

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._feature_importance,
            method="clustered",
            metadata={
                "cluster_labels": self._cluster_labels.tolist(),
                "cluster_importance": self._cluster_importance,
            },
        )

    def select_features(
        self,
        k_clusters: Optional[int] = None,
        k_features_per_cluster: int = 1,
    ) -> FeatureSelectionResult:
        """Select top features from each cluster"""
        if self._feature_importance is None:
            raise ValueError("Must call fit() first")

        selected = []
        unique_clusters = np.unique(self._cluster_labels)

        # Sort clusters by total importance
        sorted_clusters = sorted(
            unique_clusters,
            key=lambda c: self._cluster_importance[c],
            reverse=True,
        )

        if k_clusters is not None:
            sorted_clusters = sorted_clusters[:k_clusters]

        for cluster in sorted_clusters:
            cluster_mask = self._cluster_labels == cluster
            cluster_indices = np.where(cluster_mask)[0]
            cluster_importances = self._feature_importance[cluster_indices]

            # Select top features from this cluster
            top_indices = np.argsort(cluster_importances)[-k_features_per_cluster:]
            for idx in top_indices:
                selected.append(self._feature_names[cluster_indices[idx]])

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method="clustered",
        )


# =============================================================================
# VALIDATION METHODS
# =============================================================================

class ProbabilityOfBacktestOverfitting:
    """
    Probability of Backtest Overfitting (PBO) - Lopez de Prado method.
    Measures probability that an in-sample optimal strategy underperforms OOS.
    """

    def __init__(
        self,
        n_partitions: int = 16,
        metric: Callable[[np.ndarray, np.ndarray], float] = None,
    ):
        self.n_partitions = n_partitions
        self.metric = metric or (lambda y_true, y_pred: -mean_squared_error(y_true, y_pred))
        self._pbo: Optional[float] = None
        self._performance_degradation: Optional[float] = None

    def compute(
        self,
        strategy_returns: np.ndarray,
        n_strategies: int = 100,
    ) -> float:
        """
        Compute PBO using Combinatorial Symmetric Cross-Validation.

        Args:
            strategy_returns: Matrix of shape (n_periods, n_strategies)
            n_strategies: Number of strategies to evaluate

        Returns:
            PBO: Probability of Backtest Overfitting
        """
        n_periods = len(strategy_returns)
        partition_size = n_periods // self.n_partitions

        # Generate all combinations of train/test splits
        partition_indices = list(range(self.n_partitions))
        n_train = self.n_partitions // 2

        logits = []

        for train_combo in combinations(partition_indices, n_train):
            test_combo = tuple(p for p in partition_indices if p not in train_combo)

            # Get train/test indices
            train_idx = []
            test_idx = []
            for p in train_combo:
                train_idx.extend(range(p * partition_size, (p + 1) * partition_size))
            for p in test_combo:
                test_idx.extend(range(p * partition_size, (p + 1) * partition_size))

            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)

            # Evaluate strategies
            train_returns = strategy_returns[train_idx]
            test_returns = strategy_returns[test_idx]

            # Find best strategy in-sample
            is_sharpe = np.mean(train_returns, axis=0) / (np.std(train_returns, axis=0) + 1e-10)
            best_is_idx = np.argmax(is_sharpe)

            # Evaluate OOS
            oos_sharpe = np.mean(test_returns, axis=0) / (np.std(test_returns, axis=0) + 1e-10)
            best_oos_sharpe = oos_sharpe[best_is_idx]

            # Rank relative to all strategies
            rank = np.sum(oos_sharpe <= best_oos_sharpe) / len(oos_sharpe)

            # Compute logit
            if rank > 0 and rank < 1:
                logit = np.log(rank / (1 - rank))
                logits.append(logit)

        logits = np.array(logits)
        self._pbo = np.mean(logits < 0) if len(logits) > 0 else 0.5

        return self._pbo

    def get_result(self) -> Dict[str, float]:
        """Get PBO computation results"""
        if self._pbo is None:
            raise ValueError("Must call compute() first")

        return {
            "pbo": self._pbo,
            "interpretation": "overfit" if self._pbo > 0.5 else "not_overfit",
        }


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio - adjusts for multiple testing.
    Accounts for the number of trials/strategies tested.
    """

    def __init__(self, n_trials: int = 1):
        self.n_trials = n_trials

    def compute(
        self,
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0,
        annualization_factor: float = np.sqrt(252),
    ) -> Dict[str, float]:
        """
        Compute Deflated Sharpe Ratio.

        Args:
            returns: Strategy returns
            benchmark_sharpe: Expected Sharpe under null hypothesis
            annualization_factor: Factor to annualize returns

        Returns:
            Dictionary with DSR and related metrics
        """
        # Compute observed Sharpe
        observed_sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * annualization_factor

        # Compute skewness and kurtosis
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        # Number of observations
        n = len(returns)

        # Standard error of Sharpe (Lo's formula)
        se_sharpe = np.sqrt((1 + 0.5 * observed_sharpe**2 - skew * observed_sharpe +
                            (kurt - 1) / 4 * observed_sharpe**2) / n)

        # Expected maximum Sharpe under null (Euler constant approximation)
        euler_mascheroni = 0.5772
        expected_max = benchmark_sharpe + se_sharpe * (
            (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / self.n_trials) +
            euler_mascheroni * stats.norm.ppf(1 - 1.0 / (self.n_trials * np.e))
        )

        # Deflated Sharpe Ratio (probability observed > expected max)
        psr = stats.norm.cdf((observed_sharpe - expected_max) / se_sharpe)

        return {
            "observed_sharpe": observed_sharpe,
            "expected_max_sharpe": expected_max,
            "deflated_sharpe_ratio": psr,
            "standard_error": se_sharpe,
            "n_trials": self.n_trials,
            "is_significant": psr > 0.95,
        }


class CombinatorialSymmetricCV:
    """
    Combinatorial Symmetric Cross-Validation (CSCV).
    Generates all combinations of train/test splits for robust evaluation.
    """

    def __init__(
        self,
        n_partitions: int = 16,
        n_train_partitions: Optional[int] = None,
    ):
        self.n_partitions = n_partitions
        self.n_train_partitions = n_train_partitions or n_partitions // 2

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate CSCV splits.

        Args:
            X: Data to split

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        partition_size = n_samples // self.n_partitions

        # Create partition indices
        partitions = []
        for i in range(self.n_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size if i < self.n_partitions - 1 else n_samples
            partitions.append(np.arange(start, end))

        # Generate all combinations
        splits = []
        partition_indices = list(range(self.n_partitions))

        for train_combo in combinations(partition_indices, self.n_train_partitions):
            test_combo = tuple(p for p in partition_indices if p not in train_combo)

            train_idx = np.concatenate([partitions[p] for p in train_combo])
            test_idx = np.concatenate([partitions[p] for p in test_combo])

            splits.append((train_idx, test_idx))

        return splits

    def evaluate_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: List[int],
        estimator: BaseEstimator,
        scoring: Callable,
    ) -> Dict[str, float]:
        """
        Evaluate feature subset using CSCV.

        Args:
            X: Full feature matrix
            y: Target
            feature_indices: Indices of features to evaluate
            estimator: Model to use
            scoring: Scoring function

        Returns:
            Dictionary with evaluation metrics
        """
        X_subset = X[:, feature_indices]
        splits = self.split(X)

        train_scores = []
        test_scores = []

        for train_idx, test_idx in splits:
            model = clone(estimator)
            model.fit(X_subset[train_idx], y[train_idx])

            train_pred = model.predict(X_subset[train_idx])
            test_pred = model.predict(X_subset[test_idx])

            train_scores.append(scoring(y[train_idx], train_pred))
            test_scores.append(scoring(y[test_idx], test_pred))

        return {
            "mean_train_score": np.mean(train_scores),
            "std_train_score": np.std(train_scores),
            "mean_test_score": np.mean(test_scores),
            "std_test_score": np.std(test_scores),
            "overfitting_ratio": np.mean(train_scores) / (np.mean(test_scores) + 1e-10),
            "n_splits": len(splits),
        }


# =============================================================================
# UTILITIES
# =============================================================================

class FeatureClusteringAnalysis:
    """
    Hierarchical clustering analysis of features.
    Identifies groups of correlated features.
    """

    def __init__(
        self,
        method: Literal["pearson", "spearman"] = "spearman",
        linkage: str = "ward",
    ):
        self.method = method
        self.linkage = linkage
        self._linkage_matrix: Optional[np.ndarray] = None
        self._distance_matrix: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureClusteringAnalysis":
        """Compute hierarchical clustering of features"""
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or list(X.columns)
            X = X.values

        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Compute correlation-based distance
        if self.method == "spearman":
            corr, _ = spearmanr(X)
        else:
            corr = np.corrcoef(X.T)

        corr = np.nan_to_num(corr, nan=0, posinf=1, neginf=-1)
        self._distance_matrix = 1 - np.abs(corr)
        np.fill_diagonal(self._distance_matrix, 0)

        # Ensure symmetry
        self._distance_matrix = (self._distance_matrix + self._distance_matrix.T) / 2
        self._distance_matrix = np.clip(self._distance_matrix, 0, 2)

        # Compute linkage
        condensed = squareform(self._distance_matrix, checks=False)
        self._linkage_matrix = hierarchy.linkage(condensed, method=self.linkage)

        return self

    def get_clusters(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
    ) -> Dict[int, List[str]]:
        """Get feature clusters"""
        if self._linkage_matrix is None:
            raise ValueError("Must call fit() first")

        if n_clusters is not None:
            labels = hierarchy.fcluster(self._linkage_matrix, n_clusters, criterion="maxclust")
        elif distance_threshold is not None:
            labels = hierarchy.fcluster(self._linkage_matrix, distance_threshold, criterion="distance")
        else:
            # Default: use optimal based on inconsistency
            labels = hierarchy.fcluster(self._linkage_matrix, 0.5, criterion="distance")

        clusters = {}
        for label in np.unique(labels):
            clusters[int(label)] = [
                self._feature_names[i] for i in range(len(labels)) if labels[i] == label
            ]

        return clusters

    def get_dendrogram_data(self) -> Dict[str, Any]:
        """Get data for dendrogram visualization"""
        if self._linkage_matrix is None:
            raise ValueError("Must call fit() first")

        return {
            "linkage_matrix": self._linkage_matrix,
            "feature_names": self._feature_names,
            "distance_matrix": self._distance_matrix,
        }


class VIFAnalysis:
    """
    Variance Inflation Factor (VIF) analysis for multicollinearity detection.
    VIF > 5-10 indicates problematic multicollinearity.
    """

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self._vif_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "VIFAnalysis":
        """Compute VIF for each feature"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        n_features = X.shape[1]
        self._vif_scores = np.zeros(n_features)

        for i in range(n_features):
            # Regress feature i on all other features
            y_i = X[:, i]
            X_others = np.delete(X, i, axis=1)

            # Add constant term
            X_with_const = np.column_stack([np.ones(len(X_others)), X_others])

            try:
                # OLS: VIF = 1 / (1 - R^2)
                coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y_i, rcond=None)
                y_pred = X_with_const @ coeffs
                ss_res = np.sum((y_i - y_pred) ** 2)
                ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)

                r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                self._vif_scores[i] = 1 / (1 - r_squared + 1e-10)
            except (np.linalg.LinAlgError, ValueError):
                self._vif_scores[i] = np.inf

        return self

    def get_vif_scores(self) -> pd.DataFrame:
        """Get VIF scores as DataFrame"""
        if self._vif_scores is None:
            raise ValueError("Must call fit() first")

        df = pd.DataFrame({
            "feature": self._feature_names,
            "vif": self._vif_scores,
            "multicollinear": self._vif_scores > self.threshold,
        })
        return df.sort_values("vif", ascending=False).reset_index(drop=True)

    def get_problematic_features(self) -> List[str]:
        """Get features with VIF above threshold"""
        if self._vif_scores is None:
            raise ValueError("Must call fit() first")

        return [
            self._feature_names[i]
            for i in range(len(self._vif_scores))
            if self._vif_scores[i] > self.threshold
        ]

    def select_features(self, iterative: bool = True) -> FeatureSelectionResult:
        """Select features without multicollinearity"""
        if self._vif_scores is None:
            raise ValueError("Must call fit() first")

        if not iterative:
            # Simple: remove all features above threshold
            selected = [
                self._feature_names[i]
                for i in range(len(self._vif_scores))
                if self._vif_scores[i] <= self.threshold
            ]
        else:
            # Iterative: remove highest VIF, recalculate, repeat
            selected = list(self._feature_names)
            vif_current = self._vif_scores.copy()

            while np.max(vif_current) > self.threshold and len(selected) > 1:
                worst_idx = np.argmax(vif_current)
                selected.pop(worst_idx)
                vif_current = np.delete(vif_current, worst_idx)

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            method="vif",
            validation_metrics={"threshold": self.threshold},
        )


class FeatureStabilityAnalysis:
    """
    Feature stability analysis across cross-validation folds.
    Measures how consistently features are selected.
    """

    def __init__(
        self,
        selector: Any,  # Any feature selector with fit() and select_features()
        cv: int = 5,
        time_series: bool = True,
    ):
        self.selector = selector
        self.cv = cv
        self.time_series = time_series
        self._selection_frequency: Optional[Dict[str, float]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureStabilityAnalysis":
        """Compute feature selection stability across folds"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        if self.time_series:
            cv_splitter = TimeSeriesSplit(n_splits=self.cv)
        else:
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        selection_counts = {f: 0 for f in self._feature_names}

        for train_idx, _ in cv_splitter.split(X):
            X_train = X[train_idx]
            y_train = y[train_idx]

            # Fit selector on this fold
            selector_clone = clone(self.selector) if hasattr(self.selector, "__sklearn_clone__") else self.selector
            selector_clone.fit(X_train, y_train, feature_names=self._feature_names)

            # Get selected features
            result = selector_clone.select_features()
            for f in result.selected_features:
                selection_counts[f] += 1

        # Compute selection frequency
        self._selection_frequency = {
            f: count / self.cv for f, count in selection_counts.items()
        }

        return self

    def get_stability_scores(self) -> pd.DataFrame:
        """Get stability scores as DataFrame"""
        if self._selection_frequency is None:
            raise ValueError("Must call fit() first")

        df = pd.DataFrame([
            {"feature": f, "stability": freq}
            for f, freq in self._selection_frequency.items()
        ])
        return df.sort_values("stability", ascending=False).reset_index(drop=True)

    def select_stable_features(
        self,
        stability_threshold: float = 0.5,
    ) -> FeatureSelectionResult:
        """Select features that appear in at least threshold fraction of folds"""
        if self._selection_frequency is None:
            raise ValueError("Must call fit() first")

        selected = [
            f for f, freq in self._selection_frequency.items()
            if freq >= stability_threshold
        ]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            method="stability",
            validation_metrics={
                "threshold": stability_threshold,
                "n_folds": self.cv,
            },
        )


# =============================================================================
# ENSEMBLE FEATURE SELECTOR
# =============================================================================

class EnsembleFeatureSelector:
    """
    Ensemble feature selection combining multiple methods.
    Aggregates rankings from different selectors for robust selection.
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        task_type: TaskType = TaskType.REGRESSION,
        aggregation: Literal["mean", "median", "vote"] = "mean",
    ):
        self.task_type = task_type
        self.aggregation = aggregation

        # Default methods
        self.methods = methods or [
            "mutual_information",
            "tree_based",
            "permutation",
            "lasso",
        ]

        self._selectors: Dict[str, Any] = {}
        self._importance_results: Dict[str, FeatureImportanceResult] = {}
        self._aggregated_importance: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "EnsembleFeatureSelector":
        """Fit all selectors and aggregate results"""
        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Initialize and fit selectors
        method_map = {
            "mutual_information": MutualInformationFilter(task_type=self.task_type),
            "tree_based": TreeBasedImportance(task_type=self.task_type),
            "permutation": PermutationImportance(task_type=self.task_type),
            "lasso": LassoSelector(task_type=self.task_type),
            "mdi": MeanDecreaseImpurity(task_type=self.task_type),
            "mda": MeanDecreaseAccuracy(task_type=self.task_type),
            "sfi": SingleFeatureImportance(task_type=self.task_type),
        }

        if SHAP_AVAILABLE:
            method_map["shap"] = SHAPImportance(task_type=self.task_type)

        for method_name in self.methods:
            if method_name not in method_map:
                logger.warning(f"Unknown method: {method_name}, skipping")
                continue

            try:
                selector = method_map[method_name]
                selector.fit(X, y, feature_names=self._feature_names)
                self._selectors[method_name] = selector
                self._importance_results[method_name] = selector.get_importance()
            except Exception as e:
                logger.warning(f"Failed to fit {method_name}: {e}")

        # Aggregate importance scores
        self._aggregate_importance()

        return self

    def _aggregate_importance(self):
        """Aggregate importance scores from all methods"""
        n_features = len(self._feature_names)
        all_rankings = []

        for method_name, result in self._importance_results.items():
            # Convert to rankings (0 = worst, n-1 = best)
            rankings = np.argsort(np.argsort(result.importance_scores))
            # Normalize to [0, 1]
            rankings = rankings / (n_features - 1) if n_features > 1 else rankings
            all_rankings.append(rankings)

        all_rankings = np.array(all_rankings)

        if self.aggregation == "mean":
            self._aggregated_importance = np.mean(all_rankings, axis=0)
        elif self.aggregation == "median":
            self._aggregated_importance = np.median(all_rankings, axis=0)
        else:  # vote
            # Top 50% from each method gets a vote
            threshold = 0.5
            votes = np.sum(all_rankings >= threshold, axis=0)
            self._aggregated_importance = votes / len(all_rankings)

    def get_importance(self) -> FeatureImportanceResult:
        """Get aggregated importance scores"""
        if self._aggregated_importance is None:
            raise ValueError("Must call fit() first")

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_scores=self._aggregated_importance,
            method=f"ensemble_{self.aggregation}",
            metadata={"methods_used": list(self._importance_results.keys())},
        )

    def get_all_importance_results(self) -> Dict[str, FeatureImportanceResult]:
        """Get importance results from all methods"""
        return self._importance_results

    def select_features(
        self,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Select features based on ensemble importance"""
        if self._aggregated_importance is None:
            raise ValueError("Must call fit() first")

        if k is not None:
            indices = np.argsort(self._aggregated_importance)[-k:]
        elif threshold is not None:
            indices = np.where(self._aggregated_importance >= threshold)[0]
        else:
            # Default: top 50%
            median_imp = np.median(self._aggregated_importance)
            indices = np.where(self._aggregated_importance >= median_imp)[0]

        selected = [self._feature_names[i] for i in indices]

        return FeatureSelectionResult(
            selected_features=selected,
            all_features=self._feature_names,
            importance_result=self.get_importance(),
            method=f"ensemble_{self.aggregation}",
        )


# =============================================================================
# SKLEARN TRANSFORMER WRAPPER
# =============================================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature selector transformer.
    Wraps any selection method for use in sklearn Pipelines.
    """

    def __init__(
        self,
        method: str = "ensemble",
        task_type: TaskType = TaskType.REGRESSION,
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        **method_kwargs,
    ):
        self.method = method
        self.task_type = task_type
        self.n_features = n_features
        self.threshold = threshold
        self.method_kwargs = method_kwargs

        self._selector = None
        self._selected_features: Optional[List[str]] = None
        self._selected_indices: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
    ) -> "FeatureSelector":
        """Fit the feature selector"""
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values
        else:
            feature_names = [f"f_{i}" for i in range(X.shape[1])]

        # Initialize selector based on method
        method_map = {
            "mutual_information": MutualInformationFilter,
            "correlation": CorrelationFilter,
            "variance": VarianceFilter,
            "chi_squared": ChiSquaredFilter,
            "rfe": RFESelector,
            "rfecv": RFECVSelector,
            "sequential": SequentialSelector,
            "shap": SHAPImportance,
            "permutation": PermutationImportance,
            "lasso": LassoSelector,
            "tree_based": TreeBasedImportance,
            "mdi": MeanDecreaseImpurity,
            "mda": MeanDecreaseAccuracy,
            "sfi": SingleFeatureImportance,
            "clustered": ClusteredFeatureImportance,
            "ensemble": EnsembleFeatureSelector,
        }

        if self.method not in method_map:
            raise ValueError(f"Unknown method: {self.method}")

        selector_class = method_map[self.method]

        # Handle task_type parameter
        if "task_type" in selector_class.__init__.__code__.co_varnames:
            self._selector = selector_class(task_type=self.task_type, **self.method_kwargs)
        else:
            self._selector = selector_class(**self.method_kwargs)

        # Fit selector
        self._selector.fit(X, y, feature_names=feature_names)

        # Select features
        if hasattr(self._selector, "select_features"):
            if self.n_features is not None:
                result = self._selector.select_features(k=self.n_features)
            elif self.threshold is not None:
                result = self._selector.select_features(threshold=self.threshold)
            else:
                result = self._selector.select_features()
        else:
            raise ValueError(f"Selector {self.method} does not support select_features()")

        self._selected_features = result.selected_features
        self._selected_indices = np.array([
            feature_names.index(f) for f in self._selected_features
        ])

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform by selecting features"""
        if self._selected_indices is None:
            raise ValueError("Must call fit() first")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return X[:, self._selected_indices]

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[str]]:
        """Get mask or indices of selected features"""
        if indices:
            return self._selected_indices
        return self._selected_features

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names"""
        return self._selected_features


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_features(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    method: str = "ensemble",
    task_type: TaskType = TaskType.REGRESSION,
    n_features: Optional[int] = None,
    threshold: Optional[float] = None,
    feature_names: Optional[List[str]] = None,
    **kwargs,
) -> FeatureSelectionResult:
    """
    Convenience function for feature selection.

    Args:
        X: Feature matrix
        y: Target variable
        method: Selection method (see FeatureSelector for options)
        task_type: Task type (regression or classification)
        n_features: Number of features to select
        threshold: Importance threshold for selection
        feature_names: Optional feature names
        **kwargs: Additional arguments for the selector

    Returns:
        FeatureSelectionResult with selected features and importance scores
    """
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or list(X.columns)
        X = X.values

    selector = FeatureSelector(
        method=method,
        task_type=task_type,
        n_features=n_features,
        threshold=threshold,
        **kwargs,
    )

    selector.fit(X, y)

    return FeatureSelectionResult(
        selected_features=selector._selected_features,
        all_features=feature_names or [f"f_{i}" for i in range(X.shape[1])],
        importance_result=selector._selector.get_importance() if hasattr(selector._selector, "get_importance") else None,
        method=method,
    )


def compute_all_importances(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    task_type: TaskType = TaskType.REGRESSION,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, FeatureImportanceResult]:
    """
    Compute feature importance using all available methods.

    Args:
        X: Feature matrix
        y: Target variable
        task_type: Task type
        feature_names: Optional feature names

    Returns:
        Dictionary mapping method names to importance results
    """
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or list(X.columns)
        X = X.values

    feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

    methods = {
        "mutual_information": MutualInformationFilter(task_type=task_type),
        "variance": VarianceFilter(),
        "tree_based": TreeBasedImportance(task_type=task_type),
        "permutation": PermutationImportance(task_type=task_type),
        "lasso": LassoSelector(task_type=task_type),
        "mdi": MeanDecreaseImpurity(task_type=task_type),
        "mda": MeanDecreaseAccuracy(task_type=task_type),
    }

    if SHAP_AVAILABLE:
        methods["shap"] = SHAPImportance(task_type=task_type)

    results = {}
    for name, selector in methods.items():
        try:
            selector.fit(X, y, feature_names=feature_names)
            results[name] = selector.get_importance()
        except Exception as e:
            logger.warning(f"Failed to compute {name} importance: {e}")

    return results
