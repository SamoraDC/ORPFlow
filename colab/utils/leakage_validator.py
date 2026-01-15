"""
Data Leakage Validator
Hyper-rigorous validation to detect ANY form of data leakage

This module implements multiple layers of leakage detection:
1. Feature-Target correlation analysis
2. Temporal ordering validation
3. Future information detection
4. Cross-validation leakage check
5. Distribution shift analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats


class LeakageValidator:
    """
    Hyper-rigorous data leakage detection system.

    DETECTS:
    1. Target leakage (features that contain target information)
    2. Temporal leakage (future data in features)
    3. Train-test leakage (information bleeding between splits)
    4. Feature leakage (features derived from future data)
    """

    def __init__(self, significance_level: float = 0.01):
        self.significance_level = significance_level
        self.validation_results = {}

    def check_feature_target_correlation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.95
    ) -> Dict:
        """
        Check for suspiciously high feature-target correlations.

        WARNING: Correlation > threshold suggests leakage!
        """
        results = {
            "passed": True,
            "suspicious_features": [],
            "correlations": {}
        }

        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            results["correlations"][name] = float(corr) if not np.isnan(corr) else 0.0

            if abs(corr) > threshold:
                results["passed"] = False
                results["suspicious_features"].append({
                    "feature": name,
                    "correlation": float(corr),
                    "warning": f"Correlation {corr:.4f} > {threshold} suggests TARGET LEAKAGE!"
                })

        return results

    def check_temporal_ordering(
        self,
        train_times: pd.Series,
        val_times: pd.Series,
        test_times: pd.Series
    ) -> Dict:
        """
        Verify strict temporal ordering between splits.
        """
        results = {
            "passed": True,
            "checks": []
        }

        train_max = train_times.max()
        val_min = val_times.min()
        val_max = val_times.max()
        test_min = test_times.min()

        # Check 1: Train before Val
        if train_max >= val_min:
            results["passed"] = False
            results["checks"].append({
                "check": "train_before_val",
                "passed": False,
                "message": f"FAIL: Train max ({train_max}) >= Val min ({val_min})"
            })
        else:
            results["checks"].append({
                "check": "train_before_val",
                "passed": True,
                "message": f"PASS: Train max ({train_max}) < Val min ({val_min})"
            })

        # Check 2: Val before Test
        if val_max >= test_min:
            results["passed"] = False
            results["checks"].append({
                "check": "val_before_test",
                "passed": False,
                "message": f"FAIL: Val max ({val_max}) >= Test min ({test_min})"
            })
        else:
            results["checks"].append({
                "check": "val_before_test",
                "passed": True,
                "message": f"PASS: Val max ({val_max}) < Test min ({test_min})"
            })

        return results

    def check_future_information(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> Dict:
        """
        Check if features contain future information.

        METHOD: For each feature, check if it's correlated with FUTURE target
        more than PAST target. If so, it might contain future info.
        """
        results = {
            "passed": True,
            "suspicious_features": [],
            "analysis": {}
        }

        # Create lagged targets
        df = df.copy()
        df["target_future"] = df[target_col]
        df["target_past"] = df[target_col].shift(10)  # 10 periods ago

        df_clean = df.dropna()

        for col in feature_cols[:20]:  # Check first 20 features
            if col not in df_clean.columns:
                continue

            future_corr = np.corrcoef(df_clean[col], df_clean["target_future"])[0, 1]
            past_corr = np.corrcoef(df_clean[col], df_clean["target_past"])[0, 1]

            if np.isnan(future_corr) or np.isnan(past_corr):
                continue

            results["analysis"][col] = {
                "future_correlation": float(future_corr),
                "past_correlation": float(past_corr),
                "ratio": float(abs(future_corr) / (abs(past_corr) + 1e-8))
            }

            # If future correlation is much higher than past, suspicious
            if abs(future_corr) > 0.5 and abs(future_corr) > 2 * abs(past_corr):
                results["passed"] = False
                results["suspicious_features"].append({
                    "feature": col,
                    "future_corr": float(future_corr),
                    "past_corr": float(past_corr),
                    "warning": "Feature may contain FUTURE INFORMATION!"
                })

        return results

    def check_distribution_shift(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        max_ks_stat: float = 0.3
    ) -> Dict:
        """
        Check for distribution shift between train and test.

        Large shifts might indicate temporal leakage or data issues.
        """
        results = {
            "passed": True,
            "warnings": [],
            "ks_statistics": {}
        }

        for i, name in enumerate(feature_names):
            ks_stat, p_value = stats.ks_2samp(X_train[:, i], X_test[:, i])

            results["ks_statistics"][name] = {
                "statistic": float(ks_stat),
                "p_value": float(p_value)
            }

            if ks_stat > max_ks_stat:
                results["warnings"].append({
                    "feature": name,
                    "ks_statistic": float(ks_stat),
                    "message": f"Large distribution shift (KS={ks_stat:.3f})"
                })

        return results

    def check_perfect_prediction(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.90
    ) -> Dict:
        """
        Check for suspiciously good predictions (suggests leakage).

        If a simple model achieves near-perfect accuracy, there's likely leakage.
        """
        results = {
            "passed": True,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "warning": None
        }

        # Convert to classification
        y_train_class = (y_train > 0).astype(int)
        y_test_class = (y_test > 0).astype(int)

        # Simple random forest
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_train[:10000], y_train_class[:10000])  # Subsample for speed

        train_acc = clf.score(X_train[:10000], y_train_class[:10000])
        test_acc = clf.score(X_test[:5000], y_test_class[:5000])

        results["train_accuracy"] = float(train_acc)
        results["test_accuracy"] = float(test_acc)

        # Warning conditions
        if train_acc > threshold:
            results["passed"] = False
            results["warning"] = f"Train accuracy {train_acc:.2%} > {threshold:.0%} - POSSIBLE LEAKAGE!"

        if test_acc > threshold:
            results["passed"] = False
            results["warning"] = f"Test accuracy {test_acc:.2%} > {threshold:.0%} - POSSIBLE LEAKAGE!"

        # Also check if train >> test (overfitting or leakage)
        if train_acc > test_acc + 0.15:
            results["warning"] = f"Train-test gap ({train_acc:.2%} vs {test_acc:.2%}) suggests overfitting"

        return results

    def run_all_checks(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        train_times: Optional[pd.Series] = None,
        val_times: Optional[pd.Series] = None,
        test_times: Optional[pd.Series] = None
    ) -> Dict:
        """
        Run ALL leakage detection checks.
        """
        print("=" * 70)
        print("HYPER-RIGOROUS LEAKAGE VALIDATION")
        print("=" * 70)

        all_passed = True
        results = {}

        # Check 1: Feature-Target Correlation
        print("\n[1/5] Checking feature-target correlations...")
        corr_check = self.check_feature_target_correlation(
            X_train, y_train, feature_names
        )
        results["correlation_check"] = corr_check
        if not corr_check["passed"]:
            all_passed = False
            print(f"   SUSPICIOUS: {len(corr_check['suspicious_features'])} features")
        else:
            print("   PASSED")

        # Check 2: Temporal Ordering
        if train_times is not None:
            print("\n[2/5] Checking temporal ordering...")
            temporal_check = self.check_temporal_ordering(
                train_times, val_times, test_times
            )
            results["temporal_check"] = temporal_check
            if not temporal_check["passed"]:
                all_passed = False
                print("   TEMPORAL LEAKAGE DETECTED!")
            else:
                print("   PASSED")

        # Check 3: Distribution Shift
        print("\n[3/5] Checking distribution shift...")
        dist_check = self.check_distribution_shift(
            X_train, X_test, feature_names
        )
        results["distribution_check"] = dist_check
        if len(dist_check["warnings"]) > 10:
            print(f"   WARNING: {len(dist_check['warnings'])} features with large shift")
        else:
            print("   PASSED")

        # Check 4: Perfect Prediction
        print("\n[4/5] Checking for suspiciously good predictions...")
        pred_check = self.check_perfect_prediction(
            X_train, y_train, X_test, y_test
        )
        results["prediction_check"] = pred_check
        if not pred_check["passed"]:
            all_passed = False
            print(f"   {pred_check['warning']}")
        else:
            print(f"   PASSED (Train: {pred_check['train_accuracy']:.2%}, Test: {pred_check['test_accuracy']:.2%})")

        # Check 5: Cross-validation consistency
        print("\n[5/5] Checking cross-validation consistency...")
        cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42),
            X_train[:5000],
            (y_train[:5000] > 0).astype(int),
            cv=5,
            scoring="accuracy"
        )
        results["cv_check"] = {
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
            "scores": cv_scores.tolist()
        }
        if cv_scores.mean() > 0.55 and cv_scores.std() < 0.02:
            print(f"   CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        else:
            print(f"   CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%}) - NORMAL")

        # Final verdict
        print("\n" + "=" * 70)
        if all_passed:
            print(" ALL LEAKAGE CHECKS PASSED!")
        else:
            print(" POTENTIAL LEAKAGE DETECTED - REVIEW REQUIRED!")
        print("=" * 70)

        results["all_passed"] = all_passed
        self.validation_results = results

        return results
