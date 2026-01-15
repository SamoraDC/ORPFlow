"""
Overfitting Detection Module
Detects signs of overfitting during and after training

Implements multiple detection strategies:
1. Train-validation gap monitoring
2. Learning curve analysis
3. Early stopping triggers
4. Generalization metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class OverfittingMetrics:
    """Container for overfitting detection metrics"""
    train_loss: float
    val_loss: float
    gap: float
    gap_ratio: float
    is_overfitting: bool
    severity: str  # "none", "mild", "moderate", "severe"


class OverfittingDetector:
    """
    Real-time overfitting detection during training.

    DETECTION CRITERIA:
    1. Train-Val Gap: val_loss >> train_loss
    2. Val Loss Increasing: val_loss going up while train_loss going down
    3. Generalization Ratio: val_loss / train_loss > threshold
    4. Plateau Detection: val_loss not improving for N epochs
    """

    def __init__(
        self,
        gap_threshold: float = 0.3,
        ratio_threshold: float = 2.0,
        patience: int = 10,
        min_delta: float = 1e-4
    ):
        self.gap_threshold = gap_threshold
        self.ratio_threshold = ratio_threshold
        self.patience = patience
        self.min_delta = min_delta

        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Trend analysis
        self.recent_train = deque(maxlen=5)
        self.recent_val = deque(maxlen=5)

    def reset(self):
        """Reset detector state"""
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.recent_train.clear()
        self.recent_val.clear()

    def update(self, train_loss: float, val_loss: float) -> OverfittingMetrics:
        """
        Update detector with new epoch metrics.

        Returns OverfittingMetrics with current status.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.recent_train.append(train_loss)
        self.recent_val.append(val_loss)

        # Update best val loss
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Calculate metrics
        gap = val_loss - train_loss
        gap_ratio = val_loss / (train_loss + 1e-8)

        # Determine severity
        is_overfitting = False
        severity = "none"

        if gap > self.gap_threshold:
            is_overfitting = True
            if gap > self.gap_threshold * 3:
                severity = "severe"
            elif gap > self.gap_threshold * 2:
                severity = "moderate"
            else:
                severity = "mild"

        if gap_ratio > self.ratio_threshold:
            is_overfitting = True
            if severity == "none":
                severity = "mild"

        # Check for divergence (val going up while train going down)
        if len(self.recent_train) >= 3 and len(self.recent_val) >= 3:
            train_trend = self.recent_train[-1] - self.recent_train[0]
            val_trend = self.recent_val[-1] - self.recent_val[0]

            if train_trend < 0 and val_trend > 0:
                is_overfitting = True
                severity = "moderate" if severity == "none" else severity

        return OverfittingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            gap=gap,
            gap_ratio=gap_ratio,
            is_overfitting=is_overfitting,
            severity=severity
        )

    def should_stop_early(self) -> Tuple[bool, str]:
        """
        Determine if training should stop early.

        Returns (should_stop, reason)
        """
        # Check patience
        if self.epochs_without_improvement >= self.patience:
            return True, f"No improvement for {self.patience} epochs"

        # Check severe overfitting
        if len(self.val_losses) > 5:
            recent_gaps = [
                self.val_losses[i] - self.train_losses[i]
                for i in range(-5, 0)
            ]
            if all(g > self.gap_threshold * 2 for g in recent_gaps):
                return True, "Severe overfitting detected"

        # Check val loss explosion
        if len(self.val_losses) > 1:
            if self.val_losses[-1] > self.val_losses[0] * 5:
                return True, "Validation loss exploded"

        return False, ""

    def get_summary(self) -> Dict:
        """Get training summary with overfitting analysis"""
        if not self.train_losses:
            return {"error": "No data recorded"}

        summary = {
            "total_epochs": len(self.train_losses),
            "best_val_loss": float(self.best_val_loss),
            "final_train_loss": float(self.train_losses[-1]),
            "final_val_loss": float(self.val_losses[-1]),
            "final_gap": float(self.val_losses[-1] - self.train_losses[-1]),
            "final_ratio": float(self.val_losses[-1] / (self.train_losses[-1] + 1e-8)),
            "epochs_without_improvement": self.epochs_without_improvement
        }

        # Trend analysis
        if len(self.train_losses) >= 10:
            mid = len(self.train_losses) // 2

            early_train_avg = np.mean(self.train_losses[:mid])
            late_train_avg = np.mean(self.train_losses[mid:])
            early_val_avg = np.mean(self.val_losses[:mid])
            late_val_avg = np.mean(self.val_losses[mid:])

            summary["train_trend"] = "decreasing" if late_train_avg < early_train_avg else "increasing"
            summary["val_trend"] = "decreasing" if late_val_avg < early_val_avg else "increasing"

            # Overfitting indicator
            if summary["train_trend"] == "decreasing" and summary["val_trend"] == "increasing":
                summary["overfitting_detected"] = True
                summary["recommendation"] = "STOP TRAINING - Clear overfitting pattern"
            elif summary["final_gap"] > self.gap_threshold:
                summary["overfitting_detected"] = True
                summary["recommendation"] = "Consider regularization or early stopping"
            else:
                summary["overfitting_detected"] = False
                summary["recommendation"] = "Training appears healthy"

        return summary

    def print_status(self, epoch: int):
        """Print current overfitting status"""
        if not self.train_losses:
            return

        train = self.train_losses[-1]
        val = self.val_losses[-1]
        gap = val - train

        status = ""
        if gap > self.gap_threshold * 2:
            status = " OVERFITTING!"
        elif gap > self.gap_threshold:
            status = " Warning"
        elif self.epochs_without_improvement > self.patience // 2:
            status = " Plateau"

        print(f"Epoch {epoch}: Train={train:.6f}, Val={val:.6f}, Gap={gap:.6f}{status}")


class GeneralizationAnalyzer:
    """
    Analyze model generalization after training.
    """

    @staticmethod
    def analyze(
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict
    ) -> Dict:
        """
        Analyze generalization across all splits.

        Good generalization: train ≈ val ≈ test
        Overfitting: train << val, train << test
        Data issues: val << test or test << val
        """
        analysis = {
            "passed": True,
            "issues": [],
            "metrics_comparison": {}
        }

        # Compare key metrics
        for metric in ["mse", "mae", "sharpe_ratio"]:
            if metric not in train_metrics:
                continue

            train_val = train_metrics.get(metric, 0)
            val_val = val_metrics.get(metric, 0)
            test_val = test_metrics.get(metric, 0)

            analysis["metrics_comparison"][metric] = {
                "train": train_val,
                "val": val_val,
                "test": test_val
            }

            # Check for issues
            if metric in ["mse", "mae"]:  # Lower is better
                if train_val < val_val * 0.5:
                    analysis["issues"].append(
                        f"{metric}: Train much better than Val - possible overfitting"
                    )
                    analysis["passed"] = False

                if val_val < test_val * 0.7:
                    analysis["issues"].append(
                        f"{metric}: Val much better than Test - possible data shift"
                    )

            else:  # Higher is better (sharpe, etc)
                if train_val > val_val * 2:
                    analysis["issues"].append(
                        f"{metric}: Train much better than Val - possible overfitting"
                    )
                    analysis["passed"] = False

        # Overall assessment
        if not analysis["issues"]:
            analysis["verdict"] = "GOOD GENERALIZATION"
        elif len(analysis["issues"]) == 1:
            analysis["verdict"] = "MINOR CONCERNS"
        else:
            analysis["verdict"] = "POTENTIAL OVERFITTING"

        return analysis
