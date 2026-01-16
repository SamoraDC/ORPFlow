#!/usr/bin/env python3
"""
Quick evaluation script for trained models.
Generates GO/NO-GO readiness report.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def load_and_prepare_data(data_path: Path, seed: int = 42):
    """Load data and prepare train/val/test splits."""
    np.random.seed(seed)

    df = pd.read_parquet(data_path)

    # Set index
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.set_index("open_time").sort_index()

    # Create basic features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["returns"].rolling(20).std()
    df["momentum"] = df["close"].pct_change(10)
    df["rsi"] = compute_rsi(df["close"], 14)
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]

    # Target
    df["target"] = df["close"].pct_change(5).shift(-5)

    # Feature columns
    feature_cols = ["open", "high", "low", "close", "volume",
                    "returns", "log_returns", "volatility", "momentum",
                    "rsi", "volume_ratio"]

    # Remove NaN
    df = df.dropna()

    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    # Replace inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Temporal split with embargo
    n = len(X)
    embargo = int(n * 0.01)

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_idx = np.arange(0, train_end - embargo)
    val_idx = np.arange(train_end + embargo, val_end - embargo)
    test_idx = np.arange(val_end + embargo, n)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
    }


def compute_rsi(prices, period=14):
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)

    # Regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Trading metrics
    direction_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))

    # Strategy returns (simple long/short based on prediction sign)
    strategy_returns = y_test * np.sign(y_pred)
    sharpe = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)) * np.sqrt(252 * 24 * 12)

    # Win rate
    win_rate = np.mean(strategy_returns > 0)

    # Max drawdown
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown)

    # Profit factor
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = np.abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gains / (losses + 1e-8)

    return {
        "model": model_name,
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "direction_accuracy": float(direction_accuracy),
        "sharpe_ratio": float(sharpe),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "profit_factor": float(profit_factor),
        "total_return": float(cumulative[-1]) if len(cumulative) > 0 else 0,
    }


def determine_go_nogo(metrics: dict) -> tuple:
    """
    Determine GO/NO-GO based on metrics.

    Criteria:
    - Direction accuracy > 0.50 (better than random)
    - Sharpe ratio > 0.0 (positive risk-adjusted return)
    - Win rate > 0.45
    - Profit factor > 0.8
    """
    checks = {
        "direction_accuracy": metrics["direction_accuracy"] > 0.50,
        "sharpe_positive": metrics["sharpe_ratio"] > 0.0,
        "win_rate": metrics["win_rate"] > 0.45,
        "profit_factor": metrics["profit_factor"] > 0.8,
    }

    passed = sum(checks.values())
    total = len(checks)

    # GO if at least 3/4 criteria pass
    decision = "GO" if passed >= 3 else "NO-GO"

    return decision, checks, f"{passed}/{total} criteria passed"


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("ORPFlow Model Evaluation & Readiness Report")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    data_path = PROJECT_ROOT / "data" / "raw" / "klines_90d.parquet"
    print(f"Loading data from {data_path}...")

    data = load_and_prepare_data(data_path)
    print(f"  Train: {len(data['X_train']):,} samples")
    print(f"  Val: {len(data['X_val']):,} samples")
    print(f"  Test: {len(data['X_test']):,} samples")
    print()

    results = []

    # Evaluate LightGBM
    print("[1/2] Evaluating LightGBM...")
    try:
        import pickle
        lgb_path = PROJECT_ROOT / "trained" / "models" / "lightgbm_model.pkl"
        with open(lgb_path, "rb") as f:
            lgb_model = pickle.load(f)

        lgb_metrics = evaluate_model(lgb_model, data["X_test"], data["y_test"], "lightgbm")
        lgb_decision, lgb_checks, lgb_summary = determine_go_nogo(lgb_metrics)
        lgb_metrics["decision"] = lgb_decision
        lgb_metrics["checks"] = lgb_checks
        results.append(lgb_metrics)

        print(f"  Direction Accuracy: {lgb_metrics['direction_accuracy']:.2%}")
        print(f"  Sharpe Ratio: {lgb_metrics['sharpe_ratio']:.4f}")
        print(f"  Win Rate: {lgb_metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {lgb_metrics['profit_factor']:.2f}")
        print(f"  Decision: {lgb_decision} ({lgb_summary})")
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"model": "lightgbm", "decision": "NO-GO", "error": str(e)})
    print()

    # Evaluate XGBoost
    print("[2/2] Evaluating XGBoost...")
    try:
        xgb_path = PROJECT_ROOT / "trained" / "models" / "xgboost_model.pkl"
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)

        xgb_metrics = evaluate_model(xgb_model, data["X_test"], data["y_test"], "xgboost")
        xgb_decision, xgb_checks, xgb_summary = determine_go_nogo(xgb_metrics)
        xgb_metrics["decision"] = xgb_decision
        xgb_metrics["checks"] = xgb_checks
        results.append(xgb_metrics)

        print(f"  Direction Accuracy: {xgb_metrics['direction_accuracy']:.2%}")
        print(f"  Sharpe Ratio: {xgb_metrics['sharpe_ratio']:.4f}")
        print(f"  Win Rate: {xgb_metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {xgb_metrics['profit_factor']:.2f}")
        print(f"  Decision: {xgb_decision} ({xgb_summary})")
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"model": "xgboost", "decision": "NO-GO", "error": str(e)})
    print()

    # Summary
    print("=" * 70)
    print("READINESS SUMMARY")
    print("=" * 70)

    for r in results:
        status = "✓" if r.get("decision") == "GO" else "✗"
        print(f"  {status} {r['model'].upper()}: {r.get('decision', 'ERROR')}")

    go_count = sum(1 for r in results if r.get("decision") == "GO")
    total_count = len(results)

    print()
    print(f"Overall: {go_count}/{total_count} models ready for deployment")

    if go_count == total_count:
        print("\n*** ALL MODELS READY FOR RENDER DEPLOYMENT ***")
    else:
        print("\n*** SOME MODELS NEED IMPROVEMENT BEFORE DEPLOYMENT ***")

    print("=" * 70)

    # Save results
    report_path = PROJECT_ROOT / "trained" / "readiness_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "go_count": go_count,
                "total_count": total_count,
                "ready_for_deploy": go_count == total_count,
            }
        }, f, indent=2)

    print(f"\nReport saved to {report_path}")

    return results


if __name__ == "__main__":
    main()
