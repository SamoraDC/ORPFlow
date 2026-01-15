#!/usr/bin/env python3
"""
Direct training and evaluation script.
Trains ML models and generates GO/NO-GO report.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def compute_rsi(prices, period=14):
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def load_and_prepare_data(data_path: Path, seed: int = 42):
    """Load data and prepare train/val/test splits."""
    np.random.seed(seed)

    df = pd.read_parquet(data_path)

    # Set index
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.set_index("open_time").sort_index()

    # Create features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["returns"].rolling(20).std()
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)
    df["rsi"] = compute_rsi(df["close"], 14)
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_ratio"] = df["close"] / df["open"]

    # Target: 5-bar forward return
    df["target"] = df["close"].pct_change(5).shift(-5)

    # Feature columns
    feature_cols = [
        "returns", "log_returns", "volatility",
        "momentum_5", "momentum_10", "momentum_20",
        "rsi", "volume_ratio", "high_low_ratio", "close_open_ratio"
    ]

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

    # Normalize (fit on train only)
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


def evaluate_predictions(y_true, y_pred, model_name):
    """Evaluate predictions and compute metrics."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Trading metrics
    direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

    # Strategy returns
    strategy_returns = y_true * np.sign(y_pred)
    sharpe = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)) * np.sqrt(252 * 24 * 12)

    # Win rate
    win_rate = np.mean(strategy_returns > 0)

    # Max drawdown
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Profit factor
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = np.abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gains / (losses + 1e-8)

    return {
        "model": model_name,
        "mse": float(mse),
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
    """Determine GO/NO-GO based on metrics."""
    checks = {
        "direction_accuracy": metrics["direction_accuracy"] > 0.50,
        "sharpe_positive": metrics["sharpe_ratio"] > 0.0,
        "win_rate": metrics["win_rate"] > 0.45,
        "profit_factor": metrics["profit_factor"] > 0.8,
    }

    passed = sum(checks.values())
    total = len(checks)

    decision = "GO" if passed >= 3 else "NO-GO"
    return decision, checks, f"{passed}/{total} criteria passed"


def main():
    """Main training and evaluation."""
    print("=" * 70)
    print("ORPFlow - Training & Readiness Report")
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
    print(f"  Features: {len(data['feature_names'])}")
    print()

    results = []
    output_dir = PROJECT_ROOT / "trained" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Train LightGBM
    # =========================================================================
    print("[1/2] Training LightGBM...")

    lgb_train = lgb.Dataset(data["X_train"], data["y_train"])
    lgb_val = lgb.Dataset(data["X_val"], data["y_val"], reference=lgb_train)

    lgb_params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )

    # Evaluate
    lgb_pred = lgb_model.predict(data["X_test"])
    lgb_metrics = evaluate_predictions(data["y_test"], lgb_pred, "lightgbm")
    lgb_decision, lgb_checks, lgb_summary = determine_go_nogo(lgb_metrics)
    lgb_metrics["decision"] = lgb_decision
    lgb_metrics["checks"] = lgb_checks
    results.append(lgb_metrics)

    # Save model
    lgb_model.save_model(str(output_dir / "lightgbm_model.txt"))

    print(f"  MSE: {lgb_metrics['mse']:.6f}")
    print(f"  Direction Accuracy: {lgb_metrics['direction_accuracy']:.2%}")
    print(f"  Sharpe Ratio: {lgb_metrics['sharpe_ratio']:.4f}")
    print(f"  Win Rate: {lgb_metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {lgb_metrics['profit_factor']:.2f}")
    print(f"  Decision: {lgb_decision} ({lgb_summary})")
    print()

    # =========================================================================
    # Train XGBoost
    # =========================================================================
    print("[2/2] Training XGBoost...")

    xgb_train = xgb.DMatrix(data["X_train"], label=data["y_train"])
    xgb_val = xgb.DMatrix(data["X_val"], label=data["y_val"])

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }

    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=100,
        evals=[(xgb_val, "val")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    # Evaluate
    xgb_test = xgb.DMatrix(data["X_test"])
    xgb_pred = xgb_model.predict(xgb_test)
    xgb_metrics = evaluate_predictions(data["y_test"], xgb_pred, "xgboost")
    xgb_decision, xgb_checks, xgb_summary = determine_go_nogo(xgb_metrics)
    xgb_metrics["decision"] = xgb_decision
    xgb_metrics["checks"] = xgb_checks
    results.append(xgb_metrics)

    # Save model
    xgb_model.save_model(str(output_dir / "xgboost_model.json"))

    print(f"  MSE: {xgb_metrics['mse']:.6f}")
    print(f"  Direction Accuracy: {xgb_metrics['direction_accuracy']:.2%}")
    print(f"  Sharpe Ratio: {xgb_metrics['sharpe_ratio']:.4f}")
    print(f"  Win Rate: {xgb_metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {xgb_metrics['profit_factor']:.2f}")
    print(f"  Decision: {xgb_decision} ({xgb_summary})")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("READINESS SUMMARY")
    print("=" * 70)

    for r in results:
        status = "✓" if r.get("decision") == "GO" else "✗"
        print(f"  {status} {r['model'].upper()}: {r.get('decision', 'ERROR')}")
        for check, passed in r.get("checks", {}).items():
            check_status = "✓" if passed else "✗"
            print(f"      {check_status} {check}")

    go_count = sum(1 for r in results if r.get("decision") == "GO")
    total_count = len(results)

    print()
    print(f"Overall: {go_count}/{total_count} models ready for deployment")
    print("=" * 70)

    # Save report
    report_path = PROJECT_ROOT / "trained" / "readiness_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "data_source": str(data_path),
            "train_samples": len(data["X_train"]),
            "val_samples": len(data["X_val"]),
            "test_samples": len(data["X_test"]),
            "results": results,
            "summary": {
                "go_count": go_count,
                "total_count": total_count,
                "ready_for_deploy": go_count == total_count,
            }
        }, f, indent=2)

    print(f"\nReport saved to {report_path}")
    print(f"Models saved to {output_dir}")

    return results


if __name__ == "__main__":
    main()
