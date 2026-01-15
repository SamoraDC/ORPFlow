# ORPFlow Training Runbook

**Version:** 1.0.0
**Last Updated:** January 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Data Preparation](#3-data-preparation)
4. [Training Commands](#4-training-commands)
5. [Validation Framework](#5-validation-framework)
6. [ONNX Export & Deployment](#6-onnx-export--deployment)
7. [Reproducing Experiments](#7-reproducing-experiments)
8. [Troubleshooting](#8-troubleshooting)
9. [Metrics Reference](#9-metrics-reference)

---

## 1. Overview

### Pipeline Architecture

ORPFlow implements a comprehensive quantitative trading pipeline combining Machine Learning (ML), Deep Learning (DL), and Reinforcement Learning (RL) models with rigorous validation using Combinatorial Purged Cross-Validation (CPCV).

```
                           ORPFlow Training Pipeline
+===========================================================================+
|                                                                           |
|   DATA LAYER                                                              |
|   +-------+     +------------------+     +-------------------+           |
|   | Klines| --> | Feature Engineer | --> | Quant Features    |           |
|   | (LFS) |     | quant_features   |     | + Microstructure  |           |
|   +-------+     +------------------+     +-------------------+           |
|                                                 |                        |
+===========================================================================+
|                                                 v                        |
|   VALIDATION LAYER                                                       |
|   +-------------------+     +-------------------+                        |
|   | Leakage Guards    | --> | CPCV Validator    |                        |
|   | - Temporal Order  |     | - Embargo/Purge   |                        |
|   | - Feature Leak    |     | - PBO/DSR         |                        |
|   | - Normalization   |     | - NSMI            |                        |
|   +-------------------+     +-------------------+                        |
|                                    |                                     |
+===========================================================================+
|                                    v                                     |
|   TRAINING LAYER                                                         |
|   +------------------+    +------------------+    +------------------+   |
|   | ML Models        |    | DL Models        |    | RL Agents        |   |
|   | - LightGBM       |    | - LSTM           |    | - D4PG + EVT     |   |
|   | - XGBoost        |    | - CNN            |    | - MARL System    |   |
|   +------------------+    +------------------+    +------------------+   |
|             |                     |                       |              |
+===========================================================================+
|             v                     v                       v              |
|   EXPORT LAYER                                                           |
|   +------------------+    +------------------+    +------------------+   |
|   | ONNX Export      | -> | Parity Testing   | -> | Rust Integration |   |
|   | - onnxmltools    |    | - MAE < 1e-6     |    | - tract/ort      |   |
|   | - torch.onnx     |    | - Max diff < 1e-5|    | - NSMI hot path  |   |
|   +------------------+    +------------------+    +------------------+   |
|                                                                          |
+===========================================================================+
```

### Data Flow Diagram

```
                              Data Flow Architecture

    +--------+
    | Binance|
    | API    |
    +---+----+
        |
        v
+-------+--------+
| data/raw/      |     Raw kline data (OHLCV + volume metrics)
| klines_90d.parq|     Stored via Git LFS
+-------+--------+
        |
        | preprocessor.py
        v
+-------+--------+
| QuantFeatures  |     Technical indicators, returns, volatility
| Microstructure |     Order flow, VPIN, Kyle's Lambda
+-------+--------+
        |
        | leakage_guards.py
        v
+-------+--------+
| Leakage Check  |     Temporal ordering, feature-target scan
| Validation     |     Normalization leak detection
+-------+--------+
        |
        | cpcv.py
        v
+-------+--------+     +------------------+
| CPCV Folds     |---->| Train/Val/Test   |
| (Purged+Embargo)|    | Splits           |
+-------+--------+     +--------+---------+
        |                       |
        |   +-----------+-------+-----------+
        |   |           |                   |
        v   v           v                   v
   +----+---+--+   +----+------+      +-----+------+
   | LightGBM  |   | LSTM/CNN  |      | D4PG/MARL  |
   | XGBoost   |   | PyTorch   |      | RL Agents  |
   +-----------+   +-----------+      +------------+
        |               |                   |
        +-------+-------+-------+-----------+
                |               |
                v               v
        +-------+-----+   +-----+-------+
        | trained/    |   | trained/    |
        | models/     |   | onnx/       |
        +-------------+   +-------------+
                               |
                               v
                     +-------------------+
                     | Rust Strategy     |
                     | NSMI Hot Path     |
                     +-------------------+
```

### Model Registry Summary

| Model | Category | Input Type | ONNX | GPU | Typical Training Time |
|-------|----------|------------|------|-----|----------------------|
| LightGBM | ML | Tabular | Yes | Yes | 1-5 min / 100K samples |
| XGBoost | ML | Tabular | Yes | Yes | 1-5 min / 100K samples |
| LSTM | DL | Sequence | Yes | Yes | 10-30 min / 100K samples |
| CNN | DL | Sequence | Yes | Yes | 5-20 min / 100K samples |
| D4PG + EVT | RL | State | Yes | Yes | 1-4 hours / 200 episodes |
| MARL System | RL | Multi-Agent | Yes | Yes | 2-6 hours / 200 episodes |

---

## 2. Prerequisites

### Python Dependencies

**Core Requirements:**

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually:
pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0
pip install scikit-learn>=1.3.0 joblib>=1.3.0
pip install lightgbm>=4.0.0 xgboost>=2.0.0
pip install torch>=2.0.0 tqdm>=4.65.0
pip install pyarrow>=12.0.0  # For parquet support
```

**ONNX Export Dependencies (Optional):**

```bash
pip install onnx>=1.14.0
pip install onnxmltools>=1.11.0      # For LightGBM/XGBoost
pip install onnxruntime>=1.15.0      # For validation
pip install onnxruntime-tools>=1.7.0 # For optimization
```

**Verification:**

```python
# Verify installation
from models.registry import ModelRegistry
registry = ModelRegistry()
print(registry.summary())

# Check dependencies for specific model
deps = registry.get_dependencies("lstm")
print(f"Required: {deps['required']}")
print(f"Optional: {deps['optional']}")
```

### Rust Toolchain

For NSMI hot path execution:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version  # Should be >= 1.70.0
cargo --version

# Required crates (in Cargo.toml)
# ort = "1.16"          # ONNX Runtime bindings
# tract-onnx = "0.20"   # Pure Rust ONNX inference (alternative)
# ndarray = "0.15"      # N-dimensional arrays
# polars = "0.35"       # DataFrame operations
```

### Data Requirements

**Raw Data:**

| File | Path | Size | Source | Notes |
|------|------|------|--------|-------|
| klines_90d.parquet | `data/raw/` | ~50MB | Binance API | Stored via Git LFS |

**Expected Columns in klines_90d.parquet:**

```
open_time       : int64     # Unix timestamp (ms)
open            : float64   # Open price
high            : float64   # High price
low             : float64   # Low price
close           : float64   # Close price
volume          : float64   # Base asset volume
close_time      : int64     # Close timestamp (ms)
quote_volume    : float64   # Quote asset volume
trades          : int64     # Number of trades
taker_buy_base  : float64   # Taker buy base volume
taker_buy_quote : float64   # Taker buy quote volume
```

**Fetching Data:**

```bash
# If using Git LFS
git lfs pull

# Verify data
python -c "import pandas as pd; df = pd.read_parquet('data/raw/klines_90d.parquet'); print(df.info())"
```

---

## 3. Data Preparation

### Expected Data Format

**Input Schema:**

```python
import pandas as pd

# Required columns for feature engineering
REQUIRED_COLUMNS = [
    'open_time',   # Timestamp (datetime or int64 ms)
    'open',        # float64
    'high',        # float64
    'low',         # float64
    'close',       # float64
    'volume',      # float64
]

# Optional but recommended
OPTIONAL_COLUMNS = [
    'quote_volume',
    'trades',
    'taker_buy_base',
    'taker_buy_quote',
]
```

### How to Verify Data Integrity

```python
from pathlib import Path
import pandas as pd
import numpy as np

def verify_data_integrity(path: str) -> dict:
    """Verify data file integrity for training."""

    df = pd.read_parquet(path)
    issues = []

    # 1. Check required columns
    required = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    # 2. Check for NaN values
    nan_counts = df[required].isna().sum()
    if nan_counts.sum() > 0:
        issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

    # 3. Check temporal ordering
    if 'open_time' in df.columns:
        if not df['open_time'].is_monotonic_increasing:
            issues.append("Data is not sorted by time")

    # 4. Check OHLC validity (H >= max(O, C), L <= min(O, C))
    invalid_ohlc = (
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()
    if invalid_ohlc > 0:
        issues.append(f"Invalid OHLC bars: {invalid_ohlc}")

    # 5. Check for duplicates
    duplicates = df.duplicated(subset=['open_time']).sum()
    if duplicates > 0:
        issues.append(f"Duplicate timestamps: {duplicates}")

    return {
        "path": path,
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": f"{df['open_time'].min()} to {df['open_time'].max()}",
        "issues": issues,
        "valid": len(issues) == 0,
    }

# Usage
result = verify_data_integrity("data/raw/klines_90d.parquet")
print(f"Valid: {result['valid']}")
for issue in result['issues']:
    print(f"  - {issue}")
```

### Feature Engineering Process

The feature engineering pipeline is implemented in `models/features/`:

**1. Quantitative Features (`quant_features.py`):**

```python
from models.features.quant_features import QuantFeatureEngineer

engineer = QuantFeatureEngineer(
    return_windows=[1, 5, 15, 60],      # Return calculation windows
    volatility_windows=[20, 60],        # Volatility windows
    momentum_windows=[14, 28],          # Momentum windows
    volume_windows=[20],                # Volume analysis windows
)

# Generate features
features_df = engineer.transform(df)
print(f"Generated {len(engineer.feature_names)} features")
```

**2. Microstructure Features (`microstructure.py`):**

```python
from models.features.microstructure import MicrostructureFeatures

micro = MicrostructureFeatures(
    vpin_buckets=50,                    # VPIN volume buckets
    kyle_window=100,                    # Kyle's Lambda window
    roll_window=100,                    # Roll spread window
)

# Add microstructure features
features_df = micro.transform(features_df)
```

**Generated Feature Categories:**

| Category | Examples | Count |
|----------|----------|-------|
| Returns | return_1, return_5, return_15, return_60 | 4 |
| Volatility | realized_vol_20, realized_vol_60, parkinson_vol | 5 |
| Momentum | rsi_14, rsi_28, macd, macd_signal, macd_hist | 8 |
| Moving Averages | sma_20, sma_50, ema_12, ema_26 | 6 |
| Bollinger Bands | bb_upper, bb_middle, bb_lower, bb_width | 4 |
| ATR/ADX | atr_14, adx_14, plus_di, minus_di | 4 |
| Volume | volume_sma_20, volume_ratio, obv | 5 |
| Microstructure | vpin, kyle_lambda, roll_spread, amihud | 8 |
| **Total** | | **~44** |

---

## 4. Training Commands

### Quick Start (One Command)

Train all models with default configuration:

```bash
# Full pipeline - trains all models with CPCV validation
python scripts/train_all.py

# With custom config
python scripts/train_all.py --config config/training.yaml

# Specify output directory
python scripts/train_all.py --output-dir trained/experiment_001
```

### Training Individual Models

**Machine Learning Models:**

```bash
# LightGBM
python -m models.ml.lightgbm_model --data data/processed/features.parquet \
    --target target_return_5 \
    --n-estimators 1000 \
    --early-stopping 50 \
    --output trained/models/lightgbm_model.pkl

# XGBoost
python -m models.ml.xgboost_model --data data/processed/features.parquet \
    --target target_return_5 \
    --n-estimators 1000 \
    --max-depth 6 \
    --output trained/models/xgboost_model.pkl
```

**Deep Learning Models:**

```bash
# LSTM
python -m models.dl.lstm_model --data data/processed/features.parquet \
    --sequence-length 60 \
    --hidden-size 128 \
    --num-layers 2 \
    --epochs 100 \
    --batch-size 64 \
    --device cuda \
    --output trained/models/lstm_model.pt

# CNN
python -m models.dl.cnn_model --data data/processed/features.parquet \
    --sequence-length 60 \
    --conv-channels 32,64,128 \
    --epochs 100 \
    --device cuda \
    --output trained/models/cnn_model.pt
```

**Reinforcement Learning Agents:**

```bash
# D4PG + EVT
python -m models.rl.d4pg_evt --env-config config/trading_env.yaml \
    --episodes 200 \
    --state-dim 64 \
    --hidden-dim 256 \
    --risk-aversion 0.5 \
    --device cuda \
    --output trained/models/d4pg_agent.pt

# MARL System
python -m models.rl.marl --env-config config/trading_env.yaml \
    --episodes 200 \
    --n-agents 5 \
    --state-dim 64 \
    --device cuda \
    --output trained/models/marl_system.pt
```

### Training with Custom Config

**Example Configuration File (`config/training.yaml`):**

```yaml
# Data Configuration
data:
  raw_path: data/raw/klines_90d.parquet
  processed_path: data/processed/features.parquet
  target_column: target_return_5

# Feature Engineering
features:
  return_windows: [1, 5, 15, 60]
  volatility_windows: [20, 60]
  include_microstructure: true

# Validation Configuration
validation:
  method: cpcv
  n_splits: 5
  embargo_pct: 0.01
  purge_pct: 0.01
  test_size: 0.2

# Model Configuration
models:
  lightgbm:
    enabled: true
    params:
      objective: regression
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 1000
      early_stopping_rounds: 50

  lstm:
    enabled: true
    params:
      hidden_size: 128
      num_layers: 2
      dropout: 0.2
      epochs: 100
      batch_size: 64
      learning_rate: 0.001

  d4pg:
    enabled: true
    params:
      hidden_dim: 256
      n_atoms: 51
      risk_aversion: 0.5
      episodes: 200

# ONNX Export
export:
  enabled: true
  output_dir: trained/onnx
  quantize: true
  optimize: true

# Reproducibility
seed: 42
```

**Running with Custom Config:**

```bash
python scripts/train_all.py --config config/training.yaml
```

### Training Using the Orchestrator

```python
from models.training.orchestrator import TrainingOrchestrator

# Initialize orchestrator
orchestrator = TrainingOrchestrator(
    data_path="data/processed/features.parquet",
    output_dir="trained/experiment_001",
    config_path="config/training.yaml",
)

# Run full pipeline
results = orchestrator.run_pipeline(
    models=["lightgbm", "xgboost", "lstm", "d4pg"],
    validate=True,
    export_onnx=True,
)

# View results
print(f"Best model: {results['best_model']}")
print(f"Metrics: {results['metrics']}")
```

---

## 5. Validation Framework

### CPCV Explanation

**Combinatorial Purged Cross-Validation (CPCV)** addresses financial ML's unique challenges:

1. **Non-IID Data**: Financial returns are serially correlated
2. **Information Leakage**: Labels often span multiple observations
3. **Temporal Dependency**: Future information cannot be used

```
Traditional CV (WRONG for finance):
  Fold 1: [Train: 1-80] [Test: 81-100]   # Contaminated
  Fold 2: [Train: 21-100] [Test: 1-20]   # Future -> Past leak!

CPCV (CORRECT):
  Fold 1: [Train: 1-70] [Embargo] [Test: 75-100]  # Purged
  Fold 2: [Train: 1-40, 60-100] [Test: 45-55]     # Combinatorial
```

**Key Concepts:**

| Term | Description |
|------|-------------|
| **Purge** | Remove training samples whose labels overlap with test |
| **Embargo** | Add gap between train and test to prevent leakage |
| **Combinatorial** | All possible test set combinations for unbiased estimates |

### Embargo/Purge Settings

```python
from models.validation.cpcv import CPCV

# Initialize CPCV validator
cpcv = CPCV(
    n_splits=5,                  # Number of test groups
    embargo_pct=0.01,           # 1% embargo between splits
    purge_pct=0.01,             # Purge samples with overlapping labels
)

# Generate splits
for fold, (train_idx, test_idx) in enumerate(cpcv.split(X, y, times)):
    print(f"Fold {fold}: Train={len(train_idx)}, Test={len(test_idx)}")
```

**Recommended Settings by Data Frequency:**

| Frequency | Embargo % | Purge % | Notes |
|-----------|-----------|---------|-------|
| Daily | 1-2% | 1-2% | ~5-10 trading days |
| Hourly | 0.5-1% | 0.5-1% | ~12-24 hours |
| Minute | 0.1-0.5% | 0.1-0.5% | ~60-300 minutes |
| Tick | 0.01-0.1% | 0.01-0.1% | Depends on volume |

### Leakage Checks

The leakage guard suite (`models/validation/leakage_guards.py`) implements comprehensive checks:

**1. Temporal Order Validation:**

```python
from models.validation.leakage_guards import TemporalOrderValidator

validator = TemporalOrderValidator(min_embargo_bars=5, strict=True)
report = validator.validate(
    train_times=train_df['timestamp'],
    val_times=val_df['timestamp'],
    test_times=test_df['timestamp'],
)

print(f"Passed: {report.passed}")
print(f"Message: {report.message}")
```

**2. Feature-Target Leakage Scanner:**

```python
from models.validation.leakage_guards import FeatureTargetLeakageScanner

scanner = FeatureTargetLeakageScanner(
    correlation_threshold=0.95,  # Flag features with |corr| > 0.95
)

report = scanner.scan(X, y, feature_names)
for suspicious in report.details.get('suspicious', []):
    print(f"WARNING: {suspicious['feature']} - {suspicious['type']}")
```

**3. Normalization Leakage Detector:**

```python
from models.validation.leakage_guards import NormalizationLeakageDetector

detector = NormalizationLeakageDetector()
report = detector.detect(
    X_train,
    X_test,
    scaler=fitted_scaler,
)

if not report.passed:
    print("CRITICAL: Scaler may have seen test data!")
```

**4. Look-Ahead Bias Detector:**

```python
from models.validation.leakage_guards import LookAheadBiasDetector

detector = LookAheadBiasDetector(
    future_corr_ratio=2.0,  # Flag if future_corr > 2x past_corr
    lag_periods=10,
)

report = detector.detect(df, feature_cols, target_col)
```

**5. Complete Validation Suite:**

```python
from models.validation.leakage_guards import LeakageGuardSuite

suite = LeakageGuardSuite(
    min_embargo_bars=5,
    correlation_threshold=0.95,
    strict=True,
)

summary = suite.run_all_checks(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    train_times=train_times,
    val_times=val_times,
    test_times=test_times,
    scaler=fitted_scaler,
    n_strategies_tested=1,
)

summary.print_report()

if not summary.all_passed:
    print(f"FAILED: {summary.summary()['failed']} checks failed")
    for failure in summary.critical_failures:
        print(f"  CRITICAL: {failure.check_name}")
```

### PBO/DSR Interpretation

**Probability of Backtest Overfitting (PBO):**

```python
from models.validation.cpcv import compute_pbo

pbo_score, dsr_metrics = compute_pbo(
    train_sharpes=train_sharpe_ratios,
    test_sharpes=test_sharpe_ratios,
)

print(f"PBO: {pbo_score:.2%}")
```

**Interpretation Guide:**

| PBO | Interpretation | Action |
|-----|----------------|--------|
| < 10% | Low overfitting risk | Strategy likely robust |
| 10-30% | Moderate risk | Review feature selection |
| 30-50% | High risk | Simplify model, add regularization |
| > 50% | Very high risk | Likely spurious, reconsider approach |

**Deflated Sharpe Ratio (DSR):**

Adjusts Sharpe for multiple testing:

```python
from models.validation.cpcv import compute_dsr

dsr = compute_dsr(
    sharpe_ratio=observed_sharpe,
    n_trials=number_of_strategies_tested,
    n_observations=len(returns),
    var_sharpes=variance_of_sharpes,  # Optional
)

print(f"Observed Sharpe: {observed_sharpe:.2f}")
print(f"Deflated Sharpe: {dsr:.2f}")
```

**DSR Interpretation:**

| Condition | Interpretation |
|-----------|----------------|
| DSR > 0 | Statistically significant after adjusting for trials |
| DSR close to Sharpe | Few trials, results credible |
| DSR << Sharpe | Many trials, adjust expectations |

---

## 6. ONNX Export & Deployment

### Export Commands

**Using ONNXExporter:**

```python
from models.export.onnx_exporter import ONNXExporter

exporter = ONNXExporter(output_dir="trained/onnx")

# Export individual models
exporter.export_lightgbm(lgb_model, feature_names, "lightgbm_v1")
exporter.export_xgboost(xgb_model, feature_names, "xgboost_v1")
exporter.export_lstm(lstm_model, sequence_length=60, num_features=44, model_name="lstm_v1")
exporter.export_cnn(cnn_model, sequence_length=60, num_features=44, model_name="cnn_v1")
exporter.export_d4pg_actor(d4pg_agent, "d4pg_actor_v1")
exporter.export_marl_agents(marl_system, "marl_v1")

# Export all at once
exported = exporter.export_all(
    models={
        "lightgbm_v1": lgb_model,
        "xgboost_v1": xgb_model,
        "lstm_v1": lstm_model,
        "d4pg_v1": d4pg_agent,
    },
    feature_names=feature_names,
    sequence_length=60,
)
```

**Command Line Export:**

```bash
# Export all trained models
python -m models.export.onnx_exporter \
    --models-dir trained/models \
    --output-dir trained/onnx \
    --optimize \
    --quantize

# Export specific model
python -m models.export.onnx_exporter \
    --model trained/models/lstm_model.pt \
    --type lstm \
    --sequence-length 60 \
    --num-features 44 \
    --output trained/onnx/lstm_v1.onnx
```

### Parity Testing

Verify ONNX model produces identical outputs:

```python
import numpy as np
import onnxruntime as ort

def test_onnx_parity(
    original_model,
    onnx_path: str,
    test_input: np.ndarray,
    tolerance: float = 1e-6,
) -> dict:
    """Test ONNX export parity with original model."""

    # Original prediction
    original_output = original_model.predict(test_input)

    # ONNX prediction
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input.astype(np.float32)})[0]

    # Compare
    mae = np.mean(np.abs(original_output - onnx_output))
    max_diff = np.max(np.abs(original_output - onnx_output))

    return {
        "passed": mae < tolerance and max_diff < tolerance * 10,
        "mae": float(mae),
        "max_diff": float(max_diff),
        "tolerance": tolerance,
    }

# Usage
result = test_onnx_parity(
    original_model=lstm_model,
    onnx_path="trained/onnx/lstm_v1.onnx",
    test_input=X_test[:100],
)
print(f"Parity test: {'PASSED' if result['passed'] else 'FAILED'}")
print(f"MAE: {result['mae']:.2e}, Max diff: {result['max_diff']:.2e}")
```

**Parity Requirements:**

| Metric | Threshold | Notes |
|--------|-----------|-------|
| MAE | < 1e-6 | Mean absolute error |
| Max Diff | < 1e-5 | Maximum absolute difference |
| Relative Error | < 0.01% | For large outputs |

### Rust Integration

**Loading ONNX in Rust (using `ort` crate):**

```rust
use ort::{Environment, SessionBuilder, Value};
use ndarray::{Array2, ArrayD};

fn load_model(model_path: &str) -> Result<ort::Session, ort::Error> {
    let environment = Environment::builder()
        .with_name("trading_inference")
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_model_from_file(model_path)?;

    Ok(session)
}

fn predict(session: &ort::Session, input: Array2<f32>) -> Result<ArrayD<f32>, ort::Error> {
    let input_tensor = Value::from_array(session.allocator(), &input)?;
    let outputs = session.run(vec![input_tensor])?;
    let output = outputs[0].try_extract::<f32>()?;
    Ok(output.view().to_owned())
}

// Usage
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = load_model("trained/onnx/lightgbm_v1.onnx")?;

    let input = Array2::<f32>::zeros((1, 44)); // batch_size=1, features=44
    let prediction = predict(&session, input)?;

    println!("Prediction: {:?}", prediction);
    Ok(())
}
```

**Alternative: Using `tract-onnx` (Pure Rust):**

```rust
use tract_onnx::prelude::*;

fn load_tract_model(model_path: &str, input_shape: &[usize]) -> TractResult<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> {
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(0, f32::fact(input_shape))?
        .into_optimized()?
        .into_runnable()?;

    Ok(model)
}

fn predict_tract(model: &SimplePlan<...>, input: &[f32]) -> TractResult<Vec<f32>> {
    let input_tensor = tract_ndarray::Array::from_shape_vec(
        (1, input.len()),
        input.to_vec()
    )?.into_tensor();

    let result = model.run(tvec!(input_tensor.into()))?;
    let output: &[f32] = result[0].as_slice()?;

    Ok(output.to_vec())
}
```

### NSMI Hot Path

**Normalized Strategy Metadata Index (NSMI)** integration for real-time inference:

```rust
// src/strategy/nsmi.rs

pub struct NSMIHotPath {
    lightgbm_session: ort::Session,
    lstm_session: ort::Session,
    feature_scaler: StandardScaler,
    sequence_buffer: VecDeque<Vec<f32>>,
    sequence_length: usize,
}

impl NSMIHotPath {
    pub fn new(config: &NSMIConfig) -> Result<Self, Error> {
        let lightgbm_session = load_model(&config.lightgbm_path)?;
        let lstm_session = load_model(&config.lstm_path)?;
        let feature_scaler = StandardScaler::load(&config.scaler_path)?;

        Ok(Self {
            lightgbm_session,
            lstm_session,
            feature_scaler,
            sequence_buffer: VecDeque::with_capacity(config.sequence_length),
            sequence_length: config.sequence_length,
        })
    }

    /// Hot path inference - called on every tick
    #[inline]
    pub fn predict(&mut self, features: &[f32]) -> Result<Signal, Error> {
        // 1. Scale features
        let scaled = self.feature_scaler.transform(features);

        // 2. Update sequence buffer
        self.sequence_buffer.push_back(scaled.clone());
        if self.sequence_buffer.len() > self.sequence_length {
            self.sequence_buffer.pop_front();
        }

        // 3. ML prediction (tabular)
        let ml_pred = self.predict_lightgbm(&scaled)?;

        // 4. DL prediction (sequence) - only if buffer full
        let dl_pred = if self.sequence_buffer.len() == self.sequence_length {
            self.predict_lstm()?
        } else {
            0.0
        };

        // 5. Ensemble signal
        let signal = Signal {
            ml_score: ml_pred,
            dl_score: dl_pred,
            combined: 0.6 * ml_pred + 0.4 * dl_pred,
            timestamp: chrono::Utc::now(),
        };

        Ok(signal)
    }

    #[inline]
    fn predict_lightgbm(&self, features: &[f32]) -> Result<f32, Error> {
        let input = Array2::from_shape_vec((1, features.len()), features.to_vec())?;
        let output = predict(&self.lightgbm_session, input)?;
        Ok(output[[0, 0]])
    }

    #[inline]
    fn predict_lstm(&self) -> Result<f32, Error> {
        let sequence: Vec<f32> = self.sequence_buffer.iter().flatten().cloned().collect();
        let input = Array3::from_shape_vec(
            (1, self.sequence_length, self.feature_dim()),
            sequence
        )?;
        let output = predict_3d(&self.lstm_session, input)?;
        Ok(output[[0, 0]])
    }
}
```

---

## 7. Reproducing Experiments

### Seed Management

**Setting Global Seeds:**

```python
import random
import numpy as np
import torch

def set_all_seeds(seed: int = 42):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For CuDNN reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# In training script
set_all_seeds(42)
```

**Model-Specific Seeds:**

```python
# LightGBM
lgb_params = {
    "seed": 42,
    "bagging_seed": 42,
    "feature_fraction_seed": 42,
}

# XGBoost
xgb_params = {
    "seed": 42,
    "random_state": 42,
}

# PyTorch (per-model)
torch.manual_seed(42)
model = LSTMModel(input_size=64)  # Weights initialized with seed
```

### Data Hashing

**Verify Data Integrity:**

```python
import hashlib
import pandas as pd

def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute deterministic hash of DataFrame."""
    # Convert to bytes and hash
    data_bytes = df.to_parquet()  # Or df.to_csv(index=False).encode()
    return hashlib.sha256(data_bytes).hexdigest()

def verify_data_hash(path: str, expected_hash: str) -> bool:
    """Verify data file matches expected hash."""
    df = pd.read_parquet(path)
    actual_hash = compute_data_hash(df)
    return actual_hash == expected_hash

# Usage
data_hash = compute_data_hash(df)
print(f"Data hash: {data_hash}")

# In experiment tracking
experiment_config = {
    "data_hash": data_hash,
    "data_path": "data/raw/klines_90d.parquet",
    "features_hash": compute_data_hash(features_df),
}
```

### Config Tracking

**Experiment Configuration Schema:**

```yaml
# experiments/exp_001/config.yaml
experiment:
  name: "lstm_baseline"
  version: "1.0.0"
  timestamp: "2026-01-15T10:30:00Z"

reproducibility:
  seed: 42
  data_hash: "sha256:a1b2c3d4..."
  features_hash: "sha256:e5f6g7h8..."

environment:
  python: "3.11.0"
  torch: "2.1.0"
  numpy: "1.26.0"
  cuda: "12.1"

model:
  type: "lstm"
  params:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  early_stopping: 10

validation:
  method: "cpcv"
  n_splits: 5
  embargo_pct: 0.01
```

**Programmatic Config Tracking:**

```python
import yaml
import json
from datetime import datetime
from pathlib import Path

def save_experiment_config(
    output_dir: str,
    config: dict,
    model,
    data_hash: str,
):
    """Save complete experiment configuration."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect environment info
    import torch
    import numpy as np

    full_config = {
        "experiment": {
            "timestamp": datetime.utcnow().isoformat(),
            "output_dir": str(output_path),
        },
        "reproducibility": {
            "seed": config.get("seed", 42),
            "data_hash": data_hash,
        },
        "environment": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "config": config,
    }

    # Save YAML
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(full_config, f, default_flow_style=False)

    # Save JSON (machine-readable)
    with open(output_path / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)

    return full_config
```

### Artifact Versioning

**Directory Structure:**

```
trained/
  experiments/
    exp_001_lstm_baseline/
      config.yaml           # Full configuration
      config.json           # Machine-readable config
      models/
        lstm_model.pt       # PyTorch checkpoint
        lstm_model.onnx     # ONNX export
        scaler.pkl          # Feature scaler
      metrics/
        train_metrics.json  # Training metrics
        cpcv_results.json   # CPCV fold results
        leakage_report.json # Leakage validation
      logs/
        training.log        # Training log
        tensorboard/        # TensorBoard logs
      data/
        feature_names.json  # Feature list
        data_split_info.json # Train/val/test indices
```

**Version Control with DVC:**

```bash
# Initialize DVC
dvc init

# Track large files
dvc add data/raw/klines_90d.parquet
dvc add trained/experiments/exp_001

# Push to remote storage
dvc push

# Reproduce experiment
dvc repro  # Runs full pipeline
```

---

## 8. Troubleshooting

### Common Errors

**1. ImportError: Module Not Found**

```
Error: ModuleNotFoundError: No module named 'lightgbm'

Solution:
pip install lightgbm>=4.0.0

# For GPU support
pip install lightgbm --install-option=--gpu
```

**2. CUDA Out of Memory**

```
Error: RuntimeError: CUDA out of memory

Solutions:
1. Reduce batch size:
   --batch-size 32  # Instead of 64

2. Enable gradient checkpointing:
   model.gradient_checkpointing_enable()

3. Use mixed precision:
   scaler = torch.cuda.amp.GradScaler()

4. Clear cache:
   torch.cuda.empty_cache()
```

**3. Data Shape Mismatch**

```
Error: ValueError: Shape mismatch: expected (60, 44), got (60, 42)

Solution:
1. Check feature engineering output:
   print(f"Features: {len(feature_names)}")

2. Verify sequence length:
   print(f"Sequence shape: {X.shape}")

3. Update model config:
   model = LSTMModel(input_size=42)  # Match actual features
```

**4. NaN in Training Loss**

```
Error: Training loss is NaN

Solutions:
1. Lower learning rate:
   --learning-rate 0.0001

2. Add gradient clipping:
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

3. Check data for NaN/Inf:
   assert not np.isnan(X).any(), "NaN in features"
   assert not np.isinf(X).any(), "Inf in features"

4. Use robust loss function:
   criterion = nn.HuberLoss()
```

### Leakage Check Failures

**1. Temporal Order Violation**

```
Error: TEMPORAL LEAKAGE: Train max >= Val min

Root cause: Data not properly sorted or split.

Solution:
# Sort by time before splitting
df = df.sort_values('timestamp').reset_index(drop=True)

# Use time-based split
train_end = int(len(df) * 0.7)
val_end = int(len(df) * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]
```

**2. Feature-Target Leakage**

```
Error: Found 3 suspicious features with high correlation

Root cause: Feature contains target information.

Solution:
1. Review flagged features:
   for feature in report.details['suspicious']:
       print(f"Review: {feature['feature']} (corr={feature['correlation']:.3f})")

2. Check feature calculation:
   # Wrong - uses future data
   df['feature'] = df['close'].shift(-1)

   # Correct - uses past data only
   df['feature'] = df['close'].shift(1)
```

**3. Normalization Leakage**

```
Error: Scaler may have seen test data

Root cause: Scaler fitted on full dataset.

Solution:
# Wrong
scaler.fit(X)  # All data including test
X_scaled = scaler.transform(X)

# Correct
scaler.fit(X_train)  # Train only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### ONNX Export Issues

**1. Unsupported Operations**

```
Error: Unsupported ONNX opset version for operation X

Solutions:
1. Update opset version:
   torch.onnx.export(model, ..., opset_version=17)  # Use latest

2. Replace unsupported ops:
   # Instead of custom ops, use standard PyTorch
   # torch.special.* -> implement manually

3. Use onnx-simplifier:
   pip install onnx-simplifier
   python -m onnxsim model.onnx model_simplified.onnx
```

**2. Dynamic Shape Issues**

```
Error: ONNX export failed for dynamic shapes

Solution:
# Specify dynamic axes explicitly
torch.onnx.export(
    model, input,
    "model.onnx",
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

**3. ONNX Validation Failure**

```
Error: onnx.checker.ValidationError

Solutions:
1. Run model checker:
   onnx.checker.check_model(onnx.load("model.onnx"))

2. Simplify model:
   from onnx import optimizer
   optimized = optimizer.optimize(onnx.load("model.onnx"))

3. Check for inf/nan in weights:
   for param in model.parameters():
       assert torch.isfinite(param).all()
```

### Rust Compilation Issues

**1. ORT Linking Error**

```
Error: linking with `cc` failed: cannot find -lonnxruntime

Solution:
# Set library path
export ORT_LIB_LOCATION=/path/to/onnxruntime/lib
export LD_LIBRARY_PATH=$ORT_LIB_LOCATION:$LD_LIBRARY_PATH

# Or in Cargo.toml, use bundled version:
[dependencies]
ort = { version = "1.16", features = ["download-binaries"] }
```

**2. Tract Inference Mismatch**

```
Error: tract output differs from onnxruntime

Cause: Numerical precision differences.

Solution:
1. Use f32 consistently (not f64)
2. Compare with tolerance:
   assert!((tract_out - ort_out).abs() < 1e-5);
3. Check input normalization is identical
```

**3. SIMD Compilation Issues**

```
Error: target feature `avx2` is not enabled

Solution:
# In .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]

# Or compile with specific features
cargo build --release --target x86_64-unknown-linux-gnu
```

---

## 9. Metrics Reference

### Trading Metrics

| Metric | Formula | Good Value | Notes |
|--------|---------|------------|-------|
| **Sharpe Ratio** | (Return - Rf) / Std(Return) | > 1.5 | Risk-adjusted return |
| **Sortino Ratio** | (Return - Rf) / DownsideStd | > 2.0 | Penalizes downside only |
| **Calmar Ratio** | Annual Return / Max Drawdown | > 1.0 | Return per drawdown |
| **Max Drawdown** | Max peak-to-trough decline | < 20% | Maximum loss from peak |
| **Win Rate** | Winning trades / Total trades | > 50% | Trade accuracy |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 | Risk-reward ratio |
| **Recovery Factor** | Net profit / Max drawdown | > 3.0 | Recovery efficiency |

**Calculation Examples:**

```python
def calculate_sharpe(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe Ratio."""
    excess_returns = returns - rf / periods
    return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()

def calculate_sortino(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sortino Ratio."""
    excess_returns = returns - rf / periods
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(periods) * excess_returns.mean() / downside_std

def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum drawdown as percentage."""
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

def calculate_calmar(returns: np.ndarray, periods: int = 252) -> float:
    """Calmar Ratio."""
    annual_return = returns.mean() * periods
    cumulative = (1 + returns).cumprod()
    max_dd = abs(calculate_max_drawdown(cumulative))
    return annual_return / max_dd if max_dd > 0 else np.inf
```

### CPCV Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **IS Sharpe** | In-sample Sharpe Ratio | Training performance |
| **OOS Sharpe** | Out-of-sample Sharpe | Generalization |
| **IS/OOS Ratio** | IS Sharpe / OOS Sharpe | > 2 indicates overfitting |
| **PBO** | Probability of Backtest Overfitting | < 30% acceptable |
| **DSR** | Deflated Sharpe Ratio | Adjusted for trials |
| **Stationarity** | Performance stability across folds | CV < 0.5 acceptable |

**CPCV Results Interpretation:**

```python
cpcv_results = {
    "fold_metrics": [
        {"is_sharpe": 2.1, "oos_sharpe": 0.8},
        {"is_sharpe": 1.9, "oos_sharpe": 0.9},
        {"is_sharpe": 2.3, "oos_sharpe": 0.7},
        {"is_sharpe": 2.0, "oos_sharpe": 0.85},
        {"is_sharpe": 2.2, "oos_sharpe": 0.75},
    ],
}

# Aggregate metrics
is_sharpes = [f["is_sharpe"] for f in cpcv_results["fold_metrics"]]
oos_sharpes = [f["oos_sharpe"] for f in cpcv_results["fold_metrics"]]

summary = {
    "mean_is_sharpe": np.mean(is_sharpes),       # 2.1
    "mean_oos_sharpe": np.mean(oos_sharpes),     # 0.8
    "is_oos_ratio": np.mean(is_sharpes) / np.mean(oos_sharpes),  # 2.6 - some overfitting
    "oos_cv": np.std(oos_sharpes) / np.mean(oos_sharpes),  # 0.09 - stable
    "pbo": compute_pbo(is_sharpes, oos_sharpes), # ~25%
}
```

### NSMI Metrics

**Normalized Strategy Metadata Index** measures model-strategy performance:

| Metric | Description | Target |
|--------|-------------|--------|
| **Latency (p50)** | Median inference time | < 1ms |
| **Latency (p99)** | 99th percentile latency | < 5ms |
| **Throughput** | Predictions per second | > 10,000 |
| **Memory Usage** | Runtime memory footprint | < 500MB |
| **Signal Quality** | Correlation with returns | > 0.1 |
| **Signal IC** | Information Coefficient | > 0.05 |
| **Turnover** | Signal change frequency | < 0.2/bar |

**NSMI Benchmark:**

```python
import time
import numpy as np

def benchmark_nsmi(model, test_data: np.ndarray, n_iterations: int = 1000) -> dict:
    """Benchmark NSMI hot path performance."""

    latencies = []

    for i in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(test_data[i % len(test_data)])
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)

    return {
        "latency_p50_ms": np.percentile(latencies, 50),
        "latency_p99_ms": np.percentile(latencies, 99),
        "latency_mean_ms": np.mean(latencies),
        "throughput_per_sec": 1000 / np.mean(latencies),
        "iterations": n_iterations,
    }

# Target benchmarks
NSMI_TARGETS = {
    "latency_p50_ms": 1.0,
    "latency_p99_ms": 5.0,
    "throughput_per_sec": 10000,
}
```

### Model-Specific Metrics

**ML Models (LightGBM/XGBoost):**

| Metric | Good Value | Notes |
|--------|------------|-------|
| Feature Importance Gini | < 0.8 | Diversity in feature usage |
| SHAP Stability | > 0.9 | Consistent explanations |
| Leaf Depth | 5-7 | Balance accuracy/overfitting |

**DL Models (LSTM/CNN):**

| Metric | Good Value | Notes |
|--------|------------|-------|
| Validation Loss | Decreasing | No early plateau |
| Gradient Norm | 0.1-10 | Stable training |
| Attention Entropy | 0.5-2.0 | Focused but not collapsed |

**RL Agents (D4PG/MARL):**

| Metric | Good Value | Notes |
|--------|------------|-------|
| Episode Return | Increasing | Learning progress |
| Policy Entropy | 0.1-1.0 | Exploration balance |
| Q-Value Stability | CV < 0.5 | Stable value estimates |
| VaR Violation Rate | < 5% | Risk constraint satisfaction |

---

## Appendix A: Quick Reference Card

```
# TRAINING QUICK START
==========================================
# Full pipeline
python scripts/train_all.py

# With config
python scripts/train_all.py --config config/training.yaml

# Individual models
python -m models.ml.lightgbm_model --data data/processed/features.parquet
python -m models.dl.lstm_model --data data/processed/features.parquet --device cuda
python -m models.rl.d4pg_evt --episodes 200 --device cuda

# VALIDATION
==========================================
# Leakage check
python -m models.validation.leakage_guards --data data/processed/features.parquet

# CPCV validation
python -m models.validation.cpcv --model trained/models/model.pkl --n-splits 5

# EXPORT
==========================================
# ONNX export
python -m models.export.onnx_exporter --models-dir trained/models --output-dir trained/onnx

# Parity test
python -m models.export.test_parity --original trained/models/model.pkl --onnx trained/onnx/model.onnx

# RUST INTEGRATION
==========================================
# Build Rust strategy
cd rust_strategy && cargo build --release

# Run with ONNX
./target/release/strategy --model trained/onnx/model.onnx --data data/test.parquet
```

---

## Appendix B: File Locations

| Component | Path |
|-----------|------|
| Training Script | `scripts/train_all.py` |
| Orchestrator | `models/training/orchestrator.py` |
| CPCV Validator | `models/validation/cpcv.py` |
| Leakage Guards | `models/validation/leakage_guards.py` |
| Feature Engineering | `models/features/quant_features.py` |
| Microstructure | `models/features/microstructure.py` |
| ONNX Exporter | `models/export/onnx_exporter.py` |
| Model Registry | `models/registry.py` |
| Raw Data | `data/raw/klines_90d.parquet` |
| Processed Features | `data/processed/features.parquet` |
| Trained Models | `trained/models/` |
| ONNX Exports | `trained/onnx/` |
| Configs | `config/` |

---

*End of Training Runbook*
