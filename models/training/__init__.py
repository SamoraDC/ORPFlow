"""
Training Module for ORPFlow
===========================

This module provides unified training orchestration for ML, DL, and RL models
with proper validation, leakage checks, and experiment tracking.

Core Components:
- TrainingOrchestrator: Unified training pipeline with leakage guards
- DataManager: Data loading, feature engineering, dataset hashing
- ExperimentTracker: Full experiment tracking with reproducibility manifests
- TrainingConfig: Configuration dataclass for training parameters

RL Training Infrastructure (requires torch):
- ReplayEnvironment: Historical data replay with realistic fill simulation
- D4PGTrainer: Trainer for D4PG+EVT agents
- MARLTrainer: Trainer for Multi-Agent RL systems
- RLValidationMetrics: Comprehensive trading performance metrics
- FillSimulator: Realistic order execution simulation (no lookahead)

Usage:
    from models.training.orchestrator import TrainingOrchestrator, TrainingConfig

    # ML/DL Training
    config = TrainingConfig(data_path=Path("data/raw/klines_90d.parquet"))
    orchestrator = TrainingOrchestrator(config)
    results = orchestrator.train_all()

    # RL Training (requires torch)
    from models.training.rl_trainer import (
        ReplayEnvironment, D4PGTrainer, MARLTrainer
    )
    env = ReplayEnvironment(data, features)
    trainer = D4PGTrainer(env, agent)
    results = trainer.walk_forward_train()
"""

# Core orchestrator components (always available)
from .orchestrator import (
    TrainingConfig,
    DataManager,
    ExperimentTracker,
    TrainingOrchestrator,
)

__all__ = [
    # Orchestrator exports
    "TrainingConfig",
    "DataManager",
    "ExperimentTracker",
    "TrainingOrchestrator",
]

# Model-specific trainers
try:
    from .train_xgboost import XGBoostTrainer, XGBoostConfig
    from .train_lightgbm import LightGBMTrainer, LightGBMConfig
    from .train_lstm import LSTMTrainer, LSTMTrainingConfig
    from .train_cnn import CNNTrainer, CNNTrainingConfig

    __all__.extend([
        "XGBoostTrainer",
        "XGBoostConfig",
        "LightGBMTrainer",
        "LightGBMConfig",
        "LSTMTrainer",
        "LSTMTrainingConfig",
        "CNNTrainer",
        "CNNTrainingConfig",
    ])
except ImportError:
    pass

# Optional RL training components (require torch)
try:
    from .rl_trainer import (
        # Order types and fill simulation
        OrderType,
        OrderSide,
        Order,
        FillSimulatorConfig,
        FillSimulator,
        # Environment
        EnvironmentConfig,
        ReplayEnvironment,
        # Metrics
        TradingMetrics,
        RLValidationMetrics,
        # Training configuration
        TrainerConfig,
        # Trainers
        D4PGTrainer,
        MARLTrainer,
        # Utilities
        create_walk_forward_folds,
        seed_everything,
        run_multi_seed_training,
    )

    __all__.extend([
        # RL Training exports
        "OrderType",
        "OrderSide",
        "Order",
        "FillSimulatorConfig",
        "FillSimulator",
        "EnvironmentConfig",
        "ReplayEnvironment",
        "TradingMetrics",
        "RLValidationMetrics",
        "TrainerConfig",
        "D4PGTrainer",
        "MARLTrainer",
        "create_walk_forward_folds",
        "seed_everything",
        "run_multi_seed_training",
    ])

except ImportError:
    # torch not available, RL training components not loaded
    pass
