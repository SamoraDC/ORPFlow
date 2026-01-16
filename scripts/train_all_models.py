#!/usr/bin/env python3
"""
Train All Models Pipeline
=========================
Trains all 6 models (XGBoost, LightGBM, LSTM, CNN, D4PG+EVT, MARL),
exports to ONNX, and validates parity for Rust inference.

Usage:
    # Train all models
    python scripts/train_all_models.py

    # Train specific model
    python scripts/train_all_models.py --model xgboost

    # Skip ONNX export
    python scripts/train_all_models.py --no-export

    # Run parity tests only
    python scripts/train_all_models.py --parity-only
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_xgboost(config_path: Optional[Path] = None) -> Any:
    """Train XGBoost model with CPCV validation."""
    logger.info("=" * 60)
    logger.info("Training XGBoost Model")
    logger.info("=" * 60)

    from models.training import XGBoostTrainer, XGBoostConfig

    config_path = config_path or project_root / "models" / "config" / "xgboost_config.yaml"
    config = load_config(config_path)

    trainer_config = XGBoostConfig(
        data_path=project_root / config["data"]["data_path"],
        target_column=config["data"]["target_column"],
        output_dir=project_root / config["output"]["output_dir"],
        model_name=config["output"]["model_name"],
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        early_stopping_rounds=config["model"]["early_stopping_rounds"],
        cpcv_enabled=config["cpcv"]["enabled"],
        cpcv_n_splits=config["cpcv"]["n_splits"],
        cpcv_n_test_groups=config["cpcv"]["n_test_groups"],
        optuna_enabled=config["optuna"]["enabled"],
        optuna_n_trials=config["optuna"]["n_trials"],
    )

    trainer = XGBoostTrainer(trainer_config)
    result = trainer.train()

    logger.info(f"XGBoost training completed. Metrics: {result.get('metrics', {})}")
    return result


def train_lightgbm(config_path: Optional[Path] = None) -> Any:
    """Train LightGBM model with CPCV validation."""
    logger.info("=" * 60)
    logger.info("Training LightGBM Model")
    logger.info("=" * 60)

    from models.training import LightGBMTrainer, LightGBMConfig

    config_path = config_path or project_root / "models" / "config" / "lightgbm_config.yaml"
    config = load_config(config_path)

    trainer_config = LightGBMConfig(
        data_path=project_root / config["data"]["data_path"],
        target_column=config["data"]["target_column"],
        output_dir=project_root / config["output"]["output_dir"],
        model_name=config["output"]["model_name"],
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        early_stopping_rounds=config["model"]["early_stopping_rounds"],
        cpcv_enabled=config["cpcv"]["enabled"],
        cpcv_n_splits=config["cpcv"]["n_splits"],
        cpcv_n_test_groups=config["cpcv"]["n_test_groups"],
        optuna_enabled=config["optuna"]["enabled"],
        optuna_n_trials=config["optuna"]["n_trials"],
    )

    trainer = LightGBMTrainer(trainer_config)
    result = trainer.train()

    logger.info(f"LightGBM training completed. Metrics: {result.get('metrics', {})}")
    return result


def train_lstm(config_path: Optional[Path] = None) -> Any:
    """Train LSTM model with CPCV validation."""
    logger.info("=" * 60)
    logger.info("Training LSTM Model")
    logger.info("=" * 60)

    from models.training import LSTMTrainer, LSTMTrainingConfig

    config_path = config_path or project_root / "models" / "config" / "lstm_config.yaml"
    config = load_config(config_path)

    trainer_config = LSTMTrainingConfig(
        data_path=project_root / config["data"]["data_path"],
        target_column=config["data"]["target_column"],
        sequence_length=config["data"]["sequence_length"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
        use_attention=config["model"]["use_attention"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        patience=config["training"]["patience"],
        cpcv_enabled=config["cpcv"]["enabled"],
        cpcv_n_splits=config["cpcv"]["n_splits"],
        optuna_enabled=config["optuna"]["enabled"],
        output_dir=project_root / config["output"]["output_dir"],
        model_name=config["output"]["model_name"],
        export_onnx=config["output"]["export_onnx"],
    )

    trainer = LSTMTrainer(trainer_config)
    result = trainer.train()

    logger.info(f"LSTM training completed. Metrics: {result.get('metrics', {})}")
    return result


def train_cnn(config_path: Optional[Path] = None) -> Any:
    """Train CNN (TCN) model with CPCV validation."""
    logger.info("=" * 60)
    logger.info("Training CNN (TCN) Model")
    logger.info("=" * 60)

    from models.training import CNNTrainer, CNNTrainingConfig

    config_path = config_path or project_root / "models" / "config" / "cnn_config.yaml"
    config = load_config(config_path)

    trainer_config = CNNTrainingConfig(
        data_path=project_root / config["data"]["data_path"],
        target_column=config["data"]["target_column"],
        sequence_length=config["data"]["sequence_length"],
        conv_channels=config["model"]["conv_channels"],
        kernel_sizes=config["model"]["kernel_sizes"],
        fc_units=config["model"]["fc_units"],
        dropout=config["model"]["dropout"],
        use_residual=config["model"]["use_residual"],
        use_attention=config["model"]["use_attention"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        patience=config["training"]["patience"],
        cpcv_enabled=config["cpcv"]["enabled"],
        cpcv_n_splits=config["cpcv"]["n_splits"],
        optuna_enabled=config["optuna"]["enabled"],
        output_dir=project_root / config["output"]["output_dir"],
        model_name=config["output"]["model_name"],
        export_onnx=config["output"]["export_onnx"],
    )

    trainer = CNNTrainer(trainer_config)
    result = trainer.train()

    logger.info(f"CNN training completed. Metrics: {result.get('metrics', {})}")
    return result


def train_d4pg(config_path: Optional[Path] = None) -> Any:
    """Train D4PG+EVT agent."""
    logger.info("=" * 60)
    logger.info("Training D4PG+EVT Agent")
    logger.info("=" * 60)

    from models.training import D4PGTrainer, TrainerConfig
    from models.rl import D4PGAgent, TradingEnvironment
    import pandas as pd

    config_path = config_path or project_root / "models" / "config" / "training_config.yaml"
    config = load_config(config_path)
    rl_config = config["rl"]["d4pg_evt"]["params"]

    # Load data
    data_path = project_root / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(data_path)

    # Get feature columns
    feature_cols = [c for c in df.columns if not c.startswith("target_")
                    and c not in ["open_time", "close_time", "symbol"]]

    # Create environment
    env = TradingEnvironment(
        data=df,
        feature_columns=feature_cols,
        initial_balance=10000.0,
    )

    # Create agent
    state_dim = len(feature_cols) + 4  # features + portfolio state
    action_dim = 1  # position sizing

    agent = D4PGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=rl_config["actor_lr"],
        critic_lr=rl_config["critic_lr"],
        gamma=rl_config["gamma"],
        tau=rl_config["tau"],
        n_atoms=rl_config["n_atoms"],
        v_min=rl_config["v_min"],
        v_max=rl_config["v_max"],
        n_step=rl_config["n_step"],
    )

    # Create trainer
    trainer_config = TrainerConfig(
        total_timesteps=rl_config["total_timesteps"],
        eval_freq=rl_config["eval_freq"],
        output_dir=project_root / "trained",
    )

    trainer = D4PGTrainer(env, agent, trainer_config)
    result = trainer.train()

    logger.info(f"D4PG training completed. Metrics: {result.get('metrics', {})}")
    return result


def train_marl(config_path: Optional[Path] = None) -> Any:
    """Train MARL system."""
    logger.info("=" * 60)
    logger.info("Training MARL System")
    logger.info("=" * 60)

    from models.training import MARLTrainer, TrainerConfig
    from models.rl import MARLSystem, TradingEnvironment
    import pandas as pd

    config_path = config_path or project_root / "models" / "config" / "training_config.yaml"
    config = load_config(config_path)
    marl_config = config["rl"]["marl"]["params"]

    # Load data
    data_path = project_root / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(data_path)

    # Get feature columns
    feature_cols = [c for c in df.columns if not c.startswith("target_")
                    and c not in ["open_time", "close_time", "symbol"]]

    # Create environment
    env = TradingEnvironment(
        data=df,
        feature_columns=feature_cols,
        initial_balance=10000.0,
    )

    # Create MARL system
    state_dim = len(feature_cols) + 4
    action_dim = 1

    marl_system = MARLSystem(
        num_agents=marl_config["num_agents"],
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=marl_config["hidden_size"],
        learning_rate=marl_config["learning_rate"],
        gamma=marl_config["gamma"],
        agent_types=marl_config["agent_types"],
    )

    # Create trainer
    trainer_config = TrainerConfig(
        total_timesteps=marl_config["total_timesteps"],
        n_steps=marl_config["n_steps"],
        batch_size=marl_config["batch_size"],
        output_dir=project_root / "trained",
    )

    trainer = MARLTrainer(env, marl_system, trainer_config)
    result = trainer.train()

    logger.info(f"MARL training completed. Metrics: {result.get('metrics', {})}")
    return result


def export_all_to_onnx() -> Dict[str, str]:
    """Export all trained models to ONNX format."""
    logger.info("=" * 60)
    logger.info("Exporting Models to ONNX")
    logger.info("=" * 60)

    from models.export import ONNXExporter

    exporter = ONNXExporter(output_dir=str(project_root / "trained" / "onnx"))
    exported_paths = {}

    trained_dir = project_root / "trained"

    # Export XGBoost
    xgb_path = trained_dir / "xgboost_model.json"
    if xgb_path.exists():
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(xgb_path))
            # Get feature names from model
            feature_names = [f"f{i}" for i in range(model.num_features())]
            path = exporter.export_xgboost(model, feature_names, "xgboost_model")
            exported_paths["xgboost"] = path
            logger.info(f"Exported XGBoost to {path}")
        except Exception as e:
            logger.error(f"Failed to export XGBoost: {e}")

    # Export LightGBM
    lgb_path = trained_dir / "lightgbm_model.txt"
    if lgb_path.exists():
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(lgb_path))
            feature_names = model.feature_name()
            path = exporter.export_lightgbm(model, feature_names, "lightgbm_model")
            exported_paths["lightgbm"] = path
            logger.info(f"Exported LightGBM to {path}")
        except Exception as e:
            logger.error(f"Failed to export LightGBM: {e}")

    # Export LSTM
    lstm_path = trained_dir / "lstm_model.pt"
    if lstm_path.exists():
        try:
            import torch
            from models.dl import LSTMModel
            checkpoint = torch.load(lstm_path, map_location="cpu")
            model = LSTMModel(**checkpoint["config"])
            model.load_state_dict(checkpoint["model_state_dict"])

            input_shape = (1, checkpoint["config"]["sequence_length"],
                          checkpoint["config"]["input_size"])
            path = exporter.export_lstm(model, input_shape, "lstm_model")
            exported_paths["lstm"] = path
            logger.info(f"Exported LSTM to {path}")
        except Exception as e:
            logger.error(f"Failed to export LSTM: {e}")

    # Export CNN
    cnn_path = trained_dir / "cnn_model.pt"
    if cnn_path.exists():
        try:
            import torch
            from models.dl import CNNModel
            checkpoint = torch.load(cnn_path, map_location="cpu")
            model = CNNModel(**checkpoint["config"])
            model.load_state_dict(checkpoint["model_state_dict"])

            input_shape = (1, checkpoint["config"]["sequence_length"],
                          checkpoint["config"]["num_features"])
            path = exporter.export_cnn(model, input_shape, "cnn_model")
            exported_paths["cnn"] = path
            logger.info(f"Exported CNN to {path}")
        except Exception as e:
            logger.error(f"Failed to export CNN: {e}")

    # Export D4PG actor
    d4pg_path = trained_dir / "d4pg_evt_agent.pt"
    if d4pg_path.exists():
        try:
            import torch
            checkpoint = torch.load(d4pg_path, map_location="cpu")
            # Export actor network only
            actor = checkpoint["actor"]
            state_dim = checkpoint["state_dim"]
            path = exporter.export_d4pg_actor(actor, state_dim, "d4pg_actor")
            exported_paths["d4pg"] = path
            logger.info(f"Exported D4PG to {path}")
        except Exception as e:
            logger.error(f"Failed to export D4PG: {e}")

    # Export MARL agents
    marl_path = trained_dir / "marl_system.pt"
    if marl_path.exists():
        try:
            import torch
            checkpoint = torch.load(marl_path, map_location="cpu")
            for i, agent in enumerate(checkpoint["agents"]):
                path = exporter.export_marl_agent(
                    agent,
                    checkpoint["state_dim"],
                    checkpoint["message_dim"],
                    f"marl_agent_{i}"
                )
                exported_paths[f"marl_agent_{i}"] = path
                logger.info(f"Exported MARL agent {i} to {path}")
        except Exception as e:
            logger.error(f"Failed to export MARL: {e}")

    return exported_paths


def run_parity_tests() -> bool:
    """Run ONNX parity tests."""
    logger.info("=" * 60)
    logger.info("Running ONNX Parity Tests")
    logger.info("=" * 60)

    from models.export.onnx_parity import run_all_parity_tests

    results = run_all_parity_tests(
        model_dir=project_root / "trained",
        onnx_dir=project_root / "trained" / "onnx",
        golden_dir=project_root / "tests" / "golden_data",
        n_samples=100,
    )

    passed = all(r.passed for r in results)

    if passed:
        logger.info("All parity tests PASSED!")
    else:
        failed = [r.model_name for r in results if not r.passed]
        logger.error(f"Parity tests FAILED for: {failed}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Train All ORPFlow Models")
    parser.add_argument(
        "--model",
        choices=["all", "xgboost", "lightgbm", "lstm", "cnn", "d4pg", "marl"],
        default="all",
        help="Model to train (default: all)"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX export"
    )
    parser.add_argument(
        "--parity-only",
        action="store_true",
        help="Run parity tests only (no training)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to custom config file"
    )

    args = parser.parse_args()

    if args.parity_only:
        success = run_parity_tests()
        sys.exit(0 if success else 1)

    # Training functions
    trainers = {
        "xgboost": train_xgboost,
        "lightgbm": train_lightgbm,
        "lstm": train_lstm,
        "cnn": train_cnn,
        "d4pg": train_d4pg,
        "marl": train_marl,
    }

    results = {}

    if args.model == "all":
        # Train ML models first (faster)
        for name in ["xgboost", "lightgbm"]:
            try:
                results[name] = trainers[name](args.config)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {"error": str(e)}

        # Train DL models
        for name in ["lstm", "cnn"]:
            try:
                results[name] = trainers[name](args.config)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {"error": str(e)}

        # Train RL models (slowest)
        for name in ["d4pg", "marl"]:
            try:
                results[name] = trainers[name](args.config)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {"error": str(e)}
    else:
        try:
            results[args.model] = trainers[args.model](args.config)
        except Exception as e:
            logger.error(f"Failed to train {args.model}: {e}")
            results[args.model] = {"error": str(e)}

    # Export to ONNX
    if not args.no_export:
        exported = export_all_to_onnx()
        logger.info(f"Exported {len(exported)} models to ONNX")

    # Run parity tests
    if not args.no_export:
        run_parity_tests()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: FAILED - {result['error']}")
        else:
            metrics = result.get("metrics", {})
            print(f"  {name}: SUCCESS")
            if "sharpe" in metrics:
                print(f"    Sharpe: {metrics['sharpe']:.4f}")
            if "rmse" in metrics:
                print(f"    RMSE: {metrics['rmse']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
