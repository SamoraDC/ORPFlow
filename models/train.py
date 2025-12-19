#!/usr/bin/env python3
"""
Main Training Orchestration Script
Trains all ML/DL/RL models and exports to ONNX
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrate training of all models"""

    def __init__(self, config_path: str = "config/training_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict] = {}

        self.output_dir = Path(self.config.get("output_dir", "trained"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict:
        """Load training configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default training configuration"""
        return {
            "data": {
                "symbol": "BTCUSDT",
                "days": 90,
                "sequence_length": 60,
            },
            "ml": {
                "lightgbm": {"enabled": True, "n_estimators": 1000},
                "xgboost": {"enabled": True, "n_estimators": 1000},
            },
            "dl": {
                "lstm": {"enabled": True, "epochs": 100, "hidden_size": 128},
                "cnn": {"enabled": True, "epochs": 100},
            },
            "rl": {
                "d4pg": {"enabled": True, "episodes": 200},
                "marl": {"enabled": True, "episodes": 200, "n_agents": 5},
            },
            "output_dir": "trained",
        }

    def collect_data(self) -> pd.DataFrame:
        """Collect historical market data"""
        from data.collector import BinanceDataCollector

        logger.info("Collecting market data...")

        collector = BinanceDataCollector()
        data_config = self.config.get("data", {})

        symbol = data_config.get("symbol", "BTCUSDT")
        days = data_config.get("days", 90)

        df = collector.get_historical_klines(symbol, days=days)

        if df is not None and not df.empty:
            collector.save_data(df, f"klines_{days}d.parquet")
            logger.info(f"Collected {len(df)} rows for {symbol}")
            return df

        # Try to load existing data
        existing = collector.load_data(f"klines_{days}d.parquet")
        if not existing.empty:
            logger.info(f"Loaded existing data: {len(existing)} rows")
            return existing

        raise ValueError("No data available. Check Binance API connection.")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and engineer features"""
        from data.preprocessor import FeatureEngineer

        logger.info("Preprocessing data and engineering features...")

        engineer = FeatureEngineer()
        processed = engineer.process_symbol(df)

        output_path = Path("data/processed")
        output_path.mkdir(parents=True, exist_ok=True)
        processed.to_parquet(output_path / "features.parquet", index=False)

        logger.info(f"Processed {len(processed)} rows, {len(engineer.get_feature_columns(processed))} features")

        return processed

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
    ) -> Dict:
        """Train LightGBM model"""
        from ml.lightgbm_model import LightGBMModel

        logger.info("Training LightGBM...")

        config = self.config.get("ml", {}).get("lightgbm", {})
        model = LightGBMModel()

        metrics = model.train(
            X_train, y_train, X_val, y_val,
            feature_names=feature_names,
            n_estimators=config.get("n_estimators", 1000),
        )

        self.models["lightgbm"] = model
        self.metrics["lightgbm"] = metrics

        model.save(str(self.output_dir / "lightgbm_model.pkl"))

        return metrics

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
    ) -> Dict:
        """Train XGBoost model"""
        from ml.xgboost_model import XGBoostModel

        logger.info("Training XGBoost...")

        config = self.config.get("ml", {}).get("xgboost", {})
        model = XGBoostModel()

        metrics = model.train(
            X_train, y_train, X_val, y_val,
            feature_names=feature_names,
            n_estimators=config.get("n_estimators", 1000),
        )

        self.models["xgboost"] = model
        self.metrics["xgboost"] = metrics

        model.save(str(self.output_dir / "xgboost_model.pkl"))

        return metrics

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sequence_length: int,
        num_features: int,
    ) -> Dict:
        """Train LSTM model"""
        from dl.lstm_model import LSTMModel

        logger.info("Training LSTM...")

        config = self.config.get("dl", {}).get("lstm", {})
        model = LSTMModel(
            input_size=num_features,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
        )

        metrics = model.train(
            X_train, y_train, X_val, y_val,
            epochs=config.get("epochs", 100),
            patience=config.get("patience", 10),
        )

        self.models["lstm"] = model
        self.metrics["lstm"] = metrics

        model.save(str(self.output_dir / "lstm_model.pt"))
        model.export_onnx(
            str(self.output_dir / "onnx" / "lstm_model.onnx"),
            sequence_length=sequence_length,
            num_features=num_features,
        )

        return metrics

    def train_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sequence_length: int,
        num_features: int,
    ) -> Dict:
        """Train CNN model"""
        from dl.cnn_model import CNNModel

        logger.info("Training CNN...")

        config = self.config.get("dl", {}).get("cnn", {})
        model = CNNModel(
            num_features=num_features,
            sequence_length=sequence_length,
            dropout=config.get("dropout", 0.3),
        )

        metrics = model.train(
            X_train, y_train, X_val, y_val,
            epochs=config.get("epochs", 100),
            patience=config.get("patience", 10),
        )

        self.models["cnn"] = model
        self.metrics["cnn"] = metrics

        model.save(str(self.output_dir / "cnn_model.pt"))
        model.export_onnx(str(self.output_dir / "onnx" / "cnn_model.onnx"))

        return metrics

    def train_d4pg(
        self,
        data: np.ndarray,
        features: np.ndarray,
    ) -> Dict:
        """Train D4PG+EVT agent"""
        from rl.d4pg_evt import D4PGAgent, TradingEnvironment, train_d4pg

        logger.info("Training D4PG+EVT agent...")

        config = self.config.get("rl", {}).get("d4pg", {})

        agent = train_d4pg(
            data, features,
            episodes=config.get("episodes", 200),
        )

        self.models["d4pg"] = agent

        agent.save(str(self.output_dir / "d4pg_evt_agent.pt"))
        agent.export_onnx(str(self.output_dir / "onnx" / "d4pg_actor.onnx"))

        metrics = {
            "training_steps": agent.training_step,
            "var_99": agent.evt_model.var(),
            "cvar_99": agent.evt_model.cvar(),
        }
        self.metrics["d4pg"] = metrics

        return metrics

    def train_marl(
        self,
        data: np.ndarray,
        features: np.ndarray,
    ) -> Dict:
        """Train MARL system"""
        from rl.marl import MARLSystem, train_marl

        logger.info("Training MARL system...")

        config = self.config.get("rl", {}).get("marl", {})

        marl = train_marl(
            data, features,
            n_agents=config.get("n_agents", 5),
            episodes=config.get("episodes", 200),
        )

        self.models["marl"] = marl

        marl.save(str(self.output_dir / "marl_system.pt"))
        marl.export_onnx(str(self.output_dir / "onnx" / "marl"))

        metrics = {
            "training_steps": marl.training_step,
            "n_agents": marl.n_agents,
        }
        self.metrics["marl"] = metrics

        return metrics

    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """Evaluate all models on test data"""
        from ensemble.model_selector import ModelEvaluator, ModelType

        logger.info("Evaluating models on test data...")

        evaluator = ModelEvaluator()
        results = {}

        for name, model in self.models.items():
            if hasattr(model, "predict"):
                try:
                    y_pred = model.predict(X_test)
                    model_type = ModelType[name.upper()]
                    metrics = evaluator.calculate_metrics(y_test, y_pred, model_type)
                    results[name] = {
                        "mse": metrics.mse,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "win_rate": metrics.win_rate,
                        "profit_factor": metrics.profit_factor,
                        "max_drawdown": metrics.max_drawdown,
                        "score": metrics.score(),
                    }
                    logger.info(f"{name}: Sharpe={metrics.sharpe_ratio:.3f}, Score={metrics.score():.3f}")
                except Exception as e:
                    logger.error(f"Evaluation failed for {name}: {e}")

        return results

    def select_best_models(self, results: Dict[str, Dict], n: int = 3) -> list:
        """Select best performing models"""
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )

        best = [m[0] for m in sorted_models[:n]]
        logger.info(f"Best models: {best}")

        return best

    def export_to_onnx(self, feature_names: list, sequence_length: int):
        """Export all models to ONNX"""
        from export.onnx_exporter import ONNXExporter

        logger.info("Exporting models to ONNX...")

        exporter = ONNXExporter(str(self.output_dir / "onnx"))

        exported = {}

        if "lightgbm" in self.models:
            try:
                path = exporter.export_lightgbm(
                    self.models["lightgbm"],
                    feature_names,
                    "lightgbm_model",
                )
                exported["lightgbm"] = path
            except Exception as e:
                logger.error(f"LightGBM export failed: {e}")

        if "xgboost" in self.models:
            try:
                path = exporter.export_xgboost(
                    self.models["xgboost"],
                    feature_names,
                    "xgboost_model",
                )
                exported["xgboost"] = path
            except Exception as e:
                logger.error(f"XGBoost export failed: {e}")

        logger.info(f"Exported {len(exported)} models to ONNX")

        return exported

    def save_results(self):
        """Save training results and metrics"""
        results = {
            "metrics": self.metrics,
            "models": list(self.models.keys()),
            "config": self.config,
        }

        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")

    def run(
        self,
        skip_data_collection: bool = False,
        skip_ml: bool = False,
        skip_dl: bool = False,
        skip_rl: bool = False,
    ):
        """Run full training pipeline"""

        logger.info("=" * 60)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 60)

        # 1. Collect and preprocess data
        if not skip_data_collection:
            raw_data = self.collect_data()
            processed_data = self.preprocess_data(raw_data)
        else:
            data_path = Path("data/processed/features.parquet")
            if not data_path.exists():
                raise ValueError("No processed data found. Run without --skip-data")
            processed_data = pd.read_parquet(data_path)
            logger.info(f"Loaded existing processed data: {len(processed_data)} rows")

        # 2. Prepare data
        from data.preprocessor import FeatureEngineer

        engineer = FeatureEngineer()
        data_config = self.config.get("data", {})
        sequence_length = data_config.get("sequence_length", 60)

        # ML data (flat features)
        X_train_ml, X_val_ml, X_test_ml, y_train_ml, y_val_ml, y_test_ml, feature_names = \
            engineer.prepare_ml_data(processed_data)

        # DL data (sequences)
        X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = \
            engineer.prepare_sequence_data(processed_data, sequence_length=sequence_length)

        num_features = X_train_seq.shape[2]

        # RL data
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        rl_data = processed_data[ohlcv_cols].values
        rl_features = engineer.scaler.fit_transform(
            processed_data[engineer.get_feature_columns(processed_data)].values
        )

        # 3. Train ML models
        if not skip_ml:
            ml_config = self.config.get("ml", {})

            if ml_config.get("lightgbm", {}).get("enabled", True):
                self.train_lightgbm(
                    X_train_ml, y_train_ml, X_val_ml, y_val_ml, feature_names
                )

            if ml_config.get("xgboost", {}).get("enabled", True):
                self.train_xgboost(
                    X_train_ml, y_train_ml, X_val_ml, y_val_ml, feature_names
                )

        # 4. Train DL models
        if not skip_dl:
            dl_config = self.config.get("dl", {})

            if dl_config.get("lstm", {}).get("enabled", True):
                self.train_lstm(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                    sequence_length, num_features,
                )

            if dl_config.get("cnn", {}).get("enabled", True):
                self.train_cnn(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                    sequence_length, num_features,
                )

        # 5. Train RL models
        if not skip_rl:
            rl_config = self.config.get("rl", {})

            if rl_config.get("d4pg", {}).get("enabled", True):
                self.train_d4pg(rl_data, rl_features)

            if rl_config.get("marl", {}).get("enabled", True):
                self.train_marl(rl_data, rl_features)

        # 6. Evaluate models
        evaluation_results = self.evaluate_models(X_test_ml, y_test_ml)

        # 7. Select best models
        best_models = self.select_best_models(evaluation_results)

        # 8. Export to ONNX
        self.export_to_onnx(feature_names, sequence_length)

        # 9. Save results
        self.save_results()

        # Print summary
        self._print_summary(evaluation_results, best_models)

        return evaluation_results

    def _print_summary(self, results: Dict, best_models: list):
        """Print training summary"""

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        print("\nModel Performance:")
        print("-" * 40)

        for name, metrics in sorted(results.items(), key=lambda x: x[1].get("score", 0), reverse=True):
            print(f"\n{name.upper()}:")
            print(f"  Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Win Rate:       {metrics.get('win_rate', 0):.2%}")
            print(f"  Profit Factor:  {metrics.get('profit_factor', 0):.2f}")
            print(f"  Max Drawdown:   {metrics.get('max_drawdown', 0):.4f}")
            print(f"  Composite Score: {metrics.get('score', 0):.3f}")

        print("\n" + "-" * 40)
        print(f"Best Models for Deployment: {', '.join(best_models)}")
        print(f"ONNX models exported to: {self.output_dir / 'onnx'}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train ML/DL/RL trading models")
    parser.add_argument("--config", default="config/training_config.yaml", help="Config file path")
    parser.add_argument("--skip-data", action="store_true", help="Skip data collection")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML models")
    parser.add_argument("--skip-dl", action="store_true", help="Skip DL models")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL models")
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "lstm", "cnn", "d4pg", "marl"],
                        help="Train only specific model")

    args = parser.parse_args()

    orchestrator = TrainingOrchestrator(args.config)

    if args.model:
        # Train only specific model
        skip_ml = args.model not in ["lightgbm", "xgboost"]
        skip_dl = args.model not in ["lstm", "cnn"]
        skip_rl = args.model not in ["d4pg", "marl"]
    else:
        skip_ml = args.skip_ml
        skip_dl = args.skip_dl
        skip_rl = args.skip_rl

    orchestrator.run(
        skip_data_collection=args.skip_data,
        skip_ml=skip_ml,
        skip_dl=skip_dl,
        skip_rl=skip_rl,
    )


if __name__ == "__main__":
    main()
