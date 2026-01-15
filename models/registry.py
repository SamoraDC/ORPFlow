"""
Model Registry - Comprehensive Model Discovery and Management
=============================================================

This module provides a centralized registry for all trading models in the ORPFlow
system, including Machine Learning, Deep Learning, and Reinforcement Learning models.

Features:
- Auto-discovery of all available models
- Model metadata including input requirements, dependencies, and capabilities
- Factory methods for model instantiation
- ONNX export capability tracking
- Consistent interface across all model types

Usage:
    from models.registry import ModelRegistry

    registry = ModelRegistry()

    # List all available models
    registry.list_models()

    # Get model info
    info = registry.get_model_info("lightgbm")

    # Create model instance
    model = registry.create_model("lstm", input_size=64)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCategory(Enum):
    """Model category classification"""
    ML = "machine_learning"
    DL = "deep_learning"
    RL = "reinforcement_learning"
    ENSEMBLE = "ensemble"


class InputType(Enum):
    """Input data type classification"""
    TABULAR = "tabular"          # 2D: (samples, features)
    SEQUENCE = "sequence"        # 3D: (samples, sequence_length, features)
    STATE = "state"              # 1D/2D state vector for RL
    MULTI_AGENT = "multi_agent"  # (n_agents, state_dim)


@dataclass
class ModelMetadata:
    """Metadata container for model information"""

    # Basic info
    name: str
    display_name: str
    category: ModelCategory
    description: str

    # Module info
    module_path: str
    class_name: str
    entrypoint: str

    # Input requirements
    input_type: InputType
    input_shape_description: str
    required_params: List[str] = field(default_factory=list)
    optional_params: Dict[str, Any] = field(default_factory=dict)

    # Training info
    train_method: str = "train"
    train_signature: str = ""

    # Export capabilities
    supports_onnx: bool = True
    onnx_export_method: str = "export_onnx"
    onnx_export_notes: str = ""

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)

    # Performance characteristics
    supports_gpu: bool = False
    typical_training_time: str = ""
    memory_requirements: str = ""


# ==============================================================================
# MODEL REGISTRY DEFINITIONS
# ==============================================================================

MODEL_DEFINITIONS: Dict[str, ModelMetadata] = {

    # --------------------------------------------------------------------------
    # Machine Learning Models
    # --------------------------------------------------------------------------

    "lightgbm": ModelMetadata(
        name="lightgbm",
        display_name="LightGBM Model",
        category=ModelCategory.ML,
        description="LightGBM gradient boosting model for return/signal prediction. "
                    "Fast training with excellent performance on tabular data.",
        module_path="models.ml.lightgbm_model",
        class_name="LightGBMModel",
        entrypoint="LightGBMModel",
        input_type=InputType.TABULAR,
        input_shape_description="(n_samples, n_features) - 2D array of features",
        required_params=[],
        optional_params={
            "params": {
                "objective": "regression",
                "metric": "mse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
            }
        },
        train_method="train",
        train_signature="train(X_train, y_train, X_val, y_val, feature_names=None, "
                        "n_estimators=1000, early_stopping_rounds=50) -> Dict",
        supports_onnx=True,
        onnx_export_method="export_onnx",
        onnx_export_notes="Requires onnxmltools. export_onnx(path, feature_names)",
        dependencies=["lightgbm", "sklearn", "numpy", "pandas", "joblib"],
        optional_dependencies=["onnxmltools", "onnx"],
        supports_gpu=True,
        typical_training_time="1-5 minutes on 100K samples",
        memory_requirements="Low (~1GB for 100K samples)",
    ),

    "xgboost": ModelMetadata(
        name="xgboost",
        display_name="XGBoost Model",
        category=ModelCategory.ML,
        description="XGBoost gradient boosting model with histogram-based training. "
                    "Robust performance with built-in regularization.",
        module_path="models.ml.xgboost_model",
        class_name="XGBoostModel",
        entrypoint="XGBoostModel",
        input_type=InputType.TABULAR,
        input_shape_description="(n_samples, n_features) - 2D array of features",
        required_params=[],
        optional_params={
            "params": {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "device": "cpu",
            }
        },
        train_method="train",
        train_signature="train(X_train, y_train, X_val, y_val, feature_names=None, "
                        "n_estimators=1000, early_stopping_rounds=50) -> Dict",
        supports_onnx=True,
        onnx_export_method="export_onnx",
        onnx_export_notes="Requires onnxmltools. export_onnx(path, feature_names)",
        dependencies=["xgboost", "sklearn", "numpy", "pandas", "joblib"],
        optional_dependencies=["onnxmltools", "onnx"],
        supports_gpu=True,
        typical_training_time="1-5 minutes on 100K samples",
        memory_requirements="Low (~1GB for 100K samples)",
    ),

    # --------------------------------------------------------------------------
    # Deep Learning Models
    # --------------------------------------------------------------------------

    "lstm": ModelMetadata(
        name="lstm",
        display_name="LSTM Model",
        category=ModelCategory.DL,
        description="LSTM neural network for time series prediction with attention "
                    "mechanism. Captures long-term dependencies in sequential data.",
        module_path="models.dl.lstm_model",
        class_name="LSTMModel",
        entrypoint="LSTMModel",
        input_type=InputType.SEQUENCE,
        input_shape_description="(n_samples, sequence_length, n_features) - 3D array "
                                "where sequence_length is typically 60 time steps",
        required_params=["input_size"],
        optional_params={
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "device": "auto",
        },
        train_method="train",
        train_signature="train(X_train, y_train, X_val, y_val, batch_size=64, "
                        "epochs=100, patience=10) -> Dict",
        supports_onnx=True,
        onnx_export_method="export_onnx",
        onnx_export_notes="Native PyTorch export. export_onnx(path, sequence_length, num_features)",
        dependencies=["torch", "sklearn", "numpy", "pandas", "tqdm"],
        optional_dependencies=["onnx"],
        supports_gpu=True,
        typical_training_time="10-30 minutes on 100K samples (GPU)",
        memory_requirements="Medium (~4GB GPU for 100K samples)",
    ),

    "cnn": ModelMetadata(
        name="cnn",
        display_name="CNN Model",
        category=ModelCategory.DL,
        description="1D Convolutional Neural Network for pattern recognition in market "
                    "data. Uses temporal convolutions with multi-scale feature extraction.",
        module_path="models.dl.cnn_model",
        class_name="CNNModel",
        entrypoint="CNNModel",
        input_type=InputType.SEQUENCE,
        input_shape_description="(n_samples, sequence_length, n_features) - 3D array "
                                "where sequence_length is typically 60 time steps",
        required_params=["num_features", "sequence_length"],
        optional_params={
            "conv_channels": [32, 64, 128],
            "kernel_sizes": [3, 3, 3],
            "fc_units": [256, 128],
            "dropout": 0.3,
            "learning_rate": 0.001,
            "device": "auto",
        },
        train_method="train",
        train_signature="train(X_train, y_train, X_val, y_val, batch_size=64, "
                        "epochs=100, patience=10) -> Dict",
        supports_onnx=True,
        onnx_export_method="export_onnx",
        onnx_export_notes="Native PyTorch export. export_onnx(path) - uses stored dimensions",
        dependencies=["torch", "sklearn", "numpy", "pandas"],
        optional_dependencies=["onnx"],
        supports_gpu=True,
        typical_training_time="5-20 minutes on 100K samples (GPU)",
        memory_requirements="Medium (~4GB GPU for 100K samples)",
    ),

    # --------------------------------------------------------------------------
    # Reinforcement Learning Models
    # --------------------------------------------------------------------------

    "d4pg": ModelMetadata(
        name="d4pg",
        display_name="D4PG + EVT Agent",
        category=ModelCategory.RL,
        description="Distributed Distributional DDPG with Extreme Value Theory for "
                    "risk-aware trading. Features tail risk modeling, prioritized "
                    "experience replay, and N-step returns.",
        module_path="models.rl.d4pg_evt",
        class_name="D4PGAgent",
        entrypoint="D4PGAgent",
        input_type=InputType.STATE,
        input_shape_description="(state_dim,) - 1D state vector combining market features "
                                "and portfolio state (position, balance, rolling stats)",
        required_params=["state_dim"],
        optional_params={
            "action_dim": 1,
            "hidden_dim": 256,
            "n_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "gamma": 0.99,
            "tau": 0.005,
            "actor_lr": 1e-4,
            "critic_lr": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "n_step": 5,
            "risk_aversion": 0.5,
            "var_confidence": 0.99,
            "device": "auto",
        },
        train_method="train",
        train_signature="train() -> Dict  # Trains on stored transitions in replay buffer",
        supports_onnx=True,
        onnx_export_method="export_onnx",
        onnx_export_notes="Exports actor network only. export_onnx(path)",
        dependencies=["torch", "scipy", "numpy"],
        optional_dependencies=["onnx"],
        supports_gpu=True,
        typical_training_time="1-4 hours for 200 episodes",
        memory_requirements="High (~8GB for 1M replay buffer)",
    ),

    "marl": ModelMetadata(
        name="marl",
        display_name="MARL System",
        category=ModelCategory.RL,
        description="Multi-Agent Reinforcement Learning system with specialized agent "
                    "roles (Trend Follower, Mean Reverter, Momentum Trader, Risk Manager, "
                    "Coordinator). Uses CTDE (Centralized Training, Decentralized Execution).",
        module_path="models.rl.marl",
        class_name="MARLSystem",
        entrypoint="MARLSystem",
        input_type=InputType.MULTI_AGENT,
        input_shape_description="(n_agents, state_dim) - 2D array where each agent "
                                "receives market features + portfolio state",
        required_params=["state_dim"],
        optional_params={
            "n_agents": 5,
            "action_dim": 1,
            "hidden_dim": 128,
            "message_dim": 32,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "buffer_size": 500_000,
            "device": "auto",
        },
        train_method="train",
        train_signature="train() -> Dict  # Trains all agents on stored transitions",
        supports_onnx=True,
        onnx_export_method="export_onnx",
        onnx_export_notes="Exports each agent separately. export_onnx(path_dir)",
        dependencies=["torch", "numpy"],
        optional_dependencies=["onnx"],
        supports_gpu=True,
        typical_training_time="2-6 hours for 200 episodes",
        memory_requirements="High (~6GB for 500K buffer with 5 agents)",
    ),
}


# ==============================================================================
# MODEL REGISTRY CLASS
# ==============================================================================

class ModelRegistry:
    """
    Centralized registry for all trading models.

    Provides auto-discovery, metadata access, and factory methods for
    instantiating models with consistent interfaces.

    Example:
        registry = ModelRegistry()

        # List models by category
        ml_models = registry.list_models(category=ModelCategory.ML)

        # Get detailed info
        info = registry.get_model_info("lstm")
        print(info.train_signature)

        # Create model instance
        model = registry.create_model("lstm", input_size=64)
    """

    def __init__(self):
        self._models = MODEL_DEFINITIONS.copy()
        self._loaded_classes: Dict[str, Type] = {}
        logger.info(f"ModelRegistry initialized with {len(self._models)} models")

    # --------------------------------------------------------------------------
    # Discovery Methods
    # --------------------------------------------------------------------------

    def list_models(
        self,
        category: Optional[ModelCategory] = None,
        supports_onnx: Optional[bool] = None,
        supports_gpu: Optional[bool] = None,
    ) -> List[str]:
        """
        List available models with optional filtering.

        Args:
            category: Filter by model category (ML, DL, RL)
            supports_onnx: Filter by ONNX export support
            supports_gpu: Filter by GPU support

        Returns:
            List of model names matching criteria
        """
        result = []

        for name, meta in self._models.items():
            if category is not None and meta.category != category:
                continue
            if supports_onnx is not None and meta.supports_onnx != supports_onnx:
                continue
            if supports_gpu is not None and meta.supports_gpu != supports_gpu:
                continue
            result.append(name)

        return sorted(result)

    def list_by_category(self) -> Dict[str, List[str]]:
        """
        List all models grouped by category.

        Returns:
            Dictionary mapping category names to model lists
        """
        result = {}
        for category in ModelCategory:
            models = self.list_models(category=category)
            if models:
                result[category.value] = models
        return result

    def get_model_names(self) -> List[str]:
        """Get all registered model names."""
        return sorted(self._models.keys())

    # --------------------------------------------------------------------------
    # Metadata Access
    # --------------------------------------------------------------------------

    def get_model_info(self, name: str) -> ModelMetadata:
        """
        Get detailed metadata for a model.

        Args:
            name: Model name (e.g., 'lstm', 'lightgbm')

        Returns:
            ModelMetadata object with full model information

        Raises:
            KeyError: If model name is not registered
        """
        if name not in self._models:
            available = ", ".join(self.get_model_names())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._models[name]

    def get_input_requirements(self, name: str) -> Dict[str, Any]:
        """
        Get input shape requirements for a model.

        Args:
            name: Model name

        Returns:
            Dictionary with input type, shape description, and parameters
        """
        meta = self.get_model_info(name)
        return {
            "input_type": meta.input_type.value,
            "shape_description": meta.input_shape_description,
            "required_params": meta.required_params,
            "optional_params": meta.optional_params,
        }

    def get_dependencies(self, name: str) -> Dict[str, List[str]]:
        """
        Get dependencies for a model.

        Args:
            name: Model name

        Returns:
            Dictionary with required and optional dependencies
        """
        meta = self.get_model_info(name)
        return {
            "required": meta.dependencies,
            "optional": meta.optional_dependencies,
        }

    def get_all_dependencies(self) -> Dict[str, List[str]]:
        """
        Get all unique dependencies across all models.

        Returns:
            Dictionary with required and optional dependency lists
        """
        required = set()
        optional = set()

        for meta in self._models.values():
            required.update(meta.dependencies)
            optional.update(meta.optional_dependencies)

        # Remove from optional if also in required
        optional -= required

        return {
            "required": sorted(required),
            "optional": sorted(optional),
        }

    # --------------------------------------------------------------------------
    # Factory Methods
    # --------------------------------------------------------------------------

    def _load_class(self, name: str) -> Type:
        """Dynamically load model class from module."""
        if name in self._loaded_classes:
            return self._loaded_classes[name]

        meta = self.get_model_info(name)

        try:
            import importlib
            module = importlib.import_module(meta.module_path)
            cls = getattr(module, meta.class_name)
            self._loaded_classes[name] = cls
            return cls
        except ImportError as e:
            raise ImportError(
                f"Failed to import {meta.module_path}: {e}. "
                f"Required dependencies: {meta.dependencies}"
            )

    def create_model(self, name: str, **kwargs) -> Any:
        """
        Create a model instance with given parameters.

        Args:
            name: Model name
            **kwargs: Model initialization parameters

        Returns:
            Instantiated model object

        Example:
            model = registry.create_model("lstm", input_size=64, hidden_size=256)
        """
        meta = self.get_model_info(name)
        cls = self._load_class(name)

        # Merge with default optional params
        params = meta.optional_params.copy()
        params.update(kwargs)

        # Validate required params
        for param in meta.required_params:
            if param not in params:
                raise ValueError(
                    f"Missing required parameter '{param}' for {name}. "
                    f"Required: {meta.required_params}"
                )

        logger.info(f"Creating {meta.display_name} with params: {list(params.keys())}")
        return cls(**params)

    def get_model_class(self, name: str) -> Type:
        """
        Get the model class without instantiation.

        Args:
            name: Model name

        Returns:
            Model class type
        """
        return self._load_class(name)

    # --------------------------------------------------------------------------
    # ONNX Export Support
    # --------------------------------------------------------------------------

    def get_onnx_info(self, name: str) -> Dict[str, Any]:
        """
        Get ONNX export information for a model.

        Args:
            name: Model name

        Returns:
            Dictionary with ONNX support details
        """
        meta = self.get_model_info(name)
        return {
            "supported": meta.supports_onnx,
            "method": meta.onnx_export_method,
            "notes": meta.onnx_export_notes,
            "optional_deps": [d for d in meta.optional_dependencies if "onnx" in d.lower()],
        }

    def list_onnx_exportable(self) -> List[str]:
        """List all models that support ONNX export."""
        return self.list_models(supports_onnx=True)

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------

    def summary(self) -> str:
        """
        Generate a summary of all registered models.

        Returns:
            Formatted string summary
        """
        lines = [
            "=" * 70,
            "ORPFlow Model Registry Summary",
            "=" * 70,
            "",
        ]

        by_category = self.list_by_category()

        for category, models in by_category.items():
            lines.append(f"{category.upper().replace('_', ' ')}")
            lines.append("-" * 40)

            for model in models:
                meta = self._models[model]
                lines.append(f"  {meta.display_name} ({model})")
                lines.append(f"    Input: {meta.input_type.value}")
                lines.append(f"    ONNX: {'Yes' if meta.supports_onnx else 'No'}")
                lines.append(f"    GPU: {'Yes' if meta.supports_gpu else 'No'}")
                lines.append("")

        lines.append("=" * 70)
        lines.append(f"Total Models: {len(self._models)}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Dict]:
        """
        Export registry as dictionary.

        Returns:
            Dictionary representation of all model metadata
        """
        result = {}
        for name, meta in self._models.items():
            result[name] = {
                "display_name": meta.display_name,
                "category": meta.category.value,
                "description": meta.description,
                "module_path": meta.module_path,
                "class_name": meta.class_name,
                "input_type": meta.input_type.value,
                "input_shape": meta.input_shape_description,
                "required_params": meta.required_params,
                "optional_params": meta.optional_params,
                "train_signature": meta.train_signature,
                "supports_onnx": meta.supports_onnx,
                "onnx_method": meta.onnx_export_method,
                "dependencies": meta.dependencies,
                "supports_gpu": meta.supports_gpu,
            }
        return result

    def __repr__(self) -> str:
        return f"ModelRegistry({len(self._models)} models)"

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_registry() -> ModelRegistry:
    """Get a singleton instance of the model registry."""
    if not hasattr(get_registry, "_instance"):
        get_registry._instance = ModelRegistry()
    return get_registry._instance


def list_models(category: Optional[str] = None) -> List[str]:
    """
    Convenience function to list available models.

    Args:
        category: Optional category filter ('ml', 'dl', 'rl')

    Returns:
        List of model names
    """
    registry = get_registry()
    if category:
        cat_map = {
            "ml": ModelCategory.ML,
            "dl": ModelCategory.DL,
            "rl": ModelCategory.RL,
        }
        return registry.list_models(category=cat_map.get(category.lower()))
    return registry.get_model_names()


def create_model(name: str, **kwargs) -> Any:
    """
    Convenience function to create a model instance.

    Args:
        name: Model name
        **kwargs: Model parameters

    Returns:
        Model instance
    """
    return get_registry().create_model(name, **kwargs)


def get_model_info(name: str) -> ModelMetadata:
    """
    Convenience function to get model metadata.

    Args:
        name: Model name

    Returns:
        ModelMetadata object
    """
    return get_registry().get_model_info(name)


# ==============================================================================
# DATA SCHEMA DOCUMENTATION
# ==============================================================================

DATA_SCHEMA = {
    "klines_90d": {
        "path": "data/raw/klines_90d.parquet",
        "description": "90-day historical kline (candlestick) data from Binance",
        "expected_columns": [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote"
        ],
        "notes": "Stored via Git LFS due to size. Use git lfs pull to fetch.",
    },
    "features": {
        "path": "data/processed/features.parquet",
        "description": "Preprocessed feature matrix with technical indicators",
        "expected_columns": [
            # OHLCV base
            "open", "high", "low", "close", "volume",
            # Technical indicators (examples)
            "return_1", "return_5", "return_15",
            "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower",
            "atr_14", "adx_14",
            # Target columns
            "target_return_1", "target_return_5", "target_return_15",
        ],
        "notes": "Generated by data/preprocessor.py. Contains normalized features.",
    },
}


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Display registry summary and usage examples."""
    registry = ModelRegistry()

    print(registry.summary())
    print()

    print("USAGE EXAMPLES")
    print("=" * 70)
    print()

    print("1. List all models:")
    print("   >>> from models.registry import list_models")
    print("   >>> list_models()")
    print(f"   {list_models()}")
    print()

    print("2. List ML models only:")
    print("   >>> list_models('ml')")
    print(f"   {list_models('ml')}")
    print()

    print("3. Get model info:")
    print("   >>> info = get_model_info('lstm')")
    print("   >>> print(info.train_signature)")
    info = get_model_info('lstm')
    print(f"   {info.train_signature}")
    print()

    print("4. Create model instance:")
    print("   >>> model = create_model('lightgbm')")
    print("   >>> model = create_model('lstm', input_size=64)")
    print()

    print("5. Check ONNX export support:")
    print("   >>> registry.get_onnx_info('d4pg')")
    print(f"   {registry.get_onnx_info('d4pg')}")
    print()

    print("DATA FILES")
    print("=" * 70)
    for name, schema in DATA_SCHEMA.items():
        print(f"\n{name}:")
        print(f"  Path: {schema['path']}")
        print(f"  Description: {schema['description']}")
        print(f"  Notes: {schema['notes']}")


if __name__ == "__main__":
    main()
