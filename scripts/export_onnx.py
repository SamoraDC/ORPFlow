#!/usr/bin/env python3
"""
Export LightGBM and XGBoost models to ONNX format for Rust inference.
"""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import onnx
import xgboost as xgb
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAINED_DIR = PROJECT_ROOT / "trained"
ONNX_DIR = TRAINED_DIR / "onnx"


def get_feature_names_from_lightgbm(model_path: Path) -> list[str]:
    """Extract feature names from LightGBM model file."""
    with open(model_path) as f:
        for line in f:
            if line.startswith("feature_names="):
                return line.strip().split("=")[1].split()
    return []


def export_lightgbm():
    """Export LightGBM model to ONNX."""
    model_path = TRAINED_DIR / "lightgbm_advanced.txt"
    if not model_path.exists():
        model_path = TRAINED_DIR / "lightgbm_model.txt"

    if not model_path.exists():
        logger.warning("LightGBM model not found")
        return None

    logger.info(f"Loading LightGBM from {model_path}")
    booster = lgb.Booster(model_file=str(model_path))

    # Get feature names
    feature_names = get_feature_names_from_lightgbm(model_path)
    num_features = booster.num_feature()

    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    logger.info(f"Model has {num_features} features")

    # Convert to ONNX
    initial_types = [("input", FloatTensorType([None, num_features]))]
    onnx_model = convert_lightgbm(
        booster,
        initial_types=initial_types,
        target_opset=15,
    )

    # Save
    output_path = ONNX_DIR / "lightgbm_model.onnx"
    onnx.save_model(onnx_model, str(output_path))

    # Validate
    onnx.checker.check_model(onnx_model)
    logger.info(f"LightGBM exported to {output_path}")

    # Save metadata
    metadata = {
        "model_name": "lightgbm_model",
        "model_type": "lightgbm",
        "onnx_path": "lightgbm_model.onnx",
        "feature_names": feature_names,
        "num_features": num_features,
    }

    with open(ONNX_DIR / "lightgbm_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def export_xgboost():
    """Export XGBoost model to ONNX."""
    # Try different model formats
    model_path = None
    for ext in ["json", "ubj"]:
        p = TRAINED_DIR / f"xgboost_advanced.{ext}"
        if p.exists():
            model_path = p
            break
        p = TRAINED_DIR / f"xgboost_model.{ext}"
        if p.exists():
            model_path = p
            break

    if not model_path:
        logger.warning("XGBoost model not found")
        return None

    logger.info(f"Loading XGBoost from {model_path}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Get feature names from model config
    config = json.loads(booster.save_config())
    num_features = int(config.get("learner", {}).get("learner_model_param", {}).get("num_feature", 118))

    # Get original feature names for metadata
    original_feature_names = booster.feature_names
    if original_feature_names:
        feature_names = list(original_feature_names)
    else:
        feature_names = [f"f{i}" for i in range(num_features)]

    logger.info(f"Model has {num_features} features")

    # Rename features to f%d format for onnxmltools compatibility
    booster.feature_names = [f"f{i}" for i in range(num_features)]

    # Convert to ONNX
    initial_types = [("input", FloatTensorType([None, num_features]))]
    onnx_model = convert_xgboost(
        booster,
        initial_types=initial_types,
        target_opset=15,
    )

    # Save
    output_path = ONNX_DIR / "xgboost_model.onnx"
    onnx.save_model(onnx_model, str(output_path))

    # Validate
    onnx.checker.check_model(onnx_model)
    logger.info(f"XGBoost exported to {output_path}")

    # Save metadata
    metadata = {
        "model_name": "xgboost_model",
        "model_type": "xgboost",
        "onnx_path": "xgboost_model.onnx",
        "feature_names": feature_names,
        "num_features": num_features,
    }

    with open(ONNX_DIR / "xgboost_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def create_manifest(exported_models: list[Path]):
    """Create manifest file for Rust loader."""
    models = []
    metadata = {}

    for model_path in exported_models:
        if model_path:
            model_name = model_path.stem
            models.append(model_name)

            meta_path = ONNX_DIR / f"{model_name}_metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata[model_name] = json.load(f)

    manifest = {
        "models": models,
        "export_dir": str(ONNX_DIR),
        "metadata": metadata,
    }

    with open(ONNX_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest created with {len(models)} models")


def main():
    """Export all models."""
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ONNX export...")

    exported = []

    # Export LightGBM
    try:
        lgb_path = export_lightgbm()
        exported.append(lgb_path)
    except Exception as e:
        logger.error(f"LightGBM export failed: {e}")

    # Export XGBoost
    try:
        xgb_path = export_xgboost()
        exported.append(xgb_path)
    except Exception as e:
        logger.error(f"XGBoost export failed: {e}")

    # Create manifest
    create_manifest(exported)

    logger.info(f"Export complete! {len([x for x in exported if x])} models exported to {ONNX_DIR}")


if __name__ == "__main__":
    main()
