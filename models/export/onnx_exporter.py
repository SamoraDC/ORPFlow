"""
ONNX Export Utilities
Export trained models to ONNX format for Rust inference
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import shutil

import numpy as np
import onnx
from onnx import checker, helper, TensorProto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export models to ONNX format with validation"""

    def __init__(self, output_dir: str = "trained/onnx"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exported_models: Dict[str, Dict] = {}

    def export_lightgbm(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "lightgbm_model",
    ) -> str:
        """Export LightGBM model to ONNX"""

        output_path = self.output_dir / f"{model_name}.onnx"

        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_types = [("input", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_lightgbm(
                model.model,
                initial_types=initial_types,
                target_opset=17,
            )

            onnx.save_model(onnx_model, str(output_path))

            self._validate_model(str(output_path))
            self._save_metadata(model_name, feature_names, "lightgbm")

            logger.info(f"LightGBM exported to {output_path}")
            return str(output_path)

        except ImportError:
            logger.warning("onnxmltools not installed, saving native format")
            native_path = self.output_dir / f"{model_name}.lgb"
            model.model.save_model(str(native_path))
            return str(native_path)

    def export_xgboost(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "xgboost_model",
    ) -> str:
        """Export XGBoost model to ONNX"""

        output_path = self.output_dir / f"{model_name}.onnx"

        try:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_types = [("input", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_xgboost(
                model.model,
                initial_types=initial_types,
                target_opset=17,
            )

            onnx.save_model(onnx_model, str(output_path))

            self._validate_model(str(output_path))
            self._save_metadata(model_name, feature_names, "xgboost")

            logger.info(f"XGBoost exported to {output_path}")
            return str(output_path)

        except ImportError:
            logger.warning("onnxmltools not installed, saving native format")
            native_path = self.output_dir / f"{model_name}.xgb"
            model.model.save_model(str(native_path))
            return str(native_path)

    def export_pytorch(
        self,
        model: Any,
        input_shape: tuple,
        model_name: str,
        input_names: List[str] = ["input"],
        output_names: List[str] = ["output"],
    ) -> str:
        """Export PyTorch model to ONNX"""
        import torch

        output_path = self.output_dir / f"{model_name}.onnx"

        model.model.eval()
        device = next(model.model.parameters()).device

        dummy_input = torch.randn(*input_shape).to(device)

        dynamic_axes = {
            name: {0: "batch_size"} for name in input_names + output_names
        }

        torch.onnx.export(
            model.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        self._validate_model(str(output_path))
        self._save_metadata(
            model_name,
            feature_names=None,
            model_type="pytorch",
            extra={
                "input_shape": list(input_shape),
                "input_names": input_names,
                "output_names": output_names,
            },
        )

        logger.info(f"PyTorch model exported to {output_path}")
        return str(output_path)

    def export_lstm(
        self,
        model: Any,
        sequence_length: int,
        num_features: int,
        model_name: str = "lstm_model",
    ) -> str:
        """Export LSTM model to ONNX"""

        return self.export_pytorch(
            model,
            input_shape=(1, sequence_length, num_features),
            model_name=model_name,
        )

    def export_cnn(
        self,
        model: Any,
        sequence_length: int,
        num_features: int,
        model_name: str = "cnn_model",
    ) -> str:
        """Export CNN model to ONNX"""

        return self.export_pytorch(
            model,
            input_shape=(1, sequence_length, num_features),
            model_name=model_name,
        )

    def export_d4pg_actor(
        self,
        agent: Any,
        model_name: str = "d4pg_actor",
    ) -> str:
        """Export D4PG actor network to ONNX"""
        import torch

        output_path = self.output_dir / f"{model_name}.onnx"

        agent.actor.eval()
        device = agent.device

        dummy_input = torch.randn(1, agent.state_dim).to(device)

        torch.onnx.export(
            agent.actor,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={
                "state": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
        )

        self._validate_model(str(output_path))
        self._save_metadata(
            model_name,
            feature_names=None,
            model_type="d4pg",
            extra={"state_dim": agent.state_dim, "action_dim": agent.action_dim},
        )

        logger.info(f"D4PG actor exported to {output_path}")
        return str(output_path)

    def export_marl_agents(
        self,
        marl_system: Any,
        base_name: str = "marl",
    ) -> List[str]:
        """Export all MARL agent networks to ONNX"""
        import torch

        exported_paths = []

        for i, agent in enumerate(marl_system.agents):
            role = agent.config.role.value
            model_name = f"{base_name}_agent_{i}_{role}"
            output_path = self.output_dir / f"{model_name}.onnx"

            agent.network.eval()
            device = marl_system.device

            dummy_state = torch.randn(1, marl_system.state_dim).to(device)
            dummy_messages = torch.randn(
                1, marl_system.n_agents - 1, marl_system.message_dim
            ).to(device)

            # Create wrapper for action-only output
            class ActionExtractor(torch.nn.Module):
                def __init__(self, network):
                    super().__init__()
                    self.network = network

                def forward(self, state, messages):
                    action, _, _ = self.network(state, messages)
                    return action

            extractor = ActionExtractor(agent.network)

            torch.onnx.export(
                extractor,
                (dummy_state, dummy_messages),
                str(output_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["state", "messages"],
                output_names=["action"],
                dynamic_axes={
                    "state": {0: "batch_size"},
                    "messages": {0: "batch_size"},
                    "action": {0: "batch_size"},
                },
            )

            self._validate_model(str(output_path))
            self._save_metadata(
                model_name,
                feature_names=None,
                model_type="marl",
                extra={
                    "agent_id": i,
                    "role": role,
                    "state_dim": marl_system.state_dim,
                    "message_dim": marl_system.message_dim,
                    "n_agents": marl_system.n_agents,
                },
            )

            exported_paths.append(str(output_path))
            logger.info(f"MARL agent {i} ({role}) exported to {output_path}")

        return exported_paths

    def _validate_model(self, path: str):
        """Validate ONNX model"""
        model = onnx.load(path)
        checker.check_model(model)
        logger.info(f"Model validation passed: {path}")

    def _save_metadata(
        self,
        model_name: str,
        feature_names: Optional[List[str]],
        model_type: str,
        extra: Optional[Dict] = None,
    ):
        """Save model metadata for Rust inference"""

        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "onnx_path": f"{model_name}.onnx",
        }

        if feature_names:
            metadata["feature_names"] = feature_names
            metadata["num_features"] = len(feature_names)

        if extra:
            metadata.update(extra)

        metadata_path = self.output_dir / f"{model_name}_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.exported_models[model_name] = metadata

    def export_all(
        self,
        models: Dict[str, Any],
        feature_names: List[str],
        sequence_length: int = 60,
    ) -> Dict[str, str]:
        """Export all models to ONNX"""

        exported = {}

        for model_name, model in models.items():
            try:
                if "lightgbm" in model_name.lower():
                    path = self.export_lightgbm(model, feature_names, model_name)
                elif "xgboost" in model_name.lower():
                    path = self.export_xgboost(model, feature_names, model_name)
                elif "lstm" in model_name.lower():
                    path = self.export_lstm(
                        model, sequence_length, len(feature_names), model_name
                    )
                elif "cnn" in model_name.lower():
                    path = self.export_cnn(
                        model, sequence_length, len(feature_names), model_name
                    )
                elif "d4pg" in model_name.lower():
                    path = self.export_d4pg_actor(model, model_name)
                elif "marl" in model_name.lower():
                    paths = self.export_marl_agents(model, model_name)
                    for p in paths:
                        exported[Path(p).stem] = p
                    continue
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue

                exported[model_name] = path

            except Exception as e:
                logger.error(f"Failed to export {model_name}: {e}")

        # Save manifest
        manifest = {
            "models": list(exported.keys()),
            "export_dir": str(self.output_dir),
            "metadata": self.exported_models,
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Exported {len(exported)} models. Manifest: {manifest_path}")

        return exported

    def optimize_model(self, model_path: str) -> str:
        """Optimize ONNX model for inference"""
        try:
            from onnxruntime.transformers import optimizer

            optimized_path = model_path.replace(".onnx", "_optimized.onnx")

            optimized_model = optimizer.optimize_model(
                model_path,
                model_type="bert",  # Generic optimization
                num_heads=0,
                hidden_size=0,
            )

            optimized_model.save_model_to_file(optimized_path)
            logger.info(f"Optimized model saved to {optimized_path}")

            return optimized_path

        except ImportError:
            logger.warning("onnxruntime-tools not installed, skipping optimization")
            return model_path

    def quantize_model(
        self,
        model_path: str,
        quantization_type: str = "dynamic",
    ) -> str:
        """Quantize ONNX model for faster inference"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = model_path.replace(".onnx", "_quantized.onnx")

            quantize_dynamic(
                model_path,
                quantized_path,
                weight_type=QuantType.QUInt8,
            )

            logger.info(f"Quantized model saved to {quantized_path}")
            return quantized_path

        except ImportError:
            logger.warning("onnxruntime not installed, skipping quantization")
            return model_path


def main():
    """Test ONNX export utilities"""

    exporter = ONNXExporter()

    logger.info("ONNX Exporter initialized")
    logger.info("Use export_* methods to export individual models")
    logger.info("Use export_all() to export all models at once")


if __name__ == "__main__":
    main()
