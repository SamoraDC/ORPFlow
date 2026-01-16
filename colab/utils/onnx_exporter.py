"""
ONNX Exporter for Rust Consumption
Exports trained models to ONNX format with validation and metadata

Output Directory Structure:
/trained/onnx/
  ├── lightgbm_model.onnx
  ├── lightgbm_metadata.json
  ├── xgboost_model.onnx
  ├── xgboost_metadata.json
  ├── lstm_model.onnx
  ├── cnn_model.onnx
  ├── d4pg_actor.onnx
  ├── marl_agent_0.onnx
  └── manifest.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Export trained models to ONNX format for Rust inference.

    Features:
    1. Model validation after export
    2. Metadata generation for each model
    3. Manifest file for Rust loader
    4. Quantization support (optional)
    """

    def __init__(self, output_dir: str = "trained/onnx"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exported_models = {}
        self.manifest = {
            "models": [],
            "version": "1.0",
            "export_info": {}
        }

    def export_lightgbm(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "lightgbm_model"
    ) -> str:
        """Export LightGBM model to ONNX"""
        import onnx
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        output_path = self.output_dir / f"{model_name}.onnx"

        initial_types = [("input", FloatTensorType([None, len(feature_names)]))]

        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=17
        )

        onnx.save_model(onnx_model, str(output_path))

        # Validate
        self._validate_onnx(str(output_path))

        # Save metadata
        self._save_metadata(model_name, {
            "model_type": "lightgbm",
            "feature_names": feature_names,
            "num_features": len(feature_names),
            "input_shape": [None, len(feature_names)],
            "output_shape": [None, 1]
        })

        logger.info(f"LightGBM exported to {output_path}")
        return str(output_path)

    def export_xgboost(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "xgboost_model"
    ) -> str:
        """Export XGBoost model to ONNX"""
        import onnx
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        output_path = self.output_dir / f"{model_name}.onnx"

        initial_types = [("input", FloatTensorType([None, len(feature_names)]))]

        onnx_model = convert_xgboost(
            model,
            initial_types=initial_types,
            target_opset=17
        )

        onnx.save_model(onnx_model, str(output_path))

        # Validate
        self._validate_onnx(str(output_path))

        # Save metadata
        self._save_metadata(model_name, {
            "model_type": "xgboost",
            "feature_names": feature_names,
            "num_features": len(feature_names),
            "input_shape": [None, len(feature_names)],
            "output_shape": [None, 1]
        })

        logger.info(f"XGBoost exported to {output_path}")
        return str(output_path)

    def export_pytorch(
        self,
        model: Any,
        input_shape: tuple,
        model_name: str,
        input_names: List[str] = ["input"],
        output_names: List[str] = ["output"]
    ) -> str:
        """Export PyTorch model to ONNX"""
        import torch

        output_path = self.output_dir / f"{model_name}.onnx"

        model.eval()
        device = next(model.parameters()).device

        dummy_input = torch.randn(*input_shape).to(device)

        dynamic_axes = {
            name: {0: "batch_size"} for name in input_names + output_names
        }

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        # Validate
        self._validate_onnx(str(output_path))

        # Save metadata
        self._save_metadata(model_name, {
            "model_type": "pytorch",
            "input_shape": list(input_shape),
            "input_names": input_names,
            "output_names": output_names
        })

        logger.info(f"PyTorch model exported to {output_path}")
        return str(output_path)

    def export_lstm(
        self,
        model: Any,
        sequence_length: int,
        num_features: int,
        model_name: str = "lstm_model"
    ) -> str:
        """Export LSTM model to ONNX"""
        return self.export_pytorch(
            model,
            input_shape=(1, sequence_length, num_features),
            model_name=model_name
        )

    def export_cnn(
        self,
        model: Any,
        sequence_length: int,
        num_features: int,
        model_name: str = "cnn_model"
    ) -> str:
        """Export CNN model to ONNX"""
        return self.export_pytorch(
            model,
            input_shape=(1, sequence_length, num_features),
            model_name=model_name
        )

    def export_d4pg_actor(
        self,
        actor: Any,
        state_dim: int,
        model_name: str = "d4pg_actor"
    ) -> str:
        """Export D4PG actor network to ONNX"""
        import torch

        output_path = self.output_dir / f"{model_name}.onnx"

        actor.eval()
        device = next(actor.parameters()).device

        dummy_input = torch.randn(1, state_dim).to(device)

        torch.onnx.export(
            actor,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={
                "state": {0: "batch_size"},
                "action": {0: "batch_size"}
            }
        )

        # Validate
        self._validate_onnx(str(output_path))

        # Save metadata
        self._save_metadata(model_name, {
            "model_type": "d4pg_actor",
            "state_dim": state_dim,
            "action_dim": 1,
            "input_names": ["state"],
            "output_names": ["action"]
        })

        logger.info(f"D4PG actor exported to {output_path}")
        return str(output_path)

    def export_marl_agent(
        self,
        agent: Any,
        state_dim: int,
        message_dim: int,
        n_agents: int,
        agent_id: int,
        model_name: str = "marl_agent"
    ) -> str:
        """Export MARL agent to ONNX"""
        import torch
        import torch.nn as nn

        output_path = self.output_dir / f"{model_name}_{agent_id}.onnx"

        # Wrapper to extract action only
        class ActionExtractor(nn.Module):
            def __init__(self, agent_model):
                super().__init__()
                self.agent = agent_model

            def forward(self, state, messages):
                action, _ = self.agent(state, messages)
                return action

        extractor = ActionExtractor(agent)
        extractor.eval()
        device = next(agent.parameters()).device

        dummy_state = torch.randn(1, state_dim).to(device)
        dummy_msgs = torch.randn(1, n_agents - 1, message_dim).to(device)

        torch.onnx.export(
            extractor,
            (dummy_state, dummy_msgs),
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["state", "messages"],
            output_names=["action"],
            dynamic_axes={
                "state": {0: "batch_size"},
                "messages": {0: "batch_size"},
                "action": {0: "batch_size"}
            }
        )

        # Validate
        self._validate_onnx(str(output_path))

        # Save metadata
        self._save_metadata(f"{model_name}_{agent_id}", {
            "model_type": "marl_agent",
            "agent_id": agent_id,
            "state_dim": state_dim,
            "message_dim": message_dim,
            "n_agents": n_agents
        })

        logger.info(f"MARL agent {agent_id} exported to {output_path}")
        return str(output_path)

    def _validate_onnx(self, path: str):
        """Validate ONNX model structure"""
        import onnx
        from onnx import checker

        model = onnx.load(path)
        checker.check_model(model)
        logger.info(f"ONNX validation passed: {path}")

    def _save_metadata(self, model_name: str, metadata: Dict):
        """Save model metadata for Rust inference"""
        metadata["onnx_file"] = f"{model_name}.onnx"

        metadata_path = self.output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.exported_models[model_name] = metadata
        self.manifest["models"].append(model_name)

    def test_inference(
        self,
        model_path: str,
        test_input: np.ndarray
    ) -> np.ndarray:
        """Test ONNX inference to verify export"""
        import onnxruntime as ort

        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        result = session.run(None, {input_name: test_input.astype(np.float32)})

        return result[0]

    def save_manifest(self):
        """Save manifest file for Rust loader"""
        self.manifest["exported_models"] = self.exported_models
        self.manifest["export_dir"] = str(self.output_dir)

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

        logger.info(f"Manifest saved to {manifest_path}")

    def print_summary(self):
        """Print export summary"""
        print("\n" + "=" * 60)
        print("ONNX EXPORT SUMMARY")
        print("=" * 60)

        print(f"\nOutput Directory: {self.output_dir}")
        print(f"Total Models: {len(self.exported_models)}")

        print("\nExported Models:")
        for name, meta in self.exported_models.items():
            size = (self.output_dir / f"{name}.onnx").stat().st_size / 1024
            print(f"  - {name}.onnx ({size:.1f} KB)")
            print(f"    Type: {meta.get('model_type', 'unknown')}")

        print(f"\nManifest: {self.output_dir / 'manifest.json'}")
        print("\n Ready for Rust consumption!")
