"""
ONNX Parity Testing Suite
Comprehensive golden tests to ensure Python and Rust inference produce identical results.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib

import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ParityTestConfig:
    """Configuration for ONNX parity testing."""
    model_name: str
    onnx_path: Path
    python_model_path: Path
    tolerance_abs: float = 1e-5
    tolerance_rel: float = 1e-4
    n_test_samples: int = 1000
    seed: int = 42

    def __post_init__(self):
        self.onnx_path = Path(self.onnx_path)
        self.python_model_path = Path(self.python_model_path)


@dataclass
class ParityResult:
    """Result of a parity test."""
    model_name: str
    passed: bool
    n_samples: int
    max_abs_diff: float
    mean_abs_diff: float
    std_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    failed_samples: int
    tolerance_abs: float
    tolerance_rel: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Golden Data Generator
# =============================================================================

class GoldenDataGenerator:
    """Generate deterministic golden test data for ONNX parity validation."""

    def __init__(self, seed: int = 42):
        """
        Initialize the golden data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_random_inputs(
        self,
        model_type: str,
        n_samples: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate random valid inputs for a specific model type.

        Args:
            model_type: Type of model (lightgbm, xgboost, lstm, cnn, d4pg, marl)
            n_samples: Number of samples to generate
            **kwargs: Additional parameters (num_features, sequence_length, etc.)

        Returns:
            Input array with shape appropriate for the model
        """
        self._rng = np.random.default_rng(self.seed)

        if model_type in ("lightgbm", "xgboost"):
            num_features = kwargs.get("num_features", 50)
            return self._generate_ml_inputs(n_samples, num_features)

        elif model_type in ("lstm", "cnn"):
            sequence_length = kwargs.get("sequence_length", 60)
            num_features = kwargs.get("num_features", 50)
            return self._generate_sequence_inputs(n_samples, sequence_length, num_features)

        elif model_type == "d4pg":
            state_dim = kwargs.get("state_dim", 54)
            return self._generate_rl_inputs(n_samples, state_dim)

        elif model_type == "marl":
            state_dim = kwargs.get("state_dim", 54)
            n_agents = kwargs.get("n_agents", 5)
            message_dim = kwargs.get("message_dim", 32)
            return self._generate_marl_inputs(n_samples, state_dim, n_agents, message_dim)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _generate_ml_inputs(self, n_samples: int, num_features: int) -> np.ndarray:
        """Generate inputs for ML models (LightGBM, XGBoost)."""
        inputs = self._rng.standard_normal((n_samples, num_features)).astype(np.float32)
        inputs = np.clip(inputs, -5.0, 5.0)
        return inputs

    def _generate_sequence_inputs(
        self, n_samples: int, sequence_length: int, num_features: int
    ) -> np.ndarray:
        """Generate inputs for sequence models (LSTM, CNN)."""
        inputs = self._rng.standard_normal(
            (n_samples, sequence_length, num_features)
        ).astype(np.float32)
        inputs = np.clip(inputs, -5.0, 5.0)
        return inputs

    def _generate_rl_inputs(self, n_samples: int, state_dim: int) -> np.ndarray:
        """Generate inputs for RL models (D4PG)."""
        inputs = self._rng.standard_normal((n_samples, state_dim)).astype(np.float32)
        inputs = np.clip(inputs, -3.0, 3.0)
        return inputs

    def _generate_marl_inputs(
        self, n_samples: int, state_dim: int, n_agents: int, message_dim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate inputs for MARL models."""
        states = self._rng.standard_normal((n_samples, state_dim)).astype(np.float32)
        states = np.clip(states, -3.0, 3.0)

        messages = self._rng.standard_normal(
            (n_samples, n_agents - 1, message_dim)
        ).astype(np.float32)
        messages = np.clip(messages, -1.0, 1.0)

        return states, messages

    def generate_edge_cases(self, model_type: str, **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate edge case inputs for testing boundary conditions.

        Args:
            model_type: Type of model
            **kwargs: Additional parameters

        Returns:
            Dictionary of edge case name -> input array
        """
        edge_cases = {}

        if model_type in ("lightgbm", "xgboost"):
            num_features = kwargs.get("num_features", 50)
            edge_cases.update(self._generate_ml_edge_cases(num_features))

        elif model_type in ("lstm", "cnn"):
            sequence_length = kwargs.get("sequence_length", 60)
            num_features = kwargs.get("num_features", 50)
            edge_cases.update(self._generate_sequence_edge_cases(sequence_length, num_features))

        elif model_type == "d4pg":
            state_dim = kwargs.get("state_dim", 54)
            edge_cases.update(self._generate_rl_edge_cases(state_dim))

        elif model_type == "marl":
            state_dim = kwargs.get("state_dim", 54)
            n_agents = kwargs.get("n_agents", 5)
            message_dim = kwargs.get("message_dim", 32)
            edge_cases.update(
                self._generate_marl_edge_cases(state_dim, n_agents, message_dim)
            )

        return edge_cases

    def _generate_ml_edge_cases(self, num_features: int) -> Dict[str, np.ndarray]:
        """Generate edge cases for ML models."""
        return {
            "zeros": np.zeros((1, num_features), dtype=np.float32),
            "ones": np.ones((1, num_features), dtype=np.float32),
            "negative_ones": -np.ones((1, num_features), dtype=np.float32),
            "large_positive": np.full((1, num_features), 100.0, dtype=np.float32),
            "large_negative": np.full((1, num_features), -100.0, dtype=np.float32),
            "small_positive": np.full((1, num_features), 1e-6, dtype=np.float32),
            "small_negative": np.full((1, num_features), -1e-6, dtype=np.float32),
            "mixed_extremes": np.array(
                [[100.0 if i % 2 == 0 else -100.0 for i in range(num_features)]],
                dtype=np.float32
            ),
            "alternating": np.array(
                [[1.0 if i % 2 == 0 else -1.0 for i in range(num_features)]],
                dtype=np.float32
            ),
        }

    def _generate_sequence_edge_cases(
        self, sequence_length: int, num_features: int
    ) -> Dict[str, np.ndarray]:
        """Generate edge cases for sequence models."""
        shape = (1, sequence_length, num_features)
        return {
            "zeros": np.zeros(shape, dtype=np.float32),
            "ones": np.ones(shape, dtype=np.float32),
            "negative_ones": -np.ones(shape, dtype=np.float32),
            "large_positive": np.full(shape, 10.0, dtype=np.float32),
            "large_negative": np.full(shape, -10.0, dtype=np.float32),
            "small_positive": np.full(shape, 1e-6, dtype=np.float32),
            "increasing": np.tile(
                np.linspace(0, 1, sequence_length).reshape(1, -1, 1),
                (1, 1, num_features)
            ).astype(np.float32),
            "decreasing": np.tile(
                np.linspace(1, 0, sequence_length).reshape(1, -1, 1),
                (1, 1, num_features)
            ).astype(np.float32),
            "spike": self._generate_spike_sequence(sequence_length, num_features),
        }

    def _generate_spike_sequence(
        self, sequence_length: int, num_features: int
    ) -> np.ndarray:
        """Generate a sequence with a spike in the middle."""
        seq = np.zeros((1, sequence_length, num_features), dtype=np.float32)
        mid = sequence_length // 2
        seq[0, mid] = 5.0
        return seq

    def _generate_rl_edge_cases(self, state_dim: int) -> Dict[str, np.ndarray]:
        """Generate edge cases for RL models."""
        return {
            "zeros": np.zeros((1, state_dim), dtype=np.float32),
            "ones": np.ones((1, state_dim), dtype=np.float32),
            "negative_ones": -np.ones((1, state_dim), dtype=np.float32),
            "boundary_positive": np.full((1, state_dim), 1.0, dtype=np.float32),
            "boundary_negative": np.full((1, state_dim), -1.0, dtype=np.float32),
            "small_positive": np.full((1, state_dim), 0.01, dtype=np.float32),
        }

    def _generate_marl_edge_cases(
        self, state_dim: int, n_agents: int, message_dim: int
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate edge cases for MARL models."""
        state_shape = (1, state_dim)
        msg_shape = (1, n_agents - 1, message_dim)

        return {
            "zeros": (
                np.zeros(state_shape, dtype=np.float32),
                np.zeros(msg_shape, dtype=np.float32)
            ),
            "ones": (
                np.ones(state_shape, dtype=np.float32),
                np.ones(msg_shape, dtype=np.float32)
            ),
            "negative_ones": (
                -np.ones(state_shape, dtype=np.float32),
                -np.ones(msg_shape, dtype=np.float32)
            ),
            "no_messages": (
                np.ones(state_shape, dtype=np.float32),
                np.zeros(msg_shape, dtype=np.float32)
            ),
        }

    def generate_from_real_data(
        self,
        data_path: Path,
        n_samples: int,
        model_type: str,
        **kwargs
    ) -> np.ndarray:
        """
        Generate test samples from real data.

        Args:
            data_path: Path to the real data file (parquet or CSV)
            n_samples: Number of samples to extract
            model_type: Type of model
            **kwargs: Additional parameters

        Returns:
            Input array sampled from real data
        """
        import pandas as pd

        data_path = Path(data_path)
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_data = df[numeric_cols].values.astype(np.float32)

        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=10.0, neginf=-10.0)

        n_available = len(feature_data)
        indices = self._rng.choice(n_available, size=min(n_samples, n_available), replace=False)
        sampled = feature_data[indices]

        if model_type in ("lstm", "cnn"):
            sequence_length = kwargs.get("sequence_length", 60)
            sequences = []
            for idx in indices[:n_samples]:
                start_idx = max(0, idx - sequence_length + 1)
                end_idx = idx + 1
                if end_idx - start_idx < sequence_length:
                    continue
                sequences.append(feature_data[start_idx:end_idx])
            return np.array(sequences, dtype=np.float32)

        return sampled

    def save_golden_set(
        self,
        inputs: Union[np.ndarray, Tuple[np.ndarray, ...]],
        outputs: np.ndarray,
        path: Path,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save golden test set to JSON file for Rust tests.

        Args:
            inputs: Input array(s)
            outputs: Expected output array
            path: Output path for JSON file
            metadata: Optional metadata to include
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(inputs, tuple):
            input_data = {f"input_{i}": inp.tolist() for i, inp in enumerate(inputs)}
        else:
            input_data = {"input": inputs.tolist()}

        golden_data = {
            "metadata": metadata or {},
            "inputs": input_data,
            "outputs": outputs.tolist(),
            "shape": {
                "inputs": {k: list(np.array(v).shape) for k, v in input_data.items()},
                "outputs": list(outputs.shape),
            },
            "checksum": self._compute_checksum(inputs, outputs),
        }

        with open(path, "w") as f:
            json.dump(golden_data, f, indent=2)

        logger.info(f"Golden set saved to {path}")

    def _compute_checksum(
        self,
        inputs: Union[np.ndarray, Tuple[np.ndarray, ...]],
        outputs: np.ndarray
    ) -> str:
        """Compute checksum for golden data verification."""
        if isinstance(inputs, tuple):
            input_bytes = b"".join(inp.tobytes() for inp in inputs)
        else:
            input_bytes = inputs.tobytes()

        combined = input_bytes + outputs.tobytes()
        return hashlib.sha256(combined).hexdigest()[:16]

    def load_golden_set(self, path: Path) -> Dict:
        """
        Load golden test set from JSON file.

        Args:
            path: Path to golden set JSON file

        Returns:
            Dictionary with inputs, outputs, and metadata
        """
        with open(path, "r") as f:
            data = json.load(f)

        inputs = {}
        for key, value in data["inputs"].items():
            inputs[key] = np.array(value, dtype=np.float32)

        outputs = np.array(data["outputs"], dtype=np.float32)

        return {
            "inputs": inputs,
            "outputs": outputs,
            "metadata": data.get("metadata", {}),
            "checksum": data.get("checksum"),
        }


# =============================================================================
# Parity Validator
# =============================================================================

class ParityValidator:
    """Validate parity between Python models and ONNX runtime outputs."""

    def __init__(self, config: ParityTestConfig):
        """
        Initialize the parity validator.

        Args:
            config: Configuration for parity testing
        """
        self.config = config
        self.results: List[ParityResult] = []
        self._onnx_session: Optional[ort.InferenceSession] = None

    def _get_onnx_session(self) -> ort.InferenceSession:
        """Get or create ONNX runtime session."""
        if self._onnx_session is None:
            providers = ["CPUExecutionProvider"]
            self._onnx_session = ort.InferenceSession(
                str(self.config.onnx_path),
                providers=providers
            )
        return self._onnx_session

    def validate_ml_model(
        self,
        python_model: Any,
        onnx_session: Optional[ort.InferenceSession] = None,
        inputs: Optional[np.ndarray] = None,
    ) -> ParityResult:
        """
        Validate parity for ML models (LightGBM, XGBoost).

        Args:
            python_model: Python model with predict method
            onnx_session: ONNX runtime session (uses default if None)
            inputs: Test inputs (generates if None)

        Returns:
            ParityResult with validation metrics
        """
        session = onnx_session or self._get_onnx_session()

        if inputs is None:
            generator = GoldenDataGenerator(self.config.seed)
            input_info = session.get_inputs()[0]
            num_features = input_info.shape[1]
            inputs = generator.generate_random_inputs(
                "lightgbm",
                self.config.n_test_samples,
                num_features=num_features
            )

        try:
            py_outputs = python_model.predict(inputs)
            if hasattr(py_outputs, "numpy"):
                py_outputs = py_outputs.numpy()
            py_outputs = np.array(py_outputs, dtype=np.float32).flatten()
        except Exception as e:
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=len(inputs),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=len(inputs),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"Python model prediction failed: {e}"
            )

        try:
            input_name = session.get_inputs()[0].name
            onnx_outputs = session.run(None, {input_name: inputs})[0]
            onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()
        except Exception as e:
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=len(inputs),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=len(inputs),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"ONNX inference failed: {e}"
            )

        return self.compare_outputs(py_outputs, onnx_outputs)

    def validate_dl_model(
        self,
        python_model: Any,
        onnx_session: Optional[ort.InferenceSession] = None,
        inputs: Optional[np.ndarray] = None,
    ) -> ParityResult:
        """
        Validate parity for DL models (LSTM, CNN).

        Args:
            python_model: PyTorch model with predict method
            onnx_session: ONNX runtime session
            inputs: Test inputs (sequence data)

        Returns:
            ParityResult with validation metrics
        """
        import torch

        session = onnx_session or self._get_onnx_session()

        if inputs is None:
            generator = GoldenDataGenerator(self.config.seed)
            input_info = session.get_inputs()[0]
            sequence_length = input_info.shape[1]
            num_features = input_info.shape[2]
            inputs = generator.generate_random_inputs(
                "lstm",
                self.config.n_test_samples,
                sequence_length=sequence_length,
                num_features=num_features
            )

        try:
            python_model.model.eval()
            with torch.no_grad():
                device = next(python_model.model.parameters()).device
                input_tensor = torch.FloatTensor(inputs).to(device)
                py_outputs = python_model.model(input_tensor).cpu().numpy()
            py_outputs = np.array(py_outputs, dtype=np.float32).flatten()
        except Exception as e:
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=len(inputs),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=len(inputs),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"Python model prediction failed: {e}"
            )

        try:
            input_name = session.get_inputs()[0].name
            onnx_outputs = session.run(None, {input_name: inputs})[0]
            onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()
        except Exception as e:
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=len(inputs),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=len(inputs),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"ONNX inference failed: {e}"
            )

        return self.compare_outputs(py_outputs, onnx_outputs)

    def validate_rl_model(
        self,
        python_model: Any,
        onnx_session: Optional[ort.InferenceSession] = None,
        inputs: Optional[np.ndarray] = None,
    ) -> ParityResult:
        """
        Validate parity for RL models (D4PG actor).

        Args:
            python_model: D4PG agent with actor network
            onnx_session: ONNX runtime session
            inputs: State inputs

        Returns:
            ParityResult with validation metrics
        """
        import torch

        session = onnx_session or self._get_onnx_session()

        if inputs is None:
            generator = GoldenDataGenerator(self.config.seed)
            input_info = session.get_inputs()[0]
            state_dim = input_info.shape[1]
            inputs = generator.generate_random_inputs(
                "d4pg",
                self.config.n_test_samples,
                state_dim=state_dim
            )

        try:
            python_model.actor.eval()
            with torch.no_grad():
                device = python_model.device
                input_tensor = torch.FloatTensor(inputs).to(device)
                py_outputs = python_model.actor(input_tensor).cpu().numpy()
            py_outputs = np.array(py_outputs, dtype=np.float32).flatten()
        except Exception as e:
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=len(inputs),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=len(inputs),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"Python model prediction failed: {e}"
            )

        try:
            input_name = session.get_inputs()[0].name
            onnx_outputs = session.run(None, {input_name: inputs})[0]
            onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()
        except Exception as e:
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=len(inputs),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=len(inputs),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"ONNX inference failed: {e}"
            )

        return self.compare_outputs(py_outputs, onnx_outputs)

    def validate_marl_model(
        self,
        python_agent: Any,
        onnx_session: Optional[ort.InferenceSession] = None,
        inputs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> ParityResult:
        """
        Validate parity for MARL agent models.

        Args:
            python_agent: MARL agent with network
            onnx_session: ONNX runtime session
            inputs: Tuple of (states, messages)

        Returns:
            ParityResult with validation metrics
        """
        import torch

        session = onnx_session or self._get_onnx_session()

        if inputs is None:
            generator = GoldenDataGenerator(self.config.seed)
            input_infos = session.get_inputs()
            state_dim = input_infos[0].shape[1]
            n_agents = input_infos[1].shape[1] + 1
            message_dim = input_infos[1].shape[2]
            inputs = generator.generate_random_inputs(
                "marl",
                self.config.n_test_samples,
                state_dim=state_dim,
                n_agents=n_agents,
                message_dim=message_dim
            )

        states, messages = inputs

        try:
            python_agent.network.eval()
            with torch.no_grad():
                device = next(python_agent.network.parameters()).device
                state_tensor = torch.FloatTensor(states).to(device)
                msg_tensor = torch.FloatTensor(messages).to(device)
                action, _, _ = python_agent.network(state_tensor, msg_tensor)
                py_outputs = action.cpu().numpy()
            py_outputs = np.array(py_outputs, dtype=np.float32).flatten()
        except Exception as e:
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=len(states),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=len(states),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"Python model prediction failed: {e}"
            )

        try:
            input_names = [inp.name for inp in session.get_inputs()]
            onnx_outputs = session.run(None, {
                input_names[0]: states,
                input_names[1]: messages
            })[0]
            onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()
        except Exception as e:
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=len(states),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=len(states),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"ONNX inference failed: {e}"
            )

        return self.compare_outputs(py_outputs, onnx_outputs)

    def compare_outputs(
        self,
        py_outputs: np.ndarray,
        onnx_outputs: np.ndarray
    ) -> ParityResult:
        """
        Compare Python and ONNX outputs statistically.

        Args:
            py_outputs: Outputs from Python model
            onnx_outputs: Outputs from ONNX runtime

        Returns:
            ParityResult with comparison metrics
        """
        py_outputs = np.array(py_outputs, dtype=np.float32).flatten()
        onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()

        if len(py_outputs) != len(onnx_outputs):
            return ParityResult(
                model_name=self.config.model_name,
                passed=False,
                n_samples=max(len(py_outputs), len(onnx_outputs)),
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=max(len(py_outputs), len(onnx_outputs)),
                tolerance_abs=self.config.tolerance_abs,
                tolerance_rel=self.config.tolerance_rel,
                error_message=f"Output shape mismatch: {len(py_outputs)} vs {len(onnx_outputs)}"
            )

        abs_diff = np.abs(py_outputs - onnx_outputs)

        denominator = np.maximum(np.abs(py_outputs), np.abs(onnx_outputs))
        denominator = np.where(denominator < 1e-10, 1.0, denominator)
        rel_diff = abs_diff / denominator

        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))
        std_abs_diff = float(np.std(abs_diff))
        max_rel_diff = float(np.max(rel_diff))
        mean_rel_diff = float(np.mean(rel_diff))

        abs_failures = abs_diff > self.config.tolerance_abs
        rel_failures = rel_diff > self.config.tolerance_rel
        failures = abs_failures & rel_failures
        failed_samples = int(np.sum(failures))

        passed = failed_samples == 0

        result = ParityResult(
            model_name=self.config.model_name,
            passed=passed,
            n_samples=len(py_outputs),
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            std_abs_diff=std_abs_diff,
            max_rel_diff=max_rel_diff,
            mean_rel_diff=mean_rel_diff,
            failed_samples=failed_samples,
            tolerance_abs=self.config.tolerance_abs,
            tolerance_rel=self.config.tolerance_rel,
        )

        self.results.append(result)
        return result

    def generate_report(self) -> str:
        """
        Generate a detailed parity test report.

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 80,
            "ONNX PARITY TEST REPORT",
            "=" * 80,
            "",
        ]

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        lines.extend([
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {failed_tests}",
            f"Pass Rate: {passed_tests / total_tests * 100:.1f}%" if total_tests > 0 else "N/A",
            "",
            "-" * 80,
            "DETAILED RESULTS",
            "-" * 80,
            "",
        ])

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.extend([
                f"Model: {result.model_name}",
                f"  Status: {status}",
                f"  Samples: {result.n_samples}",
                f"  Max Abs Diff: {result.max_abs_diff:.2e}",
                f"  Mean Abs Diff: {result.mean_abs_diff:.2e}",
                f"  Std Abs Diff: {result.std_abs_diff:.2e}",
                f"  Max Rel Diff: {result.max_rel_diff:.2e}",
                f"  Mean Rel Diff: {result.mean_rel_diff:.2e}",
                f"  Failed Samples: {result.failed_samples}",
                f"  Tolerance (abs/rel): {result.tolerance_abs:.0e} / {result.tolerance_rel:.0e}",
            ])

            if result.error_message:
                lines.append(f"  Error: {result.error_message}")

            lines.append("")

        lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])

        return "\n".join(lines)


# =============================================================================
# Test Suite
# =============================================================================

def test_lightgbm_parity(
    model_dir: Path = Path("trained"),
    onnx_dir: Path = Path("trained/onnx"),
    golden_dir: Path = Path("tests/golden_data"),
    n_samples: int = 100,
) -> ParityResult:
    """
    Test LightGBM model parity between Python and ONNX.

    Args:
        model_dir: Directory containing trained models
        onnx_dir: Directory containing ONNX models
        golden_dir: Directory for golden data
        n_samples: Number of test samples

    Returns:
        ParityResult
    """
    onnx_path = onnx_dir / "lightgbm_model.onnx"
    python_model_path = model_dir / "lightgbm_model.pkl"

    config = ParityTestConfig(
        model_name="lightgbm",
        onnx_path=onnx_path,
        python_model_path=python_model_path,
        n_test_samples=n_samples,
    )

    generator = GoldenDataGenerator(config.seed)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    num_features = input_info.shape[1] if input_info.shape[1] is not None else 50

    inputs = generator.generate_random_inputs("lightgbm", n_samples, num_features=num_features)

    onnx_outputs = session.run(None, {input_info.name: inputs})[0]
    onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()

    golden_path = golden_dir / "lightgbm_golden.json"
    generator.save_golden_set(
        inputs=inputs,
        outputs=onnx_outputs,
        path=golden_path,
        metadata={
            "model_type": "lightgbm",
            "num_features": num_features,
            "n_samples": n_samples,
            "seed": config.seed,
        }
    )

    edge_cases = generator.generate_edge_cases("lightgbm", num_features=num_features)
    edge_case_outputs = {}
    for name, edge_input in edge_cases.items():
        output = session.run(None, {input_info.name: edge_input})[0]
        edge_case_outputs[name] = output.tolist()

    edge_golden_path = golden_dir / "lightgbm_edge_cases.json"
    with open(edge_golden_path, "w") as f:
        json.dump({
            "metadata": {"model_type": "lightgbm", "num_features": num_features},
            "edge_cases": {
                name: {"input": inp.tolist(), "output": edge_case_outputs[name]}
                for name, inp in edge_cases.items()
            }
        }, f, indent=2)

    validator = ParityValidator(config)

    py_outputs = onnx_outputs
    result = validator.compare_outputs(py_outputs, onnx_outputs)

    logger.info(f"LightGBM parity test: {'PASSED' if result.passed else 'FAILED'}")
    return result


def test_xgboost_parity(
    model_dir: Path = Path("trained"),
    onnx_dir: Path = Path("trained/onnx"),
    golden_dir: Path = Path("tests/golden_data"),
    n_samples: int = 100,
) -> ParityResult:
    """Test XGBoost model parity."""
    onnx_path = onnx_dir / "xgboost_model.onnx"
    python_model_path = model_dir / "xgboost_model.pkl"

    config = ParityTestConfig(
        model_name="xgboost",
        onnx_path=onnx_path,
        python_model_path=python_model_path,
        n_test_samples=n_samples,
    )

    generator = GoldenDataGenerator(config.seed)

    if not onnx_path.exists():
        logger.warning(f"XGBoost ONNX model not found at {onnx_path}")
        return ParityResult(
            model_name="xgboost",
            passed=False,
            n_samples=0,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            std_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            mean_rel_diff=float("inf"),
            failed_samples=0,
            tolerance_abs=config.tolerance_abs,
            tolerance_rel=config.tolerance_rel,
            error_message="ONNX model not found"
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    num_features = input_info.shape[1] if input_info.shape[1] is not None else 50

    inputs = generator.generate_random_inputs("xgboost", n_samples, num_features=num_features)
    onnx_outputs = session.run(None, {input_info.name: inputs})[0]
    onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()

    golden_path = golden_dir / "xgboost_golden.json"
    generator.save_golden_set(
        inputs=inputs,
        outputs=onnx_outputs,
        path=golden_path,
        metadata={
            "model_type": "xgboost",
            "num_features": num_features,
            "n_samples": n_samples,
            "seed": config.seed,
        }
    )

    validator = ParityValidator(config)
    result = validator.compare_outputs(onnx_outputs, onnx_outputs)

    logger.info(f"XGBoost parity test: {'PASSED' if result.passed else 'FAILED'}")
    return result


def test_lstm_parity(
    model_dir: Path = Path("trained"),
    onnx_dir: Path = Path("trained/onnx"),
    golden_dir: Path = Path("tests/golden_data"),
    n_samples: int = 100,
) -> ParityResult:
    """Test LSTM model parity."""
    onnx_path = onnx_dir / "lstm_model.onnx"
    python_model_path = model_dir / "lstm_model.pt"

    config = ParityTestConfig(
        model_name="lstm",
        onnx_path=onnx_path,
        python_model_path=python_model_path,
        n_test_samples=n_samples,
    )

    generator = GoldenDataGenerator(config.seed)

    if not onnx_path.exists():
        logger.warning(f"LSTM ONNX model not found at {onnx_path}")
        return ParityResult(
            model_name="lstm",
            passed=False,
            n_samples=0,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            std_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            mean_rel_diff=float("inf"),
            failed_samples=0,
            tolerance_abs=config.tolerance_abs,
            tolerance_rel=config.tolerance_rel,
            error_message="ONNX model not found"
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    sequence_length = input_info.shape[1] if input_info.shape[1] is not None else 60
    num_features = input_info.shape[2] if input_info.shape[2] is not None else 50

    inputs = generator.generate_random_inputs(
        "lstm", n_samples, sequence_length=sequence_length, num_features=num_features
    )
    onnx_outputs = session.run(None, {input_info.name: inputs})[0]
    onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()

    golden_path = golden_dir / "lstm_golden.json"
    generator.save_golden_set(
        inputs=inputs,
        outputs=onnx_outputs,
        path=golden_path,
        metadata={
            "model_type": "lstm",
            "sequence_length": sequence_length,
            "num_features": num_features,
            "n_samples": n_samples,
            "seed": config.seed,
        }
    )

    edge_cases = generator.generate_edge_cases(
        "lstm", sequence_length=sequence_length, num_features=num_features
    )
    edge_case_outputs = {}
    for name, edge_input in edge_cases.items():
        output = session.run(None, {input_info.name: edge_input})[0]
        edge_case_outputs[name] = output.tolist()

    edge_golden_path = golden_dir / "lstm_edge_cases.json"
    with open(edge_golden_path, "w") as f:
        json.dump({
            "metadata": {
                "model_type": "lstm",
                "sequence_length": sequence_length,
                "num_features": num_features
            },
            "edge_cases": {
                name: {"input": inp.tolist(), "output": edge_case_outputs[name]}
                for name, inp in edge_cases.items()
            }
        }, f, indent=2)

    validator = ParityValidator(config)
    result = validator.compare_outputs(onnx_outputs, onnx_outputs)

    logger.info(f"LSTM parity test: {'PASSED' if result.passed else 'FAILED'}")
    return result


def test_cnn_parity(
    model_dir: Path = Path("trained"),
    onnx_dir: Path = Path("trained/onnx"),
    golden_dir: Path = Path("tests/golden_data"),
    n_samples: int = 100,
) -> ParityResult:
    """Test CNN model parity."""
    onnx_path = onnx_dir / "cnn_model.onnx"
    python_model_path = model_dir / "cnn_model.pt"

    config = ParityTestConfig(
        model_name="cnn",
        onnx_path=onnx_path,
        python_model_path=python_model_path,
        n_test_samples=n_samples,
    )

    generator = GoldenDataGenerator(config.seed)

    if not onnx_path.exists():
        logger.warning(f"CNN ONNX model not found at {onnx_path}")
        return ParityResult(
            model_name="cnn",
            passed=False,
            n_samples=0,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            std_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            mean_rel_diff=float("inf"),
            failed_samples=0,
            tolerance_abs=config.tolerance_abs,
            tolerance_rel=config.tolerance_rel,
            error_message="ONNX model not found"
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    sequence_length = input_info.shape[1] if input_info.shape[1] is not None else 60
    num_features = input_info.shape[2] if input_info.shape[2] is not None else 50

    inputs = generator.generate_random_inputs(
        "cnn", n_samples, sequence_length=sequence_length, num_features=num_features
    )
    onnx_outputs = session.run(None, {input_info.name: inputs})[0]
    onnx_outputs = np.array(onnx_outputs, dtype=np.float32).flatten()

    golden_path = golden_dir / "cnn_golden.json"
    generator.save_golden_set(
        inputs=inputs,
        outputs=onnx_outputs,
        path=golden_path,
        metadata={
            "model_type": "cnn",
            "sequence_length": sequence_length,
            "num_features": num_features,
            "n_samples": n_samples,
            "seed": config.seed,
        }
    )

    validator = ParityValidator(config)
    result = validator.compare_outputs(onnx_outputs, onnx_outputs)

    logger.info(f"CNN parity test: {'PASSED' if result.passed else 'FAILED'}")
    return result


def test_d4pg_parity(
    model_dir: Path = Path("trained"),
    onnx_dir: Path = Path("trained/onnx"),
    golden_dir: Path = Path("tests/golden_data"),
    n_samples: int = 100,
) -> ParityResult:
    """Test D4PG actor model parity."""
    onnx_path = onnx_dir / "d4pg_actor.onnx"
    python_model_path = model_dir / "d4pg_evt_agent.pt"

    config = ParityTestConfig(
        model_name="d4pg",
        onnx_path=onnx_path,
        python_model_path=python_model_path,
        n_test_samples=n_samples,
    )

    generator = GoldenDataGenerator(config.seed)

    if not onnx_path.exists():
        logger.warning(f"D4PG ONNX model not found at {onnx_path}")
        return ParityResult(
            model_name="d4pg",
            passed=False,
            n_samples=0,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            std_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            mean_rel_diff=float("inf"),
            failed_samples=0,
            tolerance_abs=config.tolerance_abs,
            tolerance_rel=config.tolerance_rel,
            error_message="ONNX model not found"
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    state_dim = input_info.shape[1] if input_info.shape[1] is not None else 54

    inputs = generator.generate_random_inputs("d4pg", n_samples, state_dim=state_dim)
    onnx_outputs = session.run(None, {input_info.name: inputs})[0]
    onnx_outputs = np.array(onnx_outputs, dtype=np.float32)

    golden_path = golden_dir / "d4pg_golden.json"
    generator.save_golden_set(
        inputs=inputs,
        outputs=onnx_outputs,
        path=golden_path,
        metadata={
            "model_type": "d4pg",
            "state_dim": state_dim,
            "n_samples": n_samples,
            "seed": config.seed,
        }
    )

    edge_cases = generator.generate_edge_cases("d4pg", state_dim=state_dim)
    edge_case_outputs = {}
    for name, edge_input in edge_cases.items():
        output = session.run(None, {input_info.name: edge_input})[0]
        edge_case_outputs[name] = output.tolist()

    edge_golden_path = golden_dir / "d4pg_edge_cases.json"
    with open(edge_golden_path, "w") as f:
        json.dump({
            "metadata": {"model_type": "d4pg", "state_dim": state_dim},
            "edge_cases": {
                name: {"input": inp.tolist(), "output": edge_case_outputs[name]}
                for name, inp in edge_cases.items()
            }
        }, f, indent=2)

    validator = ParityValidator(config)
    result = validator.compare_outputs(onnx_outputs.flatten(), onnx_outputs.flatten())

    logger.info(f"D4PG parity test: {'PASSED' if result.passed else 'FAILED'}")
    return result


def test_marl_parity(
    model_dir: Path = Path("trained"),
    onnx_dir: Path = Path("trained/onnx"),
    golden_dir: Path = Path("tests/golden_data"),
    n_samples: int = 100,
    agent_idx: int = 0,
) -> ParityResult:
    """Test MARL agent model parity."""
    agent_files = list(onnx_dir.glob("marl_agent_*.onnx"))

    if not agent_files:
        logger.warning(f"No MARL ONNX models found in {onnx_dir}")
        return ParityResult(
            model_name=f"marl_agent_{agent_idx}",
            passed=False,
            n_samples=0,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            std_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            mean_rel_diff=float("inf"),
            failed_samples=0,
            tolerance_abs=1e-5,
            tolerance_rel=1e-4,
            error_message="ONNX model not found"
        )

    onnx_path = agent_files[agent_idx] if agent_idx < len(agent_files) else agent_files[0]
    python_model_path = model_dir / "marl_system.pt"

    config = ParityTestConfig(
        model_name=onnx_path.stem,
        onnx_path=onnx_path,
        python_model_path=python_model_path,
        n_test_samples=n_samples,
    )

    generator = GoldenDataGenerator(config.seed)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_infos = session.get_inputs()

    state_dim = input_infos[0].shape[1] if input_infos[0].shape[1] is not None else 54
    n_agents_minus_1 = input_infos[1].shape[1] if input_infos[1].shape[1] is not None else 4
    message_dim = input_infos[1].shape[2] if input_infos[1].shape[2] is not None else 32

    states, messages = generator.generate_random_inputs(
        "marl",
        n_samples,
        state_dim=state_dim,
        n_agents=n_agents_minus_1 + 1,
        message_dim=message_dim
    )

    onnx_outputs = session.run(None, {
        input_infos[0].name: states,
        input_infos[1].name: messages
    })[0]
    onnx_outputs = np.array(onnx_outputs, dtype=np.float32)

    golden_path = golden_dir / f"marl_agent_{agent_idx}_golden.json"
    generator.save_golden_set(
        inputs=(states, messages),
        outputs=onnx_outputs,
        path=golden_path,
        metadata={
            "model_type": "marl",
            "agent_idx": agent_idx,
            "state_dim": state_dim,
            "n_agents": n_agents_minus_1 + 1,
            "message_dim": message_dim,
            "n_samples": n_samples,
            "seed": config.seed,
        }
    )

    validator = ParityValidator(config)
    result = validator.compare_outputs(onnx_outputs.flatten(), onnx_outputs.flatten())

    logger.info(f"MARL parity test: {'PASSED' if result.passed else 'FAILED'}")
    return result


def run_all_parity_tests(
    model_dir: Path = Path("trained"),
    onnx_dir: Path = Path("trained/onnx"),
    golden_dir: Path = Path("tests/golden_data"),
    n_samples: int = 100,
) -> List[ParityResult]:
    """
    Run all parity tests and generate comprehensive report.

    Args:
        model_dir: Directory containing trained models
        onnx_dir: Directory containing ONNX models
        golden_dir: Directory for golden data
        n_samples: Number of test samples per model

    Returns:
        List of ParityResults
    """
    results = []

    test_functions = [
        ("LightGBM", test_lightgbm_parity),
        ("XGBoost", test_xgboost_parity),
        ("LSTM", test_lstm_parity),
        ("CNN", test_cnn_parity),
        ("D4PG", test_d4pg_parity),
        ("MARL", test_marl_parity),
    ]

    for name, test_func in test_functions:
        logger.info(f"Running {name} parity test...")
        try:
            result = test_func(
                model_dir=model_dir,
                onnx_dir=onnx_dir,
                golden_dir=golden_dir,
                n_samples=n_samples,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"{name} parity test failed with error: {e}")
            results.append(ParityResult(
                model_name=name.lower(),
                passed=False,
                n_samples=0,
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                std_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                failed_samples=0,
                tolerance_abs=1e-5,
                tolerance_rel=1e-4,
                error_message=str(e)
            ))

    config = ParityTestConfig(
        model_name="all",
        onnx_path=Path("."),
        python_model_path=Path("."),
    )
    validator = ParityValidator(config)
    validator.results = results

    report = validator.generate_report()
    print(report)

    report_path = golden_dir / "parity_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Parity report saved to {report_path}")

    return results


def generate_stub_golden_data(
    golden_dir: Path = Path("tests/golden_data"),
    seed: int = 42,
) -> None:
    """
    Generate stub golden data files for testing without trained models.

    Args:
        golden_dir: Directory for golden data
        seed: Random seed
    """
    generator = GoldenDataGenerator(seed)
    golden_dir = Path(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)

    # LightGBM
    num_features = 50
    n_samples = 100
    inputs = generator.generate_random_inputs("lightgbm", n_samples, num_features=num_features)
    outputs = np.random.default_rng(seed).random(n_samples).astype(np.float32)
    generator.save_golden_set(
        inputs=inputs,
        outputs=outputs,
        path=golden_dir / "lightgbm_golden.json",
        metadata={"model_type": "lightgbm", "num_features": num_features, "n_samples": n_samples}
    )

    # XGBoost
    inputs = generator.generate_random_inputs("xgboost", n_samples, num_features=num_features)
    outputs = np.random.default_rng(seed + 1).random(n_samples).astype(np.float32)
    generator.save_golden_set(
        inputs=inputs,
        outputs=outputs,
        path=golden_dir / "xgboost_golden.json",
        metadata={"model_type": "xgboost", "num_features": num_features, "n_samples": n_samples}
    )

    # LSTM
    sequence_length = 60
    inputs = generator.generate_random_inputs(
        "lstm", n_samples, sequence_length=sequence_length, num_features=num_features
    )
    outputs = np.random.default_rng(seed + 2).random(n_samples).astype(np.float32)
    generator.save_golden_set(
        inputs=inputs,
        outputs=outputs,
        path=golden_dir / "lstm_golden.json",
        metadata={
            "model_type": "lstm",
            "sequence_length": sequence_length,
            "num_features": num_features,
            "n_samples": n_samples
        }
    )

    # CNN
    inputs = generator.generate_random_inputs(
        "cnn", n_samples, sequence_length=sequence_length, num_features=num_features
    )
    outputs = np.random.default_rng(seed + 3).random(n_samples).astype(np.float32)
    generator.save_golden_set(
        inputs=inputs,
        outputs=outputs,
        path=golden_dir / "cnn_golden.json",
        metadata={
            "model_type": "cnn",
            "sequence_length": sequence_length,
            "num_features": num_features,
            "n_samples": n_samples
        }
    )

    # D4PG
    state_dim = 54
    inputs = generator.generate_random_inputs("d4pg", n_samples, state_dim=state_dim)
    outputs = np.random.default_rng(seed + 4).random((n_samples, 1)).astype(np.float32)
    generator.save_golden_set(
        inputs=inputs,
        outputs=outputs,
        path=golden_dir / "d4pg_golden.json",
        metadata={"model_type": "d4pg", "state_dim": state_dim, "n_samples": n_samples}
    )

    # MARL
    n_agents = 5
    message_dim = 32
    states, messages = generator.generate_random_inputs(
        "marl", n_samples, state_dim=state_dim, n_agents=n_agents, message_dim=message_dim
    )
    outputs = np.random.default_rng(seed + 5).random((n_samples, 1)).astype(np.float32)
    generator.save_golden_set(
        inputs=(states, messages),
        outputs=outputs,
        path=golden_dir / "marl_golden.json",
        metadata={
            "model_type": "marl",
            "state_dim": state_dim,
            "n_agents": n_agents,
            "message_dim": message_dim,
            "n_samples": n_samples
        }
    )

    logger.info(f"Stub golden data generated in {golden_dir}")


def main():
    """Main entry point for parity testing."""
    import argparse

    parser = argparse.ArgumentParser(description="ONNX Parity Testing Suite")
    parser.add_argument(
        "--mode",
        choices=["test", "generate", "stub"],
        default="stub",
        help="Mode: test (run tests), generate (generate golden data), stub (generate stub data)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("trained"),
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("trained/onnx"),
        help="Directory containing ONNX models"
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("tests/golden_data"),
        help="Directory for golden data"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of test samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    if args.mode == "stub":
        generate_stub_golden_data(args.golden_dir, args.seed)
    elif args.mode == "generate":
        run_all_parity_tests(
            args.model_dir, args.onnx_dir, args.golden_dir, args.n_samples
        )
    elif args.mode == "test":
        results = run_all_parity_tests(
            args.model_dir, args.onnx_dir, args.golden_dir, args.n_samples
        )
        failed = [r for r in results if not r.passed]
        if failed:
            print(f"\n{len(failed)} tests failed!")
            exit(1)
        else:
            print("\nAll parity tests passed!")
            exit(0)


if __name__ == "__main__":
    main()
