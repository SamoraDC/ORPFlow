"""
ONNX Export Utilities
=====================

This module provides utilities for exporting trained models to ONNX format
and validating parity between Python and ONNX inference.

Components:
- ONNXExporter: Export various model types to ONNX format
- GoldenDataGenerator: Generate deterministic test data for parity validation
- ParityValidator: Validate inference parity between Python and ONNX
- run_all_parity_tests: Run comprehensive parity tests for all models
"""

from .onnx_exporter import ONNXExporter

from .onnx_parity import (
    # Configuration
    ParityTestConfig,
    ParityResult,
    # Golden data generation
    GoldenDataGenerator,
    # Parity validation
    ParityValidator,
    # Test functions
    test_lightgbm_parity,
    test_xgboost_parity,
    test_lstm_parity,
    test_cnn_parity,
    test_d4pg_parity,
    test_marl_parity,
    run_all_parity_tests,
    generate_stub_golden_data,
)

__all__ = [
    # Exporter
    "ONNXExporter",
    # Parity testing
    "ParityTestConfig",
    "ParityResult",
    "GoldenDataGenerator",
    "ParityValidator",
    "test_lightgbm_parity",
    "test_xgboost_parity",
    "test_lstm_parity",
    "test_cnn_parity",
    "test_d4pg_parity",
    "test_marl_parity",
    "run_all_parity_tests",
    "generate_stub_golden_data",
]
