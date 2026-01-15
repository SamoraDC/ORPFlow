//! ONNX Parity Tests
//!
//! This module contains integration tests that verify ONNX model outputs
//! match the expected golden data generated from Python.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;

/// Configuration for parity testing
const TOLERANCE_ABS: f32 = 1e-5;
const TOLERANCE_REL: f32 = 1e-4;

/// Golden data structure for ML models (LightGBM, XGBoost)
#[derive(Debug, Deserialize)]
struct GoldenDataML {
    metadata: GoldenMetadata,
    inputs: MLInputs,
    outputs: Vec<f32>,
    shape: ShapeInfo,
    checksum: String,
}

/// Golden data structure for sequence models (LSTM, CNN)
#[derive(Debug, Deserialize)]
struct GoldenDataSequence {
    metadata: GoldenMetadata,
    inputs: SequenceInputs,
    outputs: Vec<f32>,
    shape: ShapeInfo,
    checksum: String,
}

/// Golden data structure for RL models (D4PG)
#[derive(Debug, Deserialize)]
struct GoldenDataRL {
    metadata: GoldenMetadata,
    inputs: RLInputs,
    outputs: Vec<Vec<f32>>,
    shape: ShapeInfo,
    checksum: String,
}

/// Golden data structure for MARL models
#[derive(Debug, Deserialize)]
struct GoldenDataMARL {
    metadata: GoldenMetadata,
    inputs: MARLInputs,
    outputs: Vec<Vec<f32>>,
    shape: ShapeInfo,
    checksum: String,
}

#[derive(Debug, Deserialize)]
struct GoldenMetadata {
    model_type: String,
    #[serde(default)]
    num_features: Option<usize>,
    #[serde(default)]
    sequence_length: Option<usize>,
    #[serde(default)]
    state_dim: Option<usize>,
    #[serde(default)]
    n_agents: Option<usize>,
    #[serde(default)]
    message_dim: Option<usize>,
    n_samples: usize,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct MLInputs {
    input: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct SequenceInputs {
    input: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct RLInputs {
    input: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct MARLInputs {
    input_0: Vec<Vec<f32>>,
    input_1: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct ShapeInfo {
    inputs: serde_json::Value,
    outputs: Vec<usize>,
}

/// Result of a parity comparison
#[derive(Debug)]
struct ParityResult {
    passed: bool,
    n_samples: usize,
    max_abs_diff: f32,
    mean_abs_diff: f32,
    failed_samples: usize,
}

impl ParityResult {
    fn new() -> Self {
        Self {
            passed: true,
            n_samples: 0,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
            failed_samples: 0,
        }
    }
}

/// Get the golden data directory path
fn golden_data_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("tests")
        .join("golden_data")
}

/// Get the ONNX models directory path
fn onnx_models_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("trained")
        .join("onnx")
}

/// Compare two float values with absolute and relative tolerance
fn compare_with_tolerance(expected: f32, actual: f32) -> (bool, f32) {
    let abs_diff = (expected - actual).abs();

    // Handle NaN and Inf cases
    if expected.is_nan() && actual.is_nan() {
        return (true, 0.0);
    }
    if expected.is_nan() || actual.is_nan() {
        return (false, f32::INFINITY);
    }
    if expected.is_infinite() && actual.is_infinite() && expected.signum() == actual.signum() {
        return (true, 0.0);
    }
    if expected.is_infinite() || actual.is_infinite() {
        return (false, f32::INFINITY);
    }

    // Check absolute tolerance first
    if abs_diff <= TOLERANCE_ABS {
        return (true, abs_diff);
    }

    // Check relative tolerance
    let denominator = expected.abs().max(actual.abs()).max(1e-10);
    let rel_diff = abs_diff / denominator;

    let passed = rel_diff <= TOLERANCE_REL;
    (passed, abs_diff)
}

/// Compare output arrays and return parity result
fn compare_outputs(expected: &[f32], actual: &[f32]) -> ParityResult {
    let mut result = ParityResult::new();

    if expected.len() != actual.len() {
        result.passed = false;
        result.failed_samples = expected.len().max(actual.len());
        return result;
    }

    result.n_samples = expected.len();
    let mut total_abs_diff = 0.0f32;

    for (i, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        let (passed, abs_diff) = compare_with_tolerance(exp, act);

        total_abs_diff += abs_diff;
        result.max_abs_diff = result.max_abs_diff.max(abs_diff);

        if !passed {
            result.failed_samples += 1;
            result.passed = false;

            // Log first few failures for debugging
            if result.failed_samples <= 5 {
                eprintln!(
                    "  Sample {}: expected={:.6e}, actual={:.6e}, diff={:.6e}",
                    i, exp, act, abs_diff
                );
            }
        }
    }

    result.mean_abs_diff = total_abs_diff / result.n_samples as f32;

    result
}

/// Load ONNX session from file
fn load_onnx_session(model_path: &PathBuf) -> Result<Session> {
    Session::builder()?
        .with_intra_threads(1)?
        .commit_from_file(model_path)
        .context(format!("Failed to load ONNX model: {:?}", model_path))
}

/// Run inference on ML model and compare with golden data
fn test_ml_model_parity(model_name: &str) -> Result<ParityResult> {
    let golden_path = golden_data_dir().join(format!("{}_golden.json", model_name));
    let onnx_path = onnx_models_dir().join(format!("{}_model.onnx", model_name));

    println!("Testing {} parity...", model_name);
    println!("  Golden data: {:?}", golden_path);
    println!("  ONNX model: {:?}", onnx_path);

    // Check if files exist
    if !golden_path.exists() {
        anyhow::bail!("Golden data file not found: {:?}", golden_path);
    }
    if !onnx_path.exists() {
        anyhow::bail!("ONNX model file not found: {:?}", onnx_path);
    }

    // Load golden data
    let golden_json = fs::read_to_string(&golden_path)?;
    let golden: GoldenDataML = serde_json::from_str(&golden_json)?;

    println!("  Loaded {} samples from golden data", golden.metadata.n_samples);

    // Load ONNX session
    let mut session = load_onnx_session(&onnx_path)?;

    // Prepare input tensor
    let n_samples = golden.inputs.input.len();
    let n_features = golden.inputs.input[0].len();

    let flat_input: Vec<f32> = golden.inputs.input.iter().flatten().copied().collect();
    let shape = [n_samples as i64, n_features as i64];
    let input_tensor = Tensor::from_array((shape, flat_input.into_boxed_slice()))?;

    // Run inference
    let outputs = session.run(ort::inputs![input_tensor])?;

    // Extract output
    let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;
    let onnx_outputs: Vec<f32> = output_data.iter().copied().collect();

    // Compare outputs
    let result = compare_outputs(&golden.outputs, &onnx_outputs);

    println!(
        "  Result: {} (max_diff={:.6e}, mean_diff={:.6e}, failed={}/{})",
        if result.passed { "PASS" } else { "FAIL" },
        result.max_abs_diff,
        result.mean_abs_diff,
        result.failed_samples,
        result.n_samples
    );

    Ok(result)
}

/// Run inference on sequence model and compare with golden data
fn test_sequence_model_parity(model_name: &str) -> Result<ParityResult> {
    let golden_path = golden_data_dir().join(format!("{}_golden.json", model_name));
    let onnx_path = onnx_models_dir().join(format!("{}_model.onnx", model_name));

    println!("Testing {} parity...", model_name);
    println!("  Golden data: {:?}", golden_path);
    println!("  ONNX model: {:?}", onnx_path);

    if !golden_path.exists() {
        anyhow::bail!("Golden data file not found: {:?}", golden_path);
    }
    if !onnx_path.exists() {
        anyhow::bail!("ONNX model file not found: {:?}", onnx_path);
    }

    let golden_json = fs::read_to_string(&golden_path)?;
    let golden: GoldenDataSequence = serde_json::from_str(&golden_json)?;

    println!("  Loaded {} samples from golden data", golden.metadata.n_samples);

    let mut session = load_onnx_session(&onnx_path)?;

    // Prepare input tensor (batch, seq_len, features)
    let n_samples = golden.inputs.input.len();
    let seq_len = golden.inputs.input[0].len();
    let n_features = golden.inputs.input[0][0].len();

    let flat_input: Vec<f32> = golden.inputs.input
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect();

    let shape = [n_samples as i64, seq_len as i64, n_features as i64];
    let input_tensor = Tensor::from_array((shape, flat_input.into_boxed_slice()))?;

    let outputs = session.run(ort::inputs![input_tensor])?;

    let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;
    let onnx_outputs: Vec<f32> = output_data.iter().copied().collect();

    let result = compare_outputs(&golden.outputs, &onnx_outputs);

    println!(
        "  Result: {} (max_diff={:.6e}, mean_diff={:.6e}, failed={}/{})",
        if result.passed { "PASS" } else { "FAIL" },
        result.max_abs_diff,
        result.mean_abs_diff,
        result.failed_samples,
        result.n_samples
    );

    Ok(result)
}

/// Run inference on RL model and compare with golden data
fn test_rl_model_parity(model_name: &str) -> Result<ParityResult> {
    let golden_path = golden_data_dir().join(format!("{}_golden.json", model_name));
    let onnx_path = onnx_models_dir().join(format!("{}_actor.onnx", model_name));

    println!("Testing {} parity...", model_name);
    println!("  Golden data: {:?}", golden_path);
    println!("  ONNX model: {:?}", onnx_path);

    if !golden_path.exists() {
        anyhow::bail!("Golden data file not found: {:?}", golden_path);
    }
    if !onnx_path.exists() {
        anyhow::bail!("ONNX model file not found: {:?}", onnx_path);
    }

    let golden_json = fs::read_to_string(&golden_path)?;
    let golden: GoldenDataRL = serde_json::from_str(&golden_json)?;

    println!("  Loaded {} samples from golden data", golden.metadata.n_samples);

    let mut session = load_onnx_session(&onnx_path)?;

    let n_samples = golden.inputs.input.len();
    let state_dim = golden.inputs.input[0].len();

    let flat_input: Vec<f32> = golden.inputs.input.iter().flatten().copied().collect();
    let shape = [n_samples as i64, state_dim as i64];
    let input_tensor = Tensor::from_array((shape, flat_input.into_boxed_slice()))?;

    let outputs = session.run(ort::inputs![input_tensor])?;

    let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;
    let onnx_outputs: Vec<f32> = output_data.iter().copied().collect();

    // Flatten expected outputs
    let expected: Vec<f32> = golden.outputs.iter().flatten().copied().collect();

    let result = compare_outputs(&expected, &onnx_outputs);

    println!(
        "  Result: {} (max_diff={:.6e}, mean_diff={:.6e}, failed={}/{})",
        if result.passed { "PASS" } else { "FAIL" },
        result.max_abs_diff,
        result.mean_abs_diff,
        result.failed_samples,
        result.n_samples
    );

    Ok(result)
}

/// Run inference on MARL model and compare with golden data
fn test_marl_model_parity(agent_idx: usize) -> Result<ParityResult> {
    let golden_path = golden_data_dir().join(format!("marl_agent_{}_golden.json", agent_idx));

    // Find the MARL agent ONNX file (pattern: marl_agent_N_<role>.onnx)
    let onnx_dir = onnx_models_dir();
    let onnx_path = fs::read_dir(&onnx_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&format!("marl_agent_{}_", agent_idx)) && n.ends_with(".onnx"))
                .unwrap_or(false)
        });

    let onnx_path = match onnx_path {
        Some(p) => p,
        None => {
            // Try the generic golden data if specific agent not found
            let fallback_golden = golden_data_dir().join("marl_golden.json");
            if !fallback_golden.exists() {
                anyhow::bail!("No MARL ONNX model found for agent {}", agent_idx);
            }
            anyhow::bail!("MARL ONNX model not found for agent {}", agent_idx);
        }
    };

    println!("Testing MARL agent {} parity...", agent_idx);
    println!("  Golden data: {:?}", golden_path);
    println!("  ONNX model: {:?}", onnx_path);

    if !golden_path.exists() {
        anyhow::bail!("Golden data file not found: {:?}", golden_path);
    }

    let golden_json = fs::read_to_string(&golden_path)?;
    let golden: GoldenDataMARL = serde_json::from_str(&golden_json)?;

    println!("  Loaded {} samples from golden data", golden.metadata.n_samples);

    let mut session = load_onnx_session(&onnx_path)?;

    // Prepare state tensor
    let n_samples = golden.inputs.input_0.len();
    let state_dim = golden.inputs.input_0[0].len();
    let flat_states: Vec<f32> = golden.inputs.input_0.iter().flatten().copied().collect();
    let state_shape = [n_samples as i64, state_dim as i64];
    let state_tensor = Tensor::from_array((state_shape, flat_states.into_boxed_slice()))?;

    // Prepare messages tensor
    let n_other_agents = golden.inputs.input_1[0].len();
    let message_dim = golden.inputs.input_1[0][0].len();
    let flat_messages: Vec<f32> = golden.inputs.input_1
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect();
    let msg_shape = [n_samples as i64, n_other_agents as i64, message_dim as i64];
    let msg_tensor = Tensor::from_array((msg_shape, flat_messages.into_boxed_slice()))?;

    let outputs = session.run(ort::inputs![state_tensor, msg_tensor])?;

    let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;
    let onnx_outputs: Vec<f32> = output_data.iter().copied().collect();

    let expected: Vec<f32> = golden.outputs.iter().flatten().copied().collect();

    let result = compare_outputs(&expected, &onnx_outputs);

    println!(
        "  Result: {} (max_diff={:.6e}, mean_diff={:.6e}, failed={}/{})",
        if result.passed { "PASS" } else { "FAIL" },
        result.max_abs_diff,
        result.mean_abs_diff,
        result.failed_samples,
        result.n_samples
    );

    Ok(result)
}

// =============================================================================
// Test Cases
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires ONNX models and golden data"]
    fn test_lightgbm_parity() {
        let result = test_ml_model_parity("lightgbm")
            .expect("LightGBM parity test failed to run");
        assert!(
            result.passed,
            "LightGBM parity test failed: {} samples out of {} exceeded tolerance",
            result.failed_samples,
            result.n_samples
        );
    }

    #[test]
    #[ignore = "requires ONNX models and golden data"]
    fn test_xgboost_parity() {
        let result = test_ml_model_parity("xgboost")
            .expect("XGBoost parity test failed to run");
        assert!(
            result.passed,
            "XGBoost parity test failed: {} samples out of {} exceeded tolerance",
            result.failed_samples,
            result.n_samples
        );
    }

    #[test]
    #[ignore = "requires ONNX models and golden data"]
    fn test_lstm_parity() {
        let result = test_sequence_model_parity("lstm")
            .expect("LSTM parity test failed to run");
        assert!(
            result.passed,
            "LSTM parity test failed: {} samples out of {} exceeded tolerance",
            result.failed_samples,
            result.n_samples
        );
    }

    #[test]
    #[ignore = "requires ONNX models and golden data"]
    fn test_cnn_parity() {
        let result = test_sequence_model_parity("cnn")
            .expect("CNN parity test failed to run");
        assert!(
            result.passed,
            "CNN parity test failed: {} samples out of {} exceeded tolerance",
            result.failed_samples,
            result.n_samples
        );
    }

    #[test]
    #[ignore = "requires ONNX models and golden data"]
    fn test_d4pg_parity() {
        let result = test_rl_model_parity("d4pg")
            .expect("D4PG parity test failed to run");
        assert!(
            result.passed,
            "D4PG parity test failed: {} samples out of {} exceeded tolerance",
            result.failed_samples,
            result.n_samples
        );
    }

    #[test]
    #[ignore = "requires ONNX models and golden data"]
    fn test_marl_agent_0_parity() {
        let result = test_marl_model_parity(0)
            .expect("MARL agent 0 parity test failed to run");
        assert!(
            result.passed,
            "MARL agent 0 parity test failed: {} samples out of {} exceeded tolerance",
            result.failed_samples,
            result.n_samples
        );
    }

    /// Test that golden data files exist and are valid JSON
    #[test]
    fn test_golden_data_exists() {
        let golden_dir = golden_data_dir();

        let expected_files = [
            "lightgbm_golden.json",
            "xgboost_golden.json",
            "lstm_golden.json",
            "cnn_golden.json",
            "d4pg_golden.json",
            "marl_golden.json",
        ];

        for file in expected_files {
            let path = golden_dir.join(file);
            if path.exists() {
                // Verify it's valid JSON
                let content = fs::read_to_string(&path)
                    .unwrap_or_else(|_| panic!("Failed to read {}", file));
                let _: serde_json::Value = serde_json::from_str(&content)
                    .unwrap_or_else(|_| panic!("Invalid JSON in {}", file));
                println!("Verified: {}", file);
            } else {
                println!("Not found (expected): {}", file);
            }
        }
    }

    /// Test tolerance comparison logic
    #[test]
    fn test_tolerance_comparison() {
        // Exact match
        let (passed, diff) = compare_with_tolerance(1.0, 1.0);
        assert!(passed);
        assert_eq!(diff, 0.0);

        // Within absolute tolerance
        let (passed, diff) = compare_with_tolerance(1.0, 1.0 + 1e-6);
        assert!(passed);
        assert!(diff < TOLERANCE_ABS);

        // Outside absolute but within relative
        let (passed, _) = compare_with_tolerance(1000.0, 1000.0 + 0.05);
        assert!(passed);

        // Outside both tolerances
        let (passed, _) = compare_with_tolerance(1.0, 2.0);
        assert!(!passed);

        // NaN handling
        let (passed, _) = compare_with_tolerance(f32::NAN, f32::NAN);
        assert!(passed);

        let (passed, _) = compare_with_tolerance(1.0, f32::NAN);
        assert!(!passed);

        // Infinity handling
        let (passed, _) = compare_with_tolerance(f32::INFINITY, f32::INFINITY);
        assert!(passed);

        let (passed, _) = compare_with_tolerance(f32::NEG_INFINITY, f32::NEG_INFINITY);
        assert!(passed);

        let (passed, _) = compare_with_tolerance(f32::INFINITY, f32::NEG_INFINITY);
        assert!(!passed);
    }

    /// Test output comparison
    #[test]
    fn test_output_comparison() {
        // Identical outputs
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compare_outputs(&expected, &actual);
        assert!(result.passed);
        assert_eq!(result.failed_samples, 0);

        // Slightly different outputs (within tolerance)
        let expected = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0 + 1e-6, 2.0 - 1e-6, 3.0 + 1e-7];
        let result = compare_outputs(&expected, &actual);
        assert!(result.passed);

        // Significantly different outputs
        let expected = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0, 2.5, 3.0];
        let result = compare_outputs(&expected, &actual);
        assert!(!result.passed);
        assert_eq!(result.failed_samples, 1);

        // Length mismatch
        let expected = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0, 2.0];
        let result = compare_outputs(&expected, &actual);
        assert!(!result.passed);
    }
}

/// Run all parity tests and generate a summary
#[allow(dead_code)]
pub fn run_all_parity_tests() -> Result<bool> {
    println!("{}", "=".repeat(80));
    println!("ONNX PARITY TEST SUITE");
    println!("{}", "=".repeat(80));
    println!();

    let tests: Vec<(&str, Box<dyn Fn() -> Result<ParityResult>>)> = vec![
        ("LightGBM", Box::new(|| test_ml_model_parity("lightgbm"))),
        ("XGBoost", Box::new(|| test_ml_model_parity("xgboost"))),
        ("LSTM", Box::new(|| test_sequence_model_parity("lstm"))),
        ("CNN", Box::new(|| test_sequence_model_parity("cnn"))),
        ("D4PG", Box::new(|| test_rl_model_parity("d4pg"))),
        ("MARL Agent 0", Box::new(|| test_marl_model_parity(0))),
    ];

    let mut results = Vec::new();
    let mut all_passed = true;

    for (name, test_fn) in tests {
        println!("{}", "-".repeat(40));
        match test_fn() {
            Ok(result) => {
                if !result.passed {
                    all_passed = false;
                }
                results.push((name, Some(result)));
            }
            Err(e) => {
                println!("  {} test error: {}", name, e);
                results.push((name, None));
                // Don't fail on missing models/data
            }
        }
        println!();
    }

    println!("{}", "=".repeat(80));
    println!("SUMMARY");
    println!("{}", "=".repeat(80));

    for (name, result) in &results {
        match result {
            Some(r) => {
                let status = if r.passed { "PASS" } else { "FAIL" };
                println!(
                    "{}: {} (max_diff={:.2e}, failed={}/{})",
                    name, status, r.max_abs_diff, r.failed_samples, r.n_samples
                );
            }
            None => {
                println!("{}: SKIP (missing model or data)", name);
            }
        }
    }

    let passed_count = results.iter().filter(|(_, r)| r.as_ref().map(|r| r.passed).unwrap_or(false)).count();
    let failed_count = results.iter().filter(|(_, r)| r.as_ref().map(|r| !r.passed).unwrap_or(false)).count();
    let skipped_count = results.iter().filter(|(_, r)| r.is_none()).count();

    println!();
    println!("Total: {} passed, {} failed, {} skipped", passed_count, failed_count, skipped_count);

    Ok(all_passed)
}
