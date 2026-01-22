//! ONNX Runtime Integration for ML Model Inference
//!
//! Provides high-performance inference for trained ML/DL/RL models
//! exported from Python training pipeline.
//!
//! NSMI Integration:
//! - Dynamic model weight adjustment based on regime detection
//! - Feature augmentation with NSMI-derived features
//! - Zero-allocation hot path via pre-allocated buffers

// Allow dead_code - module has comprehensive API, not all functions used yet
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use ort::{session::{builder::GraphOptimizationLevel, Session}, value::Tensor};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::nsmi::{NSMIConfig, NSMIFeatures, NSMIResult, NSMIState};

/// Model type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum ModelType {
    LightGBM,
    XGBoost,
    Lstm,
    Cnn,
    D4pg,
    Marl,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::LightGBM => write!(f, "lightgbm"),
            ModelType::XGBoost => write!(f, "xgboost"),
            ModelType::Lstm => write!(f, "lstm"),
            ModelType::Cnn => write!(f, "cnn"),
            ModelType::D4pg => write!(f, "d4pg"),
            ModelType::Marl => write!(f, "marl"),
        }
    }
}

/// Model metadata loaded from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_name: String,
    pub model_type: String,
    pub onnx_path: String,
    #[serde(default)]
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub num_features: usize,
    #[serde(default)]
    pub input_shape: Vec<usize>,
    #[serde(default)]
    pub state_dim: usize,
    #[serde(default)]
    pub action_dim: usize,
}

/// ONNX model wrapper
pub struct OnnxModel {
    session: Session,
    metadata: ModelMetadata,
    model_type: ModelType,
}

impl OnnxModel {
    /// Load model from ONNX file
    pub fn load(model_path: &Path, metadata: ModelMetadata, model_type: ModelType) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        info!("Loaded {} model from {:?}", model_type, model_path);

        Ok(Self {
            session,
            metadata,
            model_type,
        })
    }

    /// Run inference on ML model (flat features)
    pub fn predict_ml(&mut self, features: &[f32]) -> Result<f32> {
        let shape = [1i64, features.len() as i64];
        let input_tensor = Tensor::from_array((shape, features.to_vec().into_boxed_slice()))?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;

        let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(output_data.first().copied().unwrap_or(0.0))
    }

    /// Run inference on DL model (sequence features)
    pub fn predict_sequence(&mut self, sequence: &[Vec<f32>]) -> Result<f32> {
        let seq_len = sequence.len();
        let num_features = sequence[0].len();

        let flat: Vec<f32> = sequence.iter().flatten().copied().collect();
        let shape = [1i64, seq_len as i64, num_features as i64];
        let input_tensor = Tensor::from_array((shape, flat.into_boxed_slice()))?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;

        let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(output_data.first().copied().unwrap_or(0.0))
    }

    /// Run inference on RL actor (state to action)
    pub fn predict_action(&mut self, state: &[f32]) -> Result<Vec<f32>> {
        let shape = [1i64, state.len() as i64];
        let input_tensor = Tensor::from_array((shape, state.to_vec().into_boxed_slice()))?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;

        let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(output_data.to_vec())
    }

    /// Run MARL agent inference with messages
    pub fn predict_marl_action(&mut self, state: &[f32], messages: &[Vec<f32>]) -> Result<Vec<f32>> {
        let state_shape = [1i64, state.len() as i64];
        let state_tensor = Tensor::from_array((state_shape, state.to_vec().into_boxed_slice()))?;

        let n_messages = messages.len();
        let msg_dim = messages[0].len();
        let flat_messages: Vec<f32> = messages.iter().flatten().copied().collect();
        let messages_shape = [1i64, n_messages as i64, msg_dim as i64];
        let messages_tensor = Tensor::from_array((messages_shape, flat_messages.into_boxed_slice()))?;

        let outputs = self.session.run(ort::inputs![state_tensor, messages_tensor])?;

        let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(output_data.to_vec())
    }

    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

/// Pre-allocated buffer for NSMI feature augmentation (zero-allocation hot path)
pub struct NSMIAugmentBuffer {
    /// Buffer for augmented features (raw + NSMI)
    augmented: Vec<f32>,
    /// Raw feature dimension
    raw_dim: usize,
    /// NSMI feature dimension
    nsmi_dim: usize,
}

impl NSMIAugmentBuffer {
    /// Create new buffer with specified dimensions
    pub fn new(raw_dim: usize) -> Self {
        let nsmi_dim = NSMIFeatures::NUM_FEATURES;
        Self {
            augmented: vec![0.0; raw_dim + nsmi_dim],
            raw_dim,
            nsmi_dim,
        }
    }

    /// Augment raw features with NSMI features (zero-allocation)
    #[inline(always)]
    pub fn augment(&mut self, raw: &[f32], nsmi: &NSMIFeatures) -> &[f32] {
        // Copy raw features
        let copy_len = raw.len().min(self.raw_dim);
        self.augmented[..copy_len].copy_from_slice(&raw[..copy_len]);

        // Fill remainder with zeros if raw is shorter
        if copy_len < self.raw_dim {
            self.augmented[copy_len..self.raw_dim].fill(0.0);
        }

        // Append NSMI features
        let offset = self.raw_dim;
        self.augmented[offset] = nsmi.nonstationarity as f32;
        self.augmented[offset + 1] = nsmi.regime_prob as f32;
        self.augmented[offset + 2] = nsmi.spectral_gap_norm as f32;
        self.augmented[offset + 3] = nsmi.drift_norm as f32;
        self.augmented[offset + 4] = nsmi.dim_ratio as f32;
        self.augmented[offset + 5] = nsmi.regime_stability as f32;

        &self.augmented[..self.raw_dim + self.nsmi_dim]
    }

    /// Get total augmented dimension
    #[inline]
    pub fn total_dim(&self) -> usize {
        self.raw_dim + self.nsmi_dim
    }
}

/// Model weights adjusted by NSMI regime detection
#[derive(Debug, Clone, Default)]
pub struct NSMIAdjustedWeights {
    /// Base weights (from performance tracking)
    pub base_weights: HashMap<ModelType, f32>,
    /// NSMI-adjusted weights (for current regime)
    pub adjusted_weights: HashMap<ModelType, f32>,
    /// Regime weight multiplier applied
    pub regime_weight: f32,
    /// Current regime identifier
    pub current_regime: u64,
}

/// Model ensemble with dynamic weighting and NSMI integration
pub struct ModelEnsemble {
    models: RwLock<HashMap<ModelType, OnnxModel>>,
    weights: RwLock<HashMap<ModelType, f32>>,
    predictions_history: RwLock<Vec<(ModelType, f32, f32)>>,
    /// NSMI state for regime detection (optional)
    nsmi_state: Option<RwLock<NSMIState>>,
    /// Pre-allocated augmentation buffer
    augment_buffer: RwLock<NSMIAugmentBuffer>,
    /// NSMI weight influence (0.0 = no NSMI, 1.0 = full NSMI)
    nsmi_weight_factor: f32,
    /// Cache of last NSMI result
    last_nsmi_result: RwLock<Option<NSMIResult>>,
}

impl ModelEnsemble {
    /// Create new ensemble from ONNX model directory
    pub fn new(model_dir: &Path) -> Result<Self> {
        let mut models = HashMap::new();
        let mut weights = HashMap::new();

        let manifest_path = model_dir.join("manifest.json");
        if manifest_path.exists() {
            let manifest: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;

            if let Some(model_names) = manifest["models"].as_array() {
                for name in model_names {
                    if let Some(name_str) = name.as_str() {
                        let model_type = match name_str {
                            "lightgbm" | "lightgbm_model" => Some(ModelType::LightGBM),
                            "xgboost" | "xgboost_model" => Some(ModelType::XGBoost),
                            "lstm" | "lstm_model" => Some(ModelType::Lstm),
                            "cnn" | "cnn_model" => Some(ModelType::Cnn),
                            "d4pg" | "d4pg_actor" => Some(ModelType::D4pg),
                            n if n.starts_with("marl") => Some(ModelType::Marl),
                            _ => None,
                        };

                        if let Some(mt) = model_type {
                            let metadata_path = model_dir.join(format!("{}_metadata.json", name_str));
                            let onnx_path = model_dir.join(format!("{}.onnx", name_str));

                            if onnx_path.exists() {
                                let metadata = if metadata_path.exists() {
                                    serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?
                                } else {
                                    ModelMetadata {
                                        model_name: name_str.to_string(),
                                        model_type: name_str.to_string(),
                                        onnx_path: onnx_path.to_string_lossy().to_string(),
                                        feature_names: vec![],
                                        num_features: 0,
                                        input_shape: vec![],
                                        state_dim: 0,
                                        action_dim: 1,
                                    }
                                };

                                match OnnxModel::load(&onnx_path, metadata, mt) {
                                    Ok(model) => {
                                        weights.insert(mt, 1.0 / ((models.len() + 1) as f32));
                                        models.insert(mt, model);
                                    }
                                    Err(e) => {
                                        warn!("Failed to load {}: {}", name_str, e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize weights
        let total: f32 = weights.values().sum();
        if total > 0.0 {
            for w in weights.values_mut() {
                *w /= total;
            }
        }

        info!("Loaded {} models into ensemble", models.len());

        // Default feature dimension (can be updated via with_nsmi)
        let default_feature_dim = 64;

        Ok(Self {
            models: RwLock::new(models),
            weights: RwLock::new(weights),
            predictions_history: RwLock::new(Vec::new()),
            nsmi_state: None,
            augment_buffer: RwLock::new(NSMIAugmentBuffer::new(default_feature_dim)),
            nsmi_weight_factor: 0.0,
            last_nsmi_result: RwLock::new(None),
        })
    }

    /// Create ensemble with NSMI integration enabled
    pub fn with_nsmi(model_dir: &Path, nsmi_config: NSMIConfig, nsmi_weight: f32, feature_dim: usize) -> Result<Self> {
        let mut ensemble = Self::new(model_dir)?;
        ensemble.nsmi_state = Some(RwLock::new(NSMIState::new(nsmi_config)));
        ensemble.nsmi_weight_factor = nsmi_weight.clamp(0.0, 1.0);
        ensemble.augment_buffer = RwLock::new(NSMIAugmentBuffer::new(feature_dim));
        Ok(ensemble)
    }

    /// Enable NSMI on existing ensemble
    pub fn enable_nsmi(&mut self, nsmi_config: NSMIConfig, nsmi_weight: f32, feature_dim: usize) {
        self.nsmi_state = Some(RwLock::new(NSMIState::new(nsmi_config)));
        self.nsmi_weight_factor = nsmi_weight.clamp(0.0, 1.0);
        self.augment_buffer = RwLock::new(NSMIAugmentBuffer::new(feature_dim));
    }

    /// Update NSMI state with new observation and return result
    ///
    /// This should be called with raw feature vector on each tick
    /// before running inference.
    #[inline]
    pub fn update_nsmi(&self, observation: &[f64]) -> Option<NSMIResult> {
        if let Some(ref nsmi_lock) = self.nsmi_state {
            let mut nsmi = nsmi_lock.write();
            let result = nsmi.update(observation);
            *self.last_nsmi_result.write() = Some(result.clone());
            Some(result)
        } else {
            None
        }
    }

    /// Get current NSMI features for augmentation
    #[inline]
    pub fn get_nsmi_features(&self) -> Option<NSMIFeatures> {
        self.nsmi_state.as_ref().map(|nsmi_lock| {
            nsmi_lock.read().get_features()
        })
    }

    /// Get last NSMI result (cached from last update)
    #[inline]
    pub fn get_last_nsmi_result(&self) -> Option<NSMIResult> {
        self.last_nsmi_result.read().clone()
    }

    /// Get NSMI-adjusted weights for current regime
    pub fn get_nsmi_adjusted_weights(&self) -> NSMIAdjustedWeights {
        let base_weights = self.weights.read().clone();
        let mut adjusted = NSMIAdjustedWeights {
            base_weights: base_weights.clone(),
            adjusted_weights: HashMap::new(),
            regime_weight: 1.0,
            current_regime: 0,
        };

        if let Some(ref nsmi_lock) = self.nsmi_state {
            let nsmi = nsmi_lock.read();
            adjusted.regime_weight = nsmi.get_regime_weight() as f32;
            adjusted.current_regime = nsmi.sample_count();

            // Adjust weights based on regime stability
            // In stable regimes, trust ML models more; in unstable, be more conservative
            let regime_factor = adjusted.regime_weight;
            let base_factor = 1.0 - self.nsmi_weight_factor * (1.0 - regime_factor);

            for (model_type, base_weight) in &base_weights {
                // Different adjustment per model type based on regime
                let adjustment = match model_type {
                    // Tree models are more robust to regime changes
                    ModelType::LightGBM | ModelType::XGBoost => {
                        base_weight * (base_factor * 0.9 + 0.1)
                    }
                    // DL models may overfit to recent regime
                    ModelType::Lstm | ModelType::Cnn => {
                        base_weight * base_factor
                    }
                    // RL models need stable regimes
                    ModelType::D4pg | ModelType::Marl => {
                        base_weight * (base_factor * 0.8 + 0.2 * regime_factor)
                    }
                };
                adjusted.adjusted_weights.insert(*model_type, adjustment);
            }

            // Normalize adjusted weights
            let total: f32 = adjusted.adjusted_weights.values().sum();
            if total > 0.0 {
                for w in adjusted.adjusted_weights.values_mut() {
                    *w /= total;
                }
            }
        } else {
            adjusted.adjusted_weights = base_weights;
        }

        adjusted
    }

    /// Run ensemble prediction on ML features
    pub fn predict(&self, features: &[f32]) -> Result<f32> {
        let mut models = self.models.write();
        let weights = self.weights.read();

        let mut weighted_sum = 0.0f32;
        let mut total_weight = 0.0f32;

        for model_type in [ModelType::LightGBM, ModelType::XGBoost] {
            if let Some(&weight) = weights.get(&model_type) {
                if let Some(model) = models.get_mut(&model_type) {
                    match model.predict_ml(features) {
                        Ok(pred) => {
                            weighted_sum += pred * weight;
                            total_weight += weight;
                            debug!("{}: pred={:.6}, weight={:.3}", model_type, pred, weight);
                        }
                        Err(e) => {
                            warn!("{} prediction failed: {}", model_type, e);
                        }
                    }
                }
            }
        }

        if total_weight > 0.0 {
            Ok(weighted_sum / total_weight)
        } else {
            Err(anyhow::anyhow!("No ML models available"))
        }
    }

    /// Run ensemble prediction on sequence features
    pub fn predict_sequence(&self, sequence: &[Vec<f32>]) -> Result<f32> {
        let mut models = self.models.write();
        let weights = self.weights.read();

        let mut weighted_sum = 0.0f32;
        let mut total_weight = 0.0f32;

        for model_type in [ModelType::Lstm, ModelType::Cnn] {
            if let Some(&weight) = weights.get(&model_type) {
                if let Some(model) = models.get_mut(&model_type) {
                    match model.predict_sequence(sequence) {
                        Ok(pred) => {
                            weighted_sum += pred * weight;
                            total_weight += weight;
                            debug!("{}: pred={:.6}, weight={:.3}", model_type, pred, weight);
                        }
                        Err(e) => {
                            warn!("{} prediction failed: {}", model_type, e);
                        }
                    }
                }
            }
        }

        if total_weight > 0.0 {
            Ok(weighted_sum / total_weight)
        } else {
            Err(anyhow::anyhow!("No DL models available"))
        }
    }

    /// Get RL action from D4PG actor
    pub fn get_rl_action(&self, state: &[f32]) -> Result<Vec<f32>> {
        let mut models = self.models.write();

        if let Some(model) = models.get_mut(&ModelType::D4pg) {
            model.predict_action(state)
        } else {
            Err(anyhow::anyhow!("D4PG model not available"))
        }
    }

    /// Run NSMI-augmented ensemble prediction
    ///
    /// This method:
    /// 1. Augments features with NSMI-derived features
    /// 2. Uses NSMI-adjusted model weights based on current regime
    /// 3. Returns prediction with NSMI metadata
    ///
    /// # Arguments
    /// * `features` - Raw feature vector
    ///
    /// # Returns
    /// * Tuple of (prediction, nsmi_adjustment, model_weights)
    #[inline]
    pub fn predict_with_nsmi(&self, features: &[f32]) -> Result<(f32, f32, HashMap<ModelType, f32>)> {
        // Get NSMI features and adjusted weights
        let nsmi_features = self.get_nsmi_features();
        let adjusted = self.get_nsmi_adjusted_weights();

        // Augment features if NSMI is enabled
        let effective_features = if let Some(ref nsmi_feat) = nsmi_features {
            let mut buffer = self.augment_buffer.write();
            buffer.augment(features, nsmi_feat).to_vec()
        } else {
            features.to_vec()
        };

        // Run prediction with adjusted weights
        let mut models = self.models.write();
        let mut weighted_sum = 0.0f32;
        let mut total_weight = 0.0f32;

        for model_type in [ModelType::LightGBM, ModelType::XGBoost] {
            if let Some(&weight) = adjusted.adjusted_weights.get(&model_type) {
                if let Some(model) = models.get_mut(&model_type) {
                    match model.predict_ml(&effective_features) {
                        Ok(pred) => {
                            weighted_sum += pred * weight;
                            total_weight += weight;
                            debug!("{}: pred={:.6}, nsmi_weight={:.3}", model_type, pred, weight);
                        }
                        Err(e) => {
                            warn!("{} prediction failed: {}", model_type, e);
                        }
                    }
                }
            }
        }

        if total_weight > 0.0 {
            let prediction = weighted_sum / total_weight;
            // Calculate NSMI adjustment (difference from base prediction)
            let nsmi_adjustment = 1.0 - adjusted.regime_weight;
            Ok((prediction, nsmi_adjustment, adjusted.adjusted_weights))
        } else {
            Err(anyhow::anyhow!("No ML models available"))
        }
    }

    /// Run NSMI-augmented sequence prediction for DL models
    #[inline]
    pub fn predict_sequence_with_nsmi(&self, sequence: &[Vec<f32>]) -> Result<(f32, f32, HashMap<ModelType, f32>)> {
        let nsmi_features = self.get_nsmi_features();
        let adjusted = self.get_nsmi_adjusted_weights();

        // Augment each timestep with NSMI features
        let effective_sequence = if let Some(ref nsmi_feat) = nsmi_features {
            sequence.iter().map(|step| {
                let mut buffer = self.augment_buffer.write();
                buffer.augment(step, nsmi_feat).to_vec()
            }).collect::<Vec<_>>()
        } else {
            sequence.to_vec()
        };

        let mut models = self.models.write();
        let mut weighted_sum = 0.0f32;
        let mut total_weight = 0.0f32;

        for model_type in [ModelType::Lstm, ModelType::Cnn] {
            if let Some(&weight) = adjusted.adjusted_weights.get(&model_type) {
                if let Some(model) = models.get_mut(&model_type) {
                    match model.predict_sequence(&effective_sequence) {
                        Ok(pred) => {
                            weighted_sum += pred * weight;
                            total_weight += weight;
                            debug!("{}: pred={:.6}, nsmi_weight={:.3}", model_type, pred, weight);
                        }
                        Err(e) => {
                            warn!("{} prediction failed: {}", model_type, e);
                        }
                    }
                }
            }
        }

        if total_weight > 0.0 {
            let prediction = weighted_sum / total_weight;
            let nsmi_adjustment = 1.0 - adjusted.regime_weight;
            Ok((prediction, nsmi_adjustment, adjusted.adjusted_weights))
        } else {
            Err(anyhow::anyhow!("No DL models available"))
        }
    }

    /// Check if NSMI is enabled
    #[inline]
    pub fn nsmi_enabled(&self) -> bool {
        self.nsmi_state.is_some()
    }

    /// Get NSMI weight factor
    #[inline]
    pub fn nsmi_weight_factor(&self) -> f32 {
        self.nsmi_weight_factor
    }

    /// Reset NSMI state
    pub fn reset_nsmi(&self) {
        if let Some(ref nsmi_lock) = self.nsmi_state {
            nsmi_lock.write().reset();
        }
        *self.last_nsmi_result.write() = None;
    }

    /// Update weights based on prediction performance
    pub fn update_weights(&self, actual: f32, predictions: &[(ModelType, f32)]) {
        let mut weights = self.weights.write();
        let mut history = self.predictions_history.write();

        for (model_type, pred) in predictions {
            history.push((*model_type, *pred, actual));
        }

        const LOOKBACK: usize = 100;
        if history.len() > LOOKBACK * predictions.len() {
            history.drain(0..predictions.len());
        }

        let mut scores: HashMap<ModelType, f32> = HashMap::new();

        for model_type in weights.keys() {
            let model_history: Vec<_> = history
                .iter()
                .filter(|(mt, _, _)| mt == model_type)
                .collect();

            if model_history.len() >= 10 {
                let correct: f32 = model_history
                    .iter()
                    .map(|(_, pred, actual)| if pred.signum() == actual.signum() { 1.0 } else { 0.0 })
                    .sum();

                let accuracy = correct / model_history.len() as f32;

                let mse: f32 = model_history
                    .iter()
                    .map(|(_, pred, actual)| (pred - actual).powi(2))
                    .sum::<f32>() / model_history.len() as f32;

                let mse_score = 1.0 / (1.0 + mse);

                scores.insert(*model_type, accuracy * 0.7 + mse_score * 0.3);
            } else {
                scores.insert(*model_type, 0.5);
            }
        }

        let total: f32 = scores.values().sum();
        if total > 0.0 {
            for (model_type, score) in scores {
                if let Some(w) = weights.get_mut(&model_type) {
                    *w = 0.9 * *w + 0.1 * (score / total);
                }
            }

            let new_total: f32 = weights.values().sum();
            for w in weights.values_mut() {
                *w /= new_total;
            }
        }

        debug!("Updated weights: {:?}", *weights);
    }

    /// Get current model weights
    pub fn get_weights(&self) -> HashMap<ModelType, f32> {
        self.weights.read().clone()
    }

    /// Check if specific model is loaded
    pub fn has_model(&self, model_type: ModelType) -> bool {
        self.models.read().contains_key(&model_type)
    }

    /// Get number of loaded models
    pub fn model_count(&self) -> usize {
        self.models.read().len()
    }
}

/// Feature buffer for sequence models
pub struct FeatureBuffer {
    buffer: Vec<Vec<f32>>,
    sequence_length: usize,
}

impl FeatureBuffer {
    pub fn new(sequence_length: usize, _num_features: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(sequence_length),
            sequence_length,
        }
    }

    pub fn push(&mut self, features: Vec<f32>) {
        self.buffer.push(features);
        if self.buffer.len() > self.sequence_length {
            self.buffer.remove(0);
        }
    }

    pub fn is_ready(&self) -> bool {
        self.buffer.len() >= self.sequence_length
    }

    pub fn get_sequence(&self) -> Option<Vec<Vec<f32>>> {
        if self.is_ready() {
            Some(self.buffer.clone())
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_buffer() {
        let mut buffer = FeatureBuffer::new(5, 10);
        assert!(!buffer.is_ready());

        for i in 0..5 {
            buffer.push(vec![i as f32; 10]);
        }

        assert!(buffer.is_ready());
        let seq = buffer.get_sequence().unwrap();
        assert_eq!(seq.len(), 5);
        assert_eq!(seq[0].len(), 10);
    }

    #[test]
    fn test_model_type_display() {
        assert_eq!(ModelType::LightGBM.to_string(), "lightgbm");
        assert_eq!(ModelType::D4pg.to_string(), "d4pg");
    }

    #[test]
    fn test_nsmi_augment_buffer() {
        let mut buffer = NSMIAugmentBuffer::new(10);
        assert_eq!(buffer.total_dim(), 10 + NSMIFeatures::NUM_FEATURES);

        let raw = vec![1.0f32; 10];
        let nsmi_features = NSMIFeatures {
            nonstationarity: 0.5,
            regime_prob: 0.3,
            spectral_gap_norm: 0.8,
            drift_norm: 0.2,
            dim_ratio: 0.6,
            regime_stability: 0.9,
        };

        let augmented = buffer.augment(&raw, &nsmi_features);
        assert_eq!(augmented.len(), 16); // 10 + 6 NSMI features

        // Check raw features preserved
        for i in 0..10 {
            assert_eq!(augmented[i], 1.0);
        }

        // Check NSMI features appended
        assert!((augmented[10] - 0.5).abs() < 0.001); // nonstationarity
        assert!((augmented[11] - 0.3).abs() < 0.001); // regime_prob
        assert!((augmented[15] - 0.9).abs() < 0.001); // regime_stability
    }

    #[test]
    fn test_nsmi_augment_buffer_short_input() {
        let mut buffer = NSMIAugmentBuffer::new(10);
        let raw = vec![2.0f32; 5]; // Shorter than buffer size

        let nsmi_features = NSMIFeatures::default();
        let augmented = buffer.augment(&raw, &nsmi_features);

        // First 5 should be 2.0, next 5 should be 0.0, then NSMI
        for i in 0..5 {
            assert_eq!(augmented[i], 2.0);
        }
        for i in 5..10 {
            assert_eq!(augmented[i], 0.0);
        }
        assert_eq!(augmented.len(), 16);
    }

    #[test]
    fn test_nsmi_adjusted_weights_default() {
        let adjusted = NSMIAdjustedWeights::default();
        assert!(adjusted.base_weights.is_empty());
        assert!(adjusted.adjusted_weights.is_empty());
        assert_eq!(adjusted.regime_weight, 0.0);
        assert_eq!(adjusted.current_regime, 0);
    }
}
