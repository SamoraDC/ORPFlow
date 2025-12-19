//! ONNX Runtime Integration for ML Model Inference
//!
//! Provides high-performance inference for trained ML/DL/RL models
//! exported from Python training pipeline

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use ort::{session::{builder::GraphOptimizationLevel, Session}, value::Tensor};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Model type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    LightGBM,
    XGBoost,
    LSTM,
    CNN,
    D4PG,
    MARL,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::LightGBM => write!(f, "lightgbm"),
            ModelType::XGBoost => write!(f, "xgboost"),
            ModelType::LSTM => write!(f, "lstm"),
            ModelType::CNN => write!(f, "cnn"),
            ModelType::D4PG => write!(f, "d4pg"),
            ModelType::MARL => write!(f, "marl"),
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

/// Model ensemble with dynamic weighting
pub struct ModelEnsemble {
    models: RwLock<HashMap<ModelType, OnnxModel>>,
    weights: RwLock<HashMap<ModelType, f32>>,
    predictions_history: RwLock<Vec<(ModelType, f32, f32)>>,
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
                            "lstm" | "lstm_model" => Some(ModelType::LSTM),
                            "cnn" | "cnn_model" => Some(ModelType::CNN),
                            "d4pg" | "d4pg_actor" => Some(ModelType::D4PG),
                            n if n.starts_with("marl") => Some(ModelType::MARL),
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

        Ok(Self {
            models: RwLock::new(models),
            weights: RwLock::new(weights),
            predictions_history: RwLock::new(Vec::new()),
        })
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

        for model_type in [ModelType::LSTM, ModelType::CNN] {
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

        if let Some(model) = models.get_mut(&ModelType::D4PG) {
            model.predict_action(state)
        } else {
            Err(anyhow::anyhow!("D4PG model not available"))
        }
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
        assert_eq!(ModelType::D4PG.to_string(), "d4pg");
    }
}
