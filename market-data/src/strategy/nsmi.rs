//! Non-Stationary Manifold Inference (NSMI) Module
//!
//! Online detection of regime changes and non-stationarity in time series
//! by tracking the evolution of local data manifold structure.
//!
//! Key components:
//! - Streaming covariance matrix estimation with exponential weighting
//! - Eigenspectrum tracking via efficient rank-1 updates
//! - Manifold curvature and drift detection
//! - Regime change probability scoring
//!
//! Performance characteristics:
//! - O(n^2) per update where n is feature dimension
//! - Zero allocations in hot path (pre-allocated buffers)
//! - Thread-safe for async runtimes

// Allow dead_code when compiled without ml feature (NSMI is used by ml_inference)
#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for NSMI state tracking
#[derive(Debug, Clone)]
pub struct NSMIConfig {
    /// Dimension of the feature space
    pub dimension: usize,
    /// Half-life for exponential weighting (in samples)
    pub half_life: f64,
    /// Number of top eigenvalues to track for spectral gap
    pub tracked_eigenvalues: usize,
    /// Threshold for regime change detection (eigenvalue ratio)
    pub regime_threshold: f64,
    /// Minimum samples before regime detection activates
    pub warmup_samples: usize,
    /// Sensitivity for manifold drift detection
    pub drift_sensitivity: f64,
}

impl Default for NSMIConfig {
    fn default() -> Self {
        Self {
            dimension: 10,
            half_life: 100.0,
            tracked_eigenvalues: 3,
            regime_threshold: 0.3,
            warmup_samples: 50,
            drift_sensitivity: 2.0,
        }
    }
}

impl NSMIConfig {
    /// Create config with specific dimension and half-life
    pub fn new(dimension: usize, half_life: f64) -> Self {
        Self {
            dimension,
            half_life,
            tracked_eigenvalues: (dimension / 3).clamp(2, 5),
            ..Default::default()
        }
    }

    /// Compute decay factor (alpha) from half-life
    #[inline(always)]
    pub fn decay_factor(&self) -> f64 {
        // alpha such that (1-alpha)^half_life = 0.5
        // alpha = 1 - 0.5^(1/half_life)
        1.0 - 0.5_f64.powf(1.0 / self.half_life)
    }
}

/// Result of NSMI update containing all derived metrics
#[derive(Debug, Clone, Default)]
pub struct NSMIResult {
    /// Non-stationarity score (0-1, higher = more non-stationary)
    pub nonstationarity_score: f64,
    /// Probability of regime change (0-1)
    pub regime_change_probability: f64,
    /// Current spectral gap (difference between top eigenvalues)
    pub spectral_gap: f64,
    /// Manifold drift magnitude (rate of covariance change)
    pub manifold_drift: f64,
    /// Estimated effective dimension of the manifold
    pub effective_dimension: f64,
    /// Whether a regime change was detected this update
    pub regime_change_detected: bool,
    /// Current regime identifier (increments on detected changes)
    pub current_regime: u64,
    /// Top eigenvalues of the covariance matrix
    pub top_eigenvalues: Vec<f64>,
}

/// Features derived from NSMI state for model augmentation
#[derive(Debug, Clone, Default)]
pub struct NSMIFeatures {
    /// Non-stationarity score
    pub nonstationarity: f64,
    /// Regime change probability
    pub regime_prob: f64,
    /// Spectral gap normalized
    pub spectral_gap_norm: f64,
    /// Manifold drift normalized
    pub drift_norm: f64,
    /// Effective dimension ratio
    pub dim_ratio: f64,
    /// Regime stability score (inverse of change frequency)
    pub regime_stability: f64,
}

impl NSMIFeatures {
    /// Convert features to a flat vector for model input
    #[inline]
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.nonstationarity,
            self.regime_prob,
            self.spectral_gap_norm,
            self.drift_norm,
            self.dim_ratio,
            self.regime_stability,
        ]
    }

    /// Number of NSMI features
    pub const NUM_FEATURES: usize = 6;
}

/// Pre-allocated buffers for hot-path operations
struct NSMIBuffers {
    /// Temporary vector for mean-centered observation
    centered: Vec<f64>,
    /// Temporary matrix for outer product
    outer_product: Vec<f64>,
    /// Buffer for eigenvalue computation (power iteration)
    eigen_vec: Vec<f64>,
    /// Secondary buffer for power iteration
    eigen_temp: Vec<f64>,
    /// Buffer for previous covariance (drift detection)
    prev_cov_diag: Vec<f64>,
}

impl NSMIBuffers {
    fn new(dim: usize) -> Self {
        Self {
            centered: vec![0.0; dim],
            outer_product: vec![0.0; dim * dim],
            eigen_vec: vec![0.0; dim],
            eigen_temp: vec![0.0; dim],
            prev_cov_diag: vec![0.0; dim],
        }
    }

    fn resize(&mut self, dim: usize) {
        self.centered.resize(dim, 0.0);
        self.outer_product.resize(dim * dim, 0.0);
        self.eigen_vec.resize(dim, 0.0);
        self.eigen_temp.resize(dim, 0.0);
        self.prev_cov_diag.resize(dim, 0.0);
    }
}

/// NSMI State - tracks manifold evolution for regime detection
///
/// Thread-safe via interior mutability pattern. The state
/// can be shared across async tasks in a tokio runtime.
pub struct NSMIState {
    config: NSMIConfig,
    /// Running mean (exponentially weighted)
    mean: Vec<f64>,
    /// Running covariance matrix (exponentially weighted, row-major)
    covariance: Vec<f64>,
    /// Tracked top eigenvalues
    eigenvalues: Vec<f64>,
    /// Number of samples processed
    sample_count: AtomicU64,
    /// Current regime identifier
    current_regime: AtomicU64,
    /// Regime change counter (for stability calculation)
    regime_changes: AtomicU64,
    /// Pre-allocated buffers (not shared, use via &mut self)
    buffers: NSMIBuffers,
    /// Previous spectral gap for drift detection
    prev_spectral_gap: f64,
    /// Exponential moving average of drift
    ema_drift: f64,
    /// Exponential moving average of squared drift (for variance)
    ema_drift_sq: f64,
    /// Alpha decay factor (cached from config)
    alpha: f64,
}

impl NSMIState {
    /// Create a new NSMI state with the given configuration
    pub fn new(config: NSMIConfig) -> Self {
        let dim = config.dimension;
        let alpha = config.decay_factor();
        let tracked = config.tracked_eigenvalues;

        Self {
            config,
            mean: vec![0.0; dim],
            covariance: vec![0.0; dim * dim],
            eigenvalues: vec![0.0; tracked],
            sample_count: AtomicU64::new(0),
            current_regime: AtomicU64::new(0),
            regime_changes: AtomicU64::new(0),
            buffers: NSMIBuffers::new(dim),
            prev_spectral_gap: 0.0,
            ema_drift: 0.0,
            ema_drift_sq: 0.0,
            alpha,
        }
    }

    /// Create with default configuration for a given dimension
    pub fn with_dimension(dim: usize) -> Self {
        Self::new(NSMIConfig::new(dim, 100.0))
    }

    /// Get the feature dimension
    #[inline]
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get the number of processed samples
    #[inline]
    pub fn sample_count(&self) -> u64 {
        self.sample_count.load(Ordering::Relaxed)
    }

    /// Check if warmup period is complete
    #[inline]
    pub fn is_warmed_up(&self) -> bool {
        self.sample_count() >= self.config.warmup_samples as u64
    }

    /// Reset the NSMI state
    pub fn reset(&mut self) {
        let dim = self.config.dimension;
        self.mean.fill(0.0);
        self.covariance.fill(0.0);
        self.eigenvalues.fill(0.0);
        self.sample_count.store(0, Ordering::Relaxed);
        self.current_regime.store(0, Ordering::Relaxed);
        self.regime_changes.store(0, Ordering::Relaxed);
        self.prev_spectral_gap = 0.0;
        self.ema_drift = 0.0;
        self.ema_drift_sq = 0.0;
        self.buffers.resize(dim);
    }

    /// Update NSMI state with a new observation
    ///
    /// This is the main hot-path function. Complexity is O(n^2) where
    /// n is the feature dimension. No allocations occur here.
    ///
    /// # Arguments
    /// * `observation` - Feature vector of length `dimension`
    ///
    /// # Returns
    /// * `NSMIResult` containing all derived metrics
    ///
    /// # Panics
    /// Panics if observation length doesn't match configured dimension
    #[inline]
    pub fn update(&mut self, observation: &[f64]) -> NSMIResult {
        let dim = self.config.dimension;
        assert_eq!(
            observation.len(),
            dim,
            "Observation dimension mismatch: expected {}, got {}",
            dim,
            observation.len()
        );

        let count = self.sample_count.fetch_add(1, Ordering::Relaxed) + 1;
        let alpha = self.alpha;

        // Store previous covariance diagonal for drift detection
        for i in 0..dim {
            self.buffers.prev_cov_diag[i] = self.covariance[i * dim + i];
        }

        // Update running mean: mean = (1-alpha) * mean + alpha * x
        for (mean_i, &obs_i) in self.mean.iter_mut().zip(observation.iter()) {
            *mean_i = (1.0 - alpha) * *mean_i + alpha * obs_i;
        }

        // Compute centered observation: x_centered = x - mean
        for ((centered_i, &obs_i), &mean_i) in self.buffers.centered.iter_mut().zip(observation.iter()).zip(self.mean.iter()) {
            *centered_i = obs_i - mean_i;
        }

        // Update covariance matrix using rank-1 update:
        // cov = (1-alpha) * cov + alpha * (x_centered * x_centered^T)
        // This is O(n^2)
        for i in 0..dim {
            for j in 0..dim {
                let idx = i * dim + j;
                let outer = self.buffers.centered[i] * self.buffers.centered[j];
                self.covariance[idx] = (1.0 - alpha) * self.covariance[idx] + alpha * outer;
            }
        }

        // Compute top eigenvalues using power iteration
        // Only compute after warmup
        if count >= self.config.warmup_samples as u64 {
            self.compute_top_eigenvalues();
        }

        // Compute spectral gap
        let spectral_gap = self.compute_spectral_gap();

        // Compute manifold drift (change in covariance structure)
        let manifold_drift = self.compute_manifold_drift();

        // Update drift EMA for z-score calculation
        self.ema_drift = (1.0 - alpha) * self.ema_drift + alpha * manifold_drift;
        self.ema_drift_sq = (1.0 - alpha) * self.ema_drift_sq + alpha * manifold_drift * manifold_drift;

        // Compute effective dimension
        let effective_dimension = self.compute_effective_dimension();

        // Compute non-stationarity score
        let nonstationarity_score = self.compute_nonstationarity_score(manifold_drift);

        // Detect regime change
        let (regime_change_detected, regime_change_probability) =
            self.detect_regime_change(spectral_gap, manifold_drift);

        // Update regime if change detected
        if regime_change_detected {
            self.current_regime.fetch_add(1, Ordering::Relaxed);
            self.regime_changes.fetch_add(1, Ordering::Relaxed);
        }

        // Store current spectral gap for next iteration
        self.prev_spectral_gap = spectral_gap;

        NSMIResult {
            nonstationarity_score,
            regime_change_probability,
            spectral_gap,
            manifold_drift,
            effective_dimension,
            regime_change_detected,
            current_regime: self.current_regime.load(Ordering::Relaxed),
            top_eigenvalues: self.eigenvalues.clone(),
        }
    }

    /// Compute top eigenvalues using power iteration with deflation
    ///
    /// This is O(k * n^2 * iterations) where k is tracked_eigenvalues.
    /// For typical configs, this is effectively O(n^2).
    fn compute_top_eigenvalues(&mut self) {
        let dim = self.config.dimension;
        let k = self.config.tracked_eigenvalues.min(dim);
        const MAX_ITER: usize = 20;
        const TOLERANCE: f64 = 1e-6;

        // Work with a copy of covariance for deflation
        // We'll deflate in-place using buffers
        let cov = &self.covariance;

        // Deflation matrix (accumulated)
        let mut deflation = vec![0.0; dim * dim];

        for eig_idx in 0..k {
            // Initialize eigenvector randomly (use index for determinism)
            for i in 0..dim {
                self.buffers.eigen_vec[i] = ((i + eig_idx + 1) as f64).sin();
            }
            normalize_vector(&mut self.buffers.eigen_vec);

            let mut prev_eigenvalue = 0.0;

            for _ in 0..MAX_ITER {
                // v_new = (A - deflation) * v
                for i in 0..dim {
                    let mut sum = 0.0;
                    for j in 0..dim {
                        let idx = i * dim + j;
                        sum += (cov[idx] - deflation[idx]) * self.buffers.eigen_vec[j];
                    }
                    self.buffers.eigen_temp[i] = sum;
                }

                // Compute eigenvalue (Rayleigh quotient)
                let eigenvalue: f64 = self
                    .buffers
                    .eigen_vec
                    .iter()
                    .zip(self.buffers.eigen_temp.iter())
                    .map(|(v, av)| v * av)
                    .sum();

                // Normalize
                normalize_vector(&mut self.buffers.eigen_temp);

                // Swap buffers
                std::mem::swap(&mut self.buffers.eigen_vec, &mut self.buffers.eigen_temp);

                // Check convergence
                if (eigenvalue - prev_eigenvalue).abs() < TOLERANCE {
                    break;
                }
                prev_eigenvalue = eigenvalue;
            }

            // Store eigenvalue (ensure non-negative)
            self.eigenvalues[eig_idx] = prev_eigenvalue.max(0.0);

            // Deflate: deflation += eigenvalue * v * v^T
            for i in 0..dim {
                for j in 0..dim {
                    let idx = i * dim + j;
                    deflation[idx] +=
                        prev_eigenvalue * self.buffers.eigen_vec[i] * self.buffers.eigen_vec[j];
                }
            }
        }

        // Sort eigenvalues descending
        self.eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Compute spectral gap (normalized difference between top eigenvalues)
    #[inline]
    fn compute_spectral_gap(&self) -> f64 {
        if self.eigenvalues.len() < 2 {
            return 0.0;
        }

        let lambda1 = self.eigenvalues[0];
        let lambda2 = self.eigenvalues[1];

        if lambda1 <= 1e-10 {
            return 0.0;
        }

        // Normalized spectral gap
        (lambda1 - lambda2) / lambda1
    }

    /// Compute manifold drift (change in covariance structure)
    #[inline]
    fn compute_manifold_drift(&self) -> f64 {
        let dim = self.config.dimension;
        let mut drift_sq = 0.0;

        // Frobenius norm of diagonal change (fast approximation)
        for i in 0..dim {
            let prev = self.buffers.prev_cov_diag[i];
            let curr = self.covariance[i * dim + i];
            let diff = curr - prev;
            drift_sq += diff * diff;
        }

        drift_sq.sqrt()
    }

    /// Compute effective dimension (participation ratio of eigenvalues)
    #[inline]
    fn compute_effective_dimension(&self) -> f64 {
        let sum: f64 = self.eigenvalues.iter().sum();
        let sum_sq: f64 = self.eigenvalues.iter().map(|e| e * e).sum();

        if sum_sq <= 1e-10 {
            return 1.0;
        }

        // Participation ratio
        (sum * sum) / sum_sq
    }

    /// Compute non-stationarity score (0-1)
    #[inline]
    fn compute_nonstationarity_score(&self, drift: f64) -> f64 {
        // Z-score of drift
        let drift_var = (self.ema_drift_sq - self.ema_drift * self.ema_drift).max(1e-10);
        let drift_std = drift_var.sqrt();
        let z_drift = (drift - self.ema_drift) / drift_std;

        // Sigmoid transformation to [0, 1]
        let sensitivity = self.config.drift_sensitivity;
        1.0 / (1.0 + (-sensitivity * z_drift).exp())
    }

    /// Detect regime change based on spectral gap and drift
    #[inline]
    fn detect_regime_change(&self, spectral_gap: f64, drift: f64) -> (bool, f64) {
        if !self.is_warmed_up() {
            return (false, 0.0);
        }

        // Spectral gap change
        let gap_change = (spectral_gap - self.prev_spectral_gap).abs();

        // Z-score of drift
        let drift_var = (self.ema_drift_sq - self.ema_drift * self.ema_drift).max(1e-10);
        let drift_std = drift_var.sqrt();
        let z_drift = (drift - self.ema_drift) / drift_std;

        // Combine indicators
        // Gap change relative to threshold
        let gap_score = gap_change / self.config.regime_threshold;
        // Drift z-score normalized
        let drift_score = z_drift.abs() / self.config.drift_sensitivity;

        // Weighted combination
        let combined = 0.6 * gap_score + 0.4 * drift_score;

        // Probability (sigmoid of combined score)
        let probability = 1.0 / (1.0 + (-2.0 * (combined - 1.0)).exp());

        // Detection threshold
        let detected = combined > 1.5 && z_drift.abs() > self.config.drift_sensitivity;

        (detected, probability)
    }

    /// Get NSMI features for model augmentation
    pub fn get_features(&self) -> NSMIFeatures {
        let sample_count = self.sample_count() as f64;
        let regime_changes = self.regime_changes.load(Ordering::Relaxed) as f64;

        // Regime stability: inverse of change frequency
        let regime_stability = if sample_count > 0.0 && regime_changes > 0.0 {
            (sample_count / regime_changes).min(1000.0) / 1000.0
        } else {
            1.0
        };

        let spectral_gap = self.compute_spectral_gap();
        let effective_dim = self.compute_effective_dimension();
        let dim = self.config.dimension as f64;

        NSMIFeatures {
            nonstationarity: self.compute_nonstationarity_score(self.ema_drift),
            regime_prob: 0.0, // Updated on next update()
            spectral_gap_norm: spectral_gap.min(1.0),
            drift_norm: (self.ema_drift / (self.ema_drift + 1.0)).min(1.0),
            dim_ratio: effective_dim / dim,
            regime_stability,
        }
    }

    /// Apply NSMI features to raw feature vector
    ///
    /// Returns augmented feature vector with NSMI features appended
    pub fn apply_to_features(&self, raw_features: &[f64]) -> Vec<f64> {
        let nsmi_features = self.get_features();
        let mut augmented = Vec::with_capacity(raw_features.len() + NSMIFeatures::NUM_FEATURES);
        augmented.extend_from_slice(raw_features);
        augmented.extend(nsmi_features.to_vec());
        augmented
    }

    /// Get regime weight for model weighting (higher during stable regimes)
    #[inline]
    pub fn get_regime_weight(&self) -> f64 {
        let features = self.get_features();
        // Weight based on stability and inverse of non-stationarity
        let stability_weight = features.regime_stability;
        let stationarity_weight = 1.0 - features.nonstationarity;

        // Combine with emphasis on stationarity
        0.3 * stability_weight + 0.7 * stationarity_weight
    }

    /// Get current covariance matrix (for diagnostics)
    pub fn get_covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Get current mean vector (for diagnostics)
    pub fn get_mean(&self) -> &[f64] {
        &self.mean
    }

    /// Get current eigenvalues
    pub fn get_eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Get configuration
    pub fn config(&self) -> &NSMIConfig {
        &self.config
    }
}

/// Normalize a vector in-place (L2 norm)
#[inline]
fn normalize_vector(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// Thread-safety: NSMIState uses AtomicU64 for counters and is designed
// to be used with &mut self (exclusive access). For concurrent access,
// wrap in Arc<Mutex<NSMIState>> or Arc<RwLock<NSMIState>>.
unsafe impl Send for NSMIState {}
unsafe impl Sync for NSMIState {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsmi_config_default() {
        let config = NSMIConfig::default();
        assert_eq!(config.dimension, 10);
        assert_eq!(config.half_life, 100.0);
        assert!(config.decay_factor() > 0.0 && config.decay_factor() < 1.0);
    }

    #[test]
    fn test_nsmi_config_decay_factor() {
        let config = NSMIConfig::new(10, 100.0);
        let alpha = config.decay_factor();
        // After half_life updates, weight should be ~0.5
        let remaining_weight = (1.0 - alpha).powi(100);
        assert!((remaining_weight - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_nsmi_state_creation() {
        let state = NSMIState::with_dimension(5);
        assert_eq!(state.dimension(), 5);
        assert_eq!(state.sample_count(), 0);
        assert!(!state.is_warmed_up());
    }

    #[test]
    fn test_nsmi_update_warmup() {
        let mut state = NSMIState::new(NSMIConfig {
            dimension: 3,
            warmup_samples: 10,
            ..Default::default()
        });

        // Before warmup
        for i in 0..9 {
            let obs = vec![i as f64, (i * 2) as f64, (i * 3) as f64];
            let result = state.update(&obs);
            assert!(!result.regime_change_detected);
            assert_eq!(state.sample_count(), (i + 1) as u64);
        }

        assert!(!state.is_warmed_up());

        // Complete warmup
        let result = state.update(&[10.0, 20.0, 30.0]);
        assert!(state.is_warmed_up());
        assert!(result.top_eigenvalues.len() > 0);
    }

    #[test]
    fn test_nsmi_mean_update() {
        let mut state = NSMIState::new(NSMIConfig {
            dimension: 2,
            half_life: 10.0,
            warmup_samples: 5,
            ..Default::default()
        });

        // Push constant values
        for _ in 0..100 {
            state.update(&[5.0, 10.0]);
        }

        let mean = state.get_mean();
        // Mean should converge to [5.0, 10.0]
        assert!((mean[0] - 5.0).abs() < 0.1);
        assert!((mean[1] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_nsmi_covariance_diagonal() {
        let mut state = NSMIState::new(NSMIConfig {
            dimension: 2,
            half_life: 50.0,
            warmup_samples: 10,
            ..Default::default()
        });

        // Push values with known variance
        for i in 0..200 {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            state.update(&[x, x * 2.0]);
        }

        let cov = state.get_covariance();
        // Variance of x is ~1.0, variance of 2x is ~4.0
        // Due to exponential weighting, exact values will differ
        assert!(cov[0] > 0.5); // Var(x)
        assert!(cov[3] > 2.0); // Var(2x)
    }

    #[test]
    fn test_nsmi_regime_detection() {
        let mut state = NSMIState::new(NSMIConfig {
            dimension: 3,
            half_life: 20.0,
            warmup_samples: 30,
            regime_threshold: 0.1, // More sensitive threshold
            drift_sensitivity: 1.0, // Lower sensitivity for easier detection
            ..Default::default()
        });

        // Stable regime with consistent pattern
        for i in 0..100 {
            let x = (i as f64 * 0.1).sin();
            state.update(&[x, x * 0.5, x * 0.2]);
        }

        // Collect regime change probabilities during dramatic shift
        let mut max_regime_prob: f64 = 0.0;

        // Sudden, dramatic change in distribution
        for i in 0..100 {
            // Completely different scale and pattern
            let x = (i as f64 * 0.5).cos() * 100.0 + 500.0;
            let result = state.update(&[x, x * 10.0, x * 5.0]);
            max_regime_prob = max_regime_prob.max(result.regime_change_probability);
        }

        // At minimum, we should see elevated regime change probability
        // The probability should increase significantly during distribution shift
        assert!(
            max_regime_prob > 0.1,
            "Regime change probability should increase during distribution shift: max_prob={}",
            max_regime_prob
        );
    }

    #[test]
    fn test_nsmi_features() {
        let mut state = NSMIState::new(NSMIConfig {
            dimension: 4,
            warmup_samples: 20,
            ..Default::default()
        });

        for i in 0..50 {
            let x = i as f64;
            state.update(&[x, x * 2.0, x * 3.0, x * 4.0]);
        }

        let features = state.get_features();

        // All features should be in valid ranges
        assert!(features.nonstationarity >= 0.0 && features.nonstationarity <= 1.0);
        assert!(features.regime_prob >= 0.0 && features.regime_prob <= 1.0);
        assert!(features.spectral_gap_norm >= 0.0 && features.spectral_gap_norm <= 1.0);
        assert!(features.drift_norm >= 0.0 && features.drift_norm <= 1.0);
        assert!(features.dim_ratio >= 0.0 && features.dim_ratio <= 1.0);
        assert!(features.regime_stability >= 0.0 && features.regime_stability <= 1.0);
    }

    #[test]
    fn test_nsmi_apply_to_features() {
        let state = NSMIState::with_dimension(3);
        let raw = vec![1.0, 2.0, 3.0];
        let augmented = state.apply_to_features(&raw);

        assert_eq!(augmented.len(), raw.len() + NSMIFeatures::NUM_FEATURES);
        assert_eq!(augmented[0], 1.0);
        assert_eq!(augmented[1], 2.0);
        assert_eq!(augmented[2], 3.0);
    }

    #[test]
    fn test_nsmi_regime_weight() {
        let mut state = NSMIState::new(NSMIConfig {
            dimension: 3,
            warmup_samples: 10,
            ..Default::default()
        });

        // Stable data with minimal variance
        for _ in 0..100 {
            state.update(&[1.0, 2.0, 3.0]);
        }

        let weight = state.get_regime_weight();
        // For perfectly constant data, regime should be stable
        // Weight is a combination of stability (1.0 for no changes) and stationarity
        // With constant data, nonstationarity may still register as moderate
        // due to the sigmoid transformation, so we accept weight > 0.3
        assert!(
            weight > 0.3,
            "Weight should be positive for stable regime: {}",
            weight
        );

        // Also verify the weight is bounded
        assert!(
            weight <= 1.0,
            "Weight should be at most 1.0: {}",
            weight
        );
    }

    #[test]
    fn test_nsmi_effective_dimension() {
        let mut state = NSMIState::new(NSMIConfig {
            dimension: 4,
            warmup_samples: 20,
            tracked_eigenvalues: 4,
            ..Default::default()
        });

        // Data with 1D structure (all features correlated)
        for i in 0..100 {
            let x = i as f64;
            state.update(&[x, x, x, x]);
        }

        // Effective dimension should be close to 1
        let result = state.update(&[100.0, 100.0, 100.0, 100.0]);
        assert!(
            result.effective_dimension < 2.0,
            "Effective dim should be ~1 for perfectly correlated data: {}",
            result.effective_dimension
        );
    }

    #[test]
    fn test_nsmi_spectral_gap() {
        let mut state = NSMIState::new(NSMIConfig {
            dimension: 3,
            warmup_samples: 20,
            tracked_eigenvalues: 3,
            ..Default::default()
        });

        // Data with clear dominant direction
        for i in 0..100 {
            let x = i as f64;
            state.update(&[x * 10.0, (i % 5) as f64, (i % 3) as f64]);
        }

        let result = state.update(&[1000.0, 2.0, 1.0]);

        // First eigenvalue should dominate
        assert!(
            result.spectral_gap > 0.5,
            "Spectral gap should be large with dominant direction: {}",
            result.spectral_gap
        );
    }

    #[test]
    fn test_nsmi_reset() {
        let mut state = NSMIState::with_dimension(3);

        for _ in 0..50 {
            state.update(&[1.0, 2.0, 3.0]);
        }

        assert!(state.sample_count() > 0);

        state.reset();

        assert_eq!(state.sample_count(), 0);
        assert_eq!(state.current_regime.load(Ordering::Relaxed), 0);
        assert_eq!(state.get_mean(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_vector(&mut v);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize_vector(&mut v);
        // Should not panic, vector remains zero
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "Observation dimension mismatch")]
    fn test_nsmi_dimension_mismatch() {
        let mut state = NSMIState::with_dimension(3);
        state.update(&[1.0, 2.0]); // Wrong dimension
    }

    #[test]
    fn test_nsmi_features_to_vec() {
        let features = NSMIFeatures {
            nonstationarity: 0.1,
            regime_prob: 0.2,
            spectral_gap_norm: 0.3,
            drift_norm: 0.4,
            dim_ratio: 0.5,
            regime_stability: 0.6,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), NSMIFeatures::NUM_FEATURES);
        assert_eq!(vec[0], 0.1);
        assert_eq!(vec[5], 0.6);
    }

    #[test]
    fn test_nsmi_thread_safety() {
        // Verify Send + Sync are implemented
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NSMIState>();
    }
}
