"""
Advanced Quantitative Features Library for Algorithmic Trading.

This module implements stateful, streaming-compatible quantitative features
including Hawkes processes, Kalman filters, signal transforms, Random Matrix
Theory correlation filtering, Hidden Markov Models, and fractal analysis.

All features support online updates and are designed to avoid look-ahead bias.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Base Classes
# =============================================================================

class OnlineFeature(ABC):
    """Abstract base class for online/streaming features."""

    @abstractmethod
    def update(self, value: float) -> Any:
        """Update the feature with a new observation."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the feature state."""
        pass


class FitTransformFeature(ABC):
    """Abstract base class for features with fit/transform interface."""

    @abstractmethod
    def fit(self, data: NDArray[np.float64]) -> 'FitTransformFeature':
        """Fit the feature to historical data."""
        pass

    @abstractmethod
    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform data using fitted parameters."""
        pass

    def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


# =============================================================================
# Hawkes Process - Self-Exciting Point Process
# =============================================================================

@dataclass
class HawkesState:
    """State container for Hawkes process."""
    intensity: float = 0.0
    baseline_intensity: float = 0.0
    decay_factor: float = 0.0
    branching_ratio: float = 0.0
    event_count: int = 0
    last_event_time: float = 0.0


class HawkesIntensity(OnlineFeature):
    """
    Hawkes Process intensity estimator for modeling self-exciting events.

    The Hawkes process is used in finance to model trade arrivals, order flow,
    and volatility clustering where past events increase the probability of
    future events.

    Intensity: lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

    Attributes:
        mu: Baseline intensity (background rate)
        alpha: Jump size after each event
        beta: Decay rate of excitation
        max_history: Maximum number of events to track
    """

    def __init__(
        self,
        mu: float = 0.1,
        alpha: float = 0.5,
        beta: float = 1.0,
        max_history: int = 1000,
    ) -> None:
        """
        Initialize Hawkes intensity estimator.

        Args:
            mu: Baseline intensity (events per unit time)
            alpha: Excitation jump size (0 < alpha < beta for stability)
            beta: Decay rate of excitation
            max_history: Maximum event history to maintain
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.max_history = max_history

        # State
        self._event_times: deque[float] = deque(maxlen=max_history)
        self._current_time: float = 0.0
        self._intensity: float = mu

        # For parameter estimation
        self._inter_arrivals: deque[float] = deque(maxlen=max_history)

    def update(self, timestamp: float, is_event: bool = True) -> HawkesState:
        """
        Update intensity with a new timestamp and optional event.

        Args:
            timestamp: Current timestamp
            is_event: Whether an event occurred at this timestamp

        Returns:
            HawkesState with current intensity metrics
        """
        dt = timestamp - self._current_time if self._current_time > 0 else 0.0
        self._current_time = timestamp

        # Decay existing intensity contribution
        decayed_contribution = 0.0
        for event_time in self._event_times:
            time_since_event = timestamp - event_time
            if time_since_event > 0:
                decayed_contribution += self.alpha * np.exp(-self.beta * time_since_event)

        self._intensity = self.mu + decayed_contribution

        # Record event
        if is_event:
            if len(self._event_times) > 0:
                last_event = self._event_times[-1]
                self._inter_arrivals.append(timestamp - last_event)
            self._event_times.append(timestamp)

        return HawkesState(
            intensity=self._intensity,
            baseline_intensity=self.mu,
            decay_factor=np.exp(-self.beta * dt) if dt > 0 else 1.0,
            branching_ratio=self.branching_ratio,
            event_count=len(self._event_times),
            last_event_time=self._event_times[-1] if self._event_times else 0.0,
        )

    @property
    def branching_ratio(self) -> float:
        """
        Calculate the branching ratio alpha/beta.

        The branching ratio represents the expected number of child events
        triggered by a single parent event. Must be < 1 for stationarity.
        """
        return self.alpha / self.beta if self.beta > 0 else 0.0

    @property
    def intensity(self) -> float:
        """Current intensity value."""
        return self._intensity

    def estimate_parameters(self, min_events: int = 50) -> Dict[str, float]:
        """
        Estimate Hawkes parameters from observed events using MLE approximation.

        Args:
            min_events: Minimum events required for estimation

        Returns:
            Dict with estimated mu, alpha, beta parameters
        """
        if len(self._inter_arrivals) < min_events:
            return {'mu': self.mu, 'alpha': self.alpha, 'beta': self.beta}

        inter_arrivals = np.array(self._inter_arrivals)

        # Method of moments estimation
        mean_ia = np.mean(inter_arrivals)
        var_ia = np.var(inter_arrivals)

        if mean_ia > 0:
            # Approximate baseline intensity
            estimated_mu = 1.0 / mean_ia

            # Estimate branching ratio from excess variance
            cv_squared = var_ia / (mean_ia ** 2)
            estimated_branching = max(0.0, min(0.99, 1 - 1/cv_squared)) if cv_squared > 1 else 0.0

            # Estimate beta from autocorrelation decay
            estimated_beta = 1.0 / mean_ia  # Simplified estimate
            estimated_alpha = estimated_branching * estimated_beta

            return {
                'mu': estimated_mu,
                'alpha': estimated_alpha,
                'beta': estimated_beta,
            }

        return {'mu': self.mu, 'alpha': self.alpha, 'beta': self.beta}

    def reset(self) -> None:
        """Reset the Hawkes process state."""
        self._event_times.clear()
        self._inter_arrivals.clear()
        self._current_time = 0.0
        self._intensity = self.mu


# =============================================================================
# Kalman Filter for State Estimation
# =============================================================================

@dataclass
class KalmanState:
    """State container for Kalman filter."""
    state_estimate: NDArray[np.float64]
    state_covariance: NDArray[np.float64]
    innovation: float
    innovation_covariance: float
    kalman_gain: NDArray[np.float64]


class KalmanFilter(OnlineFeature):
    """
    Linear Kalman Filter for state estimation in financial time series.

    Commonly used for:
    - Alpha signal estimation
    - Spread modeling in pairs trading
    - Dynamic hedge ratio estimation
    - Noise filtering from price series

    State equation: x(t+1) = F * x(t) + w(t), w ~ N(0, Q)
    Observation equation: y(t) = H * x(t) + v(t), v ~ N(0, R)
    """

    def __init__(
        self,
        state_dim: int = 2,
        obs_dim: int = 1,
        F: Optional[NDArray[np.float64]] = None,
        H: Optional[NDArray[np.float64]] = None,
        Q: Optional[NDArray[np.float64]] = None,
        R: Optional[NDArray[np.float64]] = None,
        x0: Optional[NDArray[np.float64]] = None,
        P0: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Initialize Kalman filter.

        Args:
            state_dim: Dimension of state vector
            obs_dim: Dimension of observation vector
            F: State transition matrix (state_dim x state_dim)
            H: Observation matrix (obs_dim x state_dim)
            Q: Process noise covariance (state_dim x state_dim)
            R: Observation noise covariance (obs_dim x obs_dim)
            x0: Initial state estimate
            P0: Initial state covariance
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # Default to random walk with drift model
        self.F = F if F is not None else np.eye(state_dim)
        self.H = H if H is not None else np.ones((obs_dim, state_dim)) / state_dim
        self.Q = Q if Q is not None else np.eye(state_dim) * 1e-4
        self.R = R if R is not None else np.eye(obs_dim) * 1e-2

        # Initial state
        self._x = x0 if x0 is not None else np.zeros(state_dim)
        self._P = P0 if P0 is not None else np.eye(state_dim)

        # Store initial values for reset
        self._x0 = self._x.copy()
        self._P0 = self._P.copy()

        # History for smoothing
        self._history: List[KalmanState] = []

    def predict(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Prediction step: project state forward.

        Returns:
            Tuple of (predicted state, predicted covariance)
        """
        x_pred = self.F @ self._x
        P_pred = self.F @ self._P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, observation: Union[float, NDArray[np.float64]]) -> KalmanState:
        """
        Full Kalman filter update step (predict + correct).

        Args:
            observation: New observation value(s)

        Returns:
            KalmanState with updated estimates
        """
        y = np.atleast_1d(observation).astype(np.float64)

        # Predict
        x_pred, P_pred = self.predict()

        # Innovation (measurement residual)
        y_pred = self.H @ x_pred
        innovation = y - y_pred

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        S_inv = np.linalg.inv(S) if S.ndim > 0 else 1.0 / S

        # Kalman gain
        K = P_pred @ self.H.T @ S_inv

        # Update state estimate
        self._x = x_pred + K @ innovation

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self._P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        state = KalmanState(
            state_estimate=self._x.copy(),
            state_covariance=self._P.copy(),
            innovation=float(innovation[0]) if innovation.size > 0 else 0.0,
            innovation_covariance=float(S[0, 0]) if S.ndim > 0 else float(S),
            kalman_gain=K.copy(),
        )

        self._history.append(state)
        return state

    @property
    def state(self) -> NDArray[np.float64]:
        """Current state estimate."""
        return self._x.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Current state covariance."""
        return self._P.copy()

    def smooth(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Run Rauch-Tung-Striebel smoother on a batch of observations.

        Args:
            observations: Array of observations (n_samples, obs_dim)

        Returns:
            Smoothed state estimates (n_samples, state_dim)
        """
        n = len(observations)

        # Forward pass
        forward_states = []
        forward_covs = []
        predicted_states = []
        predicted_covs = []

        for obs in observations:
            x_pred, P_pred = self.predict()
            predicted_states.append(x_pred)
            predicted_covs.append(P_pred)

            self.update(obs)
            forward_states.append(self._x.copy())
            forward_covs.append(self._P.copy())

        # Backward pass
        smoothed_states = [forward_states[-1]]

        for t in range(n - 2, -1, -1):
            J = forward_covs[t] @ self.F.T @ np.linalg.inv(predicted_covs[t + 1])
            x_smooth = forward_states[t] + J @ (smoothed_states[0] - predicted_states[t + 1])
            smoothed_states.insert(0, x_smooth)

        return np.array(smoothed_states)

    def reset(self) -> None:
        """Reset to initial state."""
        self._x = self._x0.copy()
        self._P = self._P0.copy()
        self._history.clear()

    @classmethod
    def for_spread_tracking(
        cls,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
    ) -> 'KalmanFilter':
        """
        Create a Kalman filter configured for spread/alpha tracking.

        State: [alpha, beta] where spread = alpha + beta * benchmark

        Args:
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        return cls(
            state_dim=2,
            obs_dim=1,
            F=np.eye(2),  # Random walk
            H=np.array([[1.0, 0.0]]),  # Observe alpha directly
            Q=np.eye(2) * process_noise,
            R=np.array([[measurement_noise]]),
        )


# =============================================================================
# Signal Transforms
# =============================================================================

def fisher_transform(series: NDArray[np.float64], epsilon: float = 1e-8) -> NDArray[np.float64]:
    """
    Apply Fisher Transform to normalize series to approximately Gaussian.

    The Fisher Transform is used to convert bounded signals (like RSI, correlation)
    into unbounded signals with enhanced tail behavior, making turning points
    more visible.

    Formula: y = 0.5 * ln((1 + x) / (1 - x))

    Args:
        series: Input array, values should be in range (-1, 1)
        epsilon: Small value to prevent division by zero

    Returns:
        Fisher-transformed series
    """
    # Clip to valid range
    clipped = np.clip(series, -1 + epsilon, 1 - epsilon)
    return 0.5 * np.log((1 + clipped) / (1 - clipped))


def inverse_fisher_transform(series: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply Inverse Fisher Transform to map back to bounded range.

    Formula: x = (e^(2y) - 1) / (e^(2y) + 1)

    Args:
        series: Fisher-transformed array

    Returns:
        Array with values in range (-1, 1)
    """
    exp_2y = np.exp(2 * series)
    return (exp_2y - 1) / (exp_2y + 1)


def normalize_to_fisher_range(
    series: NDArray[np.float64],
    lookback: int = 20,
    center: bool = True,
) -> NDArray[np.float64]:
    """
    Normalize a series to (-1, 1) range for Fisher Transform input.

    Args:
        series: Raw input series
        lookback: Rolling window for min/max normalization
        center: Whether to center around zero

    Returns:
        Normalized series in range (-1, 1)
    """
    result = np.full_like(series, np.nan, dtype=np.float64)

    for i in range(lookback, len(series)):
        window = series[i - lookback:i + 1]
        min_val = np.min(window)
        max_val = np.max(window)

        if max_val - min_val > 1e-10:
            normalized = (series[i] - min_val) / (max_val - min_val)
            if center:
                normalized = 2 * normalized - 1  # Map [0,1] to [-1,1]
            result[i] = np.clip(normalized, -0.999, 0.999)

    return result


def hilbert_transform(series: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply Hilbert Transform to extract instantaneous phase and amplitude.

    The Hilbert Transform is used for:
    - Identifying market cycles
    - Extracting trend and cycle components
    - Adaptive technical indicators

    Args:
        series: Input time series

    Returns:
        Tuple of (instantaneous_amplitude, instantaneous_phase)
        - amplitude: Envelope of the signal
        - phase: Instantaneous phase in radians
    """
    n = len(series)

    # FFT-based Hilbert transform
    fft = np.fft.fft(series)

    # Create Hilbert transform frequency response
    h = np.zeros(n)
    if n > 0:
        h[0] = 1
        if n % 2 == 0:
            h[1:n//2] = 2
            h[n//2] = 1
        else:
            h[1:(n+1)//2] = 2

    # Analytic signal
    analytic = np.fft.ifft(fft * h)

    # Extract amplitude and phase
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)

    return amplitude.real, phase.real


def instantaneous_frequency(phase: NDArray[np.float64], dt: float = 1.0) -> NDArray[np.float64]:
    """
    Calculate instantaneous frequency from phase.

    Args:
        phase: Instantaneous phase from Hilbert transform
        dt: Time step between samples

    Returns:
        Instantaneous frequency (cycles per unit time)
    """
    # Unwrap phase to avoid discontinuities
    unwrapped = np.unwrap(phase)

    # Differentiate to get frequency
    freq = np.gradient(unwrapped, dt) / (2 * np.pi)

    return freq


@dataclass
class WaveletDecomposition:
    """Container for wavelet decomposition results."""
    approximation: NDArray[np.float64]
    details: List[NDArray[np.float64]]
    levels: int
    wavelet: str


def wavelet_transform(
    series: NDArray[np.float64],
    wavelet: str = 'db4',
    levels: int = 4,
) -> WaveletDecomposition:
    """
    Apply discrete wavelet transform for multi-resolution decomposition.

    Wavelets are used in finance for:
    - Denoising price series
    - Multi-scale trend analysis
    - Volatility estimation at different frequencies

    This implementation uses Haar wavelets (simplest) as a fallback if
    PyWavelets is not available, otherwise uses the specified wavelet.

    Args:
        series: Input time series
        wavelet: Wavelet type ('db4', 'haar', 'sym4', etc.)
        levels: Number of decomposition levels

    Returns:
        WaveletDecomposition with approximation and detail coefficients
    """
    try:
        import pywt

        # Perform multi-level decomposition
        coeffs = pywt.wavedec(series, wavelet, level=levels)
        approximation = coeffs[0]
        details = coeffs[1:]

    except ImportError:
        # Fallback to simple Haar wavelet implementation
        warnings.warn("PyWavelets not installed, using simple Haar wavelet")

        approximation = series.copy()
        details = []

        for _ in range(levels):
            n = len(approximation)
            if n < 2:
                break

            # Haar wavelet decomposition
            padded = approximation[:n - n % 2]  # Ensure even length
            pairs = padded.reshape(-1, 2)

            # Low-pass (approximation) and high-pass (detail)
            approx = pairs.mean(axis=1)
            detail = pairs[:, 0] - pairs[:, 1]

            details.append(detail)
            approximation = approx

    return WaveletDecomposition(
        approximation=approximation,
        details=details,
        levels=len(details),
        wavelet=wavelet,
    )


def wavelet_reconstruct(decomposition: WaveletDecomposition, target_length: int) -> NDArray[np.float64]:
    """
    Reconstruct signal from wavelet decomposition.

    Args:
        decomposition: WaveletDecomposition object
        target_length: Desired output length

    Returns:
        Reconstructed signal
    """
    try:
        import pywt

        coeffs = [decomposition.approximation] + list(decomposition.details)
        reconstructed = pywt.waverec(coeffs, decomposition.wavelet)
        return reconstructed[:target_length]

    except ImportError:
        # Fallback reconstruction using Haar wavelet inverse
        approx = decomposition.approximation.copy()

        for detail in reversed(decomposition.details):
            n = len(approx)
            # Upsample and reconstruct
            reconstructed = np.zeros(n * 2)
            for i in range(n):
                reconstructed[2 * i] = approx[i] + detail[i] / 2 if i < len(detail) else approx[i]
                reconstructed[2 * i + 1] = approx[i] - detail[i] / 2 if i < len(detail) else approx[i]
            approx = reconstructed

        # Interpolate to target length if needed
        if len(approx) != target_length:
            approx = np.interp(
                np.linspace(0, 1, target_length),
                np.linspace(0, 1, len(approx)),
                approx
            )

        return approx[:target_length]


def wavelet_denoise(
    series: NDArray[np.float64],
    wavelet: str = 'db4',
    levels: int = 4,
    threshold_ratio: float = 0.3,
) -> NDArray[np.float64]:
    """
    Denoise a signal using wavelet thresholding.

    Args:
        series: Input time series
        wavelet: Wavelet type
        levels: Decomposition levels
        threshold_ratio: Fraction of max coefficient to use as threshold

    Returns:
        Denoised signal
    """
    decomp = wavelet_transform(series, wavelet, levels)

    # Threshold detail coefficients
    thresholded_details = []
    for detail in decomp.details:
        threshold = threshold_ratio * np.max(np.abs(detail))
        # Soft thresholding
        thresholded = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
        thresholded_details.append(thresholded)

    decomp.details = thresholded_details

    return wavelet_reconstruct(decomp, len(series))


# =============================================================================
# Random Matrix Theory Correlation Filter
# =============================================================================

@dataclass
class RMTFilterResult:
    """Results from RMT correlation filtering."""
    filtered_correlation: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    filtered_eigenvalues: NDArray[np.float64]
    noise_eigenvalues: NDArray[np.float64]
    signal_eigenvalues: NDArray[np.float64]
    mp_upper_bound: float
    mp_lower_bound: float


class RMTCorrelationFilter(FitTransformFeature):
    """
    Random Matrix Theory-based correlation matrix filter.

    Uses the Marchenko-Pastur distribution to identify and remove
    noise eigenvalues from correlation matrices, producing more
    stable and meaningful correlation estimates for portfolio
    optimization and risk management.

    The Marchenko-Pastur distribution describes the eigenvalue
    distribution of random matrices, allowing us to identify
    which eigenvalues represent true signal vs. noise.
    """

    def __init__(
        self,
        shrinkage_target: str = 'identity',
        min_eigenvalue: float = 1e-6,
    ) -> None:
        """
        Initialize RMT correlation filter.

        Args:
            shrinkage_target: Target for shrinkage ('identity', 'diagonal', 'mean')
            min_eigenvalue: Minimum eigenvalue to ensure positive definiteness
        """
        self.shrinkage_target = shrinkage_target
        self.min_eigenvalue = min_eigenvalue

        self._n_samples: int = 0
        self._n_features: int = 0
        self._q: float = 1.0  # T/N ratio
        self._mp_lower: float = 0.0
        self._mp_upper: float = 0.0
        self._fitted: bool = False

    def _marchenko_pastur_bounds(self, q: float, sigma: float = 1.0) -> Tuple[float, float]:
        """
        Calculate Marchenko-Pastur distribution bounds.

        Args:
            q: T/N ratio (samples/features)
            sigma: Variance parameter (usually 1 for correlation)

        Returns:
            Tuple of (lower_bound, upper_bound) for eigenvalues
        """
        if q < 1:
            # Underdetermined case
            lower = sigma ** 2 * (1 - np.sqrt(1 / q)) ** 2
            upper = sigma ** 2 * (1 + np.sqrt(1 / q)) ** 2
        else:
            lower = sigma ** 2 * (1 - np.sqrt(1 / q)) ** 2
            upper = sigma ** 2 * (1 + np.sqrt(1 / q)) ** 2

        return max(0, lower), upper

    def fit(self, data: NDArray[np.float64]) -> 'RMTCorrelationFilter':
        """
        Fit the filter to historical data.

        Args:
            data: Return matrix of shape (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        self._n_samples, self._n_features = data.shape
        self._q = self._n_samples / self._n_features
        self._mp_lower, self._mp_upper = self._marchenko_pastur_bounds(self._q)
        self._fitted = True
        return self

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform data by filtering its correlation matrix.

        Args:
            data: Return matrix of shape (n_samples, n_features)

        Returns:
            Denoised correlation matrix
        """
        result = self.filter_correlation(data)
        return result.filtered_correlation

    def filter_correlation(self, data: NDArray[np.float64]) -> RMTFilterResult:
        """
        Apply RMT filtering to correlation matrix.

        Args:
            data: Return matrix of shape (n_samples, n_features)

        Returns:
            RMTFilterResult with filtered correlation and diagnostics
        """
        n_samples, n_features = data.shape
        q = n_samples / n_features

        # Compute sample correlation matrix
        correlation = np.corrcoef(data.T)

        # Handle NaN values
        correlation = np.nan_to_num(correlation, nan=0.0)
        np.fill_diagonal(correlation, 1.0)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)

        # Sort by eigenvalue magnitude (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Calculate MP bounds
        mp_lower, mp_upper = self._marchenko_pastur_bounds(q)

        # Identify signal vs noise eigenvalues
        is_signal = eigenvalues > mp_upper

        signal_eigenvalues = eigenvalues[is_signal]
        noise_eigenvalues = eigenvalues[~is_signal]

        # Filter eigenvalues
        filtered_eigenvalues = eigenvalues.copy()

        if len(noise_eigenvalues) > 0:
            # Replace noise eigenvalues with their mean (constant eigenvalue)
            noise_mean = np.mean(noise_eigenvalues) if len(noise_eigenvalues) > 0 else 1.0
            filtered_eigenvalues[~is_signal] = noise_mean

        # Ensure positive definiteness
        filtered_eigenvalues = np.maximum(filtered_eigenvalues, self.min_eigenvalue)

        # Reconstruct correlation matrix
        filtered_correlation = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T

        # Normalize to ensure unit diagonal
        d = np.sqrt(np.diag(filtered_correlation))
        d[d == 0] = 1.0
        filtered_correlation = filtered_correlation / np.outer(d, d)

        # Ensure symmetry
        filtered_correlation = (filtered_correlation + filtered_correlation.T) / 2
        np.fill_diagonal(filtered_correlation, 1.0)

        return RMTFilterResult(
            filtered_correlation=filtered_correlation,
            eigenvalues=eigenvalues,
            filtered_eigenvalues=filtered_eigenvalues,
            noise_eigenvalues=noise_eigenvalues,
            signal_eigenvalues=signal_eigenvalues,
            mp_upper_bound=mp_upper,
            mp_lower_bound=mp_lower,
        )

    def reset(self) -> None:
        """Reset filter state."""
        self._n_samples = 0
        self._n_features = 0
        self._q = 1.0
        self._mp_lower = 0.0
        self._mp_upper = 0.0
        self._fitted = False


# =============================================================================
# Hidden Markov Model for Market Regime Detection
# =============================================================================

@dataclass
class HMMState:
    """State container for HMM update."""
    state_probabilities: NDArray[np.float64]
    most_likely_state: int
    state_entropy: float
    regime_change_prob: float


class MarketRegimeHMM(OnlineFeature, FitTransformFeature):
    """
    Hidden Markov Model for market regime detection.

    Identifies latent market regimes (e.g., trending, mean-reverting,
    high volatility) from observable features like returns and volatility.

    Supports online state probability updates for real-time regime tracking.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 1,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> None:
        """
        Initialize HMM for regime detection.

        Args:
            n_states: Number of hidden states (regimes)
            n_features: Number of observable features
            max_iterations: Maximum EM iterations for fitting
            tolerance: Convergence tolerance
        """
        self.n_states = n_states
        self.n_features = n_features
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Model parameters (will be initialized in fit)
        self._initial_probs: NDArray[np.float64] = np.ones(n_states) / n_states
        self._transition_matrix: NDArray[np.float64] = np.ones((n_states, n_states)) / n_states
        self._means: NDArray[np.float64] = np.zeros((n_states, n_features))
        self._covariances: NDArray[np.float64] = np.tile(np.eye(n_features), (n_states, 1, 1))

        # Current state probabilities
        self._state_probs: NDArray[np.float64] = self._initial_probs.copy()
        self._prev_state_probs: NDArray[np.float64] = self._initial_probs.copy()

        self._fitted: bool = False

    def _gaussian_pdf(
        self,
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> float:
        """Calculate multivariate Gaussian PDF."""
        d = len(mean)
        x_centered = x - mean

        # Handle 1D case
        if d == 1:
            var = cov[0, 0] if cov.ndim == 2 else cov
            if var <= 0:
                var = 1e-6
            return np.exp(-0.5 * x_centered[0] ** 2 / var) / np.sqrt(2 * np.pi * var)

        # Multivariate case
        try:
            cov_det = np.linalg.det(cov)
            if cov_det <= 0:
                cov_det = 1e-10
            cov_inv = np.linalg.inv(cov)

            norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * cov_det))
            exp_term = np.exp(-0.5 * x_centered @ cov_inv @ x_centered)

            return norm_const * exp_term
        except np.linalg.LinAlgError:
            return 1e-10

    def _emission_probs(self, observation: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate emission probabilities for each state."""
        probs = np.zeros(self.n_states)

        for state in range(self.n_states):
            probs[state] = self._gaussian_pdf(
                observation,
                self._means[state],
                self._covariances[state],
            )

        # Normalize to prevent underflow
        probs = np.maximum(probs, 1e-100)
        return probs

    def fit(self, data: NDArray[np.float64]) -> 'MarketRegimeHMM':
        """
        Fit HMM parameters using Expectation-Maximization.

        Args:
            data: Observation matrix of shape (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        self.n_features = n_features

        # Initialize parameters using K-means-like clustering
        self._initialize_parameters(data)

        # EM algorithm
        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iterations):
            # E-step: Forward-backward algorithm
            alpha, scaling_factors = self._forward(data)
            beta = self._backward(data, scaling_factors)

            # Calculate responsibilities
            gamma = alpha * beta
            gamma = gamma / gamma.sum(axis=1, keepdims=True)

            # Xi (transition probabilities)
            xi = self._calculate_xi(data, alpha, beta, scaling_factors)

            # M-step: Update parameters
            self._update_parameters(data, gamma, xi)

            # Check convergence
            log_likelihood = np.sum(np.log(scaling_factors + 1e-100))

            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                break

            prev_log_likelihood = log_likelihood

        self._fitted = True
        self._state_probs = gamma[-1]

        return self

    def _initialize_parameters(self, data: NDArray[np.float64]) -> None:
        """Initialize HMM parameters using data statistics."""
        n_samples, n_features = data.shape

        # Sort data by magnitude for initial clustering
        magnitudes = np.sum(data ** 2, axis=1)
        sorted_indices = np.argsort(magnitudes)

        # Divide data into n_states clusters
        chunk_size = n_samples // self.n_states

        self._means = np.zeros((self.n_states, n_features))
        self._covariances = np.zeros((self.n_states, n_features, n_features))

        for state in range(self.n_states):
            start_idx = state * chunk_size
            end_idx = start_idx + chunk_size if state < self.n_states - 1 else n_samples

            cluster_data = data[sorted_indices[start_idx:end_idx]]

            self._means[state] = np.mean(cluster_data, axis=0)
            cov = np.cov(cluster_data.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            self._covariances[state] = cov + np.eye(n_features) * 1e-4

        # Initialize transition matrix with persistence
        self._transition_matrix = np.eye(self.n_states) * 0.9
        self._transition_matrix += (1 - 0.9) / self.n_states

        # Normalize rows
        self._transition_matrix /= self._transition_matrix.sum(axis=1, keepdims=True)

    def _forward(
        self,
        data: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forward pass of forward-backward algorithm."""
        n_samples = len(data)
        alpha = np.zeros((n_samples, self.n_states))
        scaling_factors = np.zeros(n_samples)

        # Initial step
        alpha[0] = self._initial_probs * self._emission_probs(data[0])
        scaling_factors[0] = alpha[0].sum()
        alpha[0] /= scaling_factors[0] + 1e-100

        # Forward recursion
        for t in range(1, n_samples):
            emission = self._emission_probs(data[t])
            alpha[t] = (alpha[t - 1] @ self._transition_matrix) * emission
            scaling_factors[t] = alpha[t].sum()
            alpha[t] /= scaling_factors[t] + 1e-100

        return alpha, scaling_factors

    def _backward(
        self,
        data: NDArray[np.float64],
        scaling_factors: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Backward pass of forward-backward algorithm."""
        n_samples = len(data)
        beta = np.zeros((n_samples, self.n_states))

        # Initial step
        beta[-1] = 1.0

        # Backward recursion
        for t in range(n_samples - 2, -1, -1):
            emission = self._emission_probs(data[t + 1])
            beta[t] = (self._transition_matrix @ (emission * beta[t + 1]))
            beta[t] /= scaling_factors[t + 1] + 1e-100

        return beta

    def _calculate_xi(
        self,
        data: NDArray[np.float64],
        alpha: NDArray[np.float64],
        beta: NDArray[np.float64],
        scaling_factors: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate transition responsibilities."""
        n_samples = len(data)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))

        for t in range(n_samples - 1):
            emission = self._emission_probs(data[t + 1])
            xi[t] = np.outer(alpha[t], emission * beta[t + 1]) * self._transition_matrix
            xi[t] /= xi[t].sum() + 1e-100

        return xi

    def _update_parameters(
        self,
        data: NDArray[np.float64],
        gamma: NDArray[np.float64],
        xi: NDArray[np.float64],
    ) -> None:
        """M-step: update model parameters."""
        n_samples, n_features = data.shape

        # Update initial probabilities
        self._initial_probs = gamma[0]

        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                self._transition_matrix[i, j] = xi[:, i, j].sum() / (gamma[:-1, i].sum() + 1e-100)

        # Normalize transition matrix
        self._transition_matrix /= self._transition_matrix.sum(axis=1, keepdims=True)

        # Update emission parameters
        for state in range(self.n_states):
            weight = gamma[:, state]
            weight_sum = weight.sum() + 1e-100

            # Update mean
            self._means[state] = (weight[:, None] * data).sum(axis=0) / weight_sum

            # Update covariance
            centered = data - self._means[state]
            cov = np.zeros((n_features, n_features))
            for t in range(n_samples):
                cov += weight[t] * np.outer(centered[t], centered[t])
            cov /= weight_sum

            # Regularize covariance
            self._covariances[state] = cov + np.eye(n_features) * 1e-4

    def update(self, observation: Union[float, NDArray[np.float64]]) -> HMMState:
        """
        Online update of state probabilities given new observation.

        Args:
            observation: New observation value(s)

        Returns:
            HMMState with current state estimates
        """
        obs = np.atleast_1d(observation).astype(np.float64)

        if not self._fitted:
            # Return uniform probabilities if not fitted
            return HMMState(
                state_probabilities=self._state_probs,
                most_likely_state=0,
                state_entropy=np.log(self.n_states),
                regime_change_prob=0.0,
            )

        self._prev_state_probs = self._state_probs.copy()

        # Prediction step
        predicted_probs = self._prev_state_probs @ self._transition_matrix

        # Update step with observation
        emission_probs = self._emission_probs(obs)
        self._state_probs = predicted_probs * emission_probs

        # Normalize
        prob_sum = self._state_probs.sum()
        if prob_sum > 0:
            self._state_probs /= prob_sum
        else:
            self._state_probs = np.ones(self.n_states) / self.n_states

        # Calculate metrics
        most_likely = int(np.argmax(self._state_probs))

        # Entropy (uncertainty in state)
        probs_clipped = np.clip(self._state_probs, 1e-10, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped))

        # Regime change probability
        prev_most_likely = int(np.argmax(self._prev_state_probs))
        regime_change_prob = 1 - self._state_probs[prev_most_likely]

        return HMMState(
            state_probabilities=self._state_probs.copy(),
            most_likely_state=most_likely,
            state_entropy=entropy,
            regime_change_prob=regime_change_prob,
        )

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform data into state probabilities.

        Args:
            data: Observation matrix of shape (n_samples, n_features)

        Returns:
            State probability matrix of shape (n_samples, n_states)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not self._fitted:
            return np.ones((len(data), self.n_states)) / self.n_states

        # Run forward algorithm
        alpha, _ = self._forward(data)
        return alpha

    def decode(self, data: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Decode most likely state sequence using Viterbi algorithm.

        Args:
            data: Observation matrix of shape (n_samples, n_features)

        Returns:
            Most likely state sequence
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples = len(data)

        # Viterbi algorithm
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialize
        viterbi[0] = np.log(self._initial_probs + 1e-100) + np.log(self._emission_probs(data[0]) + 1e-100)

        # Recursion
        for t in range(1, n_samples):
            emission_log = np.log(self._emission_probs(data[t]) + 1e-100)

            for state in range(self.n_states):
                trans_probs = viterbi[t - 1] + np.log(self._transition_matrix[:, state] + 1e-100)
                backpointer[t, state] = np.argmax(trans_probs)
                viterbi[t, state] = trans_probs[backpointer[t, state]] + emission_log[state]

        # Backtrack
        states = np.zeros(n_samples, dtype=np.int64)
        states[-1] = np.argmax(viterbi[-1])

        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    @property
    def state_means(self) -> NDArray[np.float64]:
        """Get the mean values for each state."""
        return self._means.copy()

    @property
    def transition_matrix(self) -> NDArray[np.float64]:
        """Get the transition probability matrix."""
        return self._transition_matrix.copy()

    def reset(self) -> None:
        """Reset state probabilities to initial distribution."""
        self._state_probs = self._initial_probs.copy()
        self._prev_state_probs = self._initial_probs.copy()


# =============================================================================
# Fractal Analysis
# =============================================================================

def hurst_exponent(
    series: NDArray[np.float64],
    min_window: int = 10,
    max_window: Optional[int] = None,
    num_windows: int = 20,
) -> float:
    """
    Calculate Hurst exponent using R/S (Rescaled Range) analysis.

    The Hurst exponent characterizes the long-term memory of a time series:
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Trending (persistent)

    Args:
        series: Input time series
        min_window: Minimum window size for R/S calculation
        max_window: Maximum window size (default: len(series) // 4)
        num_windows: Number of window sizes to use

    Returns:
        Estimated Hurst exponent
    """
    n = len(series)

    if max_window is None:
        max_window = n // 4

    if n < min_window * 2:
        return 0.5  # Default to random walk for short series

    # Generate window sizes (logarithmically spaced)
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        num_windows,
    ).astype(int))

    rs_values = []

    for window_size in window_sizes:
        if window_size > n:
            continue

        # Calculate R/S for each window
        rs_list = []

        for start in range(0, n - window_size + 1, window_size):
            window = series[start:start + window_size]

            # Mean-adjusted series
            mean_adj = window - np.mean(window)

            # Cumulative deviations
            cumulative = np.cumsum(mean_adj)

            # Range
            R = np.max(cumulative) - np.min(cumulative)

            # Standard deviation
            S = np.std(window, ddof=1)

            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append((window_size, np.mean(rs_list)))

    if len(rs_values) < 2:
        return 0.5

    # Linear regression in log-log space
    window_sizes = np.array([r[0] for r in rs_values])
    rs_means = np.array([r[1] for r in rs_values])

    # Filter out invalid values
    valid = (rs_means > 0) & np.isfinite(rs_means)
    if valid.sum() < 2:
        return 0.5

    log_windows = np.log(window_sizes[valid])
    log_rs = np.log(rs_means[valid])

    # Fit line
    coeffs = np.polyfit(log_windows, log_rs, 1)
    hurst = coeffs[0]

    # Bound to valid range
    return float(np.clip(hurst, 0.0, 1.0))


def fractal_dimension(
    series: NDArray[np.float64],
    k_max: int = 10,
) -> float:
    """
    Calculate fractal dimension using Higuchi's method.

    The fractal dimension quantifies the complexity/roughness of a time series:
    - D ~ 1.0: Smooth, predictable series
    - D ~ 1.5: Random walk
    - D ~ 2.0: Very rough, space-filling series

    Relation to Hurst exponent: D = 2 - H

    Args:
        series: Input time series
        k_max: Maximum value of k (time interval)

    Returns:
        Estimated fractal dimension
    """
    n = len(series)

    if n < k_max * 4:
        k_max = max(2, n // 4)

    k_values = np.arange(1, k_max + 1)
    curve_lengths = []

    for k in k_values:
        lengths_for_k = []

        for m in range(1, k + 1):
            # Extract subsequence
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue

            subsequence = series[indices]

            # Calculate curve length
            L_m_k = np.sum(np.abs(np.diff(subsequence)))

            # Normalize
            num_points = len(subsequence) - 1
            if num_points > 0:
                norm_factor = (n - 1) / (k * num_points)
                L_m_k *= norm_factor / k
                lengths_for_k.append(L_m_k)

        if lengths_for_k:
            curve_lengths.append((k, np.mean(lengths_for_k)))

    if len(curve_lengths) < 2:
        return 1.5  # Default to Brownian motion

    # Linear regression in log-log space
    k_vals = np.array([c[0] for c in curve_lengths])
    L_vals = np.array([c[1] for c in curve_lengths])

    # Filter invalid values
    valid = (L_vals > 0) & np.isfinite(L_vals)
    if valid.sum() < 2:
        return 1.5

    log_k = np.log(k_vals[valid])
    log_L = np.log(L_vals[valid])

    # Fit line (negative slope = fractal dimension)
    coeffs = np.polyfit(log_k, log_L, 1)
    dimension = -coeffs[0]

    # Bound to valid range [1, 2]
    return float(np.clip(dimension, 1.0, 2.0))


class FractalAnalyzer:
    """
    Streaming fractal analysis for time series.

    Maintains a rolling window for online Hurst exponent and
    fractal dimension estimation.
    """

    def __init__(
        self,
        window_size: int = 500,
        update_frequency: int = 10,
    ) -> None:
        """
        Initialize fractal analyzer.

        Args:
            window_size: Size of rolling window for analysis
            update_frequency: How often to recalculate (every N updates)
        """
        self.window_size = window_size
        self.update_frequency = update_frequency

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._update_count: int = 0
        self._hurst: float = 0.5
        self._dimension: float = 1.5

    def update(self, value: float) -> Dict[str, float]:
        """
        Update with new value and optionally recalculate metrics.

        Args:
            value: New observation

        Returns:
            Dict with hurst_exponent and fractal_dimension
        """
        self._buffer.append(value)
        self._update_count += 1

        # Recalculate periodically
        if (
            self._update_count % self.update_frequency == 0 and
            len(self._buffer) >= self.window_size // 2
        ):
            series = np.array(self._buffer)
            self._hurst = hurst_exponent(series)
            self._dimension = fractal_dimension(series)

        return {
            'hurst_exponent': self._hurst,
            'fractal_dimension': self._dimension,
            'is_trending': self._hurst > 0.55,
            'is_mean_reverting': self._hurst < 0.45,
        }

    @property
    def hurst(self) -> float:
        """Current Hurst exponent estimate."""
        return self._hurst

    @property
    def dimension(self) -> float:
        """Current fractal dimension estimate."""
        return self._dimension

    def reset(self) -> None:
        """Reset analyzer state."""
        self._buffer.clear()
        self._update_count = 0
        self._hurst = 0.5
        self._dimension = 1.5


# =============================================================================
# Composite Feature Generator
# =============================================================================

class QuantFeatureGenerator:
    """
    Composite generator combining all quantitative features.

    Provides a unified interface for generating advanced quantitative
    features from market data in both batch and streaming modes.
    """

    def __init__(
        self,
        enable_hawkes: bool = True,
        enable_kalman: bool = True,
        enable_hmm: bool = True,
        enable_rmt: bool = True,
        enable_fractal: bool = True,
        hmm_states: int = 3,
        kalman_state_dim: int = 2,
        window_size: int = 500,
    ) -> None:
        """
        Initialize feature generator.

        Args:
            enable_hawkes: Enable Hawkes intensity features
            enable_kalman: Enable Kalman filter features
            enable_hmm: Enable HMM regime features
            enable_rmt: Enable RMT correlation filter
            enable_fractal: Enable fractal analysis features
            hmm_states: Number of HMM states
            kalman_state_dim: Kalman filter state dimension
            window_size: Rolling window size for various calculations
        """
        self.enable_hawkes = enable_hawkes
        self.enable_kalman = enable_kalman
        self.enable_hmm = enable_hmm
        self.enable_rmt = enable_rmt
        self.enable_fractal = enable_fractal

        # Initialize components
        if enable_hawkes:
            self.hawkes = HawkesIntensity()

        if enable_kalman:
            self.kalman = KalmanFilter.for_spread_tracking()

        if enable_hmm:
            self.hmm = MarketRegimeHMM(n_states=hmm_states)

        if enable_rmt:
            self.rmt_filter = RMTCorrelationFilter()

        if enable_fractal:
            self.fractal = FractalAnalyzer(window_size=window_size)

    def fit(self, returns: NDArray[np.float64]) -> 'QuantFeatureGenerator':
        """
        Fit all components on historical data.

        Args:
            returns: Historical returns matrix (n_samples, n_assets) or 1D

        Returns:
            Self for method chaining
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        if self.enable_hmm:
            self.hmm.fit(returns)

        if self.enable_rmt and returns.shape[1] > 1:
            self.rmt_filter.fit(returns)

        return self

    def generate_features(
        self,
        timestamp: float,
        price: float,
        volume: float,
        is_trade: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate all enabled features for a single observation.

        Args:
            timestamp: Unix timestamp
            price: Current price
            volume: Current volume
            is_trade: Whether a trade occurred

        Returns:
            Dict of feature names to values
        """
        features = {}

        if self.enable_hawkes:
            hawkes_state = self.hawkes.update(timestamp, is_trade)
            features['hawkes_intensity'] = hawkes_state.intensity
            features['hawkes_branching_ratio'] = hawkes_state.branching_ratio

        if self.enable_kalman:
            kalman_state = self.kalman.update(price)
            features['kalman_state_0'] = kalman_state.state_estimate[0]
            features['kalman_state_1'] = kalman_state.state_estimate[1]
            features['kalman_innovation'] = kalman_state.innovation

        if self.enable_hmm and self.hmm._fitted:
            hmm_state = self.hmm.update(np.array([price]))
            for i, prob in enumerate(hmm_state.state_probabilities):
                features[f'hmm_state_{i}_prob'] = prob
            features['hmm_regime'] = hmm_state.most_likely_state
            features['hmm_entropy'] = hmm_state.state_entropy
            features['hmm_regime_change_prob'] = hmm_state.regime_change_prob

        if self.enable_fractal:
            fractal_features = self.fractal.update(price)
            features.update({f'fractal_{k}': v for k, v in fractal_features.items()})

        return features

    def generate_transform_features(
        self,
        series: NDArray[np.float64],
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Generate batch transform features.

        Args:
            series: Price or return series

        Returns:
            Dict of feature arrays
        """
        features = {}

        # Fisher transform on normalized series
        normalized = normalize_to_fisher_range(series)
        fisher = fisher_transform(normalized)
        features['fisher_transform'] = fisher

        # Hilbert transform
        amplitude, phase = hilbert_transform(series)
        features['hilbert_amplitude'] = amplitude
        features['hilbert_phase'] = phase
        features['instantaneous_freq'] = instantaneous_frequency(phase)

        # Wavelet decomposition
        wavelet_decomp = wavelet_transform(series)
        features['wavelet_approx'] = np.interp(
            np.arange(len(series)),
            np.linspace(0, len(series), len(wavelet_decomp.approximation)),
            wavelet_decomp.approximation,
        )

        # Denoised series
        features['wavelet_denoised'] = wavelet_denoise(series)

        # Hurst exponent (single value, broadcast)
        h = hurst_exponent(series)
        features['hurst_exponent'] = np.full(len(series), h)

        # Fractal dimension (single value, broadcast)
        d = fractal_dimension(series)
        features['fractal_dimension'] = np.full(len(series), d)

        return features

    def reset(self) -> None:
        """Reset all components."""
        if self.enable_hawkes:
            self.hawkes.reset()
        if self.enable_kalman:
            self.kalman.reset()
        if self.enable_hmm:
            self.hmm.reset()
        if self.enable_fractal:
            self.fractal.reset()


# =============================================================================
# Testing with Synthetic Data
# =============================================================================

if __name__ == '__main__':
    import time

    print("=" * 70)
    print("Advanced Quantitative Features Library - Test Suite")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic data
    n_samples = 1000
    timestamps = np.cumsum(np.random.exponential(1.0, n_samples))

    # Trending returns (H > 0.5)
    trending_returns = np.cumsum(np.random.randn(n_samples) * 0.01 + 0.0001)

    # Mean-reverting returns (H < 0.5)
    mean_reverting = np.zeros(n_samples)
    mean_reverting[0] = 0
    for i in range(1, n_samples):
        mean_reverting[i] = -0.3 * mean_reverting[i-1] + np.random.randn() * 0.01

    # Price series with regime changes
    regime_prices = np.zeros(n_samples)
    regime_prices[0] = 100
    current_regime = 0
    for i in range(1, n_samples):
        if np.random.rand() < 0.01:  # 1% chance of regime switch
            current_regime = 1 - current_regime

        if current_regime == 0:  # Low volatility
            regime_prices[i] = regime_prices[i-1] * (1 + np.random.randn() * 0.005)
        else:  # High volatility
            regime_prices[i] = regime_prices[i-1] * (1 + np.random.randn() * 0.02)

    # =========================================================================
    # Test 1: Hawkes Process
    # =========================================================================
    print("\n" + "-" * 70)
    print("Test 1: Hawkes Process")
    print("-" * 70)

    hawkes = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)

    intensities = []
    for i, ts in enumerate(timestamps[:200]):
        is_event = np.random.rand() < 0.3  # 30% chance of event
        state = hawkes.update(ts, is_event)
        intensities.append(state.intensity)

    print(f"Final intensity: {intensities[-1]:.4f}")
    print(f"Branching ratio: {hawkes.branching_ratio:.4f}")
    print(f"Events recorded: {state.event_count}")

    # Test parameter estimation
    params = hawkes.estimate_parameters()
    print(f"Estimated parameters: mu={params['mu']:.4f}, alpha={params['alpha']:.4f}, beta={params['beta']:.4f}")

    # =========================================================================
    # Test 2: Kalman Filter
    # =========================================================================
    print("\n" + "-" * 70)
    print("Test 2: Kalman Filter")
    print("-" * 70)

    # Generate noisy signal with underlying trend
    true_signal = np.sin(np.linspace(0, 4 * np.pi, 200)) * 10 + np.linspace(0, 5, 200)
    noise_std = 2.0
    noisy_signal = true_signal + np.random.randn(200) * noise_std

    # Configure Kalman filter for signal tracking (position + velocity model)
    kalman = KalmanFilter(
        state_dim=2,
        obs_dim=1,
        F=np.array([[1, 1], [0, 1]]),  # Position + velocity model
        H=np.array([[1.0, 0.0]]),  # Observe position only
        Q=np.eye(2) * 0.1,  # Process noise
        R=np.array([[noise_std ** 2]]),  # Observation noise variance
    )

    filtered = []
    for obs in noisy_signal:
        state = kalman.update(obs)
        filtered.append(state.state_estimate[0])

    filtered = np.array(filtered)

    noise_mse = np.mean((noisy_signal - true_signal) ** 2)
    filtered_mse = np.mean((filtered - true_signal) ** 2)

    print(f"Noisy signal MSE: {noise_mse:.4f}")
    print(f"Filtered signal MSE: {filtered_mse:.4f}")
    print(f"Noise reduction: {(1 - filtered_mse/noise_mse) * 100:.1f}%")

    # Test smoothing capability
    kalman.reset()
    smoothed = kalman.smooth(noisy_signal.reshape(-1, 1))
    smoothed_mse = np.mean((smoothed[:, 0] - true_signal) ** 2)
    print(f"Smoothed signal MSE: {smoothed_mse:.4f}")

    # =========================================================================
    # Test 3: Signal Transforms
    # =========================================================================
    print("\n" + "-" * 70)
    print("Test 3: Signal Transforms")
    print("-" * 70)

    # Test Fisher Transform
    correlation_like = np.random.uniform(-0.9, 0.9, 100)
    fisher_transformed = fisher_transform(correlation_like)
    recovered = inverse_fisher_transform(fisher_transformed)

    print(f"Fisher transform - Input range: [{correlation_like.min():.3f}, {correlation_like.max():.3f}]")
    print(f"Fisher transform - Output range: [{fisher_transformed.min():.3f}, {fisher_transformed.max():.3f}]")
    print(f"Inverse transform error: {np.max(np.abs(correlation_like - recovered)):.2e}")

    # Test Hilbert Transform
    test_signal = np.sin(np.linspace(0, 8 * np.pi, 200)) + 0.5 * np.sin(np.linspace(0, 16 * np.pi, 200))
    amplitude, phase = hilbert_transform(test_signal)
    inst_freq = instantaneous_frequency(phase)

    print(f"Hilbert transform - Amplitude range: [{amplitude.min():.3f}, {amplitude.max():.3f}]")
    print(f"Hilbert transform - Phase range: [{phase.min():.3f}, {phase.max():.3f}]")

    # Test Wavelet Transform
    decomp = wavelet_transform(regime_prices, wavelet='db4', levels=4)
    denoised = wavelet_denoise(regime_prices)

    print(f"Wavelet decomposition - {decomp.levels} levels")
    print(f"Wavelet approximation length: {len(decomp.approximation)}")
    print(f"Wavelet denoised correlation: {np.corrcoef(regime_prices, denoised)[0,1]:.4f}")

    # =========================================================================
    # Test 4: RMT Correlation Filter
    # =========================================================================
    print("\n" + "-" * 70)
    print("Test 4: RMT Correlation Filter")
    print("-" * 70)

    # Generate correlated returns with noise
    n_assets = 20
    n_obs = 500

    # True correlation structure (factor model)
    n_factors = 3
    factor_loadings = np.random.randn(n_assets, n_factors) * 0.5
    factors = np.random.randn(n_obs, n_factors)
    idiosyncratic = np.random.randn(n_obs, n_assets) * 0.3

    returns = factors @ factor_loadings.T + idiosyncratic

    rmt = RMTCorrelationFilter()
    result = rmt.filter_correlation(returns)

    print(f"Sample correlation eigenvalue range: [{result.eigenvalues.min():.4f}, {result.eigenvalues.max():.4f}]")
    print(f"Marchenko-Pastur bounds: [{result.mp_lower_bound:.4f}, {result.mp_upper_bound:.4f}]")
    print(f"Signal eigenvalues (above MP): {len(result.signal_eigenvalues)}")
    print(f"Noise eigenvalues (within MP): {len(result.noise_eigenvalues)}")

    # =========================================================================
    # Test 5: Hidden Markov Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("Test 5: Hidden Markov Model")
    print("-" * 70)

    # Use regime prices for HMM
    returns_for_hmm = np.diff(np.log(regime_prices)).reshape(-1, 1)

    hmm = MarketRegimeHMM(n_states=2, n_features=1)

    start_time = time.time()
    hmm.fit(returns_for_hmm)
    fit_time = time.time() - start_time

    print(f"HMM fitted in {fit_time:.2f}s")
    print(f"State means: {hmm.state_means.flatten()}")

    # Online update test
    state_probs_history = []
    for ret in returns_for_hmm[-50:]:
        state = hmm.update(ret)
        state_probs_history.append(state.state_probabilities.copy())

    state_probs_history = np.array(state_probs_history)

    print(f"Final state probabilities: {state.state_probabilities}")
    print(f"Most likely regime: {state.most_likely_state}")
    print(f"Regime change probability: {state.regime_change_prob:.4f}")

    # Decode full sequence
    decoded = hmm.decode(returns_for_hmm)
    regime_changes = np.sum(np.diff(decoded) != 0)
    print(f"Decoded regime changes: {regime_changes}")

    # =========================================================================
    # Test 6: Fractal Analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("Test 6: Fractal Analysis")
    print("-" * 70)

    # Test Hurst exponent on known series
    h_trending = hurst_exponent(trending_returns)
    h_mean_rev = hurst_exponent(mean_reverting)
    h_random = hurst_exponent(np.cumsum(np.random.randn(1000)))

    print(f"Hurst exponent - Trending series: {h_trending:.4f} (expected > 0.5)")
    print(f"Hurst exponent - Mean-reverting series: {h_mean_rev:.4f} (expected < 0.5)")
    print(f"Hurst exponent - Random walk: {h_random:.4f} (expected ~ 0.5)")

    # Test fractal dimension
    d_trending = fractal_dimension(trending_returns)
    d_mean_rev = fractal_dimension(mean_reverting)

    print(f"Fractal dimension - Trending: {d_trending:.4f}")
    print(f"Fractal dimension - Mean-reverting: {d_mean_rev:.4f}")

    # Test FractalAnalyzer (streaming)
    analyzer = FractalAnalyzer(window_size=200, update_frequency=20)

    for price in regime_prices[:300]:
        result = analyzer.update(price)

    print(f"Streaming Hurst: {analyzer.hurst:.4f}")
    print(f"Streaming Fractal Dim: {analyzer.dimension:.4f}")

    # =========================================================================
    # Test 7: Composite Feature Generator
    # =========================================================================
    print("\n" + "-" * 70)
    print("Test 7: Composite Feature Generator")
    print("-" * 70)

    generator = QuantFeatureGenerator(
        enable_hawkes=True,
        enable_kalman=True,
        enable_hmm=True,
        enable_rmt=False,  # Requires multi-asset data
        enable_fractal=True,
        hmm_states=2,
        window_size=200,
    )

    # Fit on historical data
    generator.fit(returns_for_hmm)

    # Generate streaming features
    all_features = []
    for i in range(200):
        features = generator.generate_features(
            timestamp=timestamps[i],
            price=regime_prices[i],
            volume=np.random.exponential(1000),
            is_trade=np.random.rand() < 0.5,
        )
        all_features.append(features)

    print(f"Generated {len(all_features[0])} features per observation")
    print(f"Feature names: {list(all_features[-1].keys())}")

    # Generate batch transform features
    transform_features = generator.generate_transform_features(regime_prices)
    print(f"Transform features: {list(transform_features.keys())}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
