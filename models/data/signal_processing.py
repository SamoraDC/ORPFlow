"""
Signal Processing for Financial Time Series
============================================

A comprehensive module implementing advanced signal processing techniques
for financial time series analysis, including:

- Kalman Filtering (Standard, Extended)
- Empirical Mode Decomposition (EMD, EEMD, CEEMDAN)
- Hilbert-Huang Transform
- Wavelet Analysis (CWT, DWT, Denoising)
- Random Matrix Theory for correlation cleaning
- Fisher Transform for normalization

Author: ORPFlow Team
License: MIT
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Union, Callable, Dict, Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import signal as scipy_signal
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh
from scipy.stats import norm
import pywt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================
ArrayLike = Union[np.ndarray, pd.Series, List[float]]


def _to_array(data: ArrayLike) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(data, pd.Series):
        return data.values.astype(np.float64)
    elif isinstance(data, list):
        return np.array(data, dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


# =============================================================================
# KALMAN FILTER
# =============================================================================
@dataclass
class KalmanState:
    """State container for Kalman Filter."""
    x: np.ndarray  # State estimate
    P: np.ndarray  # Estimate covariance

    def copy(self) -> 'KalmanState':
        """Create a deep copy of the state."""
        return KalmanState(x=self.x.copy(), P=self.P.copy())


class KalmanFilter:
    """
    Kalman Filter for financial time series smoothing and estimation.

    Implements both standard Kalman Filter for linear systems and provides
    foundation for Extended Kalman Filter for non-linear systems.

    The state-space model:
        x(t) = F @ x(t-1) + w(t),  w ~ N(0, Q)
        z(t) = H @ x(t) + v(t),    v ~ N(0, R)

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector
    obs_dim : int
        Dimension of the observation vector
    F : np.ndarray, optional
        State transition matrix (state_dim x state_dim)
    H : np.ndarray, optional
        Observation matrix (obs_dim x state_dim)
    Q : np.ndarray, optional
        Process noise covariance (state_dim x state_dim)
    R : np.ndarray, optional
        Measurement noise covariance (obs_dim x obs_dim)

    Examples
    --------
    >>> # Simple price smoothing with local level model
    >>> prices = np.array([100, 101, 99, 102, 98, 103])
    >>> kf = KalmanFilter.local_level(process_var=0.1, obs_var=1.0)
    >>> smoothed, _, _ = kf.smooth(prices)
    >>> print(smoothed)

    >>> # Trend + noise estimation
    >>> kf = KalmanFilter.local_linear_trend(level_var=0.1, trend_var=0.01, obs_var=1.0)
    >>> smoothed, states, _ = kf.smooth(prices)
    >>> trend = states[:, 1]  # Extract trend component
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int = 1,
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # State transition matrix
        self.F = F if F is not None else np.eye(state_dim)

        # Observation matrix
        self.H = H if H is not None else np.zeros((obs_dim, state_dim))
        self.H[0, 0] = 1.0  # Default: observe first state component

        # Process noise covariance
        self.Q = Q if Q is not None else np.eye(state_dim) * 0.01

        # Measurement noise covariance
        self.R = R if R is not None else np.eye(obs_dim) * 1.0

    @classmethod
    def local_level(cls, process_var: float = 0.1, obs_var: float = 1.0) -> 'KalmanFilter':
        """
        Create a local level model (random walk + noise).

        Model: z(t) = mu(t) + v(t)
               mu(t) = mu(t-1) + w(t)

        Parameters
        ----------
        process_var : float
            Variance of the level change (w)
        obs_var : float
            Variance of the observation noise (v)

        Returns
        -------
        KalmanFilter
            Configured local level filter
        """
        return cls(
            state_dim=1,
            obs_dim=1,
            F=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[process_var]]),
            R=np.array([[obs_var]]),
        )

    @classmethod
    def local_linear_trend(
        cls,
        level_var: float = 0.1,
        trend_var: float = 0.01,
        obs_var: float = 1.0,
    ) -> 'KalmanFilter':
        """
        Create a local linear trend model.

        Model: z(t) = mu(t) + v(t)
               mu(t) = mu(t-1) + nu(t-1) + w1(t)
               nu(t) = nu(t-1) + w2(t)

        Parameters
        ----------
        level_var : float
            Variance of level innovation
        trend_var : float
            Variance of trend innovation
        obs_var : float
            Variance of observation noise

        Returns
        -------
        KalmanFilter
            Configured local linear trend filter
        """
        F = np.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        H = np.array([[1.0, 0.0]])
        Q = np.diag([level_var, trend_var])
        R = np.array([[obs_var]])

        return cls(state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R)

    def predict(self, state: KalmanState) -> KalmanState:
        """
        Prediction step of Kalman Filter.

        Parameters
        ----------
        state : KalmanState
            Current state estimate

        Returns
        -------
        KalmanState
            Predicted state
        """
        x_pred = self.F @ state.x
        P_pred = self.F @ state.P @ self.F.T + self.Q
        return KalmanState(x=x_pred, P=P_pred)

    def update(
        self,
        state: KalmanState,
        z: np.ndarray,
    ) -> Tuple[KalmanState, np.ndarray, np.ndarray]:
        """
        Update step of Kalman Filter.

        Parameters
        ----------
        state : KalmanState
            Predicted state
        z : np.ndarray
            Observation

        Returns
        -------
        Tuple[KalmanState, np.ndarray, np.ndarray]
            Updated state, innovation, Kalman gain
        """
        # Innovation (measurement residual)
        y = z - self.H @ state.x

        # Innovation covariance
        S = self.H @ state.P @ self.H.T + self.R

        # Kalman gain
        K = state.P @ self.H.T @ np.linalg.inv(S)

        # State update
        x_new = state.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        IKH = np.eye(self.state_dim) - K @ self.H
        P_new = IKH @ state.P @ IKH.T + K @ self.R @ K.T

        return KalmanState(x=x_new, P=P_new), y, K

    def filter(
        self,
        observations: ArrayLike,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the Kalman Filter forward pass.

        Parameters
        ----------
        observations : ArrayLike
            Time series of observations (T,) or (T, obs_dim)
        initial_state : np.ndarray, optional
            Initial state estimate
        initial_cov : np.ndarray, optional
            Initial state covariance

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Filtered observations, state estimates, state covariances
        """
        obs = _to_array(observations)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        T = len(obs)

        # Initialize
        x0 = initial_state if initial_state is not None else np.zeros(self.state_dim)
        P0 = initial_cov if initial_cov is not None else np.eye(self.state_dim) * 1e6

        # Storage
        filtered = np.zeros((T, self.obs_dim))
        states = np.zeros((T, self.state_dim))
        covariances = np.zeros((T, self.state_dim, self.state_dim))

        state = KalmanState(x=x0, P=P0)

        for t in range(T):
            # Predict
            state = self.predict(state)

            # Update
            state, _, _ = self.update(state, obs[t])

            # Store
            filtered[t] = self.H @ state.x
            states[t] = state.x
            covariances[t] = state.P

        return filtered.squeeze(), states, covariances

    def smooth(
        self,
        observations: ArrayLike,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the Rauch-Tung-Striebel (RTS) smoother.

        Parameters
        ----------
        observations : ArrayLike
            Time series of observations
        initial_state : np.ndarray, optional
            Initial state estimate
        initial_cov : np.ndarray, optional
            Initial state covariance

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Smoothed observations, state estimates, state covariances
        """
        obs = _to_array(observations)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        T = len(obs)

        # Forward pass (filtering)
        _, states_filt, covs_filt = self.filter(obs, initial_state, initial_cov)

        # Predicted states storage for smoother
        states_pred = np.zeros_like(states_filt)
        covs_pred = np.zeros_like(covs_filt)

        # Recompute predicted states
        x0 = initial_state if initial_state is not None else np.zeros(self.state_dim)
        P0 = initial_cov if initial_cov is not None else np.eye(self.state_dim) * 1e6

        state = KalmanState(x=x0, P=P0)
        for t in range(T):
            pred_state = self.predict(state)
            states_pred[t] = pred_state.x
            covs_pred[t] = pred_state.P
            state, _, _ = self.update(pred_state, obs[t])

        # Backward pass (smoothing)
        states_smooth = states_filt.copy()
        covs_smooth = covs_filt.copy()

        for t in range(T - 2, -1, -1):
            # Smoother gain
            J = covs_filt[t] @ self.F.T @ np.linalg.inv(covs_pred[t + 1])

            # Smoothed state
            states_smooth[t] = states_filt[t] + J @ (states_smooth[t + 1] - states_pred[t + 1])

            # Smoothed covariance
            covs_smooth[t] = covs_filt[t] + J @ (covs_smooth[t + 1] - covs_pred[t + 1]) @ J.T

        # Compute smoothed observations
        smoothed_obs = np.array([self.H @ x for x in states_smooth]).squeeze()

        return smoothed_obs, states_smooth, covs_smooth


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for non-linear state-space models.

    Handles non-linear state transition and observation functions by
    linearizing around the current estimate.

    Model:
        x(t) = f(x(t-1)) + w(t),  w ~ N(0, Q)
        z(t) = h(x(t)) + v(t),    v ~ N(0, R)

    Parameters
    ----------
    state_dim : int
        Dimension of state vector
    obs_dim : int
        Dimension of observation vector
    f : Callable
        Non-linear state transition function f(x) -> x'
    h : Callable
        Non-linear observation function h(x) -> z
    F_jacobian : Callable
        Function returning Jacobian of f at x
    H_jacobian : Callable
        Function returning Jacobian of h at x
    Q : np.ndarray
        Process noise covariance
    R : np.ndarray
        Measurement noise covariance

    Examples
    --------
    >>> # Non-linear price model with mean reversion
    >>> def f(x):
    ...     mu, theta, sigma = 100.0, 0.1, 0.5
    ...     return np.array([x[0] + theta * (mu - x[0])])
    >>> def h(x):
    ...     return x  # Direct observation
    >>> def F_jac(x):
    ...     return np.array([[1 - 0.1]])  # d/dx of f
    >>> def H_jac(x):
    ...     return np.array([[1.0]])
    >>> ekf = ExtendedKalmanFilter(1, 1, f, h, F_jac, H_jac,
    ...                            Q=np.array([[0.25]]), R=np.array([[1.0]]))
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        F_jacobian: Callable[[np.ndarray], np.ndarray],
        H_jacobian: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R

    def predict(self, state: KalmanState) -> KalmanState:
        """Prediction step with non-linear transition."""
        x_pred = self.f(state.x)
        F = self.F_jacobian(state.x)
        P_pred = F @ state.P @ F.T + self.Q
        return KalmanState(x=x_pred, P=P_pred)

    def update(
        self,
        state: KalmanState,
        z: np.ndarray,
    ) -> Tuple[KalmanState, np.ndarray, np.ndarray]:
        """Update step with non-linear observation."""
        # Linearize observation
        H = self.H_jacobian(state.x)

        # Innovation
        y = z - self.h(state.x)

        # Innovation covariance
        S = H @ state.P @ H.T + self.R

        # Kalman gain
        K = state.P @ H.T @ np.linalg.inv(S)

        # State update
        x_new = state.x + K @ y

        # Covariance update
        IKH = np.eye(self.state_dim) - K @ H
        P_new = IKH @ state.P @ IKH.T + K @ self.R @ K.T

        return KalmanState(x=x_new, P=P_new), y, K

    def filter(
        self,
        observations: ArrayLike,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the Extended Kalman Filter."""
        obs = _to_array(observations)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        T = len(obs)

        # Initialize
        x0 = initial_state if initial_state is not None else np.zeros(self.state_dim)
        P0 = initial_cov if initial_cov is not None else np.eye(self.state_dim) * 1e6

        # Storage
        filtered = np.zeros((T, self.obs_dim))
        states = np.zeros((T, self.state_dim))
        covariances = np.zeros((T, self.state_dim, self.state_dim))

        state = KalmanState(x=x0, P=P0)

        for t in range(T):
            state = self.predict(state)
            state, _, _ = self.update(state, obs[t])

            filtered[t] = self.h(state.x)
            states[t] = state.x
            covariances[t] = state.P

        return filtered.squeeze(), states, covariances


# =============================================================================
# EMPIRICAL MODE DECOMPOSITION (EMD)
# =============================================================================
class EMD:
    """
    Empirical Mode Decomposition for adaptive signal analysis.

    Decomposes a signal into Intrinsic Mode Functions (IMFs) through the
    sifting process. IMFs satisfy:
    1. Number of extrema and zero-crossings differ by at most one
    2. Mean of upper and lower envelopes is zero at any point

    Parameters
    ----------
    max_imfs : int
        Maximum number of IMFs to extract
    max_sift_iterations : int
        Maximum iterations in sifting process
    sift_threshold : float
        Stopping criterion for sifting (SD threshold)
    envelope_interp : str
        Interpolation method for envelopes ('cubic', 'linear')

    Examples
    --------
    >>> prices = np.sin(np.linspace(0, 4*np.pi, 200)) + 0.5*np.sin(np.linspace(0, 20*np.pi, 200))
    >>> emd = EMD(max_imfs=5)
    >>> imfs, residue = emd.decompose(prices)
    >>> print(f"Extracted {len(imfs)} IMFs")
    """

    def __init__(
        self,
        max_imfs: int = 10,
        max_sift_iterations: int = 100,
        sift_threshold: float = 0.05,
        envelope_interp: str = 'cubic',
    ):
        self.max_imfs = max_imfs
        self.max_sift_iterations = max_sift_iterations
        self.sift_threshold = sift_threshold
        self.envelope_interp = envelope_interp

    def _find_extrema(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find indices of local maxima and minima."""
        # Find local maxima
        diff = np.diff(x)
        maxima = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1
        minima = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1

        return maxima, minima

    def _interpolate_envelope(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Interpolate envelope through extrema points."""
        if len(indices) < 2:
            return np.zeros_like(x)

        # Mirror extrema at boundaries for better edge handling
        n = len(x)

        # Extend indices and values
        ext_indices = indices.copy()
        ext_values = values.copy()

        # Add boundary points using mirroring
        if indices[0] > 0:
            ext_indices = np.concatenate([[0], ext_indices])
            ext_values = np.concatenate([[values[0]], ext_values])
        if indices[-1] < n - 1:
            ext_indices = np.concatenate([ext_indices, [n - 1]])
            ext_values = np.concatenate([ext_values, [values[-1]]])

        # Interpolate
        if self.envelope_interp == 'cubic' and len(ext_indices) >= 4:
            cs = CubicSpline(ext_indices, ext_values)
            envelope = cs(np.arange(n))
        else:
            envelope = np.interp(np.arange(n), ext_indices, ext_values)

        return envelope

    def _compute_envelopes(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute upper and lower envelopes and their mean."""
        maxima_idx, minima_idx = self._find_extrema(x)

        if len(maxima_idx) < 2 or len(minima_idx) < 2:
            return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

        upper = self._interpolate_envelope(x, maxima_idx, x[maxima_idx])
        lower = self._interpolate_envelope(x, minima_idx, x[minima_idx])
        mean = (upper + lower) / 2

        return upper, lower, mean

    def _sift(self, x: np.ndarray) -> np.ndarray:
        """Sifting process to extract one IMF."""
        h = x.copy()

        for _ in range(self.max_sift_iterations):
            _, _, mean = self._compute_envelopes(h)

            # Standard deviation criterion
            if np.std(h) > 0:
                sd = np.sum((mean ** 2)) / np.sum((h ** 2))
            else:
                sd = 0

            h_new = h - mean

            if sd < self.sift_threshold:
                return h_new

            h = h_new

        return h

    def _is_imf(self, x: np.ndarray) -> bool:
        """Check if signal satisfies IMF conditions."""
        maxima_idx, minima_idx = self._find_extrema(x)
        n_extrema = len(maxima_idx) + len(minima_idx)

        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(x)) != 0)

        # IMF condition: |n_extrema - zero_crossings| <= 1
        return abs(n_extrema - zero_crossings) <= 1

    def _is_monotonic(self, x: np.ndarray) -> bool:
        """Check if signal is monotonic (stopping criterion)."""
        diff = np.diff(x)
        return np.all(diff >= 0) or np.all(diff <= 0)

    def decompose(
        self,
        signal: ArrayLike,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Decompose signal into IMFs.

        Parameters
        ----------
        signal : ArrayLike
            Input signal to decompose

        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            List of IMFs and the residue
        """
        x = _to_array(signal)
        residue = x.copy()
        imfs = []

        for _ in range(self.max_imfs):
            # Check stopping conditions
            maxima_idx, minima_idx = self._find_extrema(residue)
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            if self._is_monotonic(residue):
                break

            # Extract IMF
            imf = self._sift(residue)
            imfs.append(imf)

            # Update residue
            residue = residue - imf

            # Check if residue is monotonic
            if self._is_monotonic(residue):
                break

        return imfs, residue


class EEMD:
    """
    Ensemble Empirical Mode Decomposition.

    Adds white noise to the signal multiple times, performs EMD on each
    noisy signal, and averages the IMFs. This reduces mode mixing.

    Parameters
    ----------
    ensemble_size : int
        Number of ensemble members
    noise_std : float
        Standard deviation of added noise (as fraction of signal std)
    emd_params : dict
        Parameters to pass to EMD

    Examples
    --------
    >>> prices = np.sin(np.linspace(0, 4*np.pi, 200)) + 0.1*np.random.randn(200)
    >>> eemd = EEMD(ensemble_size=100, noise_std=0.2)
    >>> imfs, residue = eemd.decompose(prices)
    """

    def __init__(
        self,
        ensemble_size: int = 100,
        noise_std: float = 0.2,
        emd_params: Optional[Dict[str, Any]] = None,
    ):
        self.ensemble_size = ensemble_size
        self.noise_std = noise_std
        self.emd_params = emd_params or {}

    def decompose(
        self,
        signal: ArrayLike,
        seed: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Decompose signal using EEMD.

        Parameters
        ----------
        signal : ArrayLike
            Input signal
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            Averaged IMFs and residue
        """
        if seed is not None:
            np.random.seed(seed)

        x = _to_array(signal)
        n = len(x)
        signal_std = np.std(x)
        noise_amplitude = self.noise_std * signal_std

        emd = EMD(**self.emd_params)

        # Collect IMFs from all ensemble members
        all_imfs: List[List[np.ndarray]] = []

        for _ in range(self.ensemble_size):
            # Add noise
            noise = np.random.randn(n) * noise_amplitude
            noisy_signal = x + noise

            # Decompose
            imfs, _ = emd.decompose(noisy_signal)
            all_imfs.append(imfs)

        # Determine maximum number of IMFs
        max_n_imfs = max(len(imfs) for imfs in all_imfs)

        # Average IMFs
        averaged_imfs = []
        for i in range(max_n_imfs):
            # Collect i-th IMF from all ensembles (pad with zeros if needed)
            imf_ensemble = []
            for imfs in all_imfs:
                if i < len(imfs):
                    imf_ensemble.append(imfs[i])
                else:
                    imf_ensemble.append(np.zeros(n))

            averaged_imf = np.mean(imf_ensemble, axis=0)
            averaged_imfs.append(averaged_imf)

        # Compute residue
        residue = x - sum(averaged_imfs)

        return averaged_imfs, residue


class CEEMDAN:
    """
    Complete Ensemble Empirical Mode Decomposition with Adaptive Noise.

    An improvement over EEMD that adds noise at each stage of decomposition
    adaptively, resulting in more accurate IMFs with less computational cost.

    Parameters
    ----------
    ensemble_size : int
        Number of ensemble members
    noise_std : float
        Initial noise standard deviation
    max_imfs : int
        Maximum number of IMFs

    Examples
    --------
    >>> returns = np.random.randn(500) * 0.02  # Simulated returns
    >>> ceemdan = CEEMDAN(ensemble_size=50)
    >>> imfs, residue = ceemdan.decompose(returns)
    """

    def __init__(
        self,
        ensemble_size: int = 100,
        noise_std: float = 0.2,
        max_imfs: int = 10,
    ):
        self.ensemble_size = ensemble_size
        self.noise_std = noise_std
        self.max_imfs = max_imfs

    def decompose(
        self,
        signal: ArrayLike,
        seed: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Decompose signal using CEEMDAN.

        Parameters
        ----------
        signal : ArrayLike
            Input signal
        seed : int, optional
            Random seed

        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            IMFs and residue
        """
        if seed is not None:
            np.random.seed(seed)

        x = _to_array(signal)
        n = len(x)
        signal_std = np.std(x)

        emd = EMD(max_imfs=1)  # Extract only first IMF each iteration
        imfs = []
        residue = x.copy()

        for k in range(self.max_imfs):
            # Generate noise realizations
            noise_std_k = self.noise_std * signal_std * (0.5 ** k)  # Decrease noise

            # Decompose with noise
            imf_ensemble = []
            for _ in range(self.ensemble_size):
                noise = np.random.randn(n) * noise_std_k
                noisy_residue = residue + noise

                try:
                    imf_list, _ = emd.decompose(noisy_residue)
                    if len(imf_list) > 0:
                        imf_ensemble.append(imf_list[0])
                except Exception:
                    continue

            if len(imf_ensemble) == 0:
                break

            # Average IMF
            imf_k = np.mean(imf_ensemble, axis=0)
            imfs.append(imf_k)

            # Update residue
            residue = residue - imf_k

            # Check stopping criterion
            maxima_idx, minima_idx = EMD()._find_extrema(residue)
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break

        return imfs, residue


# =============================================================================
# HILBERT-HUANG TRANSFORM
# =============================================================================
class HilbertHuangTransform:
    """
    Hilbert-Huang Transform for instantaneous frequency and amplitude analysis.

    Applies the Hilbert Transform to IMFs from EMD to extract:
    - Instantaneous amplitude (envelope)
    - Instantaneous phase
    - Instantaneous frequency

    Parameters
    ----------
    emd : EMD, EEMD, or CEEMDAN
        EMD decomposition object (if None, uses standard EMD)

    Examples
    --------
    >>> prices = np.sin(np.linspace(0, 8*np.pi, 400)) * np.exp(-np.linspace(0, 1, 400))
    >>> hht = HilbertHuangTransform()
    >>> result = hht.analyze(prices, sampling_rate=1.0)
    >>> inst_freq = result['instantaneous_frequency']
    >>> inst_amp = result['instantaneous_amplitude']
    """

    def __init__(self, emd: Optional[Union[EMD, EEMD, CEEMDAN]] = None):
        self.emd = emd or EMD()

    def hilbert_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply Hilbert transform to get analytic signal."""
        return scipy_signal.hilbert(x)

    def instantaneous_amplitude(self, analytic_signal: np.ndarray) -> np.ndarray:
        """Extract instantaneous amplitude (envelope)."""
        return np.abs(analytic_signal)

    def instantaneous_phase(self, analytic_signal: np.ndarray) -> np.ndarray:
        """Extract instantaneous phase."""
        return np.unwrap(np.angle(analytic_signal))

    def instantaneous_frequency(
        self,
        phase: np.ndarray,
        sampling_rate: float = 1.0,
    ) -> np.ndarray:
        """
        Extract instantaneous frequency from phase.

        Parameters
        ----------
        phase : np.ndarray
            Unwrapped phase signal
        sampling_rate : float
            Sampling rate of the signal

        Returns
        -------
        np.ndarray
            Instantaneous frequency (same length as input)
        """
        # Derivative of phase
        freq = np.gradient(phase) * sampling_rate / (2 * np.pi)
        return freq

    def analyze(
        self,
        signal: ArrayLike,
        sampling_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Perform complete Hilbert-Huang Transform analysis.

        Parameters
        ----------
        signal : ArrayLike
            Input signal
        sampling_rate : float
            Sampling rate (e.g., samples per second)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'imfs': List of IMFs
            - 'residue': Residue signal
            - 'instantaneous_frequency': List of inst. freq for each IMF
            - 'instantaneous_amplitude': List of inst. amp for each IMF
            - 'instantaneous_phase': List of inst. phase for each IMF
            - 'analytic_signals': List of analytic signals for each IMF
        """
        x = _to_array(signal)

        # EMD decomposition
        imfs, residue = self.emd.decompose(x)

        # Hilbert transform on each IMF
        analytic_signals = []
        inst_amplitudes = []
        inst_phases = []
        inst_frequencies = []

        for imf in imfs:
            analytic = self.hilbert_transform(imf)
            amplitude = self.instantaneous_amplitude(analytic)
            phase = self.instantaneous_phase(analytic)
            frequency = self.instantaneous_frequency(phase, sampling_rate)

            analytic_signals.append(analytic)
            inst_amplitudes.append(amplitude)
            inst_phases.append(phase)
            inst_frequencies.append(frequency)

        return {
            'imfs': imfs,
            'residue': residue,
            'instantaneous_frequency': inst_frequencies,
            'instantaneous_amplitude': inst_amplitudes,
            'instantaneous_phase': inst_phases,
            'analytic_signals': analytic_signals,
        }

    def hilbert_spectrum(
        self,
        result: Dict[str, Any],
        n_bins: int = 100,
        freq_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Hilbert spectrum (time-frequency representation).

        Parameters
        ----------
        result : Dict[str, Any]
            Output from analyze()
        n_bins : int
            Number of frequency bins
        freq_range : Tuple[float, float], optional
            Frequency range (min, max)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Time array, frequency bins, amplitude matrix (freq x time)
        """
        inst_freq = result['instantaneous_frequency']
        inst_amp = result['instantaneous_amplitude']

        n_time = len(inst_freq[0])

        # Determine frequency range
        all_freqs = np.concatenate(inst_freq)
        if freq_range is None:
            freq_min = np.percentile(all_freqs[np.isfinite(all_freqs)], 1)
            freq_max = np.percentile(all_freqs[np.isfinite(all_freqs)], 99)
        else:
            freq_min, freq_max = freq_range

        freq_bins = np.linspace(freq_min, freq_max, n_bins)

        # Build spectrum
        spectrum = np.zeros((n_bins, n_time))

        for freq, amp in zip(inst_freq, inst_amp):
            for t in range(n_time):
                if np.isfinite(freq[t]) and freq_min <= freq[t] <= freq_max:
                    # Find bin
                    bin_idx = int((freq[t] - freq_min) / (freq_max - freq_min) * (n_bins - 1))
                    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                    spectrum[bin_idx, t] += amp[t]

        time_array = np.arange(n_time)

        return time_array, freq_bins, spectrum


# =============================================================================
# WAVELET ANALYSIS
# =============================================================================
class WaveletTransform:
    """
    Wavelet Transform toolkit for financial time series.

    Provides both Continuous Wavelet Transform (CWT) and Discrete Wavelet
    Transform (DWT), along with denoising and volatility regime detection.

    Parameters
    ----------
    wavelet : str
        Wavelet name (e.g., 'db4', 'sym4', 'coif4', 'haar', 'morl')

    Examples
    --------
    >>> prices = np.cumsum(np.random.randn(500)) + 100
    >>> wt = WaveletTransform(wavelet='db4')
    >>> # CWT analysis
    >>> cwt_matrix, freqs = wt.cwt(prices, scales=np.arange(1, 128))
    >>> # DWT decomposition
    >>> coeffs = wt.dwt(prices, level=4)
    >>> # Denoising
    >>> denoised = wt.denoise(prices, level=4, threshold_type='soft')
    """

    def __init__(self, wavelet: str = 'db4'):
        self.wavelet = wavelet

    def cwt(
        self,
        signal: ArrayLike,
        scales: Optional[np.ndarray] = None,
        sampling_period: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous Wavelet Transform.

        Parameters
        ----------
        signal : ArrayLike
            Input signal
        scales : np.ndarray, optional
            Scales for CWT (default: 1 to len(signal)/8)
        sampling_period : float
            Sampling period for frequency calculation

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            CWT coefficients matrix and corresponding frequencies
        """
        x = _to_array(signal)
        n = len(x)

        if scales is None:
            scales = np.arange(1, max(n // 8, 10))

        # Use continuous wavelet (e.g., 'morl', 'cmor', 'mexh')
        if self.wavelet in ['db4', 'sym4', 'coif4', 'haar']:
            cwt_wavelet = 'morl'  # Morlet for CWT
        else:
            cwt_wavelet = self.wavelet

        coefficients, frequencies = pywt.cwt(
            x, scales, cwt_wavelet, sampling_period=sampling_period
        )

        return coefficients, frequencies

    def dwt(
        self,
        signal: ArrayLike,
        level: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Discrete Wavelet Transform (multi-level decomposition).

        Parameters
        ----------
        signal : ArrayLike
            Input signal
        level : int, optional
            Decomposition level (default: max level)

        Returns
        -------
        List[np.ndarray]
            List of coefficients [cA_n, cD_n, cD_n-1, ..., cD_1]
        """
        x = _to_array(signal)

        if level is None:
            level = pywt.dwt_max_level(len(x), self.wavelet)

        coeffs = pywt.wavedec(x, self.wavelet, level=level)
        return coeffs

    def idwt(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Inverse Discrete Wavelet Transform (reconstruction).

        Parameters
        ----------
        coeffs : List[np.ndarray]
            Wavelet coefficients from dwt()

        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        return pywt.waverec(coeffs, self.wavelet)

    def denoise(
        self,
        signal: ArrayLike,
        level: Optional[int] = None,
        threshold_type: str = 'soft',
        threshold_mode: str = 'universal',
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Wavelet denoising using thresholding.

        Parameters
        ----------
        signal : ArrayLike
            Noisy signal
        level : int, optional
            Decomposition level
        threshold_type : str
            'soft' or 'hard' thresholding
        threshold_mode : str
            'universal': sqrt(2 * log(n)) * sigma
            'bayes': BayesShrink
            'sure': SURE shrink
        sigma : float, optional
            Noise standard deviation (estimated if None)

        Returns
        -------
        np.ndarray
            Denoised signal
        """
        x = _to_array(signal)
        n = len(x)

        # Decompose
        coeffs = self.dwt(x, level)

        # Estimate noise from finest detail coefficients
        if sigma is None:
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Calculate threshold
        if threshold_mode == 'universal':
            threshold = sigma * np.sqrt(2 * np.log(n))
        elif threshold_mode == 'bayes':
            # BayesShrink threshold for each level
            thresholds = []
            for c in coeffs[1:]:  # Detail coefficients
                var_y = np.var(c)
                var_x = max(var_y - sigma ** 2, 0)
                if var_x == 0:
                    thresholds.append(np.max(np.abs(c)))
                else:
                    thresholds.append(sigma ** 2 / np.sqrt(var_x))
            threshold = thresholds
        else:  # SURE
            threshold = sigma * np.sqrt(2 * np.log(n))

        # Apply thresholding to detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation

        for i, c in enumerate(coeffs[1:]):
            if isinstance(threshold, list):
                thresh = threshold[i]
            else:
                thresh = threshold

            if threshold_type == 'soft':
                c_thresh = pywt.threshold(c, thresh, mode='soft')
            else:  # hard
                c_thresh = pywt.threshold(c, thresh, mode='hard')

            denoised_coeffs.append(c_thresh)

        # Reconstruct
        return self.idwt(denoised_coeffs)[:n]

    def volatility_regime_detection(
        self,
        returns: ArrayLike,
        level: int = 4,
        smooth_window: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Detect volatility regimes using wavelet analysis.

        Parameters
        ----------
        returns : ArrayLike
            Return series
        level : int
            Wavelet decomposition level
        smooth_window : int
            Window for smoothing volatility estimates

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with volatility components at different scales
        """
        x = _to_array(returns)
        n = len(x)

        # Decompose
        coeffs = self.dwt(x, level)

        # Analyze volatility at each scale
        result = {}

        # Approximation (trend/long-term)
        approx = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], self.wavelet)[:n]
        result['trend'] = approx

        # Detail coefficients (different frequency volatility)
        scales = ['high_freq', 'mid_high_freq', 'mid_freq', 'low_freq']

        for i, (name, detail) in enumerate(zip(scales[:len(coeffs)-1], reversed(coeffs[1:]))):
            # Reconstruct signal from single detail level
            rec_coeffs = [np.zeros_like(coeffs[0])]
            for j in range(1, len(coeffs)):
                if j == len(coeffs) - 1 - i:
                    rec_coeffs.append(coeffs[j])
                else:
                    rec_coeffs.append(np.zeros_like(coeffs[j]))

            component = pywt.waverec(rec_coeffs, self.wavelet)[:n]

            # Estimate volatility (squared returns at this scale)
            vol_component = np.abs(component)

            # Smooth
            kernel = np.ones(smooth_window) / smooth_window
            vol_smooth = np.convolve(vol_component, kernel, mode='same')

            result[f'vol_{name}'] = vol_smooth
            result[f'component_{name}'] = component

        # Total volatility estimate
        result['total_volatility'] = np.sqrt(
            sum(result[f'vol_{s}'] ** 2 for s in scales[:len(coeffs)-1])
        )

        return result

    def multiresolution_analysis(
        self,
        signal: ArrayLike,
        level: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Perform multiresolution analysis (MRA).

        Decomposes signal into approximation and detail components
        at multiple scales.

        Parameters
        ----------
        signal : ArrayLike
            Input signal
        level : int, optional
            Number of decomposition levels

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'A_n' (approximation) and 'D_1'...'D_n' (details)
        """
        x = _to_array(signal)
        n = len(x)

        coeffs = self.dwt(x, level)
        actual_level = len(coeffs) - 1

        result = {}

        # Approximation at level n
        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        result[f'A_{actual_level}'] = pywt.waverec(approx_coeffs, self.wavelet)[:n]

        # Details at each level
        for i in range(1, len(coeffs)):
            detail_coeffs = [np.zeros_like(coeffs[0])]
            for j in range(1, len(coeffs)):
                if j == i:
                    detail_coeffs.append(coeffs[j])
                else:
                    detail_coeffs.append(np.zeros_like(coeffs[j]))

            level_idx = actual_level - i + 1
            result[f'D_{level_idx}'] = pywt.waverec(detail_coeffs, self.wavelet)[:n]

        return result


# =============================================================================
# RANDOM MATRIX THEORY
# =============================================================================
class RandomMatrixTheory:
    """
    Random Matrix Theory tools for correlation matrix cleaning.

    Uses Marcenko-Pastur distribution to separate signal from noise
    in correlation matrices, improving portfolio optimization and
    risk estimation.

    Parameters
    ----------
    q : float, optional
        Ratio T/N (observations / assets). Auto-calculated if not provided.

    Examples
    --------
    >>> returns = np.random.randn(252, 50)  # 252 days, 50 assets
    >>> rmt = RandomMatrixTheory()
    >>> corr_raw = np.corrcoef(returns.T)
    >>> corr_clean = rmt.denoise_correlation(corr_raw, returns.shape[0], returns.shape[1])
    """

    def __init__(self, q: Optional[float] = None):
        self.q = q

    def marcenko_pastur_pdf(
        self,
        eigenvalues: np.ndarray,
        q: float,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        Marcenko-Pastur probability density function.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Points at which to evaluate the PDF
        q : float
            Ratio T/N (must be > 1 for valid PDF)
        sigma : float
            Variance of the random matrix elements

        Returns
        -------
        np.ndarray
            PDF values
        """
        lambda_min = sigma ** 2 * (1 - 1 / np.sqrt(q)) ** 2
        lambda_max = sigma ** 2 * (1 + 1 / np.sqrt(q)) ** 2

        pdf = np.zeros_like(eigenvalues, dtype=float)
        mask = (eigenvalues >= lambda_min) & (eigenvalues <= lambda_max)

        pdf[mask] = (
            q / (2 * np.pi * sigma ** 2) *
            np.sqrt((lambda_max - eigenvalues[mask]) * (eigenvalues[mask] - lambda_min)) /
            eigenvalues[mask]
        )

        return pdf

    def marcenko_pastur_bounds(
        self,
        q: float,
        sigma: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Calculate Marcenko-Pastur eigenvalue bounds.

        Parameters
        ----------
        q : float
            Ratio T/N
        sigma : float
            Variance

        Returns
        -------
        Tuple[float, float]
            (lambda_min, lambda_max)
        """
        lambda_min = sigma ** 2 * (1 - 1 / np.sqrt(q)) ** 2
        lambda_max = sigma ** 2 * (1 + 1 / np.sqrt(q)) ** 2
        return lambda_min, lambda_max

    def fit_marcenko_pastur(
        self,
        eigenvalues: np.ndarray,
        q: float,
        bw: float = 0.01,
    ) -> float:
        """
        Fit Marcenko-Pastur distribution to find optimal sigma.

        Uses kernel density estimation and minimizes KL divergence.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed eigenvalues
        q : float
            Ratio T/N
        bw : float
            Bandwidth for KDE

        Returns
        -------
        float
            Estimated sigma
        """
        from scipy.optimize import minimize_scalar

        def neg_log_likelihood(sigma):
            if sigma <= 0:
                return np.inf
            pdf_vals = self.marcenko_pastur_pdf(eigenvalues, q, sigma)
            pdf_vals = np.maximum(pdf_vals, 1e-10)
            return -np.sum(np.log(pdf_vals))

        result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 5.0), method='bounded')
        return result.x

    def denoise_correlation(
        self,
        corr: np.ndarray,
        n_observations: int,
        n_assets: int,
        method: str = 'constant_residual',
    ) -> np.ndarray:
        """
        Denoise correlation matrix using RMT.

        Parameters
        ----------
        corr : np.ndarray
            Raw correlation matrix (n_assets x n_assets)
        n_observations : int
            Number of observations (T)
        n_assets : int
            Number of assets (N)
        method : str
            'constant_residual': Replace noise eigenvalues with average
            'shrinkage': Shrink noise eigenvalues toward average
            'target': Shrink toward identity

        Returns
        -------
        np.ndarray
            Denoised correlation matrix
        """
        q = n_observations / n_assets

        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(corr)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Find Marcenko-Pastur threshold
        lambda_min, lambda_max = self.marcenko_pastur_bounds(q)

        # Identify signal vs noise eigenvalues
        signal_mask = eigenvalues > lambda_max
        n_signal = np.sum(signal_mask)

        if n_signal == 0:
            logger.warning("No signal eigenvalues detected above MP threshold")
            n_signal = 1  # Keep at least one
            signal_mask[0] = True

        # Denoise based on method
        eigenvalues_clean = eigenvalues.copy()

        if method == 'constant_residual':
            # Replace noise eigenvalues with their average
            noise_eigenvalues = eigenvalues[~signal_mask]
            if len(noise_eigenvalues) > 0:
                avg_noise = np.mean(noise_eigenvalues)
                # Ensure trace preservation
                target_sum = n_assets - np.sum(eigenvalues[signal_mask])
                n_noise = n_assets - n_signal
                eigenvalues_clean[~signal_mask] = target_sum / n_noise

        elif method == 'shrinkage':
            # Shrink noise eigenvalues toward 1 (identity)
            noise_eigenvalues = eigenvalues[~signal_mask]
            if len(noise_eigenvalues) > 0:
                # Shrinkage intensity
                alpha = min(1.0, (lambda_max - lambda_min) / np.var(noise_eigenvalues))
                eigenvalues_clean[~signal_mask] = (
                    alpha * 1.0 + (1 - alpha) * noise_eigenvalues
                )

        elif method == 'target':
            # Replace noise eigenvalues with 1 and rescale
            eigenvalues_clean[~signal_mask] = 1.0
            # Rescale to preserve trace
            scale = n_assets / np.sum(eigenvalues_clean)
            eigenvalues_clean *= scale

        # Reconstruct correlation matrix
        corr_clean = eigenvectors @ np.diag(eigenvalues_clean) @ eigenvectors.T

        # Ensure proper correlation matrix properties
        np.fill_diagonal(corr_clean, 1.0)
        corr_clean = np.clip(corr_clean, -1.0, 1.0)

        # Ensure symmetry
        corr_clean = (corr_clean + corr_clean.T) / 2

        return corr_clean

    def get_signal_eigenvalues(
        self,
        corr: np.ndarray,
        q: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Separate signal and noise eigenvalues.

        Parameters
        ----------
        corr : np.ndarray
            Correlation matrix
        q : float
            Ratio T/N

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (signal_eigenvalues, noise_eigenvalues, threshold)
        """
        _, lambda_max = self.marcenko_pastur_bounds(q)

        eigenvalues, _ = eigh(corr)
        eigenvalues = np.sort(eigenvalues)[::-1]

        signal = eigenvalues[eigenvalues > lambda_max]
        noise = eigenvalues[eigenvalues <= lambda_max]

        return signal, noise, lambda_max

    def estimate_effective_dimension(
        self,
        corr: np.ndarray,
        q: float,
    ) -> int:
        """
        Estimate effective number of independent factors.

        Parameters
        ----------
        corr : np.ndarray
            Correlation matrix
        q : float
            Ratio T/N

        Returns
        -------
        int
            Number of significant eigenvalues (effective dimension)
        """
        signal, _, _ = self.get_signal_eigenvalues(corr, q)
        return len(signal)


class CorrelationCleaner:
    """
    High-level interface for correlation matrix cleaning.

    Combines RMT with additional techniques for robust correlation estimation.

    Examples
    --------
    >>> returns = pd.DataFrame(np.random.randn(252, 20))  # 252 days, 20 assets
    >>> cleaner = CorrelationCleaner()
    >>> result = cleaner.clean(returns)
    >>> clean_corr = result['correlation']
    >>> effective_dim = result['effective_dimension']
    """

    def __init__(self):
        self.rmt = RandomMatrixTheory()

    def clean(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        method: str = 'constant_residual',
        shrinkage_target: str = 'identity',
    ) -> Dict[str, Any]:
        """
        Clean correlation matrix from returns data.

        Parameters
        ----------
        returns : Union[pd.DataFrame, np.ndarray]
            Returns matrix (T observations x N assets)
        method : str
            RMT denoising method
        shrinkage_target : str
            Target for additional shrinkage ('identity', 'diagonal', 'none')

        Returns
        -------
        Dict[str, Any]
            Cleaned correlation matrix and diagnostics
        """
        if isinstance(returns, pd.DataFrame):
            returns_arr = returns.values
            columns = returns.columns
        else:
            returns_arr = returns
            columns = None

        n_obs, n_assets = returns_arr.shape
        q = n_obs / n_assets

        # Raw correlation
        corr_raw = np.corrcoef(returns_arr.T)

        # RMT denoising
        corr_denoised = self.rmt.denoise_correlation(
            corr_raw, n_obs, n_assets, method=method
        )

        # Optional additional shrinkage
        if shrinkage_target == 'identity':
            # Ledoit-Wolf style shrinkage toward identity
            alpha = 1.0 / n_obs  # Simple shrinkage intensity
            corr_final = (1 - alpha) * corr_denoised + alpha * np.eye(n_assets)
        elif shrinkage_target == 'diagonal':
            # Shrink toward diagonal
            alpha = 1.0 / n_obs
            corr_final = (1 - alpha) * corr_denoised + alpha * np.diag(np.diag(corr_denoised))
        else:
            corr_final = corr_denoised

        # Diagnostics
        signal, noise, threshold = self.rmt.get_signal_eigenvalues(corr_raw, q)

        result = {
            'correlation': corr_final,
            'correlation_raw': corr_raw,
            'effective_dimension': len(signal),
            'signal_eigenvalues': signal,
            'noise_eigenvalues': noise,
            'mp_threshold': threshold,
            'q_ratio': q,
        }

        if columns is not None:
            result['correlation'] = pd.DataFrame(
                corr_final, index=columns, columns=columns
            )
            result['correlation_raw'] = pd.DataFrame(
                corr_raw, index=columns, columns=columns
            )

        return result


# =============================================================================
# FISHER TRANSFORM
# =============================================================================
class FisherTransform:
    """
    Fisher Transform for normalizing bounded indicators.

    The Fisher Transform converts values bounded between -1 and 1
    to approximately normally distributed values, useful for:
    - RSI normalization
    - Correlation values
    - Any bounded indicator

    Transform: y = 0.5 * ln((1 + x) / (1 - x))

    Examples
    --------
    >>> ft = FisherTransform()
    >>> rsi = np.array([30, 50, 70, 80, 20]) / 100  # Normalized RSI
    >>> rsi_normalized = ft.transform_bounded(rsi, 0, 100)
    >>> print(ft.transform(rsi * 2 - 1))  # Convert to [-1, 1] first
    """

    def __init__(self, clip_value: float = 0.999):
        """
        Parameters
        ----------
        clip_value : float
            Clip input to [-clip_value, clip_value] to avoid infinity
        """
        self.clip_value = clip_value

    def transform(self, x: ArrayLike) -> np.ndarray:
        """
        Apply Fisher Transform.

        Parameters
        ----------
        x : ArrayLike
            Input values in range [-1, 1]

        Returns
        -------
        np.ndarray
            Transformed values (approximately normal)
        """
        x = _to_array(x)
        x_clipped = np.clip(x, -self.clip_value, self.clip_value)
        return 0.5 * np.log((1 + x_clipped) / (1 - x_clipped))

    def inverse_transform(self, y: ArrayLike) -> np.ndarray:
        """
        Inverse Fisher Transform.

        Parameters
        ----------
        y : ArrayLike
            Fisher-transformed values

        Returns
        -------
        np.ndarray
            Original scale values in [-1, 1]
        """
        y = _to_array(y)
        return (np.exp(2 * y) - 1) / (np.exp(2 * y) + 1)

    def transform_bounded(
        self,
        x: ArrayLike,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        """
        Transform values from [lower, upper] range.

        Parameters
        ----------
        x : ArrayLike
            Input values in [lower, upper]
        lower : float
            Lower bound
        upper : float
            Upper bound

        Returns
        -------
        np.ndarray
            Fisher-transformed values
        """
        x = _to_array(x)
        # Normalize to [-1, 1]
        x_normalized = 2 * (x - lower) / (upper - lower) - 1
        return self.transform(x_normalized)

    def inverse_transform_bounded(
        self,
        y: ArrayLike,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        """
        Inverse transform back to [lower, upper] range.

        Parameters
        ----------
        y : ArrayLike
            Fisher-transformed values
        lower : float
            Original lower bound
        upper : float
            Original upper bound

        Returns
        -------
        np.ndarray
            Values in [lower, upper]
        """
        y = _to_array(y)
        x_normalized = self.inverse_transform(y)
        return (x_normalized + 1) / 2 * (upper - lower) + lower

    def transform_rsi(
        self,
        rsi: ArrayLike,
        smooth_period: int = 5,
    ) -> np.ndarray:
        """
        Apply Fisher Transform to RSI indicator.

        Parameters
        ----------
        rsi : ArrayLike
            RSI values (typically 0-100)
        smooth_period : int
            Smoothing period for the transform

        Returns
        -------
        np.ndarray
            Fisher-transformed RSI
        """
        rsi = _to_array(rsi)

        # Normalize RSI to [-1, 1]
        rsi_norm = (rsi - 50) / 50 * 0.999  # Center at 0, scale to avoid extremes

        # Apply transform
        fisher = self.transform(rsi_norm)

        # Optional smoothing
        if smooth_period > 1:
            kernel = np.ones(smooth_period) / smooth_period
            fisher = np.convolve(fisher, kernel, mode='same')

        return fisher

    def transform_correlation(
        self,
        corr: ArrayLike,
    ) -> np.ndarray:
        """
        Apply Fisher Transform to correlation values.

        Also known as Fisher z-transformation, useful for
        statistical testing of correlations.

        Parameters
        ----------
        corr : ArrayLike
            Correlation values in [-1, 1]

        Returns
        -------
        np.ndarray
            Fisher z-transformed correlations
        """
        return self.transform(corr)

    def correlation_confidence_interval(
        self,
        corr: float,
        n: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for correlation using Fisher Transform.

        Parameters
        ----------
        corr : float
            Sample correlation
        n : int
            Sample size
        confidence : float
            Confidence level (e.g., 0.95 for 95%)

        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound)
        """
        # Fisher transform
        z = self.transform(np.array([corr]))[0]

        # Standard error
        se = 1 / np.sqrt(n - 3)

        # Z-score for confidence level
        z_score = norm.ppf((1 + confidence) / 2)

        # Confidence interval in Fisher space
        z_lower = z - z_score * se
        z_upper = z + z_score * se

        # Transform back
        lower = self.inverse_transform(np.array([z_lower]))[0]
        upper = self.inverse_transform(np.array([z_upper]))[0]

        return lower, upper


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def kalman_smooth_prices(
    prices: ArrayLike,
    process_var: float = 0.1,
    obs_var: float = 1.0,
) -> np.ndarray:
    """
    Convenience function for Kalman smoothing of prices.

    Parameters
    ----------
    prices : ArrayLike
        Price series
    process_var : float
        Process variance (level innovation)
    obs_var : float
        Observation variance (measurement noise)

    Returns
    -------
    np.ndarray
        Smoothed prices
    """
    kf = KalmanFilter.local_level(process_var, obs_var)
    smoothed, _, _ = kf.smooth(prices)
    return smoothed


def emd_decompose(
    signal: ArrayLike,
    max_imfs: int = 10,
    method: str = 'emd',
    ensemble_size: int = 100,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Convenience function for EMD decomposition.

    Parameters
    ----------
    signal : ArrayLike
        Input signal
    max_imfs : int
        Maximum IMFs
    method : str
        'emd', 'eemd', or 'ceemdan'
    ensemble_size : int
        Ensemble size for EEMD/CEEMDAN

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        IMFs and residue
    """
    if method == 'emd':
        decomposer = EMD(max_imfs=max_imfs)
    elif method == 'eemd':
        decomposer = EEMD(ensemble_size=ensemble_size, emd_params={'max_imfs': max_imfs})
    elif method == 'ceemdan':
        decomposer = CEEMDAN(ensemble_size=ensemble_size, max_imfs=max_imfs)
    else:
        raise ValueError(f"Unknown method: {method}")

    return decomposer.decompose(signal)


def wavelet_denoise(
    signal: ArrayLike,
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_type: str = 'soft',
) -> np.ndarray:
    """
    Convenience function for wavelet denoising.

    Parameters
    ----------
    signal : ArrayLike
        Noisy signal
    wavelet : str
        Wavelet type
    level : int, optional
        Decomposition level
    threshold_type : str
        'soft' or 'hard'

    Returns
    -------
    np.ndarray
        Denoised signal
    """
    wt = WaveletTransform(wavelet)
    return wt.denoise(signal, level, threshold_type)


def clean_correlation_matrix(
    returns: Union[pd.DataFrame, np.ndarray],
    method: str = 'constant_residual',
) -> np.ndarray:
    """
    Convenience function for RMT correlation cleaning.

    Parameters
    ----------
    returns : Union[pd.DataFrame, np.ndarray]
        Returns matrix (T x N)
    method : str
        Denoising method

    Returns
    -------
    np.ndarray
        Cleaned correlation matrix
    """
    cleaner = CorrelationCleaner()
    result = cleaner.clean(returns, method)
    return result['correlation']


def fisher_transform_indicator(
    indicator: ArrayLike,
    lower: float = 0,
    upper: float = 100,
) -> np.ndarray:
    """
    Convenience function for Fisher Transform of bounded indicators.

    Parameters
    ----------
    indicator : ArrayLike
        Indicator values (e.g., RSI in [0, 100])
    lower : float
        Lower bound
    upper : float
        Upper bound

    Returns
    -------
    np.ndarray
        Fisher-transformed values
    """
    ft = FisherTransform()
    return ft.transform_bounded(indicator, lower, upper)


# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    # Kalman Filter
    'KalmanFilter',
    'KalmanState',
    'ExtendedKalmanFilter',
    # EMD
    'EMD',
    'EEMD',
    'CEEMDAN',
    # Hilbert-Huang
    'HilbertHuangTransform',
    # Wavelets
    'WaveletTransform',
    # RMT
    'RandomMatrixTheory',
    'CorrelationCleaner',
    # Fisher Transform
    'FisherTransform',
    # Convenience functions
    'kalman_smooth_prices',
    'emd_decompose',
    'wavelet_denoise',
    'clean_correlation_matrix',
    'fisher_transform_indicator',
]


if __name__ == '__main__':
    # Example usage demonstration
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    np.random.seed(42)

    # Generate sample financial data
    n_points = 500
    t = np.linspace(0, 10, n_points)

    # Simulated price with trend + cycles + noise
    trend = 100 + 5 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 2) + 1.5 * np.sin(2 * np.pi * t / 0.5)
    noise = np.random.randn(n_points) * 2
    prices = trend + seasonal + noise

    print("=" * 60)
    print("SIGNAL PROCESSING MODULE DEMONSTRATION")
    print("=" * 60)

    # 1. Kalman Filter
    print("\n1. KALMAN FILTER")
    print("-" * 40)
    kf = KalmanFilter.local_linear_trend(level_var=0.5, trend_var=0.01, obs_var=4.0)
    smoothed, states, _ = kf.smooth(prices)
    print(f"   Original prices std: {np.std(prices):.2f}")
    print(f"   Smoothed prices std: {np.std(smoothed):.2f}")
    print(f"   Estimated trend slope: {np.mean(np.diff(states[:, 1])):.4f}")

    # 2. EMD Decomposition
    print("\n2. EMD DECOMPOSITION")
    print("-" * 40)
    emd = EMD(max_imfs=5)
    imfs, residue = emd.decompose(prices)
    print(f"   Extracted {len(imfs)} IMFs")
    for i, imf in enumerate(imfs):
        print(f"   IMF {i+1} variance: {np.var(imf):.2f}")
    print(f"   Residue variance: {np.var(residue):.2f}")

    # 3. Hilbert-Huang Transform
    print("\n3. HILBERT-HUANG TRANSFORM")
    print("-" * 40)
    hht = HilbertHuangTransform()
    result = hht.analyze(prices - np.mean(prices))
    print(f"   Number of IMFs analyzed: {len(result['imfs'])}")
    for i, (amp, freq) in enumerate(zip(
        result['instantaneous_amplitude'][:3],
        result['instantaneous_frequency'][:3]
    )):
        valid_freq = freq[np.isfinite(freq)]
        print(f"   IMF {i+1} - Mean amplitude: {np.mean(amp):.2f}, "
              f"Mean frequency: {np.mean(valid_freq):.4f}")

    # 4. Wavelet Transform
    print("\n4. WAVELET ANALYSIS")
    print("-" * 40)
    wt = WaveletTransform(wavelet='db4')
    denoised = wt.denoise(prices, level=4, threshold_type='soft')
    print(f"   Original noise (vs trend): {np.std(prices - trend):.2f}")
    print(f"   Denoised noise (vs trend): {np.std(denoised - trend):.2f}")

    # Volatility regime detection
    returns = np.diff(np.log(prices))
    vol_regimes = wt.volatility_regime_detection(returns, level=3)
    print(f"   High-freq volatility range: "
          f"[{np.min(vol_regimes['vol_high_freq']):.4f}, "
          f"{np.max(vol_regimes['vol_high_freq']):.4f}]")

    # 5. Random Matrix Theory
    print("\n5. RANDOM MATRIX THEORY")
    print("-" * 40)
    n_assets = 20
    n_obs = 250
    returns_matrix = np.random.randn(n_obs, n_assets) * 0.02
    # Add some structure
    factor = np.random.randn(n_obs, 1)
    returns_matrix += factor @ np.random.randn(1, n_assets) * 0.01

    cleaner = CorrelationCleaner()
    clean_result = cleaner.clean(returns_matrix)
    print(f"   Effective dimension: {clean_result['effective_dimension']}")
    print(f"   MP threshold: {clean_result['mp_threshold']:.4f}")
    print(f"   Signal eigenvalues: {clean_result['signal_eigenvalues']}")

    # 6. Fisher Transform
    print("\n6. FISHER TRANSFORM")
    print("-" * 40)
    ft = FisherTransform()

    # Transform RSI-like values
    rsi_values = np.array([20, 30, 50, 70, 80, 90])
    fisher_rsi = ft.transform_bounded(rsi_values, 0, 100)
    print(f"   RSI values: {rsi_values}")
    print(f"   Fisher RSI: {np.round(fisher_rsi, 2)}")

    # Correlation confidence interval
    sample_corr = 0.7
    sample_n = 100
    ci_low, ci_high = ft.correlation_confidence_interval(sample_corr, sample_n)
    print(f"   Correlation 0.7 (n=100) 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    print("\n" + "=" * 60)
    print("Module demonstration complete!")
    print("=" * 60)
