"""
Hawkes Processes for High-Frequency Trading
============================================

Self-exciting point processes for modeling order flow, trade clustering,
and market microstructure dynamics.

Hawkes processes capture the self-exciting nature of financial events where
the arrival of one event increases the probability of subsequent events.

Mathematical Background:
    The conditional intensity function is:
    λ(t) = μ + Σᵢ g(t - tᵢ) for tᵢ < t

    where:
    - μ is the base (exogenous) intensity
    - g(·) is the kernel (excitation) function
    - tᵢ are the historical event times

Key Applications in Trading:
    - Order flow modeling
    - Trade clustering detection
    - Market making optimization
    - Toxicity detection (VPIN-like measures)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    Protocol,
)

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import gamma as gamma_func
from scipy.integrate import quad


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

ArrayLike = Union[List[float], NDArray[np.float64]]
EventTimes = NDArray[np.float64]
KernelParams = Dict[str, float]


class KernelType(Enum):
    """Supported kernel types for Hawkes processes"""
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    SUM_EXPONENTIALS = "sum_exponentials"
    RAISED_COSINE = "raised_cosine"


# =============================================================================
# Kernel Classes
# =============================================================================

class HawkesKernel(ABC):
    """Abstract base class for Hawkes kernel functions"""

    @abstractmethod
    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate kernel at time lag(s) t"""
        pass

    @abstractmethod
    def integral(self, t_start: float, t_end: float) -> float:
        """Compute definite integral of kernel from t_start to t_end"""
        pass

    @abstractmethod
    def norm(self) -> float:
        """Compute L1 norm (integral from 0 to infinity) of kernel"""
        pass

    @abstractmethod
    def get_params(self) -> KernelParams:
        """Get kernel parameters as dictionary"""
        pass

    @abstractmethod
    def set_params(self, **params: float) -> None:
        """Set kernel parameters"""
        pass

    @property
    @abstractmethod
    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        pass


class ExponentialKernel(HawkesKernel):
    """
    Exponential decay kernel: k(t) = α * exp(-β*t)

    The most common kernel for Hawkes processes, providing exponential
    decay of excitation over time.

    Parameters
    ----------
    alpha : float
        Excitation amplitude (α > 0)
    beta : float
        Decay rate (β > 0)

    Properties
    ----------
    - Branching ratio: α/β (must be < 1 for stationarity)
    - Mean excitation time: 1/β
    - L1 norm: α/β

    Example
    -------
    >>> kernel = ExponentialKernel(alpha=0.5, beta=2.0)
    >>> kernel(np.array([0.0, 0.5, 1.0]))
    array([0.5       , 0.18393972, 0.06766764])
    >>> kernel.norm()  # branching ratio
    0.25
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        self.alpha = alpha
        self.beta = beta

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        result = np.where(t >= 0, self.alpha * np.exp(-self.beta * t), 0.0)
        return result

    def integral(self, t_start: float, t_end: float) -> float:
        """Integral from t_start to t_end"""
        if t_start < 0:
            t_start = 0
        if t_end <= t_start:
            return 0.0
        return (self.alpha / self.beta) * (
            np.exp(-self.beta * t_start) - np.exp(-self.beta * t_end)
        )

    def norm(self) -> float:
        """L1 norm = α/β (branching ratio)"""
        return self.alpha / self.beta

    def get_params(self) -> KernelParams:
        return {"alpha": self.alpha, "beta": self.beta}

    def set_params(self, **params: float) -> None:
        if "alpha" in params:
            if params["alpha"] <= 0:
                raise ValueError("alpha must be positive")
            self.alpha = params["alpha"]
        if "beta" in params:
            if params["beta"] <= 0:
                raise ValueError("beta must be positive")
            self.beta = params["beta"]

    @property
    def param_bounds(self) -> List[Tuple[float, float]]:
        return [(1e-8, 10.0), (1e-8, 100.0)]  # alpha, beta bounds

    def __repr__(self) -> str:
        return f"ExponentialKernel(alpha={self.alpha:.4f}, beta={self.beta:.4f})"


class PowerLawKernel(HawkesKernel):
    """
    Power-law decay kernel: k(t) = α * (t + c)^(-β)

    Captures long-memory effects in financial markets where events have
    persistent influence over time.

    Parameters
    ----------
    alpha : float
        Excitation amplitude (α > 0)
    beta : float
        Decay exponent (β > 1 for finite integral)
    c : float
        Shift parameter to avoid singularity at t=0 (c > 0)

    Properties
    ----------
    - Heavy-tailed decay (slower than exponential)
    - Suitable for long-memory processes
    - L1 norm: α * c^(1-β) / (β-1) for β > 1

    Example
    -------
    >>> kernel = PowerLawKernel(alpha=1.0, beta=2.0, c=0.1)
    >>> kernel(np.array([0.0, 1.0, 10.0]))
    array([100.        ,   0.82644628,   0.00990099])
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, c: float = 0.1):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if beta <= 1:
            raise ValueError(f"beta must be > 1 for finite integral, got {beta}")
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}")

        self.alpha = alpha
        self.beta = beta
        self.c = c

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        result = np.where(t >= 0, self.alpha * np.power(t + self.c, -self.beta), 0.0)
        return result

    def integral(self, t_start: float, t_end: float) -> float:
        """Integral from t_start to t_end"""
        if t_start < 0:
            t_start = 0
        if t_end <= t_start:
            return 0.0

        coef = self.alpha / (1 - self.beta)
        return coef * (
            np.power(t_end + self.c, 1 - self.beta) -
            np.power(t_start + self.c, 1 - self.beta)
        )

    def norm(self) -> float:
        """L1 norm (integral from 0 to infinity)"""
        return self.alpha * np.power(self.c, 1 - self.beta) / (self.beta - 1)

    def get_params(self) -> KernelParams:
        return {"alpha": self.alpha, "beta": self.beta, "c": self.c}

    def set_params(self, **params: float) -> None:
        if "alpha" in params:
            if params["alpha"] <= 0:
                raise ValueError("alpha must be positive")
            self.alpha = params["alpha"]
        if "beta" in params:
            if params["beta"] <= 1:
                raise ValueError("beta must be > 1")
            self.beta = params["beta"]
        if "c" in params:
            if params["c"] <= 0:
                raise ValueError("c must be positive")
            self.c = params["c"]

    @property
    def param_bounds(self) -> List[Tuple[float, float]]:
        return [(1e-8, 10.0), (1.01, 5.0), (1e-4, 1.0)]  # alpha, beta, c

    def __repr__(self) -> str:
        return f"PowerLawKernel(alpha={self.alpha:.4f}, beta={self.beta:.4f}, c={self.c:.4f})"


class SumExponentialsKernel(HawkesKernel):
    """
    Sum of exponential kernels: k(t) = Σⱼ αⱼ * exp(-βⱼ*t)

    Provides flexible multi-timescale dynamics by combining multiple
    exponential decays with different time constants.

    Parameters
    ----------
    alphas : List[float]
        Excitation amplitudes for each component
    betas : List[float]
        Decay rates for each component

    Properties
    ----------
    - Multiple timescales of excitation
    - Useful for capturing fast and slow dynamics
    - L1 norm: Σⱼ αⱼ/βⱼ

    Example
    -------
    >>> kernel = SumExponentialsKernel(alphas=[0.3, 0.2], betas=[5.0, 1.0])
    >>> # Fast decay (β=5) + slow decay (β=1)
    >>> kernel.norm()
    0.26
    """

    def __init__(
        self,
        alphas: List[float] = [0.3, 0.2],
        betas: List[float] = [5.0, 1.0]
    ):
        if len(alphas) != len(betas):
            raise ValueError("alphas and betas must have same length")
        if any(a <= 0 for a in alphas):
            raise ValueError("all alphas must be positive")
        if any(b <= 0 for b in betas):
            raise ValueError("all betas must be positive")

        self.alphas = np.array(alphas, dtype=np.float64)
        self.betas = np.array(betas, dtype=np.float64)
        self.n_components = len(alphas)

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        t = np.atleast_1d(t)

        result = np.zeros_like(t)
        mask = t >= 0

        for alpha, beta in zip(self.alphas, self.betas):
            result[mask] += alpha * np.exp(-beta * t[mask])

        return result

    def integral(self, t_start: float, t_end: float) -> float:
        """Integral from t_start to t_end"""
        if t_start < 0:
            t_start = 0
        if t_end <= t_start:
            return 0.0

        total = 0.0
        for alpha, beta in zip(self.alphas, self.betas):
            total += (alpha / beta) * (
                np.exp(-beta * t_start) - np.exp(-beta * t_end)
            )
        return total

    def norm(self) -> float:
        """L1 norm = Σ αⱼ/βⱼ"""
        return np.sum(self.alphas / self.betas)

    def get_params(self) -> KernelParams:
        params = {}
        for i, (a, b) in enumerate(zip(self.alphas, self.betas)):
            params[f"alpha_{i}"] = a
            params[f"beta_{i}"] = b
        return params

    def set_params(self, **params: float) -> None:
        for i in range(self.n_components):
            if f"alpha_{i}" in params:
                if params[f"alpha_{i}"] <= 0:
                    raise ValueError(f"alpha_{i} must be positive")
                self.alphas[i] = params[f"alpha_{i}"]
            if f"beta_{i}" in params:
                if params[f"beta_{i}"] <= 0:
                    raise ValueError(f"beta_{i} must be positive")
                self.betas[i] = params[f"beta_{i}"]

    @property
    def param_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        for _ in range(self.n_components):
            bounds.extend([(1e-8, 5.0), (1e-8, 100.0)])  # alpha, beta for each
        return bounds

    def __repr__(self) -> str:
        components = [
            f"({a:.4f}, {b:.4f})"
            for a, b in zip(self.alphas, self.betas)
        ]
        return f"SumExponentialsKernel([{', '.join(components)}])"


# =============================================================================
# Univariate Hawkes Process
# =============================================================================

@dataclass
class HawkesResult:
    """Container for Hawkes process estimation results"""

    mu: float  # Base intensity
    kernel: HawkesKernel
    branching_ratio: float
    log_likelihood: float
    aic: float
    bic: float
    n_events: int
    T: float  # Observation window
    converged: bool
    n_iterations: int
    message: str = ""


class UnivariateHawkes:
    """
    Univariate Hawkes Process for modeling self-exciting point processes.

    The conditional intensity is:
    λ(t) = μ + Σᵢ g(t - tᵢ)

    where μ is the base intensity and g(·) is the kernel function.

    Parameters
    ----------
    kernel : HawkesKernel
        Excitation kernel function
    mu : float
        Base (exogenous) intensity

    Example
    -------
    >>> # Create Hawkes process with exponential kernel
    >>> kernel = ExponentialKernel(alpha=0.5, beta=2.0)
    >>> hawkes = UnivariateHawkes(kernel=kernel, mu=1.0)
    >>>
    >>> # Simulate events
    >>> times = hawkes.simulate(T=100.0)
    >>> print(f"Simulated {len(times)} events")
    >>>
    >>> # Estimate parameters from data
    >>> hawkes_fitted = UnivariateHawkes.fit(times, T=100.0, kernel_type=KernelType.EXPONENTIAL)
    """

    def __init__(
        self,
        kernel: Optional[HawkesKernel] = None,
        mu: float = 1.0
    ):
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")

        self.kernel = kernel or ExponentialKernel()
        self.mu = mu
        self._fitted = False

    @property
    def branching_ratio(self) -> float:
        """
        Branching ratio n* = ∫g(t)dt

        Must be < 1 for the process to be stationary.
        Represents average number of descendants per event.
        """
        return self.kernel.norm()

    @property
    def is_stationary(self) -> bool:
        """Check if process is stationary (branching ratio < 1)"""
        return self.branching_ratio < 1.0

    @property
    def stationary_intensity(self) -> float:
        """
        Stationary mean intensity: μ / (1 - n*)

        Only valid when branching_ratio < 1
        """
        if not self.is_stationary:
            return np.inf
        return self.mu / (1 - self.branching_ratio)

    def intensity(self, t: float, history: EventTimes) -> float:
        """
        Compute conditional intensity λ(t) given event history.

        Parameters
        ----------
        t : float
            Time at which to compute intensity
        history : EventTimes
            Array of past event times (< t)

        Returns
        -------
        float
            Conditional intensity at time t
        """
        history = np.asarray(history, dtype=np.float64)
        past_events = history[history < t]

        if len(past_events) == 0:
            return self.mu

        time_lags = t - past_events
        excitation = np.sum(self.kernel(time_lags))

        return self.mu + excitation

    def intensity_path(
        self,
        times: EventTimes,
        grid: Optional[ArrayLike] = None,
        n_points: int = 1000
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute intensity path over a time grid.

        Parameters
        ----------
        times : EventTimes
            Event times
        grid : Optional[ArrayLike]
            Time grid for evaluation (if None, creates uniform grid)
        n_points : int
            Number of grid points if grid is None

        Returns
        -------
        Tuple[NDArray, NDArray]
            (time_grid, intensity_values)
        """
        times = np.asarray(times, dtype=np.float64)

        if grid is None:
            T = times[-1] if len(times) > 0 else 1.0
            grid = np.linspace(0, T, n_points)
        else:
            grid = np.asarray(grid, dtype=np.float64)

        intensities = np.array([self.intensity(t, times) for t in grid])

        return grid, intensities

    def compensator(self, t: float, history: EventTimes) -> float:
        """
        Compute compensator Λ(t) = ∫₀ᵗ λ(s) ds

        Parameters
        ----------
        t : float
            Upper limit of integration
        history : EventTimes
            Event times

        Returns
        -------
        float
            Compensator value
        """
        history = np.asarray(history, dtype=np.float64)
        past_events = history[history < t]

        # Base intensity contribution
        comp = self.mu * t

        # Excitation contribution
        for ti in past_events:
            comp += self.kernel.integral(0, t - ti)

        return comp

    def log_likelihood(self, times: EventTimes, T: float) -> float:
        """
        Compute log-likelihood of event times.

        L = Σᵢ log(λ(tᵢ)) - ∫₀ᵀ λ(t) dt

        Parameters
        ----------
        times : EventTimes
            Observed event times
        T : float
            Observation window [0, T]

        Returns
        -------
        float
            Log-likelihood value
        """
        times = np.asarray(times, dtype=np.float64)
        times = np.sort(times)
        n = len(times)

        if n == 0:
            return -self.mu * T

        # Sum of log intensities
        log_intensity_sum = 0.0
        for i, ti in enumerate(times):
            lam = self.intensity(ti, times[:i])
            if lam > 0:
                log_intensity_sum += np.log(lam)
            else:
                return -np.inf

        # Compensator (integral of intensity)
        compensator = self.compensator(T, times)

        return log_intensity_sum - compensator

    def simulate(
        self,
        T: float,
        max_events: int = 100000,
        seed: Optional[int] = None
    ) -> EventTimes:
        """
        Simulate event times using Ogata's thinning algorithm.

        The algorithm uses an upper bound on the intensity and accepts/rejects
        proposed events proportional to the true intensity.

        Parameters
        ----------
        T : float
            Simulation horizon
        max_events : int
            Maximum number of events (safety limit)
        seed : Optional[int]
            Random seed for reproducibility

        Returns
        -------
        EventTimes
            Simulated event times

        Notes
        -----
        Ogata's thinning algorithm:
        1. Set upper bound λ* ≥ λ(t) for current time
        2. Generate exponential waiting time with rate λ*
        3. Accept event with probability λ(t)/λ*
        4. Update upper bound and repeat
        """
        if not self.is_stationary:
            logger.warning("Process is not stationary (branching ratio >= 1)")

        if seed is not None:
            np.random.seed(seed)

        events: List[float] = []
        t = 0.0

        while t < T and len(events) < max_events:
            # Compute intensity upper bound
            lam = self.intensity(t, np.array(events))
            kernel_at_zero = self.kernel(np.array([0.0]))
            k0 = kernel_at_zero[0] if kernel_at_zero.ndim > 0 else float(kernel_at_zero)
            lam_bar = lam + k0

            # Generate exponential waiting time
            u1 = np.random.uniform()
            w = -np.log(u1) / lam_bar
            t_new = t + w

            if t_new > T:
                break

            # Accept/reject
            lam_new = self.intensity(t_new, np.array(events))
            u2 = np.random.uniform()

            if u2 <= lam_new / lam_bar:
                events.append(t_new)

            t = t_new

        return np.array(events, dtype=np.float64)

    @classmethod
    def fit(
        cls,
        times: EventTimes,
        T: float,
        kernel_type: KernelType = KernelType.EXPONENTIAL,
        method: str = "MLE",
        **kwargs: Any
    ) -> HawkesResult:
        """
        Fit Hawkes process to observed event times.

        Parameters
        ----------
        times : EventTimes
            Observed event times
        T : float
            Observation window [0, T]
        kernel_type : KernelType
            Type of kernel to fit
        method : str
            Estimation method: "MLE" or "EM"
        **kwargs
            Additional arguments passed to optimizer

        Returns
        -------
        HawkesResult
            Estimation results

        Example
        -------
        >>> times = np.array([0.1, 0.3, 0.35, 0.8, 1.2, 1.25, 1.3, 2.0])
        >>> result = UnivariateHawkes.fit(times, T=2.5, kernel_type=KernelType.EXPONENTIAL)
        >>> print(f"Base intensity: {result.mu:.4f}")
        >>> print(f"Branching ratio: {result.branching_ratio:.4f}")
        """
        times = np.asarray(times, dtype=np.float64)
        times = np.sort(times)
        n = len(times)

        if n < 5:
            raise ValueError("Need at least 5 events for estimation")

        if method.upper() == "MLE":
            return cls._fit_mle(times, T, kernel_type, **kwargs)
        elif method.upper() == "EM":
            return cls._fit_em(times, T, kernel_type, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    @classmethod
    def _fit_mle(
        cls,
        times: EventTimes,
        T: float,
        kernel_type: KernelType,
        max_iter: int = 1000,
        tol: float = 1e-8
    ) -> HawkesResult:
        """Maximum Likelihood Estimation using Newton-Raphson"""

        n = len(times)

        # Initialize kernel based on type
        if kernel_type == KernelType.EXPONENTIAL:
            kernel = ExponentialKernel(alpha=0.5, beta=1.0)
            n_kernel_params = 2
        elif kernel_type == KernelType.POWER_LAW:
            kernel = PowerLawKernel(alpha=1.0, beta=2.0, c=0.1)
            n_kernel_params = 3
        elif kernel_type == KernelType.SUM_EXPONENTIALS:
            kernel = SumExponentialsKernel(alphas=[0.3, 0.2], betas=[5.0, 1.0])
            n_kernel_params = 4
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        # Initial parameter guess
        mu_init = n / T / 2  # Start with half the empirical rate

        def neg_log_likelihood(params: NDArray[np.float64]) -> float:
            """Negative log-likelihood for minimization"""
            mu = params[0]
            kernel_params = params[1:]

            if mu <= 0:
                return np.inf

            # Update kernel parameters
            try:
                if kernel_type == KernelType.EXPONENTIAL:
                    if kernel_params[0] <= 0 or kernel_params[1] <= 0:
                        return np.inf
                    kernel.set_params(alpha=kernel_params[0], beta=kernel_params[1])
                elif kernel_type == KernelType.POWER_LAW:
                    if kernel_params[0] <= 0 or kernel_params[1] <= 1 or kernel_params[2] <= 0:
                        return np.inf
                    kernel.set_params(
                        alpha=kernel_params[0],
                        beta=kernel_params[1],
                        c=kernel_params[2]
                    )
                elif kernel_type == KernelType.SUM_EXPONENTIALS:
                    if any(p <= 0 for p in kernel_params):
                        return np.inf
                    kernel.set_params(
                        alpha_0=kernel_params[0],
                        beta_0=kernel_params[1],
                        alpha_1=kernel_params[2],
                        beta_1=kernel_params[3]
                    )
            except ValueError:
                return np.inf

            # Check stationarity
            if kernel.norm() >= 1:
                return np.inf

            # Create model and compute log-likelihood
            model = cls(kernel=kernel, mu=mu)
            ll = model.log_likelihood(times, T)

            return -ll if np.isfinite(ll) else np.inf

        # Initial parameters
        if kernel_type == KernelType.EXPONENTIAL:
            x0 = np.array([mu_init, 0.3, 2.0])
            bounds = [(1e-8, None), (1e-8, 10.0), (1e-8, 100.0)]
        elif kernel_type == KernelType.POWER_LAW:
            x0 = np.array([mu_init, 0.5, 2.5, 0.1])
            bounds = [(1e-8, None), (1e-8, 10.0), (1.01, 5.0), (1e-4, 1.0)]
        elif kernel_type == KernelType.SUM_EXPONENTIALS:
            x0 = np.array([mu_init, 0.2, 5.0, 0.1, 1.0])
            bounds = [(1e-8, None), (1e-8, 5.0), (1e-8, 100.0), (1e-8, 5.0), (1e-8, 100.0)]

        # Optimize
        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": tol}
        )

        # Extract results
        mu_opt = result.x[0]
        kernel_params_opt = result.x[1:]

        if kernel_type == KernelType.EXPONENTIAL:
            kernel.set_params(alpha=kernel_params_opt[0], beta=kernel_params_opt[1])
        elif kernel_type == KernelType.POWER_LAW:
            kernel.set_params(
                alpha=kernel_params_opt[0],
                beta=kernel_params_opt[1],
                c=kernel_params_opt[2]
            )
        elif kernel_type == KernelType.SUM_EXPONENTIALS:
            kernel.set_params(
                alpha_0=kernel_params_opt[0],
                beta_0=kernel_params_opt[1],
                alpha_1=kernel_params_opt[2],
                beta_1=kernel_params_opt[3]
            )

        ll = -result.fun
        n_params = 1 + n_kernel_params
        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n) - 2 * ll

        return HawkesResult(
            mu=mu_opt,
            kernel=kernel,
            branching_ratio=kernel.norm(),
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            n_events=n,
            T=T,
            converged=result.success,
            n_iterations=result.nit,
            message=result.message
        )

    @classmethod
    def _fit_em(
        cls,
        times: EventTimes,
        T: float,
        kernel_type: KernelType,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> HawkesResult:
        """
        EM Algorithm for Hawkes process estimation.

        E-step: Compute expected branching structure
        M-step: Update parameters
        """
        n = len(times)

        if kernel_type != KernelType.EXPONENTIAL:
            logger.warning("EM algorithm only supports exponential kernel, using MLE")
            return cls._fit_mle(times, T, kernel_type)

        # Initialize parameters
        mu = n / T / 2
        alpha = 0.3
        beta = 2.0

        kernel = ExponentialKernel(alpha=alpha, beta=beta)

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step: Compute p_ij = P(event j triggered by event i)
            # p_ij = g(t_j - t_i) / λ(t_j)

            p_matrix = np.zeros((n, n))
            p_background = np.zeros(n)

            for j in range(n):
                # Compute intensity at t_j
                lam_j = mu
                for i in range(j):
                    dt = times[j] - times[i]
                    lam_j += kernel(dt)

                # Background probability
                p_background[j] = mu / lam_j

                # Trigger probabilities
                for i in range(j):
                    dt = times[j] - times[i]
                    p_matrix[i, j] = kernel(dt) / lam_j

            # M-step: Update parameters

            # Update mu
            mu_new = np.sum(p_background) / T

            # Update alpha and beta (for exponential kernel)
            # Expected number of children per event
            expected_children = np.sum(p_matrix, axis=1)

            # Total expected children
            total_children = np.sum(expected_children)

            # Update alpha using expected branching ratio
            branching_ratio_emp = total_children / n

            # Update beta using weighted average of inter-arrival times
            weighted_sum_dt = 0.0
            weight_sum = 0.0

            for j in range(n):
                for i in range(j):
                    dt = times[j] - times[i]
                    weight = p_matrix[i, j]
                    weighted_sum_dt += weight * dt
                    weight_sum += weight

            if weight_sum > 0:
                mean_dt = weighted_sum_dt / weight_sum
                beta_new = 1.0 / mean_dt if mean_dt > 0 else beta
            else:
                beta_new = beta

            alpha_new = branching_ratio_emp * beta_new

            # Ensure stationarity
            if alpha_new / beta_new >= 0.99:
                alpha_new = 0.99 * beta_new

            # Update parameters
            mu = max(mu_new, 1e-8)
            alpha = max(alpha_new, 1e-8)
            beta = max(beta_new, 1e-8)

            kernel.set_params(alpha=alpha, beta=beta)

            # Check convergence
            model = cls(kernel=kernel, mu=mu)
            ll = model.log_likelihood(times, T)

            if ll - prev_ll < tol and iteration > 5:
                break

            prev_ll = ll

        n_params = 3
        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n) - 2 * ll

        return HawkesResult(
            mu=mu,
            kernel=kernel,
            branching_ratio=kernel.norm(),
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            n_events=n,
            T=T,
            converged=True,
            n_iterations=iteration + 1,
            message="EM converged"
        )


# =============================================================================
# Multivariate Hawkes Process
# =============================================================================

@dataclass
class MultivariateHawkesResult:
    """Container for multivariate Hawkes estimation results"""

    mu: NDArray[np.float64]  # Base intensities (D,)
    kernels: List[List[HawkesKernel]]  # Kernel matrix (D x D)
    branching_matrix: NDArray[np.float64]  # (D x D)
    spectral_radius: float
    log_likelihood: float
    n_events: List[int]
    T: float
    converged: bool


class MultivariateHawkes:
    """
    Multivariate Hawkes Process for modeling cross-excitation.

    The conditional intensity for dimension d is:
    λᵈ(t) = μᵈ + Σₑ Σᵢ gᵈᵉ(t - tᵢᵉ)

    where gᵈᵉ captures how events in dimension e excite events in dimension d.

    Applications
    ------------
    - Bid/ask order flow modeling
    - Multi-asset contagion
    - Cross-exchange dynamics

    Parameters
    ----------
    dim : int
        Number of dimensions (event types)
    kernels : Optional[List[List[HawkesKernel]]]
        D x D matrix of kernels, kernels[d][e] is excitation from e to d
    mu : Optional[NDArray]
        Base intensities (D,)

    Example
    -------
    >>> # 2D Hawkes: bid (0) and ask (1) orders
    >>> hawkes = MultivariateHawkes(dim=2)
    >>>
    >>> # Set cross-excitation: bids trigger asks and vice versa
    >>> hawkes.set_kernel(0, 1, ExponentialKernel(0.3, 2.0))  # ask -> bid
    >>> hawkes.set_kernel(1, 0, ExponentialKernel(0.3, 2.0))  # bid -> ask
    >>>
    >>> # Simulate
    >>> times, marks = hawkes.simulate(T=100.0)
    """

    def __init__(
        self,
        dim: int = 2,
        kernels: Optional[List[List[HawkesKernel]]] = None,
        mu: Optional[ArrayLike] = None
    ):
        self.dim = dim

        # Initialize kernels (D x D matrix)
        if kernels is None:
            self.kernels = [
                [ExponentialKernel(0.1, 1.0) for _ in range(dim)]
                for _ in range(dim)
            ]
        else:
            if len(kernels) != dim or any(len(row) != dim for row in kernels):
                raise ValueError(f"kernels must be {dim}x{dim}")
            self.kernels = kernels

        # Initialize base intensities
        if mu is None:
            self.mu = np.ones(dim) * 0.5
        else:
            self.mu = np.asarray(mu, dtype=np.float64)
            if len(self.mu) != dim:
                raise ValueError(f"mu must have length {dim}")

    def set_kernel(self, d: int, e: int, kernel: HawkesKernel) -> None:
        """Set kernel for excitation from dimension e to dimension d"""
        if not (0 <= d < self.dim and 0 <= e < self.dim):
            raise IndexError(f"Invalid indices ({d}, {e}) for dim={self.dim}")
        self.kernels[d][e] = kernel

    @property
    def branching_matrix(self) -> NDArray[np.float64]:
        """
        Compute branching matrix where entry (d,e) is the expected number
        of type-d events triggered by a type-e event.
        """
        matrix = np.zeros((self.dim, self.dim))
        for d in range(self.dim):
            for e in range(self.dim):
                matrix[d, e] = self.kernels[d][e].norm()
        return matrix

    @property
    def spectral_radius(self) -> float:
        """
        Spectral radius of branching matrix.
        Must be < 1 for stationarity.
        """
        eigenvalues = np.linalg.eigvals(self.branching_matrix)
        return np.max(np.abs(eigenvalues))

    @property
    def is_stationary(self) -> bool:
        """Check if process is stationary"""
        return self.spectral_radius < 1.0

    def intensity(
        self,
        d: int,
        t: float,
        times: List[EventTimes],
        marks: List[int]
    ) -> float:
        """
        Compute conditional intensity for dimension d at time t.

        Parameters
        ----------
        d : int
            Dimension (event type)
        t : float
            Time
        times : List[EventTimes]
            Event times by dimension
        marks : List[int]
            Not used (for API compatibility)

        Returns
        -------
        float
            Conditional intensity
        """
        lam = self.mu[d]

        for e in range(self.dim):
            past_events = times[e][times[e] < t]
            if len(past_events) > 0:
                time_lags = t - past_events
                lam += np.sum(self.kernels[d][e](time_lags))

        return lam

    def simulate(
        self,
        T: float,
        max_events: int = 100000,
        seed: Optional[int] = None
    ) -> Tuple[EventTimes, NDArray[np.int64]]:
        """
        Simulate multivariate Hawkes process using thinning.

        Parameters
        ----------
        T : float
            Simulation horizon
        max_events : int
            Maximum total events
        seed : Optional[int]
            Random seed

        Returns
        -------
        Tuple[EventTimes, NDArray[np.int64]]
            (event_times, event_marks) where marks indicate dimension
        """
        if not self.is_stationary:
            logger.warning("Process is not stationary (spectral radius >= 1)")

        if seed is not None:
            np.random.seed(seed)

        # Track events by dimension
        events_by_dim: List[List[float]] = [[] for _ in range(self.dim)]
        all_events: List[Tuple[float, int]] = []

        t = 0.0

        while t < T and len(all_events) < max_events:
            # Compute intensity upper bound
            intensities = [
                self.intensity(d, t, [np.array(e) for e in events_by_dim], [])
                for d in range(self.dim)
            ]

            # Add kernel values at t=0 for upper bound
            lam_bar = sum(intensities)
            for d in range(self.dim):
                for e in range(self.dim):
                    k0 = self.kernels[d][e](np.array([0.0]))[0]
                    lam_bar += k0

            if lam_bar <= 0:
                lam_bar = sum(self.mu)

            # Generate waiting time
            u1 = np.random.uniform()
            w = -np.log(u1) / lam_bar
            t_new = t + w

            if t_new > T:
                break

            # Compute actual intensities
            intensities_new = [
                self.intensity(d, t_new, [np.array(e) for e in events_by_dim], [])
                for d in range(self.dim)
            ]
            total_intensity = sum(intensities_new)

            # Accept/reject
            u2 = np.random.uniform()
            if u2 <= total_intensity / lam_bar:
                # Determine which dimension
                u3 = np.random.uniform() * total_intensity
                cumsum = 0.0
                for d in range(self.dim):
                    cumsum += intensities_new[d]
                    if u3 <= cumsum:
                        events_by_dim[d].append(t_new)
                        all_events.append((t_new, d))
                        break

            t = t_new

        # Sort by time
        all_events.sort(key=lambda x: x[0])

        if len(all_events) == 0:
            return np.array([]), np.array([], dtype=np.int64)

        times = np.array([e[0] for e in all_events])
        marks = np.array([e[1] for e in all_events], dtype=np.int64)

        return times, marks

    def log_likelihood(
        self,
        times: EventTimes,
        marks: NDArray[np.int64],
        T: float
    ) -> float:
        """
        Compute log-likelihood of multivariate event data.

        Parameters
        ----------
        times : EventTimes
            All event times (sorted)
        marks : NDArray[np.int64]
            Event types/dimensions
        T : float
            Observation window

        Returns
        -------
        float
            Log-likelihood
        """
        n = len(times)

        # Organize by dimension
        times_by_dim = [times[marks == d] for d in range(self.dim)]

        ll = 0.0

        # Log intensity terms
        for i in range(n):
            ti = times[i]
            di = marks[i]

            # Past events by dimension
            past_by_dim = [times_by_dim[d][times_by_dim[d] < ti] for d in range(self.dim)]

            lam = self.intensity(di, ti, past_by_dim, [])
            if lam > 0:
                ll += np.log(lam)
            else:
                return -np.inf

        # Compensator terms
        for d in range(self.dim):
            # Base intensity contribution
            ll -= self.mu[d] * T

            # Excitation contribution
            for e in range(self.dim):
                for te in times_by_dim[e]:
                    ll -= self.kernels[d][e].integral(0, T - te)

        return ll

    @classmethod
    def fit(
        cls,
        times: EventTimes,
        marks: NDArray[np.int64],
        T: float,
        dim: Optional[int] = None,
        max_iter: int = 1000,
        tol: float = 1e-8
    ) -> MultivariateHawkesResult:
        """
        Fit multivariate Hawkes process to data.

        Parameters
        ----------
        times : EventTimes
            Event times
        marks : NDArray[np.int64]
            Event types (0 to dim-1)
        T : float
            Observation window
        dim : Optional[int]
            Number of dimensions (inferred if None)
        max_iter : int
            Maximum optimization iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        MultivariateHawkesResult
            Estimation results
        """
        if dim is None:
            dim = int(np.max(marks)) + 1

        times = np.asarray(times)
        marks = np.asarray(marks, dtype=np.int64)

        # Count events by dimension
        n_events = [np.sum(marks == d) for d in range(dim)]

        # Initialize
        model = cls(dim=dim)

        # Initial base intensities
        for d in range(dim):
            model.mu[d] = n_events[d] / T / 2

        # Optimize using L-BFGS-B
        def neg_ll(params: NDArray[np.float64]) -> float:
            # Parse parameters: mu (dim) + alpha, beta for each kernel (dim^2 * 2)
            mu = params[:dim]
            kernel_params = params[dim:].reshape(dim, dim, 2)

            if np.any(mu <= 0):
                return np.inf
            if np.any(kernel_params <= 0):
                return np.inf

            # Update model
            model.mu = mu
            for d in range(dim):
                for e in range(dim):
                    alpha, beta = kernel_params[d, e]
                    model.kernels[d][e].set_params(alpha=alpha, beta=beta)

            # Check stationarity
            if not model.is_stationary:
                return np.inf

            ll = model.log_likelihood(times, marks, T)
            return -ll if np.isfinite(ll) else np.inf

        # Initial guess
        x0 = np.concatenate([
            model.mu,
            np.array([[0.2, 2.0] for _ in range(dim * dim)]).flatten()
        ])

        # Bounds
        bounds = (
            [(1e-8, None)] * dim +  # mu bounds
            [(1e-8, 5.0), (1e-8, 50.0)] * (dim * dim)  # kernel bounds
        )

        result = optimize.minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": tol}
        )

        # Extract final parameters
        mu = result.x[:dim]
        kernel_params = result.x[dim:].reshape(dim, dim, 2)

        model.mu = mu
        for d in range(dim):
            for e in range(dim):
                alpha, beta = kernel_params[d, e]
                model.kernels[d][e].set_params(alpha=alpha, beta=beta)

        return MultivariateHawkesResult(
            mu=mu,
            kernels=model.kernels,
            branching_matrix=model.branching_matrix,
            spectral_radius=model.spectral_radius,
            log_likelihood=-result.fun,
            n_events=n_events,
            T=T,
            converged=result.success
        )


# =============================================================================
# Trading Applications
# =============================================================================

@dataclass
class OrderFlowMetrics:
    """Metrics from Hawkes-based order flow analysis"""

    intensity_mean: float
    intensity_std: float
    branching_ratio: float
    self_excitation_ratio: float  # Proportion from self-excitation
    cluster_count: int
    avg_cluster_size: float
    toxicity_score: float  # Higher = more informed trading


class HawkesTradingAnalyzer:
    """
    Trading applications of Hawkes processes.

    Provides tools for:
    - Trade clustering detection
    - Order flow toxicity measurement
    - Next event prediction
    - Bid/ask dynamics modeling

    Example
    -------
    >>> analyzer = HawkesTradingAnalyzer()
    >>>
    >>> # Fit to trade times
    >>> trade_times = np.array([...])  # seconds from start
    >>> result = analyzer.fit_order_flow(trade_times, T=3600)  # 1 hour
    >>>
    >>> # Detect clusters
    >>> clusters = analyzer.detect_clusters(trade_times, result)
    >>> print(f"Found {len(clusters)} trade clusters")
    >>>
    >>> # Compute toxicity
    >>> toxicity = analyzer.compute_toxicity(trade_times, result)
    """

    def __init__(self):
        self._fitted_model: Optional[UnivariateHawkes] = None
        self._fitted_result: Optional[HawkesResult] = None

    def fit_order_flow(
        self,
        trade_times: ArrayLike,
        T: float,
        kernel_type: KernelType = KernelType.EXPONENTIAL,
        method: str = "MLE"
    ) -> HawkesResult:
        """
        Fit Hawkes process to trade arrival times.

        Parameters
        ----------
        trade_times : ArrayLike
            Trade timestamps (relative to start)
        T : float
            Observation window length
        kernel_type : KernelType
            Kernel type for excitation
        method : str
            Estimation method

        Returns
        -------
        HawkesResult
            Fitted model results
        """
        times = np.asarray(trade_times, dtype=np.float64)
        times = np.sort(times)

        result = UnivariateHawkes.fit(times, T, kernel_type, method)

        self._fitted_result = result
        self._fitted_model = UnivariateHawkes(
            kernel=result.kernel,
            mu=result.mu
        )

        return result

    def detect_clusters(
        self,
        trade_times: ArrayLike,
        result: Optional[HawkesResult] = None,
        threshold_factor: float = 2.0
    ) -> List[Tuple[int, int, float]]:
        """
        Detect trade clusters using intensity threshold.

        A cluster starts when intensity exceeds threshold and ends when
        it falls below.

        Parameters
        ----------
        trade_times : ArrayLike
            Trade timestamps
        result : Optional[HawkesResult]
            Fitted result (uses internal if None)
        threshold_factor : float
            Multiple of base intensity for cluster threshold

        Returns
        -------
        List[Tuple[int, int, float]]
            List of (start_idx, end_idx, peak_intensity) for each cluster
        """
        if result is None:
            result = self._fitted_result
        if result is None:
            raise ValueError("Must fit model first or provide result")

        times = np.asarray(trade_times, dtype=np.float64)
        model = UnivariateHawkes(kernel=result.kernel, mu=result.mu)

        # Compute threshold
        threshold = result.mu * threshold_factor

        clusters: List[Tuple[int, int, float]] = []
        in_cluster = False
        cluster_start = 0
        peak_intensity = 0.0

        for i, t in enumerate(times):
            intensity = model.intensity(t, times[:i])

            if not in_cluster and intensity > threshold:
                # Start new cluster
                in_cluster = True
                cluster_start = i
                peak_intensity = intensity
            elif in_cluster:
                peak_intensity = max(peak_intensity, intensity)
                if intensity < threshold:
                    # End cluster
                    clusters.append((cluster_start, i - 1, peak_intensity))
                    in_cluster = False

        # Handle cluster at end
        if in_cluster:
            clusters.append((cluster_start, len(times) - 1, peak_intensity))

        return clusters

    def compute_toxicity(
        self,
        trade_times: ArrayLike,
        result: Optional[HawkesResult] = None,
        window: int = 50
    ) -> NDArray[np.float64]:
        """
        Compute order flow toxicity using Hawkes-based measure.

        Toxicity is high when self-excitation dominates (informed trading
        triggers cascade of follow-on trades).

        Parameters
        ----------
        trade_times : ArrayLike
            Trade timestamps
        result : Optional[HawkesResult]
            Fitted result
        window : int
            Rolling window size

        Returns
        -------
        NDArray[np.float64]
            Toxicity scores for each trade (0 to 1)
        """
        if result is None:
            result = self._fitted_result
        if result is None:
            raise ValueError("Must fit model first or provide result")

        times = np.asarray(trade_times, dtype=np.float64)
        n = len(times)
        model = UnivariateHawkes(kernel=result.kernel, mu=result.mu)

        toxicity = np.zeros(n)

        for i in range(n):
            intensity = model.intensity(times[i], times[:i])

            # Toxicity = fraction from self-excitation
            if intensity > 0:
                excitation_part = intensity - result.mu
                toxicity[i] = max(0, excitation_part / intensity)
            else:
                toxicity[i] = 0.0

        # Apply rolling average for smoothing
        if window > 1 and n >= window:
            toxicity_smooth = np.convolve(
                toxicity,
                np.ones(window) / window,
                mode='same'
            )
            return toxicity_smooth

        return toxicity

    def predict_next_event(
        self,
        trade_times: ArrayLike,
        result: Optional[HawkesResult] = None,
        horizon: float = 60.0,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Predict time to next trade using Monte Carlo simulation.

        Parameters
        ----------
        trade_times : ArrayLike
            Historical trade times
        result : Optional[HawkesResult]
            Fitted result
        horizon : float
            Maximum prediction horizon
        n_samples : int
            Number of Monte Carlo samples

        Returns
        -------
        Dict[str, float]
            Prediction statistics: mean, std, median, p10, p90
        """
        if result is None:
            result = self._fitted_result
        if result is None:
            raise ValueError("Must fit model first or provide result")

        times = np.asarray(trade_times, dtype=np.float64)
        current_time = times[-1] if len(times) > 0 else 0.0

        model = UnivariateHawkes(kernel=result.kernel, mu=result.mu)

        # Simulate next event times
        next_times = []

        for _ in range(n_samples):
            t = current_time

            while t < current_time + horizon:
                # Intensity at current time
                intensity = model.intensity(t, times)

                # Upper bound (intensity can only decrease without new events)
                lam_bar = intensity + 0.1  # Small buffer

                # Wait time
                u1 = np.random.uniform()
                w = -np.log(u1) / lam_bar
                t_new = t + w

                if t_new > current_time + horizon:
                    next_times.append(horizon)
                    break

                # Accept/reject
                intensity_new = model.intensity(t_new, times)
                u2 = np.random.uniform()

                if u2 <= intensity_new / lam_bar:
                    next_times.append(t_new - current_time)
                    break

                t = t_new

        next_times = np.array(next_times)

        return {
            "mean": float(np.mean(next_times)),
            "std": float(np.std(next_times)),
            "median": float(np.median(next_times)),
            "p10": float(np.percentile(next_times, 10)),
            "p90": float(np.percentile(next_times, 90))
        }

    def compute_metrics(
        self,
        trade_times: ArrayLike,
        result: Optional[HawkesResult] = None,
        T: Optional[float] = None
    ) -> OrderFlowMetrics:
        """
        Compute comprehensive order flow metrics.

        Parameters
        ----------
        trade_times : ArrayLike
            Trade timestamps
        result : Optional[HawkesResult]
            Fitted result
        T : Optional[float]
            Observation window (inferred if None)

        Returns
        -------
        OrderFlowMetrics
            Computed metrics
        """
        if result is None:
            result = self._fitted_result
        if result is None:
            raise ValueError("Must fit model first or provide result")

        times = np.asarray(trade_times, dtype=np.float64)

        if T is None:
            T = times[-1] - times[0] if len(times) > 1 else 1.0

        model = UnivariateHawkes(kernel=result.kernel, mu=result.mu)

        # Compute intensity path
        _, intensities = model.intensity_path(times, n_points=500)

        # Detect clusters
        clusters = self.detect_clusters(times, result)

        # Compute toxicity
        toxicity = self.compute_toxicity(times, result)

        # Average cluster size
        if len(clusters) > 0:
            cluster_sizes = [end - start + 1 for start, end, _ in clusters]
            avg_cluster_size = np.mean(cluster_sizes)
        else:
            avg_cluster_size = 0.0

        return OrderFlowMetrics(
            intensity_mean=float(np.mean(intensities)),
            intensity_std=float(np.std(intensities)),
            branching_ratio=result.branching_ratio,
            self_excitation_ratio=float(result.branching_ratio),
            cluster_count=len(clusters),
            avg_cluster_size=float(avg_cluster_size),
            toxicity_score=float(np.mean(toxicity))
        )


class BidAskHawkes:
    """
    Bid-Ask Order Flow Modeling with Bivariate Hawkes.

    Models the cross-excitation between bid and ask orders:
    - Self-excitation: bids trigger more bids, asks trigger more asks
    - Cross-excitation: bids can trigger asks (and vice versa)

    This is useful for:
    - Understanding market maker behavior
    - Detecting order flow imbalances
    - Modeling spread dynamics

    Example
    -------
    >>> model = BidAskHawkes()
    >>>
    >>> # Fit to order flow data
    >>> bid_times = np.array([...])
    >>> ask_times = np.array([...])
    >>> model.fit(bid_times, ask_times, T=3600)
    >>>
    >>> # Analyze cross-excitation
    >>> print(f"Bid->Ask excitation: {model.bid_to_ask_excitation:.3f}")
    >>> print(f"Ask->Bid excitation: {model.ask_to_bid_excitation:.3f}")
    """

    def __init__(self):
        self.hawkes = MultivariateHawkes(dim=2)
        self._fitted = False
        self._result: Optional[MultivariateHawkesResult] = None

    @property
    def bid_self_excitation(self) -> float:
        """Self-excitation intensity for bids (bid->bid)"""
        return self.hawkes.kernels[0][0].norm()

    @property
    def ask_self_excitation(self) -> float:
        """Self-excitation intensity for asks (ask->ask)"""
        return self.hawkes.kernels[1][1].norm()

    @property
    def bid_to_ask_excitation(self) -> float:
        """Cross-excitation: how much bids trigger asks"""
        return self.hawkes.kernels[1][0].norm()

    @property
    def ask_to_bid_excitation(self) -> float:
        """Cross-excitation: how much asks trigger bids"""
        return self.hawkes.kernels[0][1].norm()

    def fit(
        self,
        bid_times: ArrayLike,
        ask_times: ArrayLike,
        T: float
    ) -> MultivariateHawkesResult:
        """
        Fit bivariate Hawkes to bid/ask order flow.

        Parameters
        ----------
        bid_times : ArrayLike
            Bid order timestamps
        ask_times : ArrayLike
            Ask order timestamps
        T : float
            Observation window

        Returns
        -------
        MultivariateHawkesResult
            Fitted model results
        """
        bid_times = np.asarray(bid_times, dtype=np.float64)
        ask_times = np.asarray(ask_times, dtype=np.float64)

        # Combine and create marks
        all_times = np.concatenate([bid_times, ask_times])
        marks = np.concatenate([
            np.zeros(len(bid_times), dtype=np.int64),
            np.ones(len(ask_times), dtype=np.int64)
        ])

        # Sort by time
        sort_idx = np.argsort(all_times)
        all_times = all_times[sort_idx]
        marks = marks[sort_idx]

        result = MultivariateHawkes.fit(all_times, marks, T, dim=2)

        self._result = result
        self.hawkes.mu = result.mu
        self.hawkes.kernels = result.kernels
        self._fitted = True

        return result

    def compute_imbalance(
        self,
        bid_times: ArrayLike,
        ask_times: ArrayLike,
        eval_times: Optional[ArrayLike] = None
    ) -> NDArray[np.float64]:
        """
        Compute order flow imbalance over time.

        Imbalance = (λ_bid - λ_ask) / (λ_bid + λ_ask)

        Positive = more bid pressure, Negative = more ask pressure

        Parameters
        ----------
        bid_times : ArrayLike
            Bid timestamps
        ask_times : ArrayLike
            Ask timestamps
        eval_times : Optional[ArrayLike]
            Times to evaluate (uses all order times if None)

        Returns
        -------
        NDArray[np.float64]
            Imbalance values at evaluation times
        """
        if not self._fitted:
            raise ValueError("Must fit model first")

        bid_times = np.asarray(bid_times, dtype=np.float64)
        ask_times = np.asarray(ask_times, dtype=np.float64)

        times_by_dim = [bid_times, ask_times]

        if eval_times is None:
            all_times = np.concatenate([bid_times, ask_times])
            eval_times = np.sort(all_times)
        else:
            eval_times = np.asarray(eval_times, dtype=np.float64)

        imbalance = np.zeros(len(eval_times))

        for i, t in enumerate(eval_times):
            # Compute intensities
            past_bid = bid_times[bid_times < t]
            past_ask = ask_times[ask_times < t]

            lam_bid = self.hawkes.intensity(0, t, [past_bid, past_ask], [])
            lam_ask = self.hawkes.intensity(1, t, [past_bid, past_ask], [])

            total = lam_bid + lam_ask
            if total > 0:
                imbalance[i] = (lam_bid - lam_ask) / total

        return imbalance

    def simulate(
        self,
        T: float,
        seed: Optional[int] = None
    ) -> Tuple[EventTimes, EventTimes]:
        """
        Simulate bid and ask order flow.

        Parameters
        ----------
        T : float
            Simulation horizon
        seed : Optional[int]
            Random seed

        Returns
        -------
        Tuple[EventTimes, EventTimes]
            (bid_times, ask_times)
        """
        times, marks = self.hawkes.simulate(T, seed=seed)

        bid_times = times[marks == 0]
        ask_times = times[marks == 1]

        return bid_times, ask_times


# =============================================================================
# Utility Functions
# =============================================================================

def goodness_of_fit_test(
    times: EventTimes,
    T: float,
    result: HawkesResult,
    n_bins: int = 50
) -> Dict[str, float]:
    """
    Perform goodness-of-fit test for Hawkes model.

    Uses the time-rescaling theorem: if the model is correct, the transformed
    times τᵢ = Λ(tᵢ) should be unit-rate Poisson (exponential interarrivals).

    Parameters
    ----------
    times : EventTimes
        Observed event times
    T : float
        Observation window
    result : HawkesResult
        Fitted model result
    n_bins : int
        Number of bins for KS test

    Returns
    -------
    Dict[str, float]
        Test statistics: ks_statistic, ks_pvalue, qq_slope, qq_intercept
    """
    from scipy import stats

    times = np.asarray(times, dtype=np.float64)
    model = UnivariateHawkes(kernel=result.kernel, mu=result.mu)

    # Compute compensator values (time-rescaled times)
    rescaled = np.zeros(len(times))
    for i, ti in enumerate(times):
        rescaled[i] = model.compensator(ti, times[:i+1])

    # Inter-arrival times of rescaled process
    rescaled_diffs = np.diff(np.concatenate([[0], rescaled]))

    # KS test against exponential(1)
    ks_stat, ks_pvalue = stats.kstest(rescaled_diffs, 'expon', args=(0, 1))

    # QQ plot statistics
    theoretical_quantiles = stats.expon.ppf(
        np.linspace(0.01, 0.99, len(rescaled_diffs))
    )
    empirical_quantiles = np.sort(rescaled_diffs)[:len(theoretical_quantiles)]

    # Linear regression for QQ slope
    slope, intercept, r_value, _, _ = stats.linregress(
        theoretical_quantiles, empirical_quantiles
    )

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "qq_slope": float(slope),
        "qq_intercept": float(intercept),
        "qq_r_squared": float(r_value ** 2)
    }


def estimate_branching_ratio_nonparametric(
    times: EventTimes,
    T: float,
    max_lag: float = 10.0,
    n_bins: int = 100
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Non-parametric estimation of branching ratio from autocorrelation.

    Uses the relationship between the autocorrelation function and
    the branching ratio for Hawkes processes.

    Parameters
    ----------
    times : EventTimes
        Event times
    T : float
        Observation window
    max_lag : float
        Maximum lag for autocorrelation
    n_bins : int
        Number of histogram bins

    Returns
    -------
    Tuple[float, NDArray, NDArray]
        (branching_ratio, lags, autocorrelation)
    """
    times = np.asarray(times, dtype=np.float64)
    n = len(times)

    # Compute all inter-event times
    inter_times = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = times[j] - times[i]
            if dt <= max_lag:
                inter_times.append(dt)
            else:
                break

    inter_times = np.array(inter_times)

    # Histogram
    hist, bin_edges = np.histogram(inter_times, bins=n_bins, range=(0, max_lag))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Normalize to get density estimate
    density = hist / (n * bin_width)

    # Empirical rate
    rate = n / T

    # Autocorrelation estimate
    acf = density / rate

    # Branching ratio estimate from integral of kernel
    # For large lags, acf -> 1, so branching_ratio ≈ integral(acf - 1) / rate
    branching_ratio = np.trapezoid(np.maximum(acf - 1, 0), bin_centers)
    branching_ratio = min(max(branching_ratio, 0), 0.99)

    return float(branching_ratio), bin_centers, acf


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hawkes Processes for Trading - Demo")
    print("=" * 60)

    # 1. Univariate Hawkes Simulation and Estimation
    print("\n1. Univariate Hawkes Process")
    print("-" * 40)

    # Create process
    kernel = ExponentialKernel(alpha=0.6, beta=2.0)
    hawkes = UnivariateHawkes(kernel=kernel, mu=1.0)

    print(f"True parameters: mu=1.0, alpha=0.6, beta=2.0")
    print(f"Branching ratio: {hawkes.branching_ratio:.3f}")
    print(f"Stationary: {hawkes.is_stationary}")

    # Simulate
    times = hawkes.simulate(T=500.0, seed=42)
    print(f"Simulated {len(times)} events")

    # Fit
    result = UnivariateHawkes.fit(times, T=500.0, kernel_type=KernelType.EXPONENTIAL)
    print(f"\nEstimated parameters:")
    print(f"  mu: {result.mu:.4f}")
    print(f"  alpha: {result.kernel.get_params()['alpha']:.4f}")
    print(f"  beta: {result.kernel.get_params()['beta']:.4f}")
    print(f"  Branching ratio: {result.branching_ratio:.4f}")
    print(f"  Log-likelihood: {result.log_likelihood:.2f}")
    print(f"  AIC: {result.aic:.2f}")

    # Goodness of fit
    gof = goodness_of_fit_test(times, 500.0, result)
    print(f"\nGoodness of fit:")
    print(f"  KS statistic: {gof['ks_statistic']:.4f}")
    print(f"  KS p-value: {gof['ks_pvalue']:.4f}")
    print(f"  QQ R-squared: {gof['qq_r_squared']:.4f}")

    # 2. Trading Analysis
    print("\n2. Trading Application")
    print("-" * 40)

    analyzer = HawkesTradingAnalyzer()
    analyzer.fit_order_flow(times, T=500.0)

    # Detect clusters
    clusters = analyzer.detect_clusters(times)
    print(f"Detected {len(clusters)} trade clusters")

    # Compute metrics
    metrics = analyzer.compute_metrics(times)
    print(f"Order flow metrics:")
    print(f"  Mean intensity: {metrics.intensity_mean:.4f}")
    print(f"  Toxicity score: {metrics.toxicity_score:.4f}")
    print(f"  Avg cluster size: {metrics.avg_cluster_size:.2f}")

    # Predict next event
    pred = analyzer.predict_next_event(times[:100])
    print(f"\nNext event prediction:")
    print(f"  Mean time: {pred['mean']:.2f}s")
    print(f"  Std: {pred['std']:.2f}s")
    print(f"  [10%, 90%]: [{pred['p10']:.2f}, {pred['p90']:.2f}]s")

    # 3. Bid-Ask Modeling
    print("\n3. Bid-Ask Order Flow")
    print("-" * 40)

    # Create bivariate model
    bid_ask_hawkes = MultivariateHawkes(dim=2)
    bid_ask_hawkes.mu = np.array([0.5, 0.5])
    bid_ask_hawkes.set_kernel(0, 0, ExponentialKernel(0.4, 2.0))  # bid->bid
    bid_ask_hawkes.set_kernel(1, 1, ExponentialKernel(0.4, 2.0))  # ask->ask
    bid_ask_hawkes.set_kernel(0, 1, ExponentialKernel(0.2, 3.0))  # ask->bid
    bid_ask_hawkes.set_kernel(1, 0, ExponentialKernel(0.2, 3.0))  # bid->ask

    print(f"Spectral radius: {bid_ask_hawkes.spectral_radius:.3f}")
    print(f"Stationary: {bid_ask_hawkes.is_stationary}")

    # Simulate
    all_times, marks = bid_ask_hawkes.simulate(T=200.0, seed=123)
    bid_times = all_times[marks == 0]
    ask_times = all_times[marks == 1]
    print(f"Simulated {len(bid_times)} bids, {len(ask_times)} asks")

    # Fit
    bid_ask_model = BidAskHawkes()
    ba_result = bid_ask_model.fit(bid_times, ask_times, T=200.0)

    print(f"\nEstimated cross-excitation:")
    print(f"  Bid->Bid: {bid_ask_model.bid_self_excitation:.4f}")
    print(f"  Ask->Ask: {bid_ask_model.ask_self_excitation:.4f}")
    print(f"  Bid->Ask: {bid_ask_model.bid_to_ask_excitation:.4f}")
    print(f"  Ask->Bid: {bid_ask_model.ask_to_bid_excitation:.4f}")

    # Compute imbalance
    imbalance = bid_ask_model.compute_imbalance(bid_times, ask_times)
    print(f"\nOrder imbalance statistics:")
    print(f"  Mean: {np.mean(imbalance):.4f}")
    print(f"  Std: {np.std(imbalance):.4f}")
    print(f"  Range: [{np.min(imbalance):.4f}, {np.max(imbalance):.4f}]")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
