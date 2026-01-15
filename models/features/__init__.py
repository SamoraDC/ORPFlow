"""
Advanced Quantitative Features Library.

This package provides streaming-compatible quantitative features for
algorithmic trading including:

- Hawkes Process (self-exciting point processes)
- Kalman Filter (state estimation)
- Signal Transforms (Fisher, Hilbert, Wavelet)
- RMT Correlation Filter (Random Matrix Theory)
- Hidden Markov Models (regime detection)
- Fractal Analysis (Hurst exponent, fractal dimension)
- Market Microstructure (VPIN, OFI, Spread Decomposition, Toxicity)
"""

from .microstructure import (
    # Main calculators
    VPINCalculator,
    OFICalculator,
    SpreadDecomposition,
    QueueMetrics,
    ToxicityIndicators,
    MicrostructureEngine,
    # Data classes
    Trade,
    Quote,
    OHLCV,
    VolumeBucket,
    LOBLevel,
    LOBSnapshot,
)

from .quant_features import (
    # Base classes
    OnlineFeature,
    FitTransformFeature,
    # Hawkes Process
    HawkesIntensity,
    HawkesState,
    # Kalman Filter
    KalmanFilter,
    KalmanState,
    # Signal Transforms
    fisher_transform,
    inverse_fisher_transform,
    normalize_to_fisher_range,
    hilbert_transform,
    instantaneous_frequency,
    wavelet_transform,
    wavelet_reconstruct,
    wavelet_denoise,
    WaveletDecomposition,
    # RMT Correlation Filter
    RMTCorrelationFilter,
    RMTFilterResult,
    # Hidden Markov Model
    MarketRegimeHMM,
    HMMState,
    # Fractal Analysis
    hurst_exponent,
    fractal_dimension,
    FractalAnalyzer,
    # Composite Generator
    QuantFeatureGenerator,
)

__all__ = [
    # Microstructure
    'VPINCalculator',
    'OFICalculator',
    'SpreadDecomposition',
    'QueueMetrics',
    'ToxicityIndicators',
    'MicrostructureEngine',
    'Trade',
    'Quote',
    'OHLCV',
    'VolumeBucket',
    'LOBLevel',
    'LOBSnapshot',
    # Base classes
    'OnlineFeature',
    'FitTransformFeature',
    # Hawkes Process
    'HawkesIntensity',
    'HawkesState',
    # Kalman Filter
    'KalmanFilter',
    'KalmanState',
    # Signal Transforms
    'fisher_transform',
    'inverse_fisher_transform',
    'normalize_to_fisher_range',
    'hilbert_transform',
    'instantaneous_frequency',
    'wavelet_transform',
    'wavelet_reconstruct',
    'wavelet_denoise',
    'WaveletDecomposition',
    # RMT Correlation Filter
    'RMTCorrelationFilter',
    'RMTFilterResult',
    # Hidden Markov Model
    'MarketRegimeHMM',
    'HMMState',
    # Fractal Analysis
    'hurst_exponent',
    'fractal_dimension',
    'FractalAnalyzer',
    # Composite Generator
    'QuantFeatureGenerator',
]
