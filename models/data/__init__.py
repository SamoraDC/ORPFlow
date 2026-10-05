"""Data Collection, Preprocessing, and Signal Processing"""

from .collector import BinanceDataCollector
from .preprocessor import FeatureEngineer
from .signal_processing import (
    # Kalman Filter
    KalmanFilter,
    KalmanState,
    ExtendedKalmanFilter,
    # EMD
    EMD,
    EEMD,
    CEEMDAN,
    # Hilbert-Huang
    HilbertHuangTransform,
    # Wavelets
    WaveletTransform,
    # RMT
    RandomMatrixTheory,
    CorrelationCleaner,
    # Fisher Transform
    FisherTransform,
    # Convenience functions
    kalman_smooth_prices,
    emd_decompose,
    wavelet_denoise,
    clean_correlation_matrix,
    fisher_transform_indicator,
)
from .microstructure import (
    MicrostructureAnalyzer,
    calculate_all_microstructure_features,
    # Order Flow
    order_flow_imbalance,
    vpin,
    order_flow_persistence,
    signed_trade_volume,
    # Micro-price
    microprice,
    # Book Metrics
    book_imbalance,
    book_pressure,
    queue_position_estimation,
    depth_profile_analysis,
    # Spreads
    spread_metrics,
    # Liquidity
    kyle_lambda,
    amihud_illiquidity,
    roll_measure,
    pastor_stambaugh_liquidity,
    corwin_schultz_spread,
    # Realized Volatility
    realized_variance,
    bipower_variation,
    realized_kernel,
    jump_detection_lee_mykland,
)
from .triple_barrier import (
    # Core functions
    get_daily_vol,
    get_vertical_barrier,
    get_horizontal_barriers,
    get_events,
    get_labels,
    # Dynamic barriers
    get_atr,
    get_atr_barriers,
    get_asymmetric_barriers,
    get_adaptive_barriers,
    # Meta-labeling
    get_meta_labels,
    combine_predictions,
    # Sample weights
    get_num_co_events,
    get_sample_uniqueness,
    get_sample_tw,
    get_decay_weights,
    get_sample_weights,
    # Regime detection
    detect_volatility_regime,
    detect_trend_regime,
    calculate_hurst_exponent,
    detect_hurst_regime,
    detect_combined_regime,
    # Event-based sampling
    cusum_filter,
    cusum_filter_symmetric,
    fractional_diff,
    find_min_ffd,
    event_driven_bars,
    # High-level interfaces
    TripleBarrierLabeler,
    MetaLabeler,
    create_labels_pipeline,
    validate_labels,
    # Configuration
    BarrierConfig,
    BarrierType,
    RegimeType,
)
from .feature_selection import (
    # Enums and Data Classes
    TaskType,
    FeatureImportanceResult,
    FeatureSelectionResult,
    # Filter Methods
    MutualInformationFilter,
    CorrelationFilter,
    VarianceFilter,
    ChiSquaredFilter,
    # Wrapper Methods
    RFESelector,
    RFECVSelector,
    SequentialSelector,
    # Embedded Methods
    SHAPImportance,
    PermutationImportance,
    LassoSelector,
    TreeBasedImportance,
    # Advanced Methods
    MeanDecreaseImpurity,
    MeanDecreaseAccuracy,
    SingleFeatureImportance,
    ClusteredFeatureImportance,
    # Validation
    ProbabilityOfBacktestOverfitting,
    DeflatedSharpeRatio,
    CombinatorialSymmetricCV,
    # Utilities
    FeatureClusteringAnalysis,
    VIFAnalysis,
    FeatureStabilityAnalysis,
    # Ensemble
    EnsembleFeatureSelector,
    # Sklearn Transformer
    FeatureSelector,
    # Convenience Functions
    select_features,
    compute_all_importances,
)
from .hawkes_processes import (
    # Kernels
    HawkesKernel,
    ExponentialKernel,
    PowerLawKernel,
    SumExponentialsKernel,
    KernelType,
    # Univariate Hawkes
    UnivariateHawkes,
    HawkesResult,
    # Multivariate Hawkes
    MultivariateHawkes,
    MultivariateHawkesResult,
    # Trading Applications
    HawkesTradingAnalyzer,
    BidAskHawkes,
    OrderFlowMetrics,
    # Utilities
    goodness_of_fit_test,
    estimate_branching_ratio_nonparametric,
)

__all__ = [
    # Original exports
    "BinanceDataCollector",
    "FeatureEngineer",
    # Signal Processing - Kalman Filter
    "KalmanFilter",
    "KalmanState",
    "ExtendedKalmanFilter",
    # Signal Processing - EMD
    "EMD",
    "EEMD",
    "CEEMDAN",
    # Signal Processing - Hilbert-Huang
    "HilbertHuangTransform",
    # Signal Processing - Wavelets
    "WaveletTransform",
    # Signal Processing - RMT
    "RandomMatrixTheory",
    "CorrelationCleaner",
    # Signal Processing - Fisher Transform
    "FisherTransform",
    # Signal Processing - Convenience functions
    "kalman_smooth_prices",
    "emd_decompose",
    "wavelet_denoise",
    "clean_correlation_matrix",
    "fisher_transform_indicator",
    # Microstructure
    "MicrostructureAnalyzer",
    "calculate_all_microstructure_features",
    # Order Flow
    "order_flow_imbalance",
    "vpin",
    "order_flow_persistence",
    "signed_trade_volume",
    # Micro-price
    "microprice",
    # Book Metrics
    "book_imbalance",
    "book_pressure",
    "queue_position_estimation",
    "depth_profile_analysis",
    # Spreads
    "spread_metrics",
    # Liquidity
    "kyle_lambda",
    "amihud_illiquidity",
    "roll_measure",
    "pastor_stambaugh_liquidity",
    "corwin_schultz_spread",
    # Realized Volatility
    "realized_variance",
    "bipower_variation",
    "realized_kernel",
    "jump_detection_lee_mykland",
    # Triple barrier core
    "get_daily_vol",
    "get_vertical_barrier",
    "get_horizontal_barriers",
    "get_events",
    "get_labels",
    # Dynamic barriers
    "get_atr",
    "get_atr_barriers",
    "get_asymmetric_barriers",
    "get_adaptive_barriers",
    # Meta-labeling
    "get_meta_labels",
    "combine_predictions",
    # Sample weights
    "get_num_co_events",
    "get_sample_uniqueness",
    "get_sample_tw",
    "get_decay_weights",
    "get_sample_weights",
    # Regime detection
    "detect_volatility_regime",
    "detect_trend_regime",
    "calculate_hurst_exponent",
    "detect_hurst_regime",
    "detect_combined_regime",
    # Event-based sampling
    "cusum_filter",
    "cusum_filter_symmetric",
    "fractional_diff",
    "find_min_ffd",
    "event_driven_bars",
    # High-level interfaces
    "TripleBarrierLabeler",
    "MetaLabeler",
    "create_labels_pipeline",
    "validate_labels",
    # Configuration
    "BarrierConfig",
    "BarrierType",
    "RegimeType",
    # Feature Selection - Core
    "TaskType",
    "FeatureImportanceResult",
    "FeatureSelectionResult",
    # Feature Selection - Filter Methods
    "MutualInformationFilter",
    "CorrelationFilter",
    "VarianceFilter",
    "ChiSquaredFilter",
    # Feature Selection - Wrapper Methods
    "RFESelector",
    "RFECVSelector",
    "SequentialSelector",
    # Feature Selection - Embedded Methods
    "SHAPImportance",
    "PermutationImportance",
    "LassoSelector",
    "TreeBasedImportance",
    # Feature Selection - Advanced Methods
    "MeanDecreaseImpurity",
    "MeanDecreaseAccuracy",
    "SingleFeatureImportance",
    "ClusteredFeatureImportance",
    # Feature Selection - Validation
    "ProbabilityOfBacktestOverfitting",
    "DeflatedSharpeRatio",
    "CombinatorialSymmetricCV",
    # Feature Selection - Utilities
    "FeatureClusteringAnalysis",
    "VIFAnalysis",
    "FeatureStabilityAnalysis",
    # Feature Selection - Ensemble
    "EnsembleFeatureSelector",
    # Feature Selection - Sklearn Transformer
    "FeatureSelector",
    # Feature Selection - Convenience Functions
    "select_features",
    "compute_all_importances",
    # Hawkes Processes - Kernels
    "HawkesKernel",
    "ExponentialKernel",
    "PowerLawKernel",
    "SumExponentialsKernel",
    "KernelType",
    # Hawkes Processes - Univariate
    "UnivariateHawkes",
    "HawkesResult",
    # Hawkes Processes - Multivariate
    "MultivariateHawkes",
    "MultivariateHawkesResult",
    # Hawkes Processes - Trading Applications
    "HawkesTradingAnalyzer",
    "BidAskHawkes",
    "OrderFlowMetrics",
    # Hawkes Processes - Utilities
    "goodness_of_fit_test",
    "estimate_branching_ratio_nonparametric",
]
