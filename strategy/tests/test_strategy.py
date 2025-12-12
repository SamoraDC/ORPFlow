"""Tests for the imbalance strategy."""

from decimal import Decimal

import pytest

from src.features.microstructure import FeatureSnapshot, MicrostructureFeatures
from src.signals.imbalance_strategy import ImbalanceStrategy, StrategyConfig


@pytest.fixture
def strategy():
    """Create a strategy instance for testing."""
    config = StrategyConfig(
        imbalance_threshold=0.3,
        min_confidence=0.3,  # Lower threshold for testing (confidence ~0.34 at streak=2)
        persistence_required=2,
        require_momentum_confirm=False,  # Disable for simpler testing
    )
    strategy = ImbalanceStrategy(config)
    # Override settings that get applied in __init__
    strategy.config.min_confidence = 0.3
    strategy.config.persistence_required = 2
    return strategy


@pytest.fixture
def features_calc():
    """Create a features calculator for testing."""
    # window_size must be >= volatility_window for volatility to be calculated
    return MicrostructureFeatures(window_size=50, volatility_window=20, momentum_window=10)


def test_no_signal_below_threshold(strategy):
    """Test that no signal is generated when imbalance is below threshold."""
    features = FeatureSnapshot(
        timestamp=1000,
        symbol="BTCUSDT",
        mid_price=Decimal("50000"),
        spread_bps=Decimal("5"),
        imbalance=0.1,  # Below 0.3 threshold
        weighted_imbalance=0.1,
    )

    signal = strategy.evaluate(features, Decimal("10000"))
    assert signal is None


def test_no_signal_without_persistence(strategy):
    """Test that no signal is generated without persistence."""
    features = FeatureSnapshot(
        timestamp=1000,
        symbol="BTCUSDT",
        mid_price=Decimal("50000"),
        spread_bps=Decimal("5"),
        imbalance=0.5,  # Above threshold
        weighted_imbalance=0.5,
    )

    # First tick - no signal (need 2 for persistence)
    signal = strategy.evaluate(features, Decimal("10000"))
    assert signal is None


def test_signal_with_persistence(strategy):
    """Test that signal is generated with persistence."""
    features = FeatureSnapshot(
        timestamp=1000,
        symbol="BTCUSDT",
        mid_price=Decimal("50000"),
        spread_bps=Decimal("5"),
        imbalance=0.5,
        weighted_imbalance=0.5,
        momentum=0.001,  # Positive momentum
    )

    # First tick
    strategy.evaluate(features, Decimal("10000"))

    # Second tick (should meet persistence requirement)
    features.timestamp = 1001
    signal = strategy.evaluate(features, Decimal("10000"))

    # Should generate signal now
    assert signal is not None
    assert signal.side.value == "buy"  # Positive imbalance = buy
    assert signal.confidence >= 0.3  # Confidence ~0.34 at imbalance=0.5 and streak=2


def test_buy_signal_on_positive_imbalance(strategy):
    """Test that positive imbalance generates buy signal."""
    # Warm up persistence
    for i in range(3):
        features = FeatureSnapshot(
            timestamp=1000 + i,
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            spread_bps=Decimal("5"),
            imbalance=0.6,  # Strong positive imbalance
            weighted_imbalance=0.6,
        )
        signal = strategy.evaluate(features, Decimal("10000"))

    assert signal is not None
    assert signal.side.value == "buy"


def test_sell_signal_on_negative_imbalance(strategy):
    """Test that negative imbalance generates sell signal."""
    # Warm up persistence
    for i in range(3):
        features = FeatureSnapshot(
            timestamp=1000 + i,
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            spread_bps=Decimal("5"),
            imbalance=-0.6,  # Strong negative imbalance
            weighted_imbalance=-0.6,
        )
        signal = strategy.evaluate(features, Decimal("10000"))

    assert signal is not None
    assert signal.side.value == "sell"


def test_spread_filter(strategy):
    """Test that wide spreads prevent signals."""
    features = FeatureSnapshot(
        timestamp=1000,
        symbol="BTCUSDT",
        mid_price=Decimal("50000"),
        spread_bps=Decimal("15"),  # Above 10 bps limit
        imbalance=0.6,
        weighted_imbalance=0.6,
    )

    signal = strategy.evaluate(features, Decimal("10000"))
    assert signal is None


def test_features_volatility_calculation(features_calc):
    """Test volatility calculation."""
    # Add price data
    for i in range(30):
        price = Decimal(str(50000 + i * 10))  # Trending up
        features_calc.update(
            symbol="BTCUSDT",
            timestamp=1000 + i * 1000,
            mid_price=price,
            imbalance=Decimal("0.1"),
            weighted_imbalance=Decimal("0.1"),
            spread_bps=Decimal("5"),
            bid_depth=Decimal("10"),
            ask_depth=Decimal("10"),
        )

    # Get latest features
    snapshot = features_calc.update(
        symbol="BTCUSDT",
        timestamp=31000,
        mid_price=Decimal("50300"),
        imbalance=Decimal("0.1"),
        weighted_imbalance=Decimal("0.1"),
        spread_bps=Decimal("5"),
        bid_depth=Decimal("10"),
        ask_depth=Decimal("10"),
    )

    assert snapshot.volatility is not None
    assert snapshot.volatility > 0


def test_features_momentum_calculation(features_calc):
    """Test momentum calculation."""
    # Add price data with upward trend
    for i in range(15):
        price = Decimal(str(50000 + i * 100))  # Strong upward trend
        features_calc.update(
            symbol="BTCUSDT",
            timestamp=1000 + i * 1000,
            mid_price=price,
            imbalance=Decimal("0.1"),
            weighted_imbalance=Decimal("0.1"),
            spread_bps=Decimal("5"),
            bid_depth=Decimal("10"),
            ask_depth=Decimal("10"),
        )

    snapshot = features_calc.update(
        symbol="BTCUSDT",
        timestamp=16000,
        mid_price=Decimal("51500"),
        imbalance=Decimal("0.1"),
        weighted_imbalance=Decimal("0.1"),
        spread_bps=Decimal("5"),
        bid_depth=Decimal("10"),
        ask_depth=Decimal("10"),
    )

    assert snapshot.momentum is not None
    assert snapshot.momentum > 0  # Positive momentum for upward trend


def test_strategy_reset(strategy):
    """Test strategy reset clears state."""
    features = FeatureSnapshot(
        timestamp=1000,
        symbol="BTCUSDT",
        mid_price=Decimal("50000"),
        spread_bps=Decimal("5"),
        imbalance=0.5,
        weighted_imbalance=0.5,
    )

    # Build up persistence
    strategy.evaluate(features, Decimal("10000"))

    # Reset
    strategy.reset()

    # Should require persistence again
    signal = strategy.evaluate(features, Decimal("10000"))
    assert signal is None  # No signal on first tick after reset
