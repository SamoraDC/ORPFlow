//! Microstructure feature calculations
//!
//! High-performance feature calculation using fixed-size ring buffers
//! for constant-time updates. All operations are O(1) or O(window_size).

use rust_decimal::Decimal;
use std::collections::HashMap;

/// Ring buffer for efficient rolling calculations
struct RingBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
            head: 0,
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        (0..self.len).map(move |i| &self.data[(start + i) % self.capacity])
    }

    fn last_n(&self, n: usize) -> impl Iterator<Item = &T> {
        let n = n.min(self.len);
        let start = if self.len >= self.capacity {
            (self.head + self.capacity - n) % self.capacity
        } else {
            self.len.saturating_sub(n)
        };
        (0..n).map(move |i| &self.data[(start + i) % self.capacity])
    }
}

/// Snapshot of calculated features at a point in time
#[derive(Debug, Clone, Default)]
pub struct FeatureSnapshot {
    pub timestamp: u64,
    pub symbol: String,

    // Order book features
    pub mid_price: Option<Decimal>,
    pub spread_bps: Option<Decimal>,
    pub imbalance: Option<f64>,
    pub weighted_imbalance: Option<f64>,

    // Volume features
    pub bid_depth: Option<Decimal>,
    pub ask_depth: Option<Decimal>,
    pub volume_ratio: Option<f64>,

    // Derived features
    pub volatility: Option<f64>,
    pub momentum: Option<f64>,
    pub imbalance_momentum: Option<f64>,

    // Normalized features (z-scores)
    pub imbalance_z: Option<f64>,
    pub volatility_z: Option<f64>,
}

/// Per-symbol feature state
struct SymbolState {
    mid_prices: RingBuffer<f64>,
    imbalances: RingBuffer<f64>,
    imbalance_mean: f64,
    imbalance_std: f64,
    volatility_mean: f64,
    volatility_std: f64,
}

impl SymbolState {
    fn new(window_size: usize) -> Self {
        Self {
            mid_prices: RingBuffer::new(window_size),
            imbalances: RingBuffer::new(window_size),
            imbalance_mean: 0.0,
            imbalance_std: 1.0,
            volatility_mean: 0.0,
            volatility_std: 1.0,
        }
    }
}

/// Microstructure feature calculator
pub struct MicrostructureFeatures {
    window_size: usize,
    volatility_window: usize,
    momentum_window: usize,
    states: HashMap<String, SymbolState>,
}

impl MicrostructureFeatures {
    pub fn new(window_size: usize, volatility_window: usize, momentum_window: usize) -> Self {
        Self {
            window_size,
            volatility_window,
            momentum_window,
            states: HashMap::new(),
        }
    }

    /// Update features with new order book data
    /// Returns a snapshot of all calculated features
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        symbol: &str,
        timestamp: u64,
        mid_price: Option<Decimal>,
        imbalance: Option<Decimal>,
        weighted_imbalance: Option<Decimal>,
        spread_bps: Option<Decimal>,
        bid_depth: Decimal,
        ask_depth: Decimal,
    ) -> FeatureSnapshot {
        // Ensure state exists for this symbol
        if !self.states.contains_key(symbol) {
            self.states.insert(symbol.to_string(), SymbolState::new(self.window_size));
        }

        let state = self.states.get_mut(symbol).unwrap();

        // Update rolling windows
        if let Some(mp) = mid_price {
            let mp_f64: f64 = mp.try_into().unwrap_or(0.0);
            state.mid_prices.push(mp_f64);
        }

        let imb_f64 = imbalance.map(|i| i.try_into().unwrap_or(0.0));
        if let Some(i) = imb_f64 {
            state.imbalances.push(i);
        }

        // Calculate derived features using state directly
        let volatility = Self::calculate_volatility_static(state, self.volatility_window);
        let momentum = Self::calculate_momentum_static(state, self.momentum_window);
        let imbalance_momentum = Self::calculate_imbalance_momentum_static(state, self.momentum_window);

        // Update statistics for normalization
        Self::update_statistics_static(state, self.volatility_window);

        // Calculate normalized features
        let imbalance_z = imb_f64.map(|i| (i - state.imbalance_mean) / state.imbalance_std);
        let volatility_z = volatility.map(|v| (v - state.volatility_mean) / state.volatility_std);

        // Volume ratio
        let volume_ratio = if ask_depth > Decimal::ZERO {
            let ratio: f64 = (bid_depth / ask_depth).try_into().unwrap_or(1.0);
            Some(ratio)
        } else {
            None
        };

        FeatureSnapshot {
            timestamp,
            symbol: symbol.to_string(),
            mid_price,
            spread_bps,
            imbalance: imb_f64,
            weighted_imbalance: weighted_imbalance.map(|w| w.try_into().unwrap_or(0.0)),
            bid_depth: Some(bid_depth),
            ask_depth: Some(ask_depth),
            volume_ratio,
            volatility,
            momentum,
            imbalance_momentum,
            imbalance_z,
            volatility_z,
        }
    }

    /// Calculate rolling volatility using log returns (static version)
    fn calculate_volatility_static(state: &SymbolState, volatility_window: usize) -> Option<f64> {
        if state.mid_prices.len() < volatility_window {
            return None;
        }

        let prices: Vec<f64> = state.mid_prices.last_n(volatility_window).copied().collect();
        if prices.len() < 2 {
            return None;
        }

        // Calculate log returns
        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            if prices[i - 1] > 0.0 {
                returns.push((prices[i] / prices[i - 1]).ln());
            }
        }

        if returns.is_empty() {
            return None;
        }

        // Calculate standard deviation
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        // Annualize (assuming 1-minute intervals, 24/7 trading)
        Some(std * (252.0 * 24.0 * 60.0_f64).sqrt())
    }

    /// Calculate price momentum (static version)
    fn calculate_momentum_static(state: &SymbolState, momentum_window: usize) -> Option<f64> {
        if state.mid_prices.len() < momentum_window {
            return None;
        }

        let prices: Vec<f64> = state.mid_prices.last_n(momentum_window).copied().collect();
        if prices.is_empty() || prices[0] == 0.0 {
            return None;
        }

        let first = prices[0];
        let last = *prices.last().unwrap();
        Some((last - first) / first)
    }

    /// Calculate momentum of imbalance (static version)
    fn calculate_imbalance_momentum_static(state: &SymbolState, momentum_window: usize) -> Option<f64> {
        if state.imbalances.len() < momentum_window {
            return None;
        }

        let imbalances: Vec<f64> = state.imbalances.last_n(momentum_window).copied().collect();
        if imbalances.len() < 2 {
            return None;
        }

        // Simple linear regression slope
        let n = imbalances.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = imbalances.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in imbalances.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator > 0.0 {
            Some(numerator / denominator)
        } else {
            Some(0.0)
        }
    }

    /// Update rolling statistics for normalization (static version)
    fn update_statistics_static(state: &mut SymbolState, volatility_window: usize) {
        // Update imbalance statistics
        if state.imbalances.len() >= 20 {
            let imbalances: Vec<f64> = state.imbalances.iter().copied().collect();
            let mean = imbalances.iter().sum::<f64>() / imbalances.len() as f64;
            let variance = imbalances.iter().map(|i| (i - mean).powi(2)).sum::<f64>() / imbalances.len() as f64;
            state.imbalance_mean = mean;
            state.imbalance_std = variance.sqrt().max(0.001);
        }

        // Update volatility statistics
        if state.mid_prices.len() >= volatility_window {
            let prices: Vec<f64> = state.mid_prices.iter().copied().collect();
            let mut returns = Vec::new();
            for i in 1..prices.len() {
                if prices[i - 1] > 0.0 {
                    returns.push((prices[i] / prices[i - 1]).ln());
                }
            }

            if returns.len() >= volatility_window {
                let vol = returns.iter().map(|r| r.powi(2)).sum::<f64>().sqrt();
                if vol > 0.0 {
                    state.volatility_mean = vol;
                    // Use a rough estimate for std
                    state.volatility_std = (vol * 0.3).max(0.0001);
                }
            }
        }
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.states.clear();
    }

    /// Reset state for a specific symbol
    pub fn reset_symbol(&mut self, symbol: &str) {
        self.states.remove(symbol);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_feature_calculation() {
        let mut features = MicrostructureFeatures::new(100, 20, 10);

        // Add some data points
        for i in 0..30 {
            let price = dec!(50000) + Decimal::from(i * 10);
            let snapshot = features.update(
                "BTCUSDT",
                i as u64 * 1000,
                Some(price),
                Some(dec!(0.1)),
                Some(dec!(0.1)),
                Some(dec!(5)),
                dec!(10),
                dec!(10),
            );

            // After enough data points, we should have derived features
            if i >= 20 {
                assert!(snapshot.volatility.is_some());
            }
            if i >= 10 {
                assert!(snapshot.momentum.is_some());
            }
        }
    }

    #[test]
    fn test_ring_buffer() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(5);

        for i in 0..10 {
            buf.push(i as f64);
        }

        assert_eq!(buf.len(), 5);

        // Should contain the last 5 values: 5, 6, 7, 8, 9
        let values: Vec<f64> = buf.iter().copied().collect();
        assert_eq!(values, vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}
