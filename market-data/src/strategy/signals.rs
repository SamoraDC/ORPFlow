//! Trading signal generation
//!
//! Order flow imbalance strategy implementation.
//! Ultra-low latency signal generation with O(1) complexity.

use rust_decimal::Decimal;
use std::collections::HashMap;

use super::features::FeatureSnapshot;
use super::models::{Side, Signal};

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyParams {
    /// Minimum imbalance to generate signal
    pub imbalance_threshold: f64,
    /// Minimum confidence for signal
    pub min_confidence: f64,
    /// Number of ticks imbalance must persist
    pub persistence_required: usize,
    /// Decay factor for persistence score (reserved for future use)
    #[allow(dead_code)]
    pub persistence_decay: f64,
    /// Low volatility multiplier for position sizing
    pub low_vol_multiplier: f64,
    /// High volatility multiplier for position sizing
    pub high_vol_multiplier: f64,
    /// Z-score threshold for low volatility
    pub vol_threshold_low: f64,
    /// Z-score threshold for high volatility
    pub vol_threshold_high: f64,
    /// Base position size as percentage of balance
    pub base_position_pct: f64,
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self {
            imbalance_threshold: 0.3,
            min_confidence: 0.6,
            persistence_required: 3,
            persistence_decay: 0.9,
            low_vol_multiplier: 1.5,
            high_vol_multiplier: 0.5,
            vol_threshold_low: -1.0,
            vol_threshold_high: 1.0,
            base_position_pct: 0.1,
        }
    }
}

/// Per-symbol tracking state
struct SymbolState {
    imbalance_streak: usize,
    last_imbalance_sign: i8, // -1, 0, 1
}

/// Order Flow Imbalance Strategy
///
/// Generates trading signals based on order book imbalance with
/// persistence filtering and volatility-adjusted position sizing.
pub struct ImbalanceStrategy {
    params: StrategyParams,
    states: HashMap<String, SymbolState>,
}

impl ImbalanceStrategy {
    pub fn new(imbalance_threshold: f64, min_confidence: f64, persistence_required: usize) -> Self {
        Self {
            params: StrategyParams {
                imbalance_threshold,
                min_confidence,
                persistence_required,
                ..Default::default()
            },
            states: HashMap::new(),
        }
    }

    /// Evaluate market conditions and generate trading signal
    /// This is the HOT PATH - must be O(1) complexity
    #[inline(always)]
    pub fn evaluate(
        &mut self,
        features: &FeatureSnapshot,
        account_balance: Decimal,
        current_position: Option<Decimal>,
    ) -> Option<Signal> {
        let symbol = &features.symbol;
        let imbalance = features.imbalance?;

        // Get or create state for this symbol
        if !self.states.contains_key(symbol) {
            self.states.insert(symbol.clone(), SymbolState {
                imbalance_streak: 0,
                last_imbalance_sign: 0,
            });
        }

        let state = self.states.get_mut(symbol).unwrap();

        // Update persistence tracking
        let imbalance_sign = if imbalance > 0.0 { 1 } else { -1 };

        if state.last_imbalance_sign == imbalance_sign {
            state.imbalance_streak += 1;
        } else {
            state.last_imbalance_sign = imbalance_sign;
            state.imbalance_streak = 1;
        }

        // Check if imbalance is significant
        if imbalance.abs() < self.params.imbalance_threshold {
            return None;
        }

        // Check persistence
        if state.imbalance_streak < self.params.persistence_required {
            return None;
        }

        // Extract streak for later use (avoid borrow issue)
        let streak = state.imbalance_streak;

        // Calculate confidence
        let confidence = self.calculate_confidence(features, streak);

        if confidence < self.params.min_confidence {
            return None;
        }

        // Check imbalance momentum direction
        if let Some(imb_momentum) = features.imbalance_momentum {
            if imb_momentum * (imbalance_sign as f64) < 0.0 {
                // Imbalance momentum diverging from imbalance direction
                return None;
            }
        }

        // Calculate position size
        let size = self.calculate_position_size(features, account_balance, current_position)?;

        if size <= Decimal::ZERO {
            return None;
        }

        // Generate signal
        let side = if imbalance > 0.0 { Side::Buy } else { Side::Sell };
        let reason = self.generate_reason(features, streak);

        Some(Signal::new(symbol.clone(), side, confidence, size, reason))
    }

    /// Calculate signal confidence based on features
    #[inline(always)]
    fn calculate_confidence(&self, features: &FeatureSnapshot, streak: usize) -> f64 {
        let mut confidence = 0.0;

        // Base confidence from imbalance strength (40%)
        if let Some(imbalance) = features.imbalance {
            let imbalance_strength = imbalance.abs().min(1.0);
            confidence += 0.4 * imbalance_strength;
        }

        // Weighted imbalance contribution (20%)
        if let Some(weighted) = features.weighted_imbalance {
            let weighted_strength = weighted.abs().min(1.0);
            confidence += 0.2 * weighted_strength;
        }

        // Persistence bonus (20%)
        let persistence_score = (streak as f64 / 10.0).min(1.0);
        confidence += 0.2 * persistence_score;

        // Volatility adjustment (10%)
        if let Some(vol_z) = features.volatility_z {
            if vol_z < self.params.vol_threshold_low {
                confidence += 0.1; // Low volatility bonus
            } else if vol_z > self.params.vol_threshold_high {
                confidence -= 0.1; // High volatility penalty
            }
        }

        // Imbalance momentum bonus (10%)
        if let Some(imb_momentum) = features.imbalance_momentum {
            if let Some(imbalance) = features.imbalance {
                let imbalance_sign = if imbalance > 0.0 { 1.0 } else { -1.0 };
                if imb_momentum * imbalance_sign > 0.0 {
                    confidence += 0.1;
                }
            }
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Calculate position size based on confidence and volatility
    #[inline(always)]
    fn calculate_position_size(
        &self,
        features: &FeatureSnapshot,
        account_balance: Decimal,
        current_position: Option<Decimal>,
    ) -> Option<Decimal> {
        let balance_f64: f64 = account_balance.try_into().ok()?;
        let mut base_size = balance_f64 * self.params.base_position_pct;

        // Adjust for volatility
        if let Some(vol_z) = features.volatility_z {
            if vol_z < self.params.vol_threshold_low {
                base_size *= self.params.low_vol_multiplier;
            } else if vol_z > self.params.vol_threshold_high {
                base_size *= self.params.high_vol_multiplier;
            }
        }

        // Convert to quantity based on mid price
        let mid_price: f64 = features.mid_price?.try_into().ok()?;
        if mid_price <= 0.0 {
            return None;
        }

        let mut quantity = base_size / mid_price;

        // Reduce size if adding to existing position
        if let Some(pos) = current_position {
            let pos_f64: f64 = pos.try_into().unwrap_or(0.0);
            if let Some(imbalance) = features.imbalance {
                let imbalance_sign = if imbalance > 0.0 { 1.0 } else { -1.0 };
                let position_sign = if pos_f64 > 0.0 { 1.0 } else { -1.0 };
                if imbalance_sign == position_sign && pos_f64 != 0.0 {
                    quantity *= 0.5; // Halve size when adding to position
                }
            }
        }

        // Round to 6 decimal places
        let rounded = (quantity * 1_000_000.0).round() / 1_000_000.0;
        Decimal::try_from(rounded).ok()
    }

    /// Generate human-readable reason for the signal
    fn generate_reason(&self, features: &FeatureSnapshot, streak: usize) -> String {
        let mut parts = Vec::new();

        if let Some(imbalance) = features.imbalance {
            let direction = if imbalance > 0.0 { "bid" } else { "ask" };
            parts.push(format!("Order book {} imbalance: {:.3}", direction, imbalance));
        }

        if streak >= self.params.persistence_required {
            parts.push(format!("Persistent for {} ticks", streak));
        }

        if let Some(vol_z) = features.volatility_z {
            if vol_z < self.params.vol_threshold_low {
                parts.push("Low volatility environment".to_string());
            } else if vol_z > self.params.vol_threshold_high {
                parts.push("High volatility (reduced size)".to_string());
            }
        }

        if parts.is_empty() {
            "Imbalance signal".to_string()
        } else {
            parts.join("; ")
        }
    }

    /// Reset strategy state
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
    fn test_no_signal_below_threshold() {
        let mut strategy = ImbalanceStrategy::new(0.3, 0.3, 2);

        let features = FeatureSnapshot {
            symbol: "BTCUSDT".to_string(),
            imbalance: Some(0.1), // Below threshold
            mid_price: Some(dec!(50000)),
            ..Default::default()
        };

        let signal = strategy.evaluate(&features, dec!(10000), None);
        assert!(signal.is_none());
    }

    #[test]
    fn test_signal_with_persistence() {
        let mut strategy = ImbalanceStrategy::new(0.3, 0.3, 2);

        let features = FeatureSnapshot {
            symbol: "BTCUSDT".to_string(),
            imbalance: Some(0.5),
            weighted_imbalance: Some(0.5),
            mid_price: Some(dec!(50000)),
            ..Default::default()
        };

        // First tick - no signal
        let signal = strategy.evaluate(&features, dec!(10000), None);
        assert!(signal.is_none());

        // Second tick - should generate signal
        let signal = strategy.evaluate(&features, dec!(10000), None);
        assert!(signal.is_some());

        let s = signal.unwrap();
        assert_eq!(s.side, Side::Buy);
        assert!(s.confidence >= 0.3);
    }

    #[test]
    fn test_sell_signal_on_negative_imbalance() {
        let mut strategy = ImbalanceStrategy::new(0.3, 0.3, 2);

        let features = FeatureSnapshot {
            symbol: "BTCUSDT".to_string(),
            imbalance: Some(-0.6),
            weighted_imbalance: Some(-0.6),
            mid_price: Some(dec!(50000)),
            ..Default::default()
        };

        // Build up persistence
        strategy.evaluate(&features, dec!(10000), None);
        let signal = strategy.evaluate(&features, dec!(10000), None);

        assert!(signal.is_some());
        assert_eq!(signal.unwrap().side, Side::Sell);
    }
}
