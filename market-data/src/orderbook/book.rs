//! Core order book implementation
//!
//! Uses BTreeMap for efficient sorted price level management.

use rust_decimal::Decimal;
use std::cmp::Reverse;
use std::collections::BTreeMap;

use super::{Level, OrderBookMetrics, OrderBookState, Side};
use crate::parser::{DepthUpdate, OrderBookSnapshot, PriceLevel};

/// Order book for a single symbol
#[derive(Debug)]
pub struct OrderBook {
    symbol: String,
    /// Bids sorted by price descending (highest first)
    bids: BTreeMap<Reverse<Decimal>, Decimal>,
    /// Asks sorted by price ascending (lowest first)
    asks: BTreeMap<Decimal, Decimal>,
    /// Last processed update ID
    last_update_id: u64,
    /// Whether the book has been initialized with a snapshot
    initialized: bool,
    /// Maximum depth levels to maintain
    max_depth: usize,
    /// Timestamp of last update
    last_update_time: u64,
}

impl OrderBook {
    /// Create a new empty order book
    pub fn new(symbol: &str, max_depth: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            initialized: false,
            max_depth,
            last_update_time: 0,
        }
    }

    /// Initialize with a snapshot from REST API
    pub fn init_snapshot(&mut self, snapshot: &OrderBookSnapshot) {
        self.bids.clear();
        self.asks.clear();

        for level in &snapshot.bids {
            if level.quantity > Decimal::ZERO {
                self.bids.insert(Reverse(level.price), level.quantity);
            }
        }

        for level in &snapshot.asks {
            if level.quantity > Decimal::ZERO {
                self.asks.insert(level.price, level.quantity);
            }
        }

        self.last_update_id = snapshot.last_update_id;
        self.initialized = true;
        self.trim_depth();
    }

    /// Apply a depth update
    ///
    /// Returns true if the update was applied successfully
    pub fn apply_update(&mut self, update: &DepthUpdate) -> bool {
        // Validate sequence - first event's U should be <= lastUpdateId + 1
        // and u should be >= lastUpdateId + 1 in the first valid event
        if !self.initialized {
            return false;
        }

        // Check if this update is relevant (not stale)
        if update.final_update_id <= self.last_update_id {
            return false; // Stale update, skip
        }

        // Apply bid updates
        for level in &update.bids {
            self.update_side(Side::Bid, level);
        }

        // Apply ask updates
        for level in &update.asks {
            self.update_side(Side::Ask, level);
        }

        self.last_update_id = update.final_update_id;
        self.last_update_time = update.event_time;
        self.trim_depth();

        true
    }

    /// Update a single price level
    fn update_side(&mut self, side: Side, level: &PriceLevel) {
        match side {
            Side::Bid => {
                if level.quantity == Decimal::ZERO {
                    self.bids.remove(&Reverse(level.price));
                } else {
                    self.bids.insert(Reverse(level.price), level.quantity);
                }
            }
            Side::Ask => {
                if level.quantity == Decimal::ZERO {
                    self.asks.remove(&level.price);
                } else {
                    self.asks.insert(level.price, level.quantity);
                }
            }
        }
    }

    /// Trim the book to max depth
    fn trim_depth(&mut self) {
        while self.bids.len() > self.max_depth {
            self.bids.pop_last();
        }
        while self.asks.len() > self.max_depth {
            self.asks.pop_last();
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.first_key_value().map(|(Reverse(p), _)| *p)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.first_key_value().map(|(p, _)| *p)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / Decimal::from(2)),
            _ => None,
        }
    }

    /// Get spread in basis points
    pub fn spread_bps(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask(), self.mid_price()) {
            (Some(bid), Some(ask), Some(mid)) if mid > Decimal::ZERO => {
                Some((ask - bid) / mid * Decimal::from(10000))
            }
            _ => None,
        }
    }

    /// Calculate order book imbalance at top N levels
    pub fn imbalance(&self, levels: usize) -> Option<Decimal> {
        let bid_volume: Decimal = self.bids.iter().take(levels).map(|(_, q)| q).sum();
        let ask_volume: Decimal = self.asks.iter().take(levels).map(|(_, q)| q).sum();

        let total = bid_volume + ask_volume;
        if total > Decimal::ZERO {
            Some((bid_volume - ask_volume) / total)
        } else {
            None
        }
    }

    /// Calculate weighted imbalance (closer to mid weighted more)
    pub fn weighted_imbalance(&self, levels: usize, decay: Decimal) -> Option<Decimal> {
        let _mid = self.mid_price()?;

        // Helper to calculate decay^i without maths feature
        let pow = |exp: usize| -> Decimal {
            (0..exp).fold(Decimal::ONE, |acc, _| acc * decay)
        };

        let bid_weighted: Decimal = self
            .bids
            .iter()
            .take(levels)
            .enumerate()
            .map(|(i, (_, q))| {
                let weight = pow(i);
                *q * weight
            })
            .sum();

        let ask_weighted: Decimal = self
            .asks
            .iter()
            .take(levels)
            .enumerate()
            .map(|(i, (_, q))| {
                let weight = pow(i);
                *q * weight
            })
            .sum();

        let total = bid_weighted + ask_weighted;
        if total > Decimal::ZERO {
            Some((bid_weighted - ask_weighted) / total)
        } else {
            None
        }
    }

    /// Check if the book is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get last update ID
    pub fn last_update_id(&self) -> u64 {
        self.last_update_id
    }

    /// Get current state for publishing
    pub fn state(&self) -> OrderBookState {
        OrderBookState {
            symbol: self.symbol.clone(),
            timestamp: self.last_update_time,
            last_update_id: self.last_update_id,
            bids: self
                .bids
                .iter()
                .map(|(Reverse(p), q)| Level {
                    price: *p,
                    quantity: *q,
                })
                .collect(),
            asks: self
                .asks
                .iter()
                .map(|(p, q)| Level {
                    price: *p,
                    quantity: *q,
                })
                .collect(),
            metrics: self.calculate_metrics(),
        }
    }

    /// Calculate order book metrics
    fn calculate_metrics(&self) -> OrderBookMetrics {
        OrderBookMetrics {
            mid_price: self.mid_price(),
            spread_bps: self.spread_bps(),
            imbalance: self.imbalance(5),
            weighted_imbalance: self
                .weighted_imbalance(10, Decimal::from_str_exact("0.9").unwrap()),
            bid_depth: self.bids.values().copied().sum(),
            ask_depth: self.asks.values().copied().sum(),
            bid_levels: self.bids.len(),
            ask_levels: self.asks.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn create_test_book() -> OrderBook {
        let mut book = OrderBook::new("BTCUSDT", 10);
        let snapshot = OrderBookSnapshot {
            last_update_id: 100,
            bids: vec![
                PriceLevel {
                    price: dec!(50000),
                    quantity: dec!(1.0),
                },
                PriceLevel {
                    price: dec!(49999),
                    quantity: dec!(2.0),
                },
            ],
            asks: vec![
                PriceLevel {
                    price: dec!(50001),
                    quantity: dec!(1.5),
                },
                PriceLevel {
                    price: dec!(50002),
                    quantity: dec!(2.5),
                },
            ],
        };
        book.init_snapshot(&snapshot);
        book
    }

    #[test]
    fn test_best_bid_ask() {
        let book = create_test_book();
        assert_eq!(book.best_bid(), Some(dec!(50000)));
        assert_eq!(book.best_ask(), Some(dec!(50001)));
    }

    #[test]
    fn test_mid_price() {
        let book = create_test_book();
        assert_eq!(book.mid_price(), Some(dec!(50000.5)));
    }

    #[test]
    fn test_imbalance() {
        let book = create_test_book();
        // Bids: 1.0 + 2.0 = 3.0, Asks: 1.5 + 2.5 = 4.0
        // Imbalance = (3.0 - 4.0) / (3.0 + 4.0) = -1/7
        let imbalance = book.imbalance(10).unwrap();
        assert!(imbalance < Decimal::ZERO);
    }

    #[test]
    fn test_apply_update() {
        let mut book = create_test_book();
        let update = DepthUpdate {
            event_type: "depthUpdate".to_string(),
            event_time: 1000,
            symbol: "BTCUSDT".to_string(),
            first_update_id: 101,
            final_update_id: 102,
            bids: vec![PriceLevel {
                price: dec!(50000),
                quantity: dec!(2.0),
            }],
            asks: vec![],
        };

        assert!(book.apply_update(&update));
        assert_eq!(book.last_update_id(), 102);
    }
}
