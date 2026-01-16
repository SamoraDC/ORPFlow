//! ORPflow HFT - Ultra-Low Latency Trading System
//!
//! Jane Street-style architecture with Rust-only hot path.
//! No Python in the critical execution path - all latency-sensitive
//! operations happen in compiled Rust code.
//!
//! ## Architecture
//!
//! - **Market Data**: WebSocket connection to Binance, order book maintenance
//! - **Strategy**: Imbalance trading strategy with microstructure features
//! - **Broker**: Paper trading simulation with realistic execution
//! - **Storage**: SQLite persistence for trades and positions
//!
//! ## Performance
//!
//! - Lock-free data structures where possible
//! - Zero-copy message passing
//! - O(1) feature calculations using ring buffers
//! - SIMD-optimized math operations

use std::sync::Arc;
use tokio::sync::RwLock;

pub mod config;
pub mod error;
pub mod orderbook;
pub mod parser;
pub mod publisher;
pub mod strategy;
pub mod websocket;

pub use config::Config;
pub use error::{MarketDataError, Result};
pub use orderbook::{OrderBook, OrderBookManager, OrderBookMetrics, OrderBookState};
pub use parser::{DepthUpdate, OrderBookSnapshot, ParsedMessage, Trade};
pub use publisher::Publisher;
pub use strategy::{StrategyConfig, StrategyEngine};
pub use websocket::WebSocketManager;

/// Application state shared across components
pub struct AppState {
    pub orderbook_manager: Arc<RwLock<OrderBookManager>>,
    pub publisher: Arc<Publisher>,
    pub config: Arc<Config>,
    pub strategy: Option<Arc<RwLock<StrategyEngine>>>,
}
