//! ORPflow HFT - Market Data Handler Library
//!
//! This crate provides high-performance market data handling for connecting
//! to Binance WebSocket streams and maintaining order book state.

use std::sync::Arc;
use tokio::sync::RwLock;

pub mod config;
pub mod error;
pub mod orderbook;
pub mod parser;
pub mod publisher;
pub mod websocket;

pub use config::Config;
pub use error::{MarketDataError, Result};
pub use orderbook::{OrderBook, OrderBookManager, OrderBookMetrics, OrderBookState};
pub use parser::{DepthUpdate, OrderBookSnapshot, ParsedMessage, Trade};
pub use publisher::Publisher;
pub use websocket::WebSocketManager;

/// Application state shared across components
pub struct AppState {
    pub orderbook_manager: Arc<RwLock<OrderBookManager>>,
    pub publisher: Arc<Publisher>,
    pub config: Arc<Config>,
}
