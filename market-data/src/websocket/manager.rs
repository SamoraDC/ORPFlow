//! WebSocket connection manager
//!
//! Handles reconnection logic and message dispatch.

use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, sleep};
use tracing::{error, info, warn};

use super::WebSocketClient;
use crate::error::{MarketDataError, Result};
use crate::parser::{OrderBookSnapshot, ParsedMessage};
use crate::AppState;

/// Manages WebSocket connections with automatic reconnection
pub struct WebSocketManager {
    state: Arc<AppState>,
    client: WebSocketClient,
    reconnect_attempts: u32,
}

impl WebSocketManager {
    /// Create a new WebSocket manager
    pub fn new(state: Arc<AppState>) -> Self {
        let client = WebSocketClient::new(&state.config.ws_endpoint, state.config.symbols.clone());

        Self {
            state,
            client,
            reconnect_attempts: 0,
        }
    }

    /// Run the WebSocket manager
    pub async fn run(&mut self) -> Result<()> {
        loop {
            match self.connect_and_process().await {
                Ok(()) => {
                    info!("WebSocket processing completed normally");
                    break;
                }
                Err(e) => {
                    error!(error = %e, "WebSocket error");

                    self.reconnect_attempts += 1;
                    if self.reconnect_attempts > self.state.config.max_reconnect_attempts {
                        error!("Max reconnection attempts exceeded");
                        return Err(MarketDataError::MaxReconnectAttemptsExceeded);
                    }

                    let delay = Duration::from_millis(
                        self.state.config.reconnect_delay_ms
                            * 2u64.pow(self.reconnect_attempts.min(5)),
                    );
                    warn!(attempt = self.reconnect_attempts, delay_ms = ?delay, "Reconnecting...");
                    sleep(delay).await;
                }
            }
        }

        Ok(())
    }

    /// Connect and process messages
    async fn connect_and_process(&mut self) -> Result<()> {
        // Connect to WebSocket
        self.client.connect().await?;
        self.reconnect_attempts = 0;

        // Fetch initial snapshots for all symbols
        self.fetch_snapshots().await?;

        // Start health check task
        let health_state = self.state.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                // Health check logging
                let books = health_state.orderbook_manager.read().await;
                let symbols = books.symbols();
                for symbol in symbols {
                    if let Some(state) = books.get_state(&symbol) {
                        if let Some(mid) = state.metrics.mid_price {
                            info!(
                                symbol = %symbol,
                                mid_price = %mid,
                                spread_bps = ?state.metrics.spread_bps,
                                imbalance = ?state.metrics.imbalance,
                                "Order book status"
                            );
                        }
                    }
                }
            }
        });

        // Process messages
        loop {
            match self.client.recv().await {
                Ok(Some(text)) => {
                    if let Err(e) = self.process_message(&text).await {
                        warn!(error = %e, "Failed to process message");
                    }
                }
                Ok(None) => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// Fetch order book snapshots from REST API
    async fn fetch_snapshots(&self) -> Result<()> {
        let client = reqwest::Client::new();

        for symbol in &self.state.config.symbols {
            let url = format!(
                "{}/depth?symbol={}&limit={}",
                self.state.config.rest_endpoint, symbol, self.state.config.depth_levels
            );

            info!(symbol = %symbol, url = %url, "Fetching order book snapshot");

            let response = client
                .get(&url)
                .send()
                .await?
                .json::<OrderBookSnapshot>()
                .await?;

            let mut manager = self.state.orderbook_manager.write().await;
            manager.init_book(symbol, response);

            info!(symbol = %symbol, "Order book initialized");
        }

        Ok(())
    }

    /// Process a single WebSocket message
    async fn process_message(&self, raw: &str) -> Result<()> {
        let parsed = ParsedMessage::parse(raw)?;

        match parsed {
            ParsedMessage::DepthUpdate(update) => {
                let mut manager = self.state.orderbook_manager.write().await;
                if manager.apply_update(&update) {
                    // Publish updated state
                    if let Some(state) = manager.get_state(&update.symbol) {
                        drop(manager); // Release lock before publishing
                        self.state.publisher.publish(&state).await?;
                    }
                }
            }
            ParsedMessage::Trade(trade) => {
                // For now, we just log trades
                // In a full implementation, you might want to publish these too
                tracing::trace!(
                    symbol = %trade.symbol,
                    price = %trade.price,
                    qty = %trade.quantity,
                    "Trade received"
                );
            }
            ParsedMessage::Unknown(msg) => {
                tracing::trace!(msg = %msg, "Unknown message type");
            }
        }

        Ok(())
    }
}
