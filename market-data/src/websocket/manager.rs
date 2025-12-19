//! WebSocket connection manager
//!
//! Handles reconnection logic and message dispatch.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::{interval, sleep, timeout};
use tracing::{error, info, warn};

use super::WebSocketClient;
use crate::error::Result;
use crate::parser::{OrderBookSnapshot, ParsedMessage};
use crate::AppState;

/// Maximum backoff delay in milliseconds (60 seconds)
const MAX_BACKOFF_MS: u64 = 60_000;
/// Cooldown period after which reconnect attempts are reset (5 minutes)
const RECONNECT_COOLDOWN_SECS: u64 = 300;

/// Manages WebSocket connections with automatic reconnection
pub struct WebSocketManager {
    state: Arc<AppState>,
    client: WebSocketClient,
    reconnect_attempts: u32,
    last_successful_connection: Option<Instant>,
}

impl WebSocketManager {
    /// Create a new WebSocket manager
    pub fn new(state: Arc<AppState>) -> Self {
        let client = WebSocketClient::new(&state.config.ws_endpoint, state.config.symbols.clone());

        Self {
            state,
            client,
            reconnect_attempts: 0,
            last_successful_connection: None,
        }
    }

    /// Run the WebSocket manager - runs indefinitely with automatic reconnection
    pub async fn run(&mut self) -> Result<()> {
        info!("Starting WebSocket manager with infinite retry");

        loop {
            // Reset reconnect attempts if we've been stable for a while
            if let Some(last_success) = self.last_successful_connection {
                if last_success.elapsed() > Duration::from_secs(RECONNECT_COOLDOWN_SECS) {
                    if self.reconnect_attempts > 0 {
                        info!(
                            previous_attempts = self.reconnect_attempts,
                            "Resetting reconnect counter after cooldown period"
                        );
                        self.reconnect_attempts = 0;
                    }
                }
            }

            match self.connect_and_process().await {
                Ok(()) => {
                    info!("WebSocket processing completed normally, reconnecting...");
                    // Brief pause before reconnecting after normal completion
                    sleep(Duration::from_secs(1)).await;
                }
                Err(e) => {
                    error!(error = %e, "WebSocket error");
                    self.reconnect_attempts += 1;

                    // Calculate delay with exponential backoff, capped at MAX_BACKOFF_MS
                    let base_delay = self.state.config.reconnect_delay_ms
                        * 2u64.pow(self.reconnect_attempts.min(6));
                    let delay = Duration::from_millis(base_delay.min(MAX_BACKOFF_MS));

                    warn!(
                        attempt = self.reconnect_attempts,
                        delay_secs = delay.as_secs(),
                        "Reconnecting after error..."
                    );
                    sleep(delay).await;
                }
            }
        }
    }

    /// Connect and process messages
    async fn connect_and_process(&mut self) -> Result<()> {
        // Connect to WebSocket
        self.client.connect().await?;

        // Mark successful connection
        self.last_successful_connection = Some(Instant::now());
        self.reconnect_attempts = 0;
        info!("WebSocket connected successfully, resetting reconnect counter");

        // Fetch initial snapshots for all symbols
        self.fetch_snapshots().await?;

        // Start health check and status logging task
        let health_state = self.state.clone();
        tokio::spawn(async move {
            let mut health_interval = interval(Duration::from_secs(30));
            loop {
                health_interval.tick().await;
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

        // Process messages with keepalive
        let mut last_message = Instant::now();
        let keepalive_timeout = Duration::from_secs(30);
        let recv_timeout = Duration::from_secs(45);

        loop {
            // Use timeout to detect stale connections
            match timeout(recv_timeout, self.client.recv()).await {
                Ok(Ok(Some(text))) => {
                    last_message = Instant::now();
                    if let Err(e) = self.process_message(&text).await {
                        warn!(error = %e, "Failed to process message");
                    }
                }
                Ok(Ok(None)) => {
                    // Ping/pong or other non-data message
                    // Send keepalive if no data received for a while
                    if last_message.elapsed() > keepalive_timeout {
                        if let Err(e) = self.client.ping().await {
                            warn!(error = %e, "Failed to send keepalive ping");
                        }
                    }
                    continue;
                }
                Ok(Err(e)) => {
                    // WebSocket error
                    return Err(e);
                }
                Err(_) => {
                    // Timeout - connection might be stale
                    warn!(
                        last_message_secs = last_message.elapsed().as_secs(),
                        "No message received within timeout, sending keepalive"
                    );
                    if let Err(e) = self.client.ping().await {
                        warn!(error = %e, "Failed to send keepalive ping, reconnecting");
                        return Err(crate::error::MarketDataError::ConnectionTimeout);
                    }
                }
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
