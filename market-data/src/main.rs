//! QuantumFlow HFT - Market Data Handler
//!
//! High-performance market data handler for connecting to Binance WebSocket streams,
//! maintaining order book state, and publishing normalized data to other system components.

mod config;
mod error;
mod orderbook;
mod parser;
mod publisher;
mod websocket;

use std::sync::Arc;
use axum::{routing::get, Json, Router};
use tokio::sync::RwLock;
use tracing::{info, warn, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::config::Config;
use crate::orderbook::OrderBookManager;
use crate::publisher::Publisher;
use crate::websocket::WebSocketManager;

/// Application state shared across components
pub struct AppState {
    pub orderbook_manager: Arc<RwLock<OrderBookManager>>,
    pub publisher: Arc<Publisher>,
    pub config: Arc<Config>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer().json())
        .with(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .init();

    info!("Starting QuantumFlow Market Data Handler");

    // Load configuration
    let config = Arc::new(Config::load()?);
    info!(symbols = ?config.symbols, "Configuration loaded");

    // Initialize order book manager
    let orderbook_manager = Arc::new(RwLock::new(OrderBookManager::new()));

    // Initialize publisher for IPC
    let publisher = Arc::new(Publisher::new(&config.ipc_socket_path).await?);

    // Create shared application state
    let state = Arc::new(AppState {
        orderbook_manager: orderbook_manager.clone(),
        publisher: publisher.clone(),
        config: config.clone(),
    });

    // Start health check server
    let health_state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = start_health_server(health_state).await {
            warn!(error = %e, "Health server error");
        }
    });

    // Start WebSocket manager
    let mut ws_manager = WebSocketManager::new(state);
    ws_manager.run().await?;

    Ok(())
}

/// Start HTTP server for health checks and metrics
async fn start_health_server(_state: Arc<AppState>) -> anyhow::Result<()> {
    use std::net::SocketAddr;

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(metrics));

    let addr = SocketAddr::from(([0, 0, 0, 0], 9090));
    info!(addr = %addr, "Starting health check server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "component": "market-data",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

async fn metrics() -> String {
    use prometheus::{Encoder, TextEncoder};
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
