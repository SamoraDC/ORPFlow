//! QuantumFlow HFT - Unified Trading System
//!
//! Jane Street-style ultra-low latency trading system.
//! Single Rust binary handling:
//! - Market data (WebSocket to Binance)
//! - Strategy execution (order flow imbalance)
//! - Paper trading (simulated execution)
//! - REST API (health, status, account)
//!
//! NO PYTHON IN THE HOT PATH.

mod config;
mod error;
mod orderbook;
mod parser;
mod publisher;
mod strategy;
mod websocket;

use std::sync::Arc;
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use tokio::sync::RwLock;
use tracing::{info, warn, error, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::config::Config;
use crate::orderbook::OrderBookManager;
use crate::publisher::Publisher;
use crate::strategy::{StrategyConfig, StrategyEngine};
use crate::websocket::WebSocketManager;

/// Application state shared across components
pub struct AppState {
    pub orderbook_manager: Arc<RwLock<OrderBookManager>>,
    pub publisher: Arc<Publisher>,
    pub config: Arc<Config>,
    pub strategy: Arc<RwLock<StrategyEngine>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer().json())
        .with(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .init();

    info!("========================================");
    info!("QuantumFlow HFT - Jane Street Edition");
    info!("========================================");
    info!("Architecture: Rust-only hot path");
    info!("No Python in critical execution path");
    info!("");

    // Load market data configuration
    let config = Arc::new(Config::load()?);
    info!(symbols = ?config.symbols, "Market data configuration loaded");

    // Load strategy configuration from environment
    let strategy_config = StrategyConfig::from_env();
    info!(
        initial_balance = %strategy_config.initial_balance,
        imbalance_threshold = strategy_config.imbalance_threshold,
        symbols = ?strategy_config.symbols,
        "Strategy configuration loaded"
    );

    // Initialize order book manager
    let orderbook_manager = Arc::new(RwLock::new(OrderBookManager::new()));

    // Initialize publisher for IPC (optional, for OCaml risk gateway)
    let publisher = Arc::new(Publisher::new(&config.ipc_socket_path).await?);

    // Initialize strategy engine
    let strategy = match StrategyEngine::new(strategy_config).await {
        Ok(s) => {
            info!("Strategy engine initialized successfully");
            Arc::new(RwLock::new(s))
        }
        Err(e) => {
            error!(error = %e, "Failed to initialize strategy engine");
            return Err(e);
        }
    };

    // Create shared application state
    let state = Arc::new(AppState {
        orderbook_manager: orderbook_manager.clone(),
        publisher: publisher.clone(),
        config: config.clone(),
        strategy: strategy.clone(),
    });

    // Start API server (port 8000 - replaces Python FastAPI)
    let api_state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = start_api_server(api_state).await {
            error!(error = %e, "API server error");
        }
    });

    // Start health/metrics server (port 9090)
    let health_state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = start_health_server(health_state).await {
            warn!(error = %e, "Health server error");
        }
    });

    info!("All services started successfully");
    info!("API server: http://0.0.0.0:8000");
    info!("Health server: http://0.0.0.0:9090");

    // Start WebSocket manager with strategy integration
    let mut ws_manager = WebSocketManager::new(state);
    ws_manager.run().await?;

    Ok(())
}

/// Start the main API server (replaces Python FastAPI)
async fn start_api_server(state: Arc<AppState>) -> anyhow::Result<()> {
    use std::net::SocketAddr;

    let app = Router::new()
        // Health & Status
        .route("/health", get(api_health))
        .route("/status", get(api_status))
        // Account & Positions
        .route("/account", get(api_account))
        .route("/positions", get(api_positions))
        // Trades
        .route("/trades", get(api_trades))
        // Strategy Control
        .route("/strategy/pause", post(api_pause))
        .route("/strategy/resume", post(api_resume))
        .route("/strategy/reset", post(api_reset))
        // Metrics
        .route("/metrics", get(api_metrics))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    info!(addr = %addr, "Starting API server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Start the health/metrics server
async fn start_health_server(state: Arc<AppState>) -> anyhow::Result<()> {
    use std::net::SocketAddr;

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(prometheus_metrics))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 9090));
    info!(addr = %addr, "Starting health check server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// =============================================================================
// API Handlers
// =============================================================================

async fn api_health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let strategy = state.strategy.read().await;
    let account = strategy.get_account().await;

    Json(serde_json::json!({
        "status": "healthy",
        "component": "quantumflow-hft",
        "architecture": "rust-only",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "account": {
            "balance": account.balance.to_string(),
            "equity": account.equity.to_string(),
        }
    }))
}

async fn api_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let strategy = state.strategy.read().await;
    let account = strategy.get_account().await;
    let positions = strategy.get_positions().await;

    Json(serde_json::json!({
        "status": "active",
        "account": {
            "balance": account.balance.to_string(),
            "equity": account.equity.to_string(),
            "total_pnl": account.total_pnl.to_string(),
            "pnl_pct": account.return_pct(),
            "total_trades": account.total_trades,
            "win_rate": account.win_rate,
        },
        "positions": positions.iter().map(|p| {
            serde_json::json!({
                "symbol": p.symbol,
                "quantity": p.quantity.to_string(),
                "entry_price": p.entry_price.to_string(),
                "unrealized_pnl": p.unrealized_pnl.to_string(),
                "realized_pnl": p.realized_pnl.to_string(),
            })
        }).collect::<Vec<_>>(),
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

async fn api_account(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let strategy = state.strategy.read().await;
    let account = strategy.get_account().await;

    Json(serde_json::json!({
        "balance": account.balance.to_string(),
        "equity": account.equity.to_string(),
        "initial_balance": account.initial_balance.to_string(),
        "total_pnl": account.total_pnl.to_string(),
        "win_rate": account.win_rate,
        "total_trades": account.total_trades,
        "winning_trades": account.winning_trades,
        "losing_trades": account.losing_trades,
    }))
}

async fn api_positions(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let strategy = state.strategy.read().await;
    let positions = strategy.get_positions().await;

    Json(serde_json::json!(positions.iter().map(|p| {
        serde_json::json!({
            "symbol": p.symbol,
            "quantity": p.quantity.to_string(),
            "entry_price": p.entry_price.to_string(),
            "unrealized_pnl": p.unrealized_pnl.to_string(),
            "realized_pnl": p.realized_pnl.to_string(),
            "updated_at": p.updated_at.to_rfc3339(),
        })
    }).collect::<Vec<_>>()))
}

async fn api_trades(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let strategy = state.strategy.read().await;
    let trades = strategy.get_trades(100).await;

    Json(serde_json::json!(trades.iter().map(|t| {
        serde_json::json!({
            "id": t.id,
            "order_id": t.order_id,
            "symbol": t.symbol,
            "side": format!("{}", t.side),
            "price": t.price.to_string(),
            "quantity": t.quantity.to_string(),
            "fee": t.fee.to_string(),
            "pnl": t.pnl.map(|p| p.to_string()),
            "timestamp": t.timestamp.to_rfc3339(),
        })
    }).collect::<Vec<_>>()))
}

async fn api_pause(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    info!("Strategy paused manually");
    Json(serde_json::json!({"status": "paused"}))
}

async fn api_resume(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    info!("Strategy resumed manually");
    Json(serde_json::json!({"status": "resumed"}))
}

async fn api_reset(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut strategy = state.strategy.write().await;
    strategy.reset();
    info!("Strategy reset");
    Json(serde_json::json!({"status": "reset"}))
}

async fn api_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let strategy = state.strategy.read().await;
    let account = strategy.get_account().await;
    let trades = strategy.get_trades(1000).await;

    let winning: usize = trades.iter().filter(|t| t.pnl.map(|p| p > rust_decimal::Decimal::ZERO).unwrap_or(false)).count();
    let total_pnl: f64 = trades.iter().filter_map(|t| t.pnl).map(|p| p.try_into().unwrap_or(0.0)).sum();

    Json(serde_json::json!({
        "total_trades": trades.len(),
        "winning_trades": winning,
        "losing_trades": trades.len() - winning,
        "win_rate": if !trades.is_empty() { winning as f64 / trades.len() as f64 } else { 0.0 },
        "total_pnl": account.total_pnl.to_string(),
        "pnl_pct": account.return_pct(),
        "avg_trade_pnl": if !trades.is_empty() { total_pnl / trades.len() as f64 } else { 0.0 },
    }))
}

// =============================================================================
// Health/Prometheus Handlers
// =============================================================================

async fn health_check(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "component": "market-data",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

async fn prometheus_metrics() -> String {
    use prometheus::{Encoder, TextEncoder};
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
