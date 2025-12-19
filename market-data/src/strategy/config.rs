//! Strategy configuration
//!
//! All configuration values with sensible defaults for HFT trading.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Strategy engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    // === Database ===
    /// Path to SQLite database
    pub database_path: String,

    // === Trading ===
    /// Initial paper trading balance
    pub initial_balance: Decimal,
    /// Symbols to trade
    pub symbols: Vec<String>,

    // === Strategy Parameters ===
    /// Minimum imbalance threshold to generate signal (0.0 - 1.0)
    pub imbalance_threshold: f64,
    /// Minimum confidence for signal execution (0.0 - 1.0)
    pub min_confidence: f64,
    /// Number of ticks imbalance must persist
    pub persistence_required: usize,
    /// Position size as percentage of balance (0.0 - 1.0)
    pub position_size_pct: f64,

    // === Risk Parameters ===
    /// Maximum position size in base currency
    pub max_position_size: Decimal,
    /// Maximum drawdown percentage before halting (0.0 - 1.0)
    pub max_drawdown_pct: f64,
    /// Maximum daily trades
    pub max_daily_trades: u32,
    /// Maximum spread in basis points to trade
    pub max_spread_bps: Decimal,

    // === Features ===
    /// Window size for feature calculations
    pub feature_window_size: usize,
    /// Window size for volatility calculation
    pub volatility_window: usize,
    /// Window size for momentum calculation
    pub momentum_window: usize,

    // === Fees ===
    /// Maker fee rate (0.001 = 0.1%)
    pub maker_fee: Decimal,
    /// Taker fee rate (0.001 = 0.1%)
    pub taker_fee: Decimal,

    // === ONNX Model (optional) ===
    /// Path to ONNX model file (None = use rule-based strategy)
    pub onnx_model_path: Option<String>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            // Database
            database_path: "/data/trades.db".to_string(),

            // Trading
            initial_balance: dec!(10000),
            symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],

            // Strategy
            imbalance_threshold: 0.3,
            min_confidence: 0.6,
            persistence_required: 3,
            position_size_pct: 0.1,

            // Risk
            max_position_size: dec!(1.0),
            max_drawdown_pct: 0.05,
            max_daily_trades: 50,
            max_spread_bps: dec!(10),

            // Features
            feature_window_size: 100,
            volatility_window: 20,
            momentum_window: 10,

            // Fees (Binance VIP 0)
            maker_fee: dec!(0.001),
            taker_fee: dec!(0.001),

            // ONNX
            onnx_model_path: None,
        }
    }
}

impl StrategyConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Database
        if let Ok(v) = std::env::var("DATABASE_URL") {
            config.database_path = v.replace("sqlite:///", "");
        }

        // Trading
        if let Ok(v) = std::env::var("INITIAL_BALANCE") {
            if let Ok(d) = v.parse() {
                config.initial_balance = d;
            }
        }
        if let Ok(v) = std::env::var("SYMBOLS") {
            config.symbols = v.split(',').map(|s| s.trim().to_string()).collect();
        }

        // Strategy
        if let Ok(v) = std::env::var("IMBALANCE_THRESHOLD") {
            if let Ok(f) = v.parse() {
                config.imbalance_threshold = f;
            }
        }
        if let Ok(v) = std::env::var("MIN_CONFIDENCE") {
            if let Ok(f) = v.parse() {
                config.min_confidence = f;
            }
        }
        if let Ok(v) = std::env::var("POSITION_SIZE_PCT") {
            if let Ok(f) = v.parse() {
                config.position_size_pct = f;
            }
        }

        // Risk
        if let Ok(v) = std::env::var("RISK_MAX_POSITION") {
            if let Ok(d) = v.parse() {
                config.max_position_size = d;
            }
        }
        if let Ok(v) = std::env::var("RISK_MAX_DRAWDOWN") {
            if let Ok(f) = v.parse() {
                config.max_drawdown_pct = f;
            }
        }
        if let Ok(v) = std::env::var("MAX_DAILY_TRADES") {
            if let Ok(n) = v.parse() {
                config.max_daily_trades = n;
            }
        }

        // ONNX
        if let Ok(v) = std::env::var("ONNX_MODEL_PATH") {
            if !v.is_empty() {
                config.onnx_model_path = Some(v);
            }
        }

        config
    }
}
