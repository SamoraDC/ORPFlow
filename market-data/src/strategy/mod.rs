//! Strategy Engine Module - Jane Street Style
//!
//! Ultra-low latency trading strategy implementation in pure Rust.
//! No Python in the hot path - all execution happens in compiled code.
//!
//! Features:
//! - Order flow imbalance strategy
//! - Microstructure feature calculation
//! - Paper broker for simulation
//! - ONNX model support for ML signals
//! - SQLite persistence for trade history
//! - NSMI regime detection and dynamic weight adjustment
//! - Zero-allocation inference pipeline

mod broker;
mod config;
mod features;
mod models;
mod nsmi;
mod signals;
mod storage;

#[cfg(feature = "ml")]
mod ml_inference;

#[cfg(feature = "ml")]
mod inference_pipeline;

pub use broker::PaperBroker;
pub use config::StrategyConfig;
pub use features::MicrostructureFeatures;
pub use models::{Account, Position, Trade};
// NSMI exports - used by ml_inference when ml feature is enabled
#[allow(unused_imports)]
pub use nsmi::{NSMIConfig, NSMIFeatures, NSMIResult, NSMIState};
pub use signals::ImbalanceStrategy;
pub use storage::TradeStorage;

#[cfg(feature = "ml")]
#[allow(unused_imports)]
pub use ml_inference::{
    FeatureBuffer, ModelEnsemble, ModelType, NSMIAdjustedWeights, NSMIAugmentBuffer, OnnxModel,
};

#[cfg(feature = "ml")]
#[allow(unused_imports)]
pub use inference_pipeline::{InferenceConfig, InferencePipeline, InferenceResult, TradingSignal};

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::orderbook::OrderBookState;

/// Strategy Engine - coordinates all trading logic
pub struct StrategyEngine {
    pub config: StrategyConfig,
    pub features: MicrostructureFeatures,
    pub strategy: ImbalanceStrategy,
    pub broker: Arc<RwLock<PaperBroker>>,
    pub storage: Arc<TradeStorage>,
}

impl StrategyEngine {
    /// Create a new strategy engine
    pub async fn new(config: StrategyConfig) -> anyhow::Result<Self> {
        let storage = Arc::new(TradeStorage::new(&config.database_path).await?);
        let broker = Arc::new(RwLock::new(
            PaperBroker::new(config.initial_balance, storage.clone()).await?,
        ));
        let features = MicrostructureFeatures::new(
            config.feature_window_size,
            config.volatility_window,
            config.momentum_window,
        );
        let strategy = ImbalanceStrategy::new(
            config.imbalance_threshold,
            config.min_confidence,
            config.persistence_required,
        );

        info!(
            initial_balance = %config.initial_balance,
            imbalance_threshold = config.imbalance_threshold,
            "Strategy engine initialized"
        );

        Ok(Self {
            config,
            features,
            strategy,
            broker,
            storage,
        })
    }

    /// Process an order book update and potentially generate trades
    /// This is the HOT PATH - must be ultra-low latency
    #[inline(always)]
    pub async fn on_orderbook_update(&mut self, state: &OrderBookState) -> Option<Trade> {
        // Update microstructure features
        let snapshot = self.features.update(
            &state.symbol,
            state.timestamp,
            state.metrics.mid_price,
            state.metrics.imbalance,
            state.metrics.weighted_imbalance,
            state.metrics.spread_bps,
            state.metrics.bid_depth,
            state.metrics.ask_depth,
        );

        // Evaluate strategy
        let (account, current_position_qty) = {
            let broker = self.broker.read().await;
            let account = broker.get_account();
            let qty = broker.get_position(&state.symbol).map(|p| p.quantity);
            (account, qty)
        };

        let signal = self.strategy.evaluate(
            &snapshot,
            account.balance,
            current_position_qty,
        )?;

        // Execute signal if valid
        if signal.confidence >= self.config.min_confidence {
            let mut broker = self.broker.write().await;

            // Check spread filter
            if let Some(spread) = state.metrics.spread_bps {
                if spread > self.config.max_spread_bps {
                    warn!(
                        symbol = %state.symbol,
                        spread = %spread,
                        max = %self.config.max_spread_bps,
                        "Spread too wide, skipping signal"
                    );
                    return None;
                }
            }

            // Execute trade
            if let Some(mid_price) = state.metrics.mid_price {
                let trade = broker
                    .execute_market_order(
                        &signal.symbol,
                        signal.side,
                        signal.suggested_size,
                        mid_price,
                    )
                    .await;

                if let Some(ref t) = trade {
                    info!(
                        trade_id = %t.id,
                        symbol = %t.symbol,
                        side = ?t.side,
                        price = %t.price,
                        quantity = %t.quantity,
                        pnl = ?t.pnl,
                        "Trade executed"
                    );
                }

                return trade;
            }
        }

        None
    }

    /// Get current account state
    pub async fn get_account(&self) -> Account {
        self.broker.read().await.get_account()
    }

    /// Get all positions
    pub async fn get_positions(&self) -> Vec<Position> {
        self.broker.read().await.get_all_positions()
    }

    /// Get recent trades
    pub async fn get_trades(&self, limit: usize) -> Vec<Trade> {
        self.storage.get_trades(None, limit).await.unwrap_or_default()
    }

    /// Reset strategy state
    pub fn reset(&mut self) {
        self.features.reset();
        self.strategy.reset();
    }
}
