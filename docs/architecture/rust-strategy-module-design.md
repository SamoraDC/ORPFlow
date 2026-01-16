# Rust Strategy Module Architecture Design

## Executive Summary

This document outlines the architecture for a high-performance trading strategy module written in Rust that will be integrated directly into the `market-data` crate, replacing the existing Python/FastAPI strategy service.

**Key Design Goals:**
- **Ultra-low latency**: Nanosecond tick-to-trade performance
- **Zero-copy architecture**: Lock-free data structures where possible
- **Single unified binary**: market-data + strategy in one process
- **ML-ready**: ONNX model loading support for future integration
- **Production-ready**: Comprehensive persistence, metrics, and observability

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Rust Binary                          │
│                                                                 │
│  ┌──────────────────┐         ┌─────────────────────────────┐  │
│  │  Market Data     │  MPSC   │    Strategy Engine          │  │
│  │  (WebSocket)     │────────▶│  (Lock-free channels)       │  │
│  │                  │         │                             │  │
│  │  - OrderBook     │         │  - Feature Calculator       │  │
│  │  - Trade Feed    │         │  - Signal Generator         │  │
│  │  - Snapshot Mgr  │         │  - Position Manager         │  │
│  └──────────────────┘         │  - Risk Checks              │  │
│           │                   │  - Paper Broker             │  │
│           │                   └─────────────────────────────┘  │
│           │                              │                     │
│           │                              │ MPSC                │
│           ▼                              ▼                     │
│  ┌──────────────────┐         ┌─────────────────────────────┐  │
│  │  Metrics/Health  │         │    Persistence Layer        │  │
│  │  (Prometheus)    │         │    (SQLite WAL mode)        │  │
│  └──────────────────┘         └─────────────────────────────┘  │
│                                                                 │
│           │                              │                     │
│           ▼                              ▼                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Optional: OCaml Risk Gateway (IPC)               │  │
│  │         (For production-critical risk checks)            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
market-data/src/
├── strategy/
│   ├── mod.rs                    # Public API and types
│   ├── engine.rs                 # Main strategy execution engine
│   ├── features/
│   │   ├── mod.rs                # Feature extraction pipeline
│   │   ├── microstructure.rs     # Volatility, momentum, imbalance
│   │   ├── orderbook_features.rs # Depth, spread, pressure
│   │   ├── trade_features.rs     # VWAP, flow toxicity
│   │   └── window.rs             # Lock-free circular buffers
│   ├── signals/
│   │   ├── mod.rs                # Signal generation framework
│   │   ├── imbalance.rs          # Imbalance-based strategy
│   │   ├── threshold.rs          # Threshold/rule engine
│   │   └── ml_inference.rs       # ONNX model integration
│   ├── execution/
│   │   ├── mod.rs                # Order execution framework
│   │   ├── paper_broker.rs       # Simulated execution
│   │   ├── position_manager.rs   # Position tracking
│   │   ├── order_manager.rs      # Order lifecycle
│   │   └── fill_simulator.rs     # Realistic fill simulation
│   ├── risk/
│   │   ├── mod.rs                # Risk management
│   │   ├── limits.rs             # Position/exposure limits
│   │   ├── validator.rs          # Pre-trade validation
│   │   └── circuit_breaker.rs    # Kill switches
│   ├── persistence/
│   │   ├── mod.rs                # Data persistence
│   │   ├── database.rs           # SQLite connection pool
│   │   ├── trade_log.rs          # Trade history
│   │   └── state_snapshot.rs     # Strategy state snapshots
│   └── types/
│       ├── mod.rs                # Common types
│       ├── market_data.rs        # Lock-free market data types
│       ├── order.rs              # Order types
│       ├── signal.rs             # Signal types
│       └── metrics.rs            # Strategy metrics
```

## Core Type System

### Lock-Free Market Data Types

```rust
/// Zero-copy market data snapshot using atomic operations
/// Designed for lock-free concurrent access from strategy thread
pub struct MarketSnapshot {
    /// Timestamp in nanoseconds (atomic for lock-free reads)
    pub timestamp_ns: AtomicU64,

    /// Best bid price (encoded as u64, actual: f64 bits)
    pub best_bid: AtomicU64,

    /// Best ask price (encoded as u64)
    pub best_ask: AtomicU64,

    /// Mid price (encoded as u64)
    pub mid_price: AtomicU64,

    /// Order book imbalance (scaled by 1e9, stored as i64)
    pub imbalance: AtomicI64,

    /// Spread in basis points (scaled by 100)
    pub spread_bps: AtomicU32,

    /// Sequence number for ordering (monotonic)
    pub sequence: AtomicU64,
}

impl MarketSnapshot {
    /// Read snapshot atomically with Acquire ordering
    pub fn read(&self) -> SnapshotView {
        SnapshotView {
            timestamp_ns: self.timestamp_ns.load(Ordering::Acquire),
            best_bid: f64::from_bits(self.best_bid.load(Ordering::Acquire)),
            best_ask: f64::from_bits(self.best_ask.load(Ordering::Acquire)),
            mid_price: f64::from_bits(self.mid_price.load(Ordering::Acquire)),
            imbalance: self.imbalance.load(Ordering::Acquire) as f64 / 1e9,
            spread_bps: self.spread_bps.load(Ordering::Acquire) as f64 / 100.0,
            sequence: self.sequence.load(Ordering::Acquire),
        }
    }

    /// Write snapshot atomically with Release ordering
    pub fn write(&self, view: &SnapshotView) {
        self.timestamp_ns.store(view.timestamp_ns, Ordering::Release);
        self.best_bid.store(view.best_bid.to_bits(), Ordering::Release);
        self.best_ask.store(view.best_ask.to_bits(), Ordering::Release);
        self.mid_price.store(view.mid_price.to_bits(), Ordering::Release);
        self.imbalance.store((view.imbalance * 1e9) as i64, Ordering::Release);
        self.spread_bps.store((view.spread_bps * 100.0) as u32, Ordering::Release);
        self.sequence.fetch_add(1, Ordering::AcqRel);
    }
}

/// Non-atomic view of market snapshot (for reading)
#[derive(Debug, Clone, Copy)]
pub struct SnapshotView {
    pub timestamp_ns: u64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub mid_price: f64,
    pub imbalance: f64,
    pub spread_bps: f64,
    pub sequence: u64,
}
```

### Order Types

```rust
/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    PostOnly,  // Maker-only orders
    IOC,       // Immediate-or-cancel
    FOK,       // Fill-or-kill
}

/// Order time-in-force
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC,  // Good-till-cancel
    IOC,  // Immediate-or-cancel
    FOK,  // Fill-or-kill
    GTD,  // Good-till-date
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID (monotonic)
    pub id: OrderId,

    /// Client order ID (strategy-generated)
    pub client_order_id: String,

    /// Trading symbol
    pub symbol: String,

    /// Order side
    pub side: Side,

    /// Order type
    pub order_type: OrderType,

    /// Time in force
    pub time_in_force: TimeInForce,

    /// Order quantity
    pub quantity: Decimal,

    /// Limit price (None for market orders)
    pub price: Option<Decimal>,

    /// Current status
    pub status: OrderStatus,

    /// Filled quantity
    pub filled_quantity: Decimal,

    /// Average fill price
    pub avg_fill_price: Option<Decimal>,

    /// Creation timestamp (nanoseconds)
    pub created_at: u64,

    /// Last update timestamp
    pub updated_at: u64,

    /// Strategy identifier
    pub strategy_id: String,
}

/// Unique order identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub u64);

impl OrderId {
    /// Generate next order ID atomically
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}
```

### Signal Types

```rust
/// Trading signal strength
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SignalStrength(f64);

impl SignalStrength {
    /// Create new signal strength (clamped to [-1.0, 1.0])
    pub fn new(value: f64) -> Self {
        Self(value.clamp(-1.0, 1.0))
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if signal exceeds threshold
    pub fn exceeds(&self, threshold: f64) -> bool {
        self.0.abs() >= threshold
    }
}

/// Trading signal
#[derive(Debug, Clone)]
pub struct Signal {
    /// Signal timestamp (nanoseconds)
    pub timestamp_ns: u64,

    /// Trading symbol
    pub symbol: String,

    /// Signal strength [-1.0, 1.0] (negative = sell, positive = buy)
    pub strength: SignalStrength,

    /// Confidence score [0.0, 1.0]
    pub confidence: f64,

    /// Expected holding period (milliseconds)
    pub holding_period_ms: u32,

    /// Strategy identifier
    pub strategy_id: String,

    /// Feature values that generated this signal
    pub features: FeatureVector,
}

/// Signal generation result
#[derive(Debug, Clone)]
pub enum SignalResult {
    /// Strong signal, ready to trade
    Signal(Signal),

    /// No signal, conditions not met
    NoSignal,

    /// Signal suppressed by risk checks
    Suppressed { reason: String },
}
```

## Strategy Engine Architecture

### Main Engine Interface

```rust
/// Strategy engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy identifier
    pub strategy_id: String,

    /// Trading symbols
    pub symbols: Vec<String>,

    /// Strategy parameters
    pub params: StrategyParams,

    /// Risk limits
    pub risk_limits: RiskLimits,

    /// Enable paper trading
    pub paper_trading: bool,

    /// Enable ONNX model inference
    pub enable_ml: bool,

    /// ONNX model path (if enable_ml = true)
    pub model_path: Option<PathBuf>,
}

/// Strategy parameters for imbalance trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParams {
    /// Imbalance threshold to trigger signal
    pub imbalance_threshold: f64,

    /// Minimum confidence score
    pub min_confidence: f64,

    /// Feature calculation window (milliseconds)
    pub feature_window_ms: u32,

    /// Signal decay half-life (milliseconds)
    pub signal_decay_ms: u32,

    /// Maximum position size
    pub max_position_size: Decimal,

    /// Order size as fraction of position limit
    pub order_size_fraction: f64,
}

/// Main strategy engine
pub struct StrategyEngine {
    /// Engine configuration
    config: Arc<StrategyConfig>,

    /// Feature calculator
    feature_calculator: Arc<FeatureCalculator>,

    /// Signal generator
    signal_generator: Arc<dyn SignalGenerator>,

    /// Position manager
    position_manager: Arc<RwLock<PositionManager>>,

    /// Order manager
    order_manager: Arc<RwLock<OrderManager>>,

    /// Risk validator
    risk_validator: Arc<RiskValidator>,

    /// Paper broker (if enabled)
    paper_broker: Option<Arc<PaperBroker>>,

    /// Persistence layer
    persistence: Arc<PersistenceManager>,

    /// Market data receiver (lock-free channel)
    market_rx: crossbeam_channel::Receiver<MarketUpdate>,

    /// Order sender (lock-free channel)
    order_tx: crossbeam_channel::Sender<Order>,

    /// Metrics collector
    metrics: Arc<StrategyMetrics>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

impl StrategyEngine {
    /// Create new strategy engine
    pub async fn new(
        config: StrategyConfig,
        market_rx: crossbeam_channel::Receiver<MarketUpdate>,
        order_tx: crossbeam_channel::Sender<Order>,
    ) -> Result<Self>;

    /// Run the strategy engine (blocking)
    pub async fn run(&self) -> Result<()>;

    /// Process a single market update
    async fn process_market_update(&self, update: MarketUpdate) -> Result<()>;

    /// Generate and evaluate signals
    async fn generate_signals(&self, features: &FeatureVector) -> Result<Vec<Signal>>;

    /// Execute a signal (create orders)
    async fn execute_signal(&self, signal: Signal) -> Result<()>;

    /// Shutdown the engine gracefully
    pub async fn shutdown(&self) -> Result<()>;
}
```

### Market Update Types

```rust
/// Market data update
#[derive(Debug, Clone)]
pub enum MarketUpdate {
    /// Order book update
    OrderBook {
        symbol: String,
        timestamp_ns: u64,
        snapshot: Arc<MarketSnapshot>,
    },

    /// Trade execution
    Trade {
        symbol: String,
        timestamp_ns: u64,
        price: Decimal,
        quantity: Decimal,
        side: Side,
    },

    /// Aggregated metrics (periodic)
    Metrics {
        symbol: String,
        timestamp_ns: u64,
        metrics: OrderBookMetrics,
    },
}
```

## Feature Calculation Pipeline

### Feature Calculator Interface

```rust
/// Feature calculator with lock-free circular buffers
pub struct FeatureCalculator {
    /// Configuration
    config: FeatureConfig,

    /// Lock-free circular buffers for each symbol
    buffers: DashMap<String, Arc<FeatureBuffer>>,

    /// Feature extractors (composable pipeline)
    extractors: Vec<Box<dyn FeatureExtractor>>,
}

/// Feature configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Window size for rolling calculations (milliseconds)
    pub window_ms: u32,

    /// Update frequency (microseconds)
    pub update_interval_us: u32,

    /// Number of quantiles to calculate
    pub num_quantiles: usize,
}

/// Lock-free circular buffer for market data
pub struct FeatureBuffer {
    /// Circular buffer of snapshots (lock-free)
    snapshots: Arc<ArrayQueue<SnapshotView>>,

    /// Buffer capacity
    capacity: usize,

    /// Current write index
    write_idx: AtomicUsize,
}

impl FeatureBuffer {
    /// Push new snapshot (lock-free)
    pub fn push(&self, snapshot: SnapshotView) {
        let _ = self.snapshots.push(snapshot);
        self.write_idx.fetch_add(1, Ordering::Release);
    }

    /// Get all snapshots in window (lock-free read)
    pub fn get_window(&self) -> Vec<SnapshotView> {
        self.snapshots.iter().collect()
    }
}

/// Feature vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub timestamp_ns: u64,
    pub symbol: String,

    // Microstructure features
    pub volatility: f64,
    pub momentum: f64,
    pub imbalance: f64,
    pub weighted_imbalance: f64,

    // Order book features
    pub spread_bps: f64,
    pub depth_ratio: f64,
    pub bid_pressure: f64,
    pub ask_pressure: f64,

    // Trade features
    pub vwap: f64,
    pub trade_flow: f64,
    pub toxicity: f64,

    // Derived features
    pub price_acceleration: f64,
    pub volume_surge: f64,
}

/// Feature extractor trait
#[async_trait]
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from market data buffer
    async fn extract(&self, buffer: &FeatureBuffer) -> Result<HashMap<String, f64>>;

    /// Feature names produced by this extractor
    fn feature_names(&self) -> Vec<String>;
}
```

### Microstructure Feature Extractor

```rust
/// Microstructure feature extractor
pub struct MicrostructureExtractor {
    config: FeatureConfig,
}

#[async_trait]
impl FeatureExtractor for MicrostructureExtractor {
    async fn extract(&self, buffer: &FeatureBuffer) -> Result<HashMap<String, f64>> {
        let snapshots = buffer.get_window();

        let mut features = HashMap::new();

        // Volatility (realized variance of mid-price returns)
        features.insert("volatility".to_string(), self.calculate_volatility(&snapshots));

        // Momentum (exponentially weighted return)
        features.insert("momentum".to_string(), self.calculate_momentum(&snapshots));

        // Imbalance (current order book imbalance)
        features.insert("imbalance".to_string(), self.calculate_imbalance(&snapshots));

        // Weighted imbalance (time-weighted average)
        features.insert("weighted_imbalance".to_string(),
                       self.calculate_weighted_imbalance(&snapshots));

        Ok(features)
    }

    fn feature_names(&self) -> Vec<String> {
        vec![
            "volatility".to_string(),
            "momentum".to_string(),
            "imbalance".to_string(),
            "weighted_imbalance".to_string(),
        ]
    }
}

impl MicrostructureExtractor {
    fn calculate_volatility(&self, snapshots: &[SnapshotView]) -> f64 {
        // Realized variance of log returns
        // σ² = Σ(log(P_t / P_{t-1}))² / (n-1)
        // Implementation details omitted for brevity
        0.0
    }

    fn calculate_momentum(&self, snapshots: &[SnapshotView]) -> f64 {
        // Exponentially weighted return
        // M = Σ(w_i * r_i) where w_i = α^i, r_i = log(P_i / P_{i-1})
        // Implementation details omitted for brevity
        0.0
    }

    fn calculate_imbalance(&self, snapshots: &[SnapshotView]) -> f64 {
        // Current order book imbalance (from latest snapshot)
        snapshots.last().map(|s| s.imbalance).unwrap_or(0.0)
    }

    fn calculate_weighted_imbalance(&self, snapshots: &[SnapshotView]) -> f64 {
        // Time-weighted average imbalance
        // Implementation details omitted for brevity
        0.0
    }
}
```

## Signal Generation

### Signal Generator Interface

```rust
/// Signal generator trait
#[async_trait]
pub trait SignalGenerator: Send + Sync {
    /// Generate signal from features
    async fn generate(&self, features: &FeatureVector) -> Result<SignalResult>;

    /// Strategy identifier
    fn strategy_id(&self) -> &str;
}

/// Imbalance-based signal generator
pub struct ImbalanceSignalGenerator {
    strategy_id: String,
    params: StrategyParams,
}

#[async_trait]
impl SignalGenerator for ImbalanceSignalGenerator {
    async fn generate(&self, features: &FeatureVector) -> Result<SignalResult> {
        // Check if imbalance exceeds threshold
        if features.imbalance.abs() < self.params.imbalance_threshold {
            return Ok(SignalResult::NoSignal);
        }

        // Calculate signal strength (scaled imbalance)
        let strength = SignalStrength::new(
            features.imbalance / self.params.imbalance_threshold
        );

        // Calculate confidence based on momentum and volatility
        let confidence = self.calculate_confidence(features);

        if confidence < self.params.min_confidence {
            return Ok(SignalResult::Suppressed {
                reason: "Confidence below threshold".to_string(),
            });
        }

        // Estimate holding period from mean reversion time
        let holding_period_ms = self.estimate_holding_period(features);

        Ok(SignalResult::Signal(Signal {
            timestamp_ns: features.timestamp_ns,
            symbol: features.symbol.clone(),
            strength,
            confidence,
            holding_period_ms,
            strategy_id: self.strategy_id.clone(),
            features: features.clone(),
        }))
    }

    fn strategy_id(&self) -> &str {
        &self.strategy_id
    }
}

impl ImbalanceSignalGenerator {
    fn calculate_confidence(&self, features: &FeatureVector) -> f64 {
        // Combine momentum strength and inverse volatility
        // High momentum + low volatility = high confidence
        let momentum_score = (features.momentum.abs() / 0.01).min(1.0);
        let volatility_score = (0.01 / features.volatility.max(0.001)).min(1.0);

        (momentum_score + volatility_score) / 2.0
    }

    fn estimate_holding_period(&self, features: &FeatureVector) -> u32 {
        // Mean reversion time ~ 1 / |imbalance|
        // Clamped between 100ms and 10s
        let base_period = 1000.0 / features.imbalance.abs().max(0.01);
        base_period.clamp(100.0, 10_000.0) as u32
    }
}
```

### ONNX ML Signal Generator

```rust
/// ONNX-based ML signal generator
pub struct MLSignalGenerator {
    strategy_id: String,
    params: StrategyParams,

    /// ONNX runtime session
    session: Arc<ort::Session>,

    /// Feature normalizer
    normalizer: FeatureNormalizer,
}

#[async_trait]
impl SignalGenerator for MLSignalGenerator {
    async fn generate(&self, features: &FeatureVector) -> Result<SignalResult> {
        // Normalize features
        let normalized = self.normalizer.normalize(features)?;

        // Run ONNX inference
        let prediction = self.run_inference(&normalized).await?;

        // Convert prediction to signal
        let strength = SignalStrength::new(prediction.direction);
        let confidence = prediction.confidence;

        if confidence < self.params.min_confidence {
            return Ok(SignalResult::Suppressed {
                reason: "ML confidence below threshold".to_string(),
            });
        }

        Ok(SignalResult::Signal(Signal {
            timestamp_ns: features.timestamp_ns,
            symbol: features.symbol.clone(),
            strength,
            confidence,
            holding_period_ms: prediction.holding_period_ms,
            strategy_id: self.strategy_id.clone(),
            features: features.clone(),
        }))
    }

    fn strategy_id(&self) -> &str {
        &self.strategy_id
    }
}

impl MLSignalGenerator {
    async fn run_inference(&self, features: &[f32]) -> Result<Prediction> {
        // Create input tensor
        let input = ndarray::Array::from_shape_vec(
            (1, features.len()),
            features.to_vec()
        )?;

        // Run ONNX session
        let outputs = self.session.run(ort::inputs!["features" => input]?)?;

        // Extract predictions
        let direction: f32 = outputs["direction"].extract_tensor()?[[0]];
        let confidence: f32 = outputs["confidence"].extract_tensor()?[[0]];
        let holding_ms: f32 = outputs["holding_period"].extract_tensor()?[[0]];

        Ok(Prediction {
            direction: direction as f64,
            confidence: confidence as f64,
            holding_period_ms: holding_ms as u32,
        })
    }
}

struct Prediction {
    direction: f64,
    confidence: f64,
    holding_period_ms: u32,
}
```

## Paper Broker

### Paper Execution Engine

```rust
/// Paper broker for simulated trading
pub struct PaperBroker {
    /// Current positions
    positions: Arc<RwLock<HashMap<String, Position>>>,

    /// Active orders
    orders: Arc<RwLock<HashMap<OrderId, Order>>>,

    /// Fill simulator
    fill_simulator: Arc<FillSimulator>,

    /// Trade log
    trade_log: Arc<RwLock<Vec<Trade>>>,

    /// Performance metrics
    metrics: Arc<RwLock<BrokerMetrics>>,
}

impl PaperBroker {
    /// Submit a new order
    pub async fn submit_order(&self, order: Order) -> Result<OrderId> {
        // Validate order
        self.validate_order(&order)?;

        // Store order
        let mut orders = self.orders.write().await;
        orders.insert(order.id, order.clone());
        drop(orders);

        Ok(order.id)
    }

    /// Process market update and simulate fills
    pub async fn process_market_update(&self, update: &MarketUpdate) -> Result<Vec<Fill>> {
        let mut fills = Vec::new();
        let mut orders = self.orders.write().await;

        for (id, order) in orders.iter_mut() {
            if let Some(fill) = self.fill_simulator.check_fill(order, update).await? {
                // Update order status
                order.filled_quantity += fill.quantity;
                order.status = if order.filled_quantity >= order.quantity {
                    OrderStatus::Filled
                } else {
                    OrderStatus::PartiallyFilled
                };

                // Update position
                self.update_position(&fill).await?;

                // Log trade
                self.log_trade(&fill).await?;

                fills.push(fill);
            }
        }

        // Remove filled/cancelled orders
        orders.retain(|_, order| {
            order.status != OrderStatus::Filled && order.status != OrderStatus::Cancelled
        });

        Ok(fills)
    }

    async fn update_position(&self, fill: &Fill) -> Result<()> {
        let mut positions = self.positions.write().await;
        let position = positions.entry(fill.symbol.clone()).or_insert(Position::default());

        match fill.side {
            Side::Buy => {
                position.quantity += fill.quantity;
                position.cost_basis =
                    (position.cost_basis * position.quantity + fill.price * fill.quantity)
                    / (position.quantity + fill.quantity);
            }
            Side::Sell => {
                position.quantity -= fill.quantity;
                if position.quantity < Decimal::ZERO {
                    position.quantity = Decimal::ZERO;
                }
            }
        }

        Ok(())
    }

    async fn log_trade(&self, fill: &Fill) -> Result<()> {
        let mut trade_log = self.trade_log.write().await;
        trade_log.push(Trade {
            id: TradeId::next(),
            order_id: fill.order_id,
            symbol: fill.symbol.clone(),
            side: fill.side,
            price: fill.price,
            quantity: fill.quantity,
            timestamp_ns: fill.timestamp_ns,
            strategy_id: fill.strategy_id.clone(),
        });
        Ok(())
    }
}

/// Fill event
#[derive(Debug, Clone)]
pub struct Fill {
    pub order_id: OrderId,
    pub symbol: String,
    pub side: Side,
    pub price: Decimal,
    pub quantity: Decimal,
    pub timestamp_ns: u64,
    pub strategy_id: String,
}

/// Position
#[derive(Debug, Clone, Default)]
pub struct Position {
    pub quantity: Decimal,
    pub cost_basis: Decimal,
    pub realized_pnl: Decimal,
}
```

### Fill Simulator

```rust
/// Realistic fill simulator
pub struct FillSimulator {
    config: FillConfig,
}

#[derive(Debug, Clone)]
pub struct FillConfig {
    /// Probability of fill at touch (0.0 = never, 1.0 = always)
    pub fill_probability_at_touch: f64,

    /// Adverse selection factor (slippage)
    pub adverse_selection_bps: f64,

    /// Queue position percentile (0.0 = back, 1.0 = front)
    pub queue_position: f64,
}

impl FillSimulator {
    /// Check if order should be filled
    pub async fn check_fill(
        &self,
        order: &Order,
        update: &MarketUpdate,
    ) -> Result<Option<Fill>> {
        match update {
            MarketUpdate::OrderBook { snapshot, .. } => {
                self.check_limit_fill(order, snapshot).await
            }
            MarketUpdate::Trade { price, .. } => {
                self.check_trade_fill(order, *price).await
            }
            _ => Ok(None),
        }
    }

    async fn check_limit_fill(
        &self,
        order: &Order,
        snapshot: &Arc<MarketSnapshot>,
    ) -> Result<Option<Fill>> {
        let view = snapshot.read();

        // Check if order price is at or through the market
        let fill_price = match (order.side, order.price) {
            (Side::Buy, Some(limit_price)) if limit_price >= view.best_ask => {
                // Aggressive buy, immediate fill at best ask
                view.best_ask
            }
            (Side::Sell, Some(limit_price)) if limit_price <= view.best_bid => {
                // Aggressive sell, immediate fill at best bid
                view.best_bid
            }
            (Side::Buy, Some(limit_price)) if limit_price == view.best_bid => {
                // Passive buy at touch, probabilistic fill
                if self.should_fill_at_touch() {
                    limit_price
                } else {
                    return Ok(None);
                }
            }
            (Side::Sell, Some(limit_price)) if limit_price == view.best_ask => {
                // Passive sell at touch, probabilistic fill
                if self.should_fill_at_touch() {
                    limit_price
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        // Apply adverse selection
        let adjusted_price = self.apply_adverse_selection(fill_price, order.side);

        Ok(Some(Fill {
            order_id: order.id,
            symbol: order.symbol.clone(),
            side: order.side,
            price: adjusted_price,
            quantity: order.quantity - order.filled_quantity,
            timestamp_ns: view.timestamp_ns,
            strategy_id: order.strategy_id.clone(),
        }))
    }

    fn should_fill_at_touch(&self) -> bool {
        // Simulate queue position and random fills
        let mut rng = rand::thread_rng();
        rng.gen::<f64>() < self.config.fill_probability_at_touch * self.config.queue_position
    }

    fn apply_adverse_selection(&self, price: f64, side: Side) -> Decimal {
        let slippage_factor = 1.0 + self.config.adverse_selection_bps / 10_000.0;
        let adjusted = match side {
            Side::Buy => price * slippage_factor,
            Side::Sell => price / slippage_factor,
        };
        Decimal::from_f64_retain(adjusted).unwrap()
    }
}
```

## Persistence Layer

### Database Schema

```sql
-- Strategy state snapshots
CREATE TABLE IF NOT EXISTS strategy_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    INDEX idx_strategy_timestamp (strategy_id, timestamp_ns)
);

-- Trade log
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL UNIQUE,
    order_id INTEGER NOT NULL,
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    INDEX idx_strategy_symbol (strategy_id, symbol),
    INDEX idx_timestamp (timestamp_ns)
);

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL UNIQUE,
    client_order_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    time_in_force TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL,
    status TEXT NOT NULL,
    filled_quantity REAL NOT NULL DEFAULT 0,
    avg_fill_price REAL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    INDEX idx_strategy_status (strategy_id, status),
    INDEX idx_created_at (created_at)
);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    cost_basis REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL,
    UNIQUE(strategy_id, symbol)
);

-- Performance metrics (periodic snapshots)
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    total_pnl REAL NOT NULL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    avg_trade_pnl REAL,
    num_trades INTEGER NOT NULL,
    INDEX idx_strategy_timestamp (strategy_id, timestamp_ns)
);
```

### Persistence Manager

```rust
/// Persistence manager with SQLite
pub struct PersistenceManager {
    /// Database connection pool
    pool: Arc<Pool<SqliteConnectionManager>>,

    /// Async write queue (lock-free)
    write_queue: Arc<crossbeam_channel::Sender<PersistenceOp>>,

    /// Background writer task
    writer_handle: Option<tokio::task::JoinHandle<()>>,
}

impl PersistenceManager {
    /// Create new persistence manager
    pub async fn new(db_path: &Path) -> Result<Self> {
        // Open SQLite in WAL mode for better concurrency
        let manager = SqliteConnectionManager::file(db_path)
            .with_init(|conn| {
                conn.execute_batch(
                    "PRAGMA journal_mode = WAL;
                     PRAGMA synchronous = NORMAL;
                     PRAGMA cache_size = -64000;
                     PRAGMA temp_store = MEMORY;"
                )?;
                Ok(())
            });

        let pool = Arc::new(Pool::new(manager)?);

        // Initialize schema
        Self::init_schema(&pool).await?;

        // Start background writer
        let (write_tx, write_rx) = crossbeam_channel::bounded(10_000);
        let writer_handle = Self::spawn_writer(pool.clone(), write_rx);

        Ok(Self {
            pool,
            write_queue: Arc::new(write_tx),
            writer_handle: Some(writer_handle),
        })
    }

    /// Log a trade (async, non-blocking)
    pub fn log_trade(&self, trade: Trade) -> Result<()> {
        self.write_queue.send(PersistenceOp::LogTrade(trade))?;
        Ok(())
    }

    /// Save order (async, non-blocking)
    pub fn save_order(&self, order: Order) -> Result<()> {
        self.write_queue.send(PersistenceOp::SaveOrder(order))?;
        Ok(())
    }

    /// Save strategy snapshot (async, non-blocking)
    pub fn save_snapshot(&self, snapshot: StrategySnapshot) -> Result<()> {
        self.write_queue.send(PersistenceOp::SaveSnapshot(snapshot))?;
        Ok(())
    }

    /// Query trades (blocking)
    pub async fn query_trades(
        &self,
        strategy_id: &str,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<Vec<Trade>> {
        let conn = self.pool.get()?;
        // SQL query implementation
        // Details omitted for brevity
        Ok(vec![])
    }

    fn spawn_writer(
        pool: Arc<Pool<SqliteConnectionManager>>,
        rx: crossbeam_channel::Receiver<PersistenceOp>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::task::spawn_blocking(move || {
            while let Ok(op) = rx.recv() {
                if let Err(e) = Self::execute_op(&pool, op) {
                    tracing::error!("Persistence error: {}", e);
                }
            }
        })
    }

    fn execute_op(pool: &Pool<SqliteConnectionManager>, op: PersistenceOp) -> Result<()> {
        match op {
            PersistenceOp::LogTrade(trade) => {
                let conn = pool.get()?;
                conn.execute(
                    "INSERT INTO trades (trade_id, order_id, strategy_id, symbol, side,
                                        price, quantity, timestamp_ns)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        trade.id.0,
                        trade.order_id.0,
                        trade.strategy_id,
                        trade.symbol,
                        format!("{:?}", trade.side),
                        trade.price.to_f64().unwrap(),
                        trade.quantity.to_f64().unwrap(),
                        trade.timestamp_ns,
                    ],
                )?;
            }
            PersistenceOp::SaveOrder(order) => {
                // Insert or update order
                // Implementation details omitted
            }
            PersistenceOp::SaveSnapshot(snapshot) => {
                // Save strategy snapshot
                // Implementation details omitted
            }
        }
        Ok(())
    }
}

enum PersistenceOp {
    LogTrade(Trade),
    SaveOrder(Order),
    SaveSnapshot(StrategySnapshot),
}
```

## Integration Points

### Market Data → Strategy Pipeline

```rust
/// Market data publisher extension for strategy integration
impl Publisher {
    /// Publish to strategy engine (lock-free channel)
    pub fn publish_to_strategy(
        &self,
        strategy_tx: &crossbeam_channel::Sender<MarketUpdate>,
        state: &OrderBookState,
    ) -> Result<()> {
        // Create atomic snapshot
        let snapshot = Arc::new(MarketSnapshot::default());
        snapshot.write(&SnapshotView {
            timestamp_ns: state.timestamp * 1_000_000, // ms to ns
            best_bid: state.metrics.mid_price.unwrap_or_default().to_f64().unwrap_or(0.0)
                      - state.metrics.spread_bps.unwrap_or_default().to_f64().unwrap_or(0.0) / 20_000.0,
            best_ask: state.metrics.mid_price.unwrap_or_default().to_f64().unwrap_or(0.0)
                      + state.metrics.spread_bps.unwrap_or_default().to_f64().unwrap_or(0.0) / 20_000.0,
            mid_price: state.metrics.mid_price.unwrap_or_default().to_f64().unwrap_or(0.0),
            imbalance: state.metrics.imbalance.unwrap_or_default().to_f64().unwrap_or(0.0),
            spread_bps: state.metrics.spread_bps.unwrap_or_default().to_f64().unwrap_or(0.0),
            sequence: 0,
        });

        // Send via lock-free channel
        strategy_tx.send(MarketUpdate::OrderBook {
            symbol: state.symbol.clone(),
            timestamp_ns: state.timestamp * 1_000_000,
            snapshot,
        })?;

        Ok(())
    }
}
```

### Main Application Integration

```rust
/// Updated AppState with strategy integration
pub struct AppState {
    pub orderbook_manager: Arc<RwLock<OrderBookManager>>,
    pub publisher: Arc<Publisher>,
    pub strategy_engine: Arc<StrategyEngine>,
    pub config: Arc<Config>,
}

/// Main application loop
pub async fn run_unified_hft_system(config: Config) -> Result<()> {
    // Create lock-free channels
    let (market_tx, market_rx) = crossbeam_channel::bounded(10_000);
    let (order_tx, order_rx) = crossbeam_channel::bounded(1_000);

    // Initialize strategy engine
    let strategy_config = StrategyConfig {
        strategy_id: "imbalance_v1".to_string(),
        symbols: config.symbols.clone(),
        params: StrategyParams {
            imbalance_threshold: 0.3,
            min_confidence: 0.7,
            feature_window_ms: 1000,
            signal_decay_ms: 500,
            max_position_size: Decimal::from(1),
            order_size_fraction: 0.5,
        },
        risk_limits: RiskLimits::default(),
        paper_trading: true,
        enable_ml: false,
        model_path: None,
    };

    let strategy_engine = Arc::new(
        StrategyEngine::new(strategy_config, market_rx, order_tx).await?
    );

    // Start strategy engine
    let strategy_handle = {
        let engine = strategy_engine.clone();
        tokio::spawn(async move {
            engine.run().await
        })
    };

    // Initialize order book manager and WebSocket
    let orderbook_manager = Arc::new(RwLock::new(OrderBookManager::new()));
    let publisher = Arc::new(Publisher::new());

    // Extended publisher to send to strategy
    let publisher_clone = publisher.clone();
    let market_tx_clone = market_tx.clone();

    // ... WebSocket setup and event loop ...

    // On each order book update:
    // 1. Update local order book
    // 2. Publish to strategy via lock-free channel
    // 3. Publish to external systems (Redis, etc.)

    strategy_handle.await??;

    Ok(())
}
```

## Performance Characteristics

### Latency Targets

| Component | Target Latency | Notes |
|-----------|---------------|-------|
| Market data → Feature calculation | < 1 μs | Lock-free read from atomic snapshot |
| Feature calculation → Signal | < 10 μs | Simple imbalance strategy |
| Signal → Order generation | < 5 μs | Pre-allocated order structs |
| Order generation → Broker | < 2 μs | Lock-free channel send |
| **Total tick-to-trade** | **< 20 μs** | End-to-end latency target |

### Memory Efficiency

- **Lock-free circular buffers**: Fixed-size pre-allocated (no dynamic allocation in hot path)
- **Zero-copy atomic snapshots**: Direct memory access via atomics
- **Object pooling**: Reuse order/signal structs where possible
- **Bounded channels**: Prevent unbounded memory growth

### Concurrency Model

- **Market data thread**: Single producer, publishes to lock-free channel
- **Strategy thread**: Single consumer, reads from channel and generates signals
- **Persistence thread**: Separate thread for async I/O, doesn't block strategy
- **No shared mutable state**: All cross-thread communication via lock-free channels or atomics

## Risk Management Integration

### Pre-Trade Validation

```rust
pub struct RiskValidator {
    limits: RiskLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum position size per symbol
    pub max_position_size: Decimal,

    /// Maximum order size
    pub max_order_size: Decimal,

    /// Maximum notional exposure
    pub max_notional_exposure: Decimal,

    /// Maximum number of orders per second
    pub max_orders_per_second: u32,

    /// Kill switch (emergency stop)
    pub kill_switch: Arc<AtomicBool>,
}

impl RiskValidator {
    /// Validate order before submission
    pub async fn validate_order(
        &self,
        order: &Order,
        current_position: &Position,
    ) -> Result<ValidationResult> {
        // Check kill switch
        if self.limits.kill_switch.load(Ordering::Acquire) {
            return Ok(ValidationResult::Rejected {
                reason: "Kill switch activated".to_string(),
            });
        }

        // Check position limits
        let new_position = self.calculate_new_position(current_position, order);
        if new_position.quantity.abs() > self.limits.max_position_size {
            return Ok(ValidationResult::Rejected {
                reason: "Position limit exceeded".to_string(),
            });
        }

        // Check order size
        if order.quantity > self.limits.max_order_size {
            return Ok(ValidationResult::Rejected {
                reason: "Order size limit exceeded".to_string(),
            });
        }

        // Check notional exposure
        let notional = order.quantity * order.price.unwrap_or_default();
        if notional > self.limits.max_notional_exposure {
            return Ok(ValidationResult::Rejected {
                reason: "Notional exposure limit exceeded".to_string(),
            });
        }

        Ok(ValidationResult::Approved)
    }
}

pub enum ValidationResult {
    Approved,
    Rejected { reason: String },
}
```

## Dependencies to Add

```toml
[dependencies]
# Existing dependencies remain...

# Lock-free data structures
crossbeam-channel = "0.5"
crossbeam-queue = "0.3"

# Concurrent hash maps
dashmap = "5.5"

# ONNX runtime
ort = { version = "2.0", features = ["load-dynamic"] }
ndarray = "0.15"

# Database
rusqlite = { version = "0.31", features = ["bundled", "modern_sqlite"] }
r2d2 = "0.8"
r2d2_sqlite = "0.24"

# Random number generation
rand = "0.8"

# Async traits
async-trait = "0.1"

# Additional serialization
bincode = "1.3"
```

## Testing Strategy

### Unit Tests
- Feature calculation accuracy
- Signal generation logic
- Fill simulation realism
- Position management correctness

### Integration Tests
- End-to-end market data → order flow
- Persistence round-trip
- Multi-threaded stress tests

### Performance Benchmarks
- Latency percentiles (p50, p99, p99.9)
- Throughput (messages/second)
- Memory usage under load
- Lock contention analysis (should be zero)

## Deployment Configuration

```toml
# Example config.toml
[strategy]
strategy_id = "imbalance_v1"
symbols = ["BTCUSDT", "ETHUSDT"]
paper_trading = true
enable_ml = false

[strategy.params]
imbalance_threshold = 0.3
min_confidence = 0.7
feature_window_ms = 1000
signal_decay_ms = 500
max_position_size = 1.0
order_size_fraction = 0.5

[strategy.risk_limits]
max_position_size = 1.0
max_order_size = 0.5
max_notional_exposure = 100000.0
max_orders_per_second = 10

[persistence]
database_path = "./data/strategy.db"
snapshot_interval_ms = 60000

[performance]
buffer_capacity = 10000
channel_capacity = 10000
```

## Migration Path

1. **Phase 1**: Implement core types and interfaces (this design)
2. **Phase 2**: Implement feature calculation with mock data
3. **Phase 3**: Implement signal generation and backtesting
4. **Phase 4**: Integrate with market-data WebSocket feed
5. **Phase 5**: Add paper broker and persistence
6. **Phase 6**: Performance optimization and testing
7. **Phase 7**: Add ONNX ML integration (optional)
8. **Phase 8**: Production deployment with OCaml risk gateway

## Conclusion

This architecture provides:

- **Ultra-low latency**: Lock-free design with < 20μs tick-to-trade
- **Unified binary**: Single Rust process for market data + strategy
- **Production-ready**: Comprehensive risk management and persistence
- **ML-ready**: ONNX integration for future model deployment
- **Scalable**: Zero-copy, lock-free design for maximum throughput

The design leverages Rust's strengths (zero-cost abstractions, fearless concurrency) while maintaining the ability to integrate with the existing OCaml risk gateway for production-critical checks.
