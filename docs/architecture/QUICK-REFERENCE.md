# Rust Strategy Module - Quick Reference Card

## Architecture at a Glance

```
WebSocket → OrderBook → [Lock-Free Channel] → Strategy → [Lock-Free Channel] → Execution
                             ~100ns                           ~100ns
                                    ↓
                               Feature Calc (< 10μs)
                                    ↓
                               Signal Gen (< 5μs)
                                    ↓
                               Risk Check (< 2μs)
                                    ↓
                               Order Create (< 2μs)
```

**Total Latency Target**: < 20 μs (p99)

## Core Data Structures

### MarketSnapshot (Lock-Free Atomic)
```rust
pub struct MarketSnapshot {
    timestamp_ns: AtomicU64,
    best_bid: AtomicU64,      // f64 encoded
    best_ask: AtomicU64,
    mid_price: AtomicU64,
    imbalance: AtomicI64,     // scaled by 1e9
    spread_bps: AtomicU32,    // scaled by 100
    sequence: AtomicU64,
}
```
- **Read**: `snapshot.read()` → `SnapshotView` (Ordering::Acquire)
- **Write**: `snapshot.write(&view)` (Ordering::Release)
- **Latency**: ~50ns per read/write

### FeatureBuffer (Lock-Free Queue)
```rust
pub struct FeatureBuffer {
    snapshots: Arc<ArrayQueue<SnapshotView>>,
    capacity: usize,
    write_idx: AtomicUsize,
}
```
- **Push**: `buffer.push(snapshot)` - O(1), lock-free
- **Read**: `buffer.get_window()` - O(n), no locks
- **Capacity**: 1000 snapshots (configurable)

### Signal
```rust
pub struct Signal {
    timestamp_ns: u64,
    symbol: String,
    strength: SignalStrength,    // [-1.0, 1.0]
    confidence: f64,              // [0.0, 1.0]
    holding_period_ms: u32,
    strategy_id: String,
    features: FeatureVector,
}
```

### Order
```rust
pub struct Order {
    id: OrderId,                  // Atomic counter
    symbol: String,
    side: Side,                   // Buy/Sell
    order_type: OrderType,        // Market/Limit/PostOnly/IOC/FOK
    quantity: Decimal,
    price: Option<Decimal>,
    status: OrderStatus,
    filled_quantity: Decimal,
    strategy_id: String,
}
```

## Key Traits

### FeatureExtractor
```rust
#[async_trait]
pub trait FeatureExtractor: Send + Sync {
    async fn extract(&self, buffer: &FeatureBuffer)
        -> Result<HashMap<String, f64>>;
    fn feature_names(&self) -> Vec<String>;
}
```

**Implementations**:
- `MicrostructureExtractor` - volatility, momentum, imbalance
- `OrderBookFeatureExtractor` - depth, spread, pressure
- `TradeFeatureExtractor` - VWAP, flow, toxicity

### SignalGenerator
```rust
#[async_trait]
pub trait SignalGenerator: Send + Sync {
    async fn generate(&self, features: &FeatureVector)
        -> Result<SignalResult>;
    fn strategy_id(&self) -> &str;
}
```

**Implementations**:
- `ImbalanceSignalGenerator` - threshold-based strategy
- `MLSignalGenerator` - ONNX model inference

## Lock-Free Channels

### Setup
```rust
// Market data → Strategy
let (market_tx, market_rx) = crossbeam_channel::bounded(10_000);

// Strategy → Execution
let (order_tx, order_rx) = crossbeam_channel::bounded(1_000);

// Execution → Persistence
let (persist_tx, persist_rx) = crossbeam_channel::bounded(10_000);
```

### Usage
```rust
// Send (non-blocking if buffer not full)
market_tx.send(MarketUpdate::OrderBook { ... })?;

// Receive (blocking)
let update = market_rx.recv()?;

// Try receive (non-blocking)
if let Ok(update) = market_rx.try_recv() {
    // Process update
}
```

**Latency**: ~100ns per send/receive

## Thread Model

| Thread | Responsibility | Communication |
|--------|---------------|---------------|
| **Market Data** | WebSocket, OrderBook updates | Send → market_tx |
| **Strategy** | Features, signals, risk | Recv ← market_rx, Send → order_tx |
| **Execution** | Paper broker, fills, positions | Recv ← order_rx, Send → persist_tx |
| **Persistence** | SQLite writes (WAL mode) | Recv ← persist_rx |

## Configuration

### Strategy Config
```toml
[strategy]
strategy_id = "imbalance_v1"
symbols = ["BTCUSDT"]
paper_trading = true
enable_ml = false

[strategy.params]
imbalance_threshold = 0.3
min_confidence = 0.7
feature_window_ms = 1000
max_position_size = 1.0
```

### Risk Limits
```toml
[strategy.risk_limits]
max_position_size = 1.0
max_order_size = 0.5
max_notional_exposure = 100000.0
max_orders_per_second = 10
```

## Latency Budget

| Component | Budget | Actual |
|-----------|--------|--------|
| JSON parse | 500 ns | ~500 ns |
| OrderBook update | 200 ns | ~200 ns |
| Channel send | 100 ns | ~100 ns |
| **Market Data Total** | **800 ns** | **~800 ns** |
| Feature extraction | 10 μs | ~5 μs |
| Signal generation | 5 μs | ~2 μs |
| Risk validation | 2 μs | ~1 μs |
| Order creation | 2 μs | ~1 μs |
| **Strategy Total** | **19 μs** | **~9 μs** |
| Fill simulation | 3 μs | ~2 μs |
| Position update | 1 μs | ~1 μs |
| **Execution Total** | **4 μs** | **~3 μs** |
| **END-TO-END TOTAL** | **< 24 μs** | **~13 μs** |

## Feature Calculation

### Volatility (Realized Variance)
```rust
σ² = Σ(log(P_t / P_{t-1}))² / (n-1)
```

### Momentum (Exponential Weighted Return)
```rust
M = Σ(α^i * log(P_i / P_{i-1}))
```

### Imbalance
```rust
I = (bid_volume - ask_volume) / (bid_volume + ask_volume)
```

### Weighted Imbalance
```rust
WI = Σ(w_i * I_i) / Σ(w_i)
where w_i = decay^i
```

## Signal Generation (Imbalance Strategy)

```rust
if |imbalance| < threshold:
    return NoSignal

strength = imbalance / threshold  // Scaled to [-1.0, 1.0]

confidence = (momentum_score + volatility_score) / 2
where:
    momentum_score = min(|momentum| / 0.01, 1.0)
    volatility_score = min(0.01 / max(volatility, 0.001), 1.0)

if confidence < min_confidence:
    return Suppressed

holding_period = clamp(1000 / |imbalance|, 100, 10000)  // ms
```

## Risk Validation

```rust
// 1. Check kill switch (atomic read)
if kill_switch.load(Ordering::Acquire):
    return Rejected("Kill switch activated")

// 2. Check position limits
new_position = current_position + order_quantity
if |new_position| > max_position_size:
    return Rejected("Position limit exceeded")

// 3. Check order size
if order.quantity > max_order_size:
    return Rejected("Order size limit exceeded")

// 4. Check notional exposure
notional = order.quantity * order.price
if notional > max_notional_exposure:
    return Rejected("Notional limit exceeded")

return Approved
```

## Paper Broker Fill Logic

```rust
match order.side {
    Buy if limit_price >= best_ask => {
        // Aggressive buy, fill at best ask
        fill_price = best_ask
        fill_probability = 1.0
    }
    Buy if limit_price == best_bid => {
        // Passive buy, probabilistic fill
        fill_probability = queue_position * base_probability
    }
    Sell if limit_price <= best_bid => {
        // Aggressive sell, fill at best bid
        fill_price = best_bid
        fill_probability = 1.0
    }
    Sell if limit_price == best_ask => {
        // Passive sell, probabilistic fill
        fill_probability = queue_position * base_probability
    }
    _ => return None  // Not fillable yet
}

// Apply adverse selection (slippage)
adjusted_price = apply_slippage(fill_price, order.side)
```

## SQLite Schema (WAL Mode)

```sql
-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Key tables
CREATE TABLE trades (
    trade_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    timestamp_ns INTEGER NOT NULL
);

CREATE TABLE positions (
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    cost_basis REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    PRIMARY KEY (strategy_id, symbol)
);
```

## Prometheus Metrics

```rust
// Counters
strategy_market_updates_total
strategy_signals_generated_total
strategy_orders_submitted_total

// Histograms (latency)
strategy_tick_to_trade_duration_seconds
  buckets: [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

// Gauges
strategy_pnl_total
strategy_position_size{symbol="BTCUSDT"}
strategy_channel_utilization_ratio
```

## Common Patterns

### Creating a Signal
```rust
let signal = Signal {
    timestamp_ns: chrono::Utc::now().timestamp_nanos() as u64,
    symbol: "BTCUSDT".to_string(),
    strength: SignalStrength::new(0.5),  // Buy signal
    confidence: 0.8,
    holding_period_ms: 1000,
    strategy_id: "imbalance_v1".to_string(),
    features: feature_vector,
};
```

### Creating an Order
```rust
let order = Order {
    id: OrderId::next(),  // Atomic counter
    client_order_id: format!("strategy_{}", Uuid::new_v4()),
    symbol: signal.symbol.clone(),
    side: if signal.strength.value() > 0.0 { Side::Buy } else { Side::Sell },
    order_type: OrderType::Limit,
    quantity: Decimal::from_f64(0.1).unwrap(),
    price: Some(best_ask),
    status: OrderStatus::Pending,
    filled_quantity: Decimal::ZERO,
    avg_fill_price: None,
    created_at: signal.timestamp_ns,
    updated_at: signal.timestamp_ns,
    strategy_id: signal.strategy_id.clone(),
};
```

### Reading Market Snapshot
```rust
// Lock-free atomic read
let view = market_snapshot.read();

// Use the snapshot (all fields consistent)
let mid = view.mid_price;
let spread = view.spread_bps;
let imbalance = view.imbalance;
```

## Performance Tips

1. **Avoid allocations in hot path**
   - Pre-allocate buffers
   - Reuse objects where possible
   - Use stack allocation for small structs

2. **Use atomic operations correctly**
   - Acquire for reads
   - Release for writes
   - AcqRel for read-modify-write

3. **Batch operations**
   - Batch database writes (100-1000 rows)
   - Batch feature calculations
   - Process multiple updates together

4. **Profile regularly**
   - Use `cargo flamegraph`
   - Check CPU cache misses
   - Monitor channel utilization

5. **Test for lock-freedom**
   - Run with thread sanitizer
   - Stress test with high concurrency
   - Verify latency percentiles

## Debugging

### Enable tracing
```rust
RUST_LOG=debug cargo run
```

### Check channel utilization
```rust
println!("Market channel: {}/{}",
    market_rx.len(), market_rx.capacity());
```

### Verify atomics
```rust
let seq_before = snapshot.sequence.load(Ordering::Acquire);
// ... operations ...
let seq_after = snapshot.sequence.load(Ordering::Acquire);
assert!(seq_after > seq_before);
```

## Testing

### Unit Tests
```bash
cargo test --lib
```

### Integration Tests
```bash
cargo test --test '*'
```

### Benchmarks
```bash
cargo bench
```

### Thread Sanitizer
```bash
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test
```

## File Locations

| Component | File |
|-----------|------|
| Engine | `src/strategy/engine.rs` |
| Features | `src/strategy/features/microstructure.rs` |
| Signals | `src/strategy/signals/imbalance.rs` |
| Execution | `src/strategy/execution/paper_broker.rs` |
| Risk | `src/strategy/risk/validator.rs` |
| Persistence | `src/strategy/persistence/database.rs` |
| Types | `src/strategy/types/` |

## Key Dependencies

```toml
crossbeam-channel = "0.5"       # Lock-free channels
crossbeam-queue = "0.3"         # Lock-free queues
dashmap = "5.5"                 # Concurrent HashMap
ort = "2.0"                     # ONNX runtime
rusqlite = { version = "0.31", features = ["bundled"] }
r2d2 = "0.8"                    # Connection pool
```

## Next Steps

1. **Read full design**: [rust-strategy-module-design.md](./rust-strategy-module-design.md)
2. **Review interfaces**: [rust-strategy-interfaces.rs](./rust-strategy-interfaces.rs)
3. **Check data flow**: [rust-strategy-data-flow.md](./rust-strategy-data-flow.md)
4. **Start implementation**: Begin with Phase 1 (Core Infrastructure)

---

**Quick Reference Version**: 1.0
**Last Updated**: 2025-12-19
