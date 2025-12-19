# ADR-001: Rust Strategy Module Architecture

**Status**: Proposed
**Date**: 2025-12-19
**Author**: System Architecture Designer
**Context**: Replacing Python strategy service with Rust implementation

## Summary

Replace the current Python/FastAPI strategy service with a high-performance Rust module integrated directly into the market-data crate, creating a unified binary for ultra-low latency trading.

## Context and Problem Statement

The current architecture has three services:
- **market-data** (Rust): WebSocket connection to Binance, order book management
- **risk-gateway** (OCaml): Pre-trade risk checks
- **strategy** (Python/FastAPI): Trading strategy implementation

**Problems with current architecture:**
1. **High latency**: Inter-process communication (IPC) between market-data and strategy adds ~100μs+ overhead
2. **Python overhead**: GIL contention, interpreted execution, garbage collection pauses
3. **Serialization costs**: Multiple serialization/deserialization steps across process boundaries
4. **Memory copying**: Data copied multiple times between processes
5. **Complex deployment**: Three separate services to manage and monitor

**Target performance**: Nanosecond-level tick-to-trade latency (< 20μs end-to-end)

## Decision

**Implement the trading strategy as a Rust module within the market-data crate, creating a single unified binary.**

### Key Architectural Principles

1. **Lock-Free Concurrency**
   - Use atomic operations for shared state
   - Lock-free channels (crossbeam) for message passing
   - No mutexes in hot path

2. **Zero-Copy Design**
   - Atomic snapshots avoid cloning large data structures
   - Reference counting (Arc) for shared ownership
   - Direct memory access where safe

3. **Single Binary Deployment**
   - market-data + strategy in one process
   - Shared memory space eliminates IPC overhead
   - Simpler deployment and monitoring

4. **Composable Architecture**
   - Trait-based abstractions (FeatureExtractor, SignalGenerator)
   - Pluggable strategies via dependency injection
   - Easy to add new strategies or features

5. **Production-Ready**
   - Comprehensive risk management
   - SQLite persistence with WAL mode
   - Prometheus metrics integration
   - Graceful shutdown and error handling

## Alternatives Considered

### Alternative 1: Keep Python Strategy with Optimizations

**Approach**: Optimize the Python service (use Cython, better IPC, etc.)

**Pros**:
- Minimal code changes
- Leverage Python ML ecosystem
- Faster to market

**Cons**:
- Cannot achieve nanosecond latency targets
- Python GIL remains a fundamental limitation
- Still requires IPC overhead
- Complex to optimize and maintain

**Verdict**: Rejected - Cannot meet performance requirements

### Alternative 2: Separate Rust Strategy Service

**Approach**: Rewrite strategy in Rust but keep as separate service communicating via shared memory or IPC

**Pros**:
- Better separation of concerns
- Independent deployment
- Could use different languages for different strategies

**Cons**:
- Still has IPC overhead (even with shared memory)
- More complex deployment and monitoring
- Serialization/deserialization costs remain
- Harder to achieve < 20μs latency target

**Verdict**: Rejected - Adds unnecessary complexity without latency benefits

### Alternative 3: Unified Rust Binary (CHOSEN)

**Approach**: Implement strategy as module within market-data crate

**Pros**:
- Minimal latency (lock-free channels ~100ns)
- Zero IPC overhead
- Shared memory space
- Simpler deployment
- Easier debugging (single process)
- Can achieve < 20μs tick-to-trade

**Cons**:
- Tighter coupling between market data and strategy
- Restart required for strategy updates
- Harder to run multiple strategies simultaneously

**Mitigation**:
- Use trait abstractions for loose coupling
- Support hot-reloadable config changes
- Support multiple strategy instances in same process
- Use feature flags for compile-time strategy selection

**Verdict**: CHOSEN - Best aligns with performance requirements

## Design Details

### Module Structure

```
market-data/src/strategy/
├── mod.rs              # Public API
├── engine.rs           # Strategy execution engine
├── features/           # Feature calculation
├── signals/            # Signal generation
├── execution/          # Order execution & paper trading
├── risk/               # Risk management
├── persistence/        # SQLite persistence
└── types/              # Common types
```

### Data Flow

```
WebSocket → OrderBook → MarketSnapshot (atomic) → FeatureCalculator
                                                        ↓
                                                  FeatureVector
                                                        ↓
                                                 SignalGenerator
                                                        ↓
                                                  RiskValidator
                                                        ↓
                                                  OrderManager
                                                        ↓
                                                   PaperBroker
```

### Key Performance Characteristics

| Component | Latency | Method |
|-----------|---------|--------|
| Market data publish | < 1 μs | Lock-free channel send |
| Feature calculation | < 10 μs | Simple imbalance features |
| Signal generation | < 5 μs | Threshold-based logic |
| Risk validation | < 2 μs | Atomic reads |
| Order submission | < 2 μs | Lock-free channel send |
| **TOTAL** | **< 20 μs** | End-to-end tick-to-trade |

### Concurrency Model

- **Market data thread**: Single producer, owns WebSocket connection
- **Strategy thread**: Single consumer, processes market updates
- **Persistence thread**: Async writer, non-blocking I/O
- **Communication**: Lock-free bounded channels (crossbeam)
- **Shared state**: Atomic types (AtomicU64, AtomicBool, etc.)

### Risk Management Integration

Strategy module includes built-in risk checks:
- Position limits
- Order size limits
- Notional exposure limits
- Rate limiting
- Kill switch (atomic bool)

**Optional OCaml integration**: For production, can still call OCaml risk-gateway via IPC for additional checks, but not in critical path.

### ML Integration Strategy

**Phase 1**: Rule-based imbalance strategy (simple thresholds)
**Phase 2**: ONNX model integration for inference
**Phase 3**: Feature extraction pipeline feeds ML models
**Phase 4**: Online learning (periodic model updates)

ONNX runtime chosen for:
- Fast inference (< 1ms for typical models)
- Cross-platform compatibility
- No Python dependency
- Support for quantized models

## Consequences

### Positive

1. **Ultra-low latency**: Achieve < 20μs tick-to-trade target
2. **Simplified deployment**: Single binary, fewer moving parts
3. **Better resource utilization**: Shared memory, no IPC overhead
4. **Type safety**: Rust's type system catches bugs at compile time
5. **Fearless concurrency**: Rust prevents data races
6. **Production-ready**: Built-in persistence, metrics, risk management

### Negative

1. **Tighter coupling**: Strategy and market data in same binary
2. **Longer compile times**: More code in single crate
3. **Harder to run multiple strategies**: Need separate processes
4. **Initial development effort**: Rewrite from Python to Rust

### Mitigation Strategies

1. **Coupling**: Use trait abstractions and dependency injection
2. **Compile times**: Use incremental compilation, workspaces if needed
3. **Multiple strategies**: Support multiple strategy instances via config
4. **Development effort**: Start with simple imbalance strategy, iterate

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Define core types (Order, Signal, FeatureVector)
- [ ] Implement lock-free MarketSnapshot
- [ ] Create lock-free channels between threads
- [ ] Basic StrategyEngine skeleton

### Phase 2: Feature Calculation (Week 2-3)
- [ ] Implement FeatureBuffer (lock-free circular buffer)
- [ ] MicrostructureExtractor (volatility, momentum, imbalance)
- [ ] OrderBookFeatureExtractor (depth, spread, pressure)
- [ ] Unit tests for feature accuracy

### Phase 3: Signal Generation (Week 3-4)
- [ ] ImbalanceSignalGenerator
- [ ] Confidence scoring
- [ ] Signal thresholding
- [ ] Backtesting framework

### Phase 4: Execution & Risk (Week 4-5)
- [ ] PaperBroker implementation
- [ ] FillSimulator with realistic fills
- [ ] RiskValidator with position limits
- [ ] PositionManager and OrderManager

### Phase 5: Persistence (Week 5-6)
- [ ] SQLite integration with WAL mode
- [ ] Trade logging
- [ ] Order history
- [ ] Strategy state snapshots

### Phase 6: Integration & Testing (Week 6-8)
- [ ] Integrate with market-data WebSocket feed
- [ ] End-to-end latency testing
- [ ] Stress testing (high message rates)
- [ ] Performance benchmarks

### Phase 7: ML Integration (Week 8-10)
- [ ] ONNX runtime integration
- [ ] Feature normalization
- [ ] Model loading and inference
- [ ] Inference latency benchmarks

### Phase 8: Production Hardening (Week 10-12)
- [ ] Comprehensive error handling
- [ ] Metrics and observability
- [ ] Documentation
- [ ] Load testing and optimization

## Metrics and Success Criteria

### Performance Metrics
- [ ] Tick-to-trade latency p50 < 10 μs
- [ ] Tick-to-trade latency p99 < 20 μs
- [ ] Throughput > 100,000 messages/second
- [ ] Zero lock contention in hot path

### Correctness Metrics
- [ ] 100% unit test coverage for critical paths
- [ ] Integration tests pass
- [ ] Backtesting results match Python implementation ± 1%
- [ ] No data races (verified by thread sanitizer)

### Operational Metrics
- [ ] Graceful shutdown in < 1 second
- [ ] Zero data loss during shutdown
- [ ] Metrics exported to Prometheus
- [ ] Logs integrated with existing system

## References

- [Lock-Free Programming](https://preshing.com/20120612/an-introduction-to-lock-free-programming/)
- [Crossbeam Channels](https://docs.rs/crossbeam-channel/)
- [ONNX Runtime Rust](https://docs.rs/ort/)
- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
- [Atomic Operations in Rust](https://doc.rust-lang.org/std/sync/atomic/)

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-12-19 | 1.0 | System Architect | Initial version |
