# ORPFlow Rust Strategy Module - Architecture Documentation

## Overview

This directory contains the complete architecture design for migrating the ORPFlow trading strategy from Python to a unified Rust implementation integrated directly into the `market-data` crate.

**Goal**: Achieve sub-20 microsecond tick-to-trade latency by eliminating inter-process communication and leveraging Rust's zero-cost abstractions and lock-free concurrency.

## Document Index

### 1. [Architecture Decision Record (ADR-001)](./ADR-001-rust-strategy-architecture.md)
**Primary document for decision rationale and alternatives considered.**

- **What**: Replace Python strategy service with Rust module
- **Why**: Achieve < 20μs tick-to-trade latency (vs 500μs - 5ms with Python)
- **How**: Lock-free concurrent architecture with atomic operations
- **Trade-offs**: Analyzed 3 alternatives, chose unified binary approach
- **Implementation Plan**: 8-phase rollout over 12 weeks

**Start here** to understand the "why" behind the architecture.

### 2. [Detailed Design Document](./rust-strategy-module-design.md)
**Comprehensive technical specification with code examples.**

Covers:
- System architecture overview
- Module structure and organization
- Complete type system with struct definitions
- Lock-free data structures (MarketSnapshot, FeatureBuffer)
- Feature calculation pipeline
- Signal generation (imbalance + ML/ONNX)
- Paper broker with realistic fill simulation
- Risk management and validation
- SQLite persistence layer (WAL mode)
- Integration points with market-data
- Performance characteristics and benchmarks
- Dependencies and testing strategy

**Read this** for implementation details and code blueprints.

### 3. [Interface Definitions](./rust-strategy-interfaces.rs)
**Compilable Rust code with trait definitions and struct declarations.**

Key interfaces:
- `MarketSnapshot` - Lock-free atomic market data
- `FeatureExtractor` trait - Composable feature calculation
- `SignalGenerator` trait - Strategy signal generation
- `PaperBroker` - Simulated execution engine
- `RiskValidator` - Pre-trade risk checks
- `PersistenceManager` - Async SQLite persistence

**Use this** as a reference for API contracts when implementing.

### 4. [Data Flow Diagram](./rust-strategy-data-flow.md)
**Visual architecture with ASCII diagrams and latency analysis.**

Includes:
- Complete system diagram (market-data → strategy → execution → persistence)
- Thread model (4 threads with lock-free channels)
- Latency breakdown (target < 20μs achieved)
- Memory layout (cache-aligned atomic structures)
- Synchronization primitives comparison
- Performance monitoring metrics
- Configuration examples
- Python vs Rust comparison table

**Reference this** to understand system flow and performance targets.

## Quick Start

### For Architects
1. Read [ADR-001](./ADR-001-rust-strategy-architecture.md) for decision rationale
2. Review [Data Flow Diagram](./rust-strategy-data-flow.md) for system overview
3. Check [Design Document](./rust-strategy-module-design.md) for architectural patterns

### For Developers
1. Study [Interface Definitions](./rust-strategy-interfaces.rs) for API contracts
2. Read [Design Document](./rust-strategy-module-design.md) implementation sections
3. Reference [Data Flow Diagram](./rust-strategy-data-flow.md) for latency budgets

### For Reviewers
1. Start with [ADR-001](./ADR-001-rust-strategy-architecture.md) for context
2. Evaluate alternatives considered and trade-offs
3. Review success criteria and metrics

## Architecture Summary

### Current State (3 Services)
```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  market-    │ IPC   │  strategy   │ IPC   │  risk-      │
│  data       ├──────▶│  (Python)   ├──────▶│  gateway    │
│  (Rust)     │~100μs │             │~100μs │  (OCaml)    │
└─────────────┘       └─────────────┘       └─────────────┘
                   Tick-to-trade: 500μs - 5ms
```

### Proposed State (Unified Binary)
```
┌─────────────────────────────────────────────┐
│  market-data + strategy (Rust)              │
│  ┌──────────┐  lock-free  ┌──────────────┐ │
│  │ OrderBook├────────────▶│ Strategy     │ │
│  │          │  ~100ns     │ Engine       │ │
│  └──────────┘             └──────────────┘ │
│                                             │
│              Optional: Call OCaml risk      │
│              gateway for extra validation   │
└─────────────────────────────────────────────┘
                   Tick-to-trade: < 20μs
```

## Key Design Principles

### 1. Lock-Free Concurrency
- **Atomic operations** for shared state (MarketSnapshot)
- **Crossbeam channels** for inter-thread communication (~100ns latency)
- **ArrayQueue** for lock-free circular buffers
- **Zero mutexes** in hot path

### 2. Zero-Copy Architecture
- **Atomic snapshots** avoid cloning large structures
- **Arc (reference counting)** for shared ownership
- **Direct memory access** via atomics (Ordering::Acquire/Release)
- **In-place updates** where safe

### 3. Composable Design
- **Trait-based abstractions** (FeatureExtractor, SignalGenerator)
- **Dependency injection** for easy testing
- **Pluggable strategies** via configuration
- **Extensible feature pipeline**

### 4. Production-Ready
- **Comprehensive risk management** (position limits, kill switch)
- **SQLite persistence** with WAL mode (crash-safe)
- **Prometheus metrics** for observability
- **Graceful shutdown** with state preservation

## Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| **Tick-to-trade (p50)** | < 10 μs | Lock-free design |
| **Tick-to-trade (p99)** | < 20 μs | No GC, no locks |
| **Throughput** | > 100k msg/s | Bounded channels |
| **Memory** | < 20 MB | Pre-allocated buffers |
| **Lock contention** | 0 | Atomic operations only |

## Technology Stack

### Core Dependencies
- `tokio` - Async runtime (multi-threaded)
- `crossbeam-channel` - Lock-free channels
- `crossbeam-queue` - Lock-free queues (ArrayQueue)
- `dashmap` - Concurrent hash map (for feature buffers)

### Optional Dependencies
- `ort` - ONNX runtime for ML inference
- `rusqlite` - SQLite database (WAL mode)
- `r2d2` - Connection pool for SQLite

### Existing Dependencies (Reused)
- `rust_decimal` - Precise decimal arithmetic
- `serde` - Serialization (config only, not in hot path)
- `prometheus` - Metrics export
- `tracing` - Structured logging

## Implementation Phases

| Phase | Duration | Focus | Deliverable |
|-------|----------|-------|-------------|
| **1. Core Infra** | Week 1-2 | Types, channels, engine skeleton | Compiles, basic tests pass |
| **2. Features** | Week 2-3 | Feature calculation pipeline | Accurate feature extraction |
| **3. Signals** | Week 3-4 | Signal generation, thresholds | Backtesting framework works |
| **4. Execution** | Week 4-5 | Paper broker, fills, risk | Realistic trade simulation |
| **5. Persistence** | Week 5-6 | SQLite integration | Trade history persisted |
| **6. Integration** | Week 6-8 | End-to-end testing | Latency targets met |
| **7. ML (Optional)** | Week 8-10 | ONNX inference | ML models deployable |
| **8. Production** | Week 10-12 | Hardening, docs, load tests | Production-ready |

## Success Criteria

### Performance
- [ ] p50 tick-to-trade latency < 10 μs
- [ ] p99 tick-to-trade latency < 20 μs
- [ ] Throughput > 100,000 messages/second
- [ ] Zero lock contention in hot path

### Correctness
- [ ] 100% unit test coverage for critical paths
- [ ] Integration tests pass
- [ ] Backtesting results match Python ± 1%
- [ ] Thread sanitizer reports no data races

### Operational
- [ ] Graceful shutdown < 1 second
- [ ] Zero data loss during shutdown
- [ ] Metrics exported to Prometheus
- [ ] Logs integrated with existing system

## File Organization

After implementation, the module structure will be:

```
market-data/src/
├── strategy/                   # New strategy module
│   ├── mod.rs                 # Public API
│   ├── engine.rs              # Strategy engine
│   ├── features/              # Feature calculation
│   ├── signals/               # Signal generation
│   ├── execution/             # Paper broker
│   ├── risk/                  # Risk management
│   ├── persistence/           # SQLite persistence
│   └── types/                 # Common types
├── orderbook/                 # Existing (unchanged)
├── parser/                    # Existing (unchanged)
├── websocket/                 # Existing (unchanged)
└── lib.rs                     # Update exports
```

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Lock-free bugs | High | Extensive testing, thread sanitizer, formal verification |
| Performance regression | High | Continuous benchmarking, latency SLOs, rollback plan |
| Memory leaks | Medium | Valgrind, address sanitizer, periodic profiling |
| SQLite contention | Low | WAL mode, async writes, batching |

### Project Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scope creep | Medium | Clear phase boundaries, MVP first |
| Skill gaps | Low | Extensive documentation, code review |
| Integration issues | Medium | Incremental integration, feature flags |
| Timeline slip | Low | Weekly checkpoints, buffer in schedule |

## Next Steps

1. **Review & Approval** (Week 0)
   - [ ] Architecture review with team
   - [ ] Approve ADR-001
   - [ ] Finalize implementation plan

2. **Environment Setup** (Week 1)
   - [ ] Create feature branch
   - [ ] Set up benchmarking infrastructure
   - [ ] Configure CI/CD for Rust builds

3. **Begin Implementation** (Week 1)
   - [ ] Implement core types (Order, Signal, FeatureVector)
   - [ ] Set up lock-free channels
   - [ ] Create basic StrategyEngine skeleton

## References

### Rust Concurrency
- [Lock-Free Programming](https://preshing.com/20120612/an-introduction-to-lock-free-programming/)
- [Crossbeam Documentation](https://docs.rs/crossbeam/)
- [Rust Atomics and Locks](https://marabos.nl/atomics/)

### Performance
- [Systems Performance](http://www.brendangregg.com/systems-performance-2nd-edition-book.html)
- [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832)

### HFT Architecture
- [Jane Street Tech Talks](https://blog.janestreet.com/category/ocaml/)
- [Low-Latency Trading Systems](https://queue.acm.org/detail.cfm?id=2536492)

### Rust in Finance
- [Rust in Finance](https://rustinfinance.github.io/)
- [Financial Services Rust Adoption](https://www.reddit.com/r/rust/comments/10x2kkr/rust_in_finance/)

## Questions & Contact

For questions about this architecture:
1. Read the relevant documentation above
2. Check the [Design Document](./rust-strategy-module-design.md) for detailed explanations
3. Review the [ADR](./ADR-001-rust-strategy-architecture.md) for decision rationale
4. Open an issue in the repository for clarifications

---

**Document Status**: Proposed Architecture (2025-12-19)
**Author**: System Architecture Designer
**Last Updated**: 2025-12-19
