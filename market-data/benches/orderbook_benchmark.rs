//! Benchmarks for order book operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use orp_flow_market_data::orderbook::OrderBook;
use orp_flow_market_data::parser::{DepthUpdate, OrderBookSnapshot, PriceLevel};
use rust_decimal::Decimal;
use std::str::FromStr;

fn create_snapshot(levels: usize) -> OrderBookSnapshot {
    let bids: Vec<PriceLevel> = (0..levels)
        .map(|i| PriceLevel {
            price: Decimal::from(50000 - i),
            quantity: Decimal::from_str("1.5").unwrap(),
        })
        .collect();

    let asks: Vec<PriceLevel> = (0..levels)
        .map(|i| PriceLevel {
            price: Decimal::from(50001 + i),
            quantity: Decimal::from_str("1.5").unwrap(),
        })
        .collect();

    OrderBookSnapshot {
        last_update_id: 1000,
        bids,
        asks,
    }
}

fn create_update(base_id: u64) -> DepthUpdate {
    DepthUpdate {
        event_type: "depthUpdate".to_string(),
        event_time: 1672531200000,
        symbol: "BTCUSDT".to_string(),
        first_update_id: base_id,
        final_update_id: base_id + 1,
        bids: vec![PriceLevel {
            price: Decimal::from(49999),
            quantity: Decimal::from_str("2.0").unwrap(),
        }],
        asks: vec![PriceLevel {
            price: Decimal::from(50001),
            quantity: Decimal::from_str("2.5").unwrap(),
        }],
    }
}

fn benchmark_init_snapshot(c: &mut Criterion) {
    let snapshot = create_snapshot(100);

    c.bench_function("init_snapshot_100_levels", |b| {
        b.iter(|| {
            let mut book = OrderBook::new("BTCUSDT", 100);
            book.init_snapshot(black_box(&snapshot));
        })
    });
}

fn benchmark_apply_update(c: &mut Criterion) {
    let snapshot = create_snapshot(100);
    let mut book = OrderBook::new("BTCUSDT", 100);
    book.init_snapshot(&snapshot);

    let update = create_update(1001);

    c.bench_function("apply_update", |b| {
        b.iter(|| {
            book.apply_update(black_box(&update));
        })
    });
}

fn benchmark_metrics_calculation(c: &mut Criterion) {
    let snapshot = create_snapshot(100);
    let mut book = OrderBook::new("BTCUSDT", 100);
    book.init_snapshot(&snapshot);

    c.bench_function("calculate_imbalance", |b| {
        b.iter(|| {
            black_box(book.imbalance(10));
        })
    });

    c.bench_function("calculate_weighted_imbalance", |b| {
        b.iter(|| {
            black_box(book.weighted_imbalance(10, Decimal::from_str("0.9").unwrap()));
        })
    });

    c.bench_function("get_state", |b| {
        b.iter(|| {
            black_box(book.state());
        })
    });
}

criterion_group!(
    benches,
    benchmark_init_snapshot,
    benchmark_apply_update,
    benchmark_metrics_calculation
);
criterion_main!(benches);
