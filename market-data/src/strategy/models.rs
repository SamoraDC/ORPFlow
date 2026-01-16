//! Data models for the strategy engine
//!
//! All models use Decimal for financial precision and are optimized
//! for low-latency access patterns.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "buy"),
            Side::Sell => write!(f, "sell"),
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Market,
    Limit,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderStatus {
    Pending,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected,
}

/// Trading signal from strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub symbol: String,
    pub side: Side,
    pub confidence: f64,
    pub suggested_size: Decimal,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

impl Signal {
    pub fn new(
        symbol: impl Into<String>,
        side: Side,
        confidence: f64,
        suggested_size: Decimal,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            confidence,
            suggested_size,
            reason: reason.into(),
            timestamp: Utc::now(),
        }
    }
}

/// Order model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub status: OrderStatus,
    pub filled_quantity: Decimal,
    pub avg_fill_price: Option<Decimal>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Order {
    pub fn new_market(id: impl Into<String>, symbol: impl Into<String>, side: Side, quantity: Decimal) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            symbol: symbol.into(),
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            status: OrderStatus::Pending,
            filled_quantity: Decimal::ZERO,
            avg_fill_price: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn fill(&mut self, price: Decimal) {
        self.status = OrderStatus::Filled;
        self.filled_quantity = self.quantity;
        self.avg_fill_price = Some(price);
        self.updated_at = Utc::now();
    }
}

/// Trade execution model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub order_id: String,
    pub symbol: String,
    pub side: Side,
    pub price: Decimal,
    pub quantity: Decimal,
    pub fee: Decimal,
    pub fee_asset: String,
    pub timestamp: DateTime<Utc>,
    pub pnl: Option<Decimal>,
}

/// Position model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    /// Positive for long, negative for short
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub updated_at: DateTime<Utc>,
}

impl Position {
    pub fn new(symbol: impl Into<String>, quantity: Decimal, entry_price: Decimal) -> Self {
        Self {
            symbol: symbol.into(),
            quantity,
            entry_price,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            updated_at: Utc::now(),
        }
    }

    /// Update unrealized PnL based on current price
    #[inline(always)]
    pub fn update_pnl(&mut self, current_price: Decimal) {
        if self.quantity != Decimal::ZERO {
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity;
        } else {
            self.unrealized_pnl = Decimal::ZERO;
        }
        self.updated_at = Utc::now();
    }

    /// Check if position is long
    #[inline(always)]
    pub fn is_long(&self) -> bool {
        self.quantity > Decimal::ZERO
    }

    /// Check if position is short
    #[inline(always)]
    pub fn is_short(&self) -> bool {
        self.quantity < Decimal::ZERO
    }

    /// Check if position is flat (no position)
    #[inline(always)]
    pub fn is_flat(&self) -> bool {
        self.quantity == Decimal::ZERO
    }
}

/// Account state model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub balance: Decimal,
    pub equity: Decimal,
    pub initial_balance: Decimal,
    pub total_pnl: Decimal,
    pub win_rate: f64,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub created_at: DateTime<Utc>,
}

impl Account {
    pub fn new(initial_balance: Decimal) -> Self {
        Self {
            balance: initial_balance,
            equity: initial_balance,
            initial_balance,
            total_pnl: Decimal::ZERO,
            win_rate: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            created_at: Utc::now(),
        }
    }

    /// Update win rate
    pub fn update_win_rate(&mut self) {
        if self.total_trades > 0 {
            self.win_rate = self.winning_trades as f64 / self.total_trades as f64;
        }
    }

    /// Get return percentage
    pub fn return_pct(&self) -> f64 {
        if self.initial_balance > Decimal::ZERO {
            let pct = (self.equity - self.initial_balance) / self.initial_balance;
            pct.try_into().unwrap_or(0.0)
        } else {
            0.0
        }
    }
}

/// Health check status (reserved for future API use)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub component: String,
    pub timestamp: DateTime<Utc>,
}

#[allow(dead_code)]
impl HealthStatus {
    pub fn healthy(component: impl Into<String>) -> Self {
        Self {
            status: "healthy".to_string(),
            component: component.into(),
            timestamp: Utc::now(),
        }
    }
}
