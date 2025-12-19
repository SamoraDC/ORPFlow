//! Paper broker for simulated order execution
//!
//! High-performance order execution simulation with realistic
//! slippage, fees, and position tracking.

use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

use super::models::{Account, Order, Position, Side, Trade};
use super::storage::TradeStorage;

/// Atomic counter for order IDs
static ORDER_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Paper broker for simulated trading
pub struct PaperBroker {
    account: Account,
    positions: HashMap<String, Position>,
    storage: Arc<TradeStorage>,

    // Fee structure (Binance VIP 0)
    maker_fee: Decimal,
    taker_fee: Decimal,

    // Daily trade counter
    trades_today: u32,
    last_trade_date: String,
}

impl PaperBroker {
    /// Create a new paper broker
    pub async fn new(initial_balance: Decimal, storage: Arc<TradeStorage>) -> anyhow::Result<Self> {
        // Load existing positions from storage
        let positions = storage.get_all_positions().await?;
        let position_map: HashMap<String, Position> = positions
            .into_iter()
            .map(|p| (p.symbol.clone(), p))
            .collect();

        let today = Utc::now().format("%Y-%m-%d").to_string();

        Ok(Self {
            account: Account::new(initial_balance),
            positions: position_map,
            storage,
            maker_fee: dec!(0.001),
            taker_fee: dec!(0.001),
            trades_today: 0,
            last_trade_date: today,
        })
    }

    /// Generate a unique order ID
    #[inline(always)]
    fn generate_order_id() -> String {
        let count = ORDER_COUNTER.fetch_add(1, Ordering::SeqCst);
        let timestamp = Utc::now().format("%Y%m%d%H%M%S");
        format!("ORD-{}-{}", timestamp, count)
    }

    /// Generate a unique trade ID
    #[inline(always)]
    fn generate_trade_id() -> String {
        let uuid_str = Uuid::new_v4().to_string();
        format!("TRD-{}", &uuid_str[..12])
    }

    /// Calculate trading fee
    #[inline(always)]
    fn calculate_fee(&self, price: Decimal, quantity: Decimal, is_maker: bool) -> Decimal {
        let fee_rate = if is_maker { self.maker_fee } else { self.taker_fee };
        price * quantity * fee_rate
    }

    /// Update account equity based on positions
    fn update_equity(&mut self) {
        let unrealized_pnl: Decimal = self.positions.values().map(|p| p.unrealized_pnl).sum();
        self.account.equity = self.account.balance + unrealized_pnl;
    }

    /// Reset daily trade counter if needed
    fn check_daily_reset(&mut self) {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        if today != self.last_trade_date {
            self.trades_today = 0;
            self.last_trade_date = today;
        }
    }

    /// Execute a market order
    pub async fn execute_market_order(
        &mut self,
        symbol: &str,
        side: Side,
        quantity: Decimal,
        current_price: Decimal,
    ) -> Option<Trade> {
        self.check_daily_reset();

        // Apply slippage (5 bps)
        let slippage = current_price * dec!(0.0005);
        let fill_price = match side {
            Side::Buy => current_price + slippage,
            Side::Sell => current_price - slippage,
        };

        // Calculate fee
        let fee = self.calculate_fee(fill_price, quantity, false);

        // Calculate P&L if closing position
        let pnl = self.calculate_pnl(symbol, side, quantity, fill_price);

        // Create order
        let order_id = Self::generate_order_id();
        let mut order = Order::new_market(&order_id, symbol, side, quantity);
        order.fill(fill_price);

        // Create trade
        let trade = Trade {
            id: Self::generate_trade_id(),
            order_id: order_id.clone(),
            symbol: symbol.to_string(),
            side,
            price: fill_price,
            quantity,
            fee,
            fee_asset: "USDT".to_string(),
            timestamp: Utc::now(),
            pnl,
        };

        // Update position
        self.update_position(symbol, side, quantity, fill_price, pnl);

        // Update account balance
        let cost = fill_price * quantity + fee;
        match side {
            Side::Buy => self.account.balance -= cost,
            Side::Sell => self.account.balance += cost - fee,
        }

        if let Some(pnl_val) = pnl {
            self.account.total_pnl += pnl_val;
            self.account.total_trades += 1;
            self.trades_today += 1;

            if pnl_val > Decimal::ZERO {
                self.account.winning_trades += 1;
            } else {
                self.account.losing_trades += 1;
            }
            self.account.update_win_rate();
        }

        self.update_equity();

        // Persist trade
        if let Err(e) = self.storage.save_trade(&trade).await {
            warn!(error = %e, "Failed to persist trade");
        }

        // Persist order
        if let Err(e) = self.storage.save_order(&order).await {
            warn!(error = %e, "Failed to persist order");
        }

        info!(
            trade_id = %trade.id,
            symbol = %symbol,
            side = ?side,
            price = %fill_price,
            quantity = %quantity,
            fee = %fee,
            pnl = ?pnl,
            "Trade executed"
        );

        Some(trade)
    }

    /// Calculate P&L for a closing trade
    #[inline(always)]
    fn calculate_pnl(
        &self,
        symbol: &str,
        side: Side,
        quantity: Decimal,
        fill_price: Decimal,
    ) -> Option<Decimal> {
        let position = self.positions.get(symbol)?;

        if position.quantity == Decimal::ZERO {
            return None;
        }

        // Check if this is a closing trade
        let is_closing = match side {
            Side::Sell => position.quantity > Decimal::ZERO,
            Side::Buy => position.quantity < Decimal::ZERO,
        };

        if !is_closing {
            return None;
        }

        let close_qty = quantity.min(position.quantity.abs());
        let pnl = if position.quantity > Decimal::ZERO {
            // Closing long position
            (fill_price - position.entry_price) * close_qty
        } else {
            // Closing short position
            (position.entry_price - fill_price) * close_qty
        };

        // Subtract fee from PnL
        let fee = self.calculate_fee(fill_price, close_qty, false);
        Some(pnl - fee)
    }

    /// Update position after a trade
    fn update_position(
        &mut self,
        symbol: &str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        pnl: Option<Decimal>,
    ) {
        let trade_qty = match side {
            Side::Buy => quantity,
            Side::Sell => -quantity,
        };

        if let Some(position) = self.positions.get_mut(symbol) {
            let current_qty = position.quantity;
            let new_qty = current_qty + trade_qty;

            if new_qty == Decimal::ZERO {
                // Position closed
                position.quantity = Decimal::ZERO;
                if let Some(p) = pnl {
                    position.realized_pnl += p;
                }
            } else if (current_qty > Decimal::ZERO && new_qty > Decimal::ZERO)
                || (current_qty < Decimal::ZERO && new_qty < Decimal::ZERO)
            {
                // Adding to position - calculate new average price
                if current_qty != Decimal::ZERO {
                    let total_cost =
                        position.entry_price * current_qty.abs() + price * quantity;
                    position.entry_price = total_cost / new_qty.abs();
                }
                position.quantity = new_qty;
            } else {
                // Position flipped
                position.quantity = new_qty;
                position.entry_price = price;
                if let Some(p) = pnl {
                    position.realized_pnl += p;
                }
            }

            position.updated_at = Utc::now();

            // Persist position
            let pos_clone = position.clone();
            let storage = self.storage.clone();
            tokio::spawn(async move {
                if let Err(e) = storage.save_position(&pos_clone).await {
                    warn!(error = %e, "Failed to persist position");
                }
            });
        } else {
            // New position
            let position = Position::new(symbol, trade_qty, price);
            let pos_clone = position.clone();
            self.positions.insert(symbol.to_string(), position);

            // Persist new position
            let storage = self.storage.clone();
            tokio::spawn(async move {
                if let Err(e) = storage.save_position(&pos_clone).await {
                    warn!(error = %e, "Failed to persist position");
                }
            });
        }
    }

    /// Get current position for a symbol
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> Vec<Position> {
        self.positions.values().cloned().collect()
    }

    /// Get current account state
    pub fn get_account(&self) -> Account {
        let mut account = self.account.clone();
        account.update_win_rate();
        account
    }

    /// Update position PnL with current price
    pub fn update_position_pnl(&mut self, symbol: &str, current_price: Decimal) {
        if let Some(position) = self.positions.get_mut(symbol) {
            position.update_pnl(current_price);
        }
        self.update_equity();
    }

    /// Check if daily trade limit reached
    pub fn daily_limit_reached(&self, max_daily_trades: u32) -> bool {
        self.trades_today >= max_daily_trades
    }
}

#[cfg(test)]
mod tests {
    // Tests would require mocking TradeStorage
}
