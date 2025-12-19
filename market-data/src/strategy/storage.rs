//! SQLite storage for trade persistence
//!
//! Async SQLite storage using tokio-rusqlite for non-blocking
//! database operations.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::path::Path;
use std::str::FromStr;
use tokio::sync::Mutex;
use tracing::info;

use super::models::{Order, Position, Side, Trade};

/// Async SQLite connection wrapper
pub struct TradeStorage {
    conn: Mutex<rusqlite::Connection>,
}

impl TradeStorage {
    /// Create new storage instance and initialize database
    pub async fn new(db_path: &str) -> anyhow::Result<Self> {
        // Create parent directories if needed
        if let Some(parent) = Path::new(db_path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let conn = rusqlite::Connection::open(db_path)?;

        // Enable WAL mode for better concurrent access
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;

        let storage = Self {
            conn: Mutex::new(conn),
        };

        storage.create_tables().await?;

        info!(path = %db_path, "Database initialized");

        Ok(storage)
    }

    /// Create database tables
    async fn create_tables(&self) -> anyhow::Result<()> {
        let conn = self.conn.lock().await;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price TEXT NOT NULL,
                quantity TEXT NOT NULL,
                fee TEXT NOT NULL,
                fee_asset TEXT NOT NULL,
                pnl TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity TEXT NOT NULL,
                price TEXT,
                status TEXT NOT NULL,
                filled_quantity TEXT NOT NULL,
                avg_fill_price TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity TEXT NOT NULL,
                entry_price TEXT NOT NULL,
                unrealized_pnl TEXT NOT NULL,
                realized_pnl TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance TEXT NOT NULL,
                equity TEXT NOT NULL,
                total_pnl TEXT NOT NULL,
                win_rate REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            "#,
        )?;

        Ok(())
    }

    /// Save a trade to the database
    pub async fn save_trade(&self, trade: &Trade) -> anyhow::Result<()> {
        let conn = self.conn.lock().await;

        conn.execute(
            r#"
            INSERT INTO trades (id, order_id, symbol, side, price, quantity, fee, fee_asset, pnl, timestamp)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            rusqlite::params![
                trade.id,
                trade.order_id,
                trade.symbol,
                format!("{}", trade.side),
                trade.price.to_string(),
                trade.quantity.to_string(),
                trade.fee.to_string(),
                trade.fee_asset,
                trade.pnl.map(|p| p.to_string()),
                trade.timestamp.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Save an order to the database
    pub async fn save_order(&self, order: &Order) -> anyhow::Result<()> {
        let conn = self.conn.lock().await;

        conn.execute(
            r#"
            INSERT OR REPLACE INTO orders
            (id, symbol, side, order_type, quantity, price, status, filled_quantity, avg_fill_price, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            "#,
            rusqlite::params![
                order.id,
                order.symbol,
                format!("{}", order.side),
                format!("{:?}", order.order_type).to_lowercase(),
                order.quantity.to_string(),
                order.price.map(|p| p.to_string()),
                format!("{:?}", order.status).to_lowercase(),
                order.filled_quantity.to_string(),
                order.avg_fill_price.map(|p| p.to_string()),
                order.created_at.to_rfc3339(),
                order.updated_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Save a position to the database
    pub async fn save_position(&self, position: &Position) -> anyhow::Result<()> {
        let conn = self.conn.lock().await;

        conn.execute(
            r#"
            INSERT OR REPLACE INTO positions
            (symbol, quantity, entry_price, unrealized_pnl, realized_pnl, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            rusqlite::params![
                position.symbol,
                position.quantity.to_string(),
                position.entry_price.to_string(),
                position.unrealized_pnl.to_string(),
                position.realized_pnl.to_string(),
                position.updated_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Get all positions from database
    pub async fn get_all_positions(&self) -> anyhow::Result<Vec<Position>> {
        let conn = self.conn.lock().await;

        let mut stmt = conn.prepare("SELECT * FROM positions")?;
        let positions = stmt
            .query_map([], |row| {
                Ok(Position {
                    symbol: row.get(0)?,
                    quantity: Decimal::from_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    entry_price: Decimal::from_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                    unrealized_pnl: Decimal::from_str(&row.get::<_, String>(3)?)
                        .unwrap_or_default(),
                    realized_pnl: Decimal::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                    updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?
            .filter_map(Result::ok)
            .collect();

        Ok(positions)
    }

    /// Get trades with optional symbol filter
    pub async fn get_trades(
        &self,
        symbol: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<Trade>> {
        let conn = self.conn.lock().await;

        let query = match symbol {
            Some(_) => {
                "SELECT * FROM trades WHERE symbol = ?1 ORDER BY timestamp DESC LIMIT ?2"
            }
            None => "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?1",
        };

        let mut stmt = conn.prepare(query)?;

        let trades = if let Some(sym) = symbol {
            stmt.query_map(rusqlite::params![sym, limit], Self::row_to_trade)?
        } else {
            stmt.query_map(rusqlite::params![limit], Self::row_to_trade)?
        };

        Ok(trades.filter_map(Result::ok).collect())
    }

    /// Convert a database row to a Trade
    fn row_to_trade(row: &rusqlite::Row) -> rusqlite::Result<Trade> {
        let side_str: String = row.get(3)?;
        let side = if side_str == "buy" { Side::Buy } else { Side::Sell };

        let pnl_str: Option<String> = row.get(8)?;
        let pnl = pnl_str.and_then(|s| Decimal::from_str(&s).ok());

        Ok(Trade {
            id: row.get(0)?,
            order_id: row.get(1)?,
            symbol: row.get(2)?,
            side,
            price: Decimal::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
            quantity: Decimal::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
            fee: Decimal::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
            fee_asset: row.get(7)?,
            pnl,
            timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }

    /// Get number of trades today
    pub async fn get_trade_count_today(&self) -> anyhow::Result<u32> {
        let conn = self.conn.lock().await;

        let today = Utc::now().format("%Y-%m-%d").to_string();
        let count: u32 = conn.query_row(
            "SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?1",
            [&today],
            |row| row.get(0),
        )?;

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_storage_init() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let storage = TradeStorage::new(db_path.to_str().unwrap()).await;
        assert!(storage.is_ok());
    }
}
