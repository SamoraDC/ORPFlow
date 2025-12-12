//! Publisher module for IPC communication
//!
//! Publishes order book state to other system components.

use std::path::Path;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::error::{MarketDataError, Result};
use crate::orderbook::OrderBookState;

/// Publisher for sending order book updates via Unix socket
pub struct Publisher {
    socket_path: String,
    stream: Mutex<Option<UnixStream>>,
}

impl Publisher {
    /// Create a new publisher
    pub async fn new(socket_path: &str) -> Result<Self> {
        let publisher = Self {
            socket_path: socket_path.to_string(),
            stream: Mutex::new(None),
        };

        // Try initial connection (may fail if core isn't ready)
        if let Err(e) = publisher.connect().await {
            warn!(error = %e, "Initial IPC connection failed, will retry on publish");
        }

        Ok(publisher)
    }

    /// Connect to the Unix socket
    async fn connect(&self) -> Result<()> {
        let path = Path::new(&self.socket_path);

        if !path.exists() {
            return Err(MarketDataError::IpcError(format!(
                "Socket path does not exist: {}",
                self.socket_path
            )));
        }

        let stream = UnixStream::connect(path).await.map_err(|e| {
            MarketDataError::IpcError(format!("Failed to connect to {}: {}", self.socket_path, e))
        })?;

        let mut guard = self.stream.lock().await;
        *guard = Some(stream);

        info!(path = %self.socket_path, "Connected to IPC socket");
        Ok(())
    }

    /// Publish order book state
    pub async fn publish(&self, state: &OrderBookState) -> Result<()> {
        // Serialize using MessagePack for efficiency
        let data = rmp_serde::to_vec(state).map_err(|e| {
            MarketDataError::SerializationError(format!("Failed to serialize: {}", e))
        })?;

        // Prepare message with length prefix
        let len = (data.len() as u32).to_be_bytes();
        let mut message = Vec::with_capacity(4 + data.len());
        message.extend_from_slice(&len);
        message.extend_from_slice(&data);

        // Try to send
        let mut guard = self.stream.lock().await;

        // Check if we need to reconnect
        if guard.is_none() {
            drop(guard);
            if let Err(e) = self.connect().await {
                debug!(error = %e, "Failed to reconnect to IPC socket");
                return Ok(()); // Don't fail on publish errors
            }
            guard = self.stream.lock().await;
        }

        if let Some(stream) = guard.as_mut() {
            match stream.write_all(&message).await {
                Ok(_) => {
                    debug!(
                        symbol = %state.symbol,
                        update_id = state.last_update_id,
                        "Published order book state"
                    );
                }
                Err(e) => {
                    warn!(error = %e, "Failed to write to IPC socket");
                    *guard = None; // Mark as disconnected
                }
            }
        }

        Ok(())
    }
}
