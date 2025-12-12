//! WebSocket client for Binance streams
//!
//! Handles connection, subscription, and message reception.

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{protocol::Message, Error as WsError},
    MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, error, info, warn};

use crate::error::{MarketDataError, Result};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// WebSocket client for a single connection
pub struct WebSocketClient {
    stream: Option<WsStream>,
    endpoint: String,
    symbols: Vec<String>,
}

impl WebSocketClient {
    /// Create a new WebSocket client
    pub fn new(endpoint: &str, symbols: Vec<String>) -> Self {
        Self {
            stream: None,
            endpoint: endpoint.to_string(),
            symbols,
        }
    }

    /// Connect to the WebSocket endpoint
    pub async fn connect(&mut self) -> Result<()> {
        // Build the combined stream URL
        let streams: Vec<String> = self
            .symbols
            .iter()
            .flat_map(|s| {
                let s_lower = s.to_lowercase();
                vec![
                    format!("{}@depth@100ms", s_lower),
                    format!("{}@trade", s_lower),
                ]
            })
            .collect();

        let url = format!("{}/stream?streams={}", self.endpoint, streams.join("/"));

        info!(url = %url, "Connecting to Binance WebSocket");

        let (ws_stream, response) = connect_async(&url).await.map_err(|e| {
            MarketDataError::WebSocketConnection(format!("Failed to connect: {}", e))
        })?;

        info!(status = ?response.status(), "WebSocket connected");
        self.stream = Some(ws_stream);

        Ok(())
    }

    /// Receive the next message
    pub async fn recv(&mut self) -> Result<Option<String>> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| MarketDataError::WebSocketConnection("Not connected".to_string()))?;

        match stream.next().await {
            Some(Ok(Message::Text(text))) => {
                debug!(len = text.len(), "Received text message");
                Ok(Some(text))
            }
            Some(Ok(Message::Binary(data))) => {
                // Convert binary to text if needed
                let text = String::from_utf8_lossy(&data).to_string();
                Ok(Some(text))
            }
            Some(Ok(Message::Ping(data))) => {
                debug!("Received ping, sending pong");
                if let Some(stream) = self.stream.as_mut() {
                    let _ = stream.send(Message::Pong(data)).await;
                }
                Ok(None)
            }
            Some(Ok(Message::Pong(_))) => {
                debug!("Received pong");
                Ok(None)
            }
            Some(Ok(Message::Close(frame))) => {
                warn!(frame = ?frame, "Received close frame");
                self.stream = None;
                Err(MarketDataError::WebSocketConnection(
                    "Connection closed".to_string(),
                ))
            }
            Some(Ok(Message::Frame(_))) => Ok(None),
            Some(Err(e)) => {
                error!(error = %e, "WebSocket error");
                self.stream = None;
                Err(MarketDataError::WebSocketMessage(e.to_string()))
            }
            None => {
                warn!("WebSocket stream ended");
                self.stream = None;
                Err(MarketDataError::WebSocketConnection(
                    "Stream ended".to_string(),
                ))
            }
        }
    }

    /// Send a ping to keep connection alive
    pub async fn ping(&mut self) -> Result<()> {
        if let Some(stream) = self.stream.as_mut() {
            stream
                .send(Message::Ping(vec![]))
                .await
                .map_err(|e| MarketDataError::WebSocketMessage(e.to_string()))?;
        }
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    /// Close the connection
    pub async fn close(&mut self) {
        if let Some(mut stream) = self.stream.take() {
            let _ = stream.close(None).await;
        }
    }
}
