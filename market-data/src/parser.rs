//! Parser module for Binance WebSocket messages
//!
//! Handles deserialization of depth updates, trades, and other market data messages.

use rust_decimal::Decimal;
use serde::{Deserialize, Deserializer};
use std::str::FromStr;

/// Binance depth update message
#[derive(Debug, Clone, Deserialize)]
pub struct DepthUpdate {
    /// Event type
    #[serde(rename = "e")]
    pub event_type: String,

    /// Event time (milliseconds)
    #[serde(rename = "E")]
    pub event_time: u64,

    /// Symbol
    #[serde(rename = "s")]
    pub symbol: String,

    /// First update ID in event
    #[serde(rename = "U")]
    pub first_update_id: u64,

    /// Final update ID in event
    #[serde(rename = "u")]
    pub final_update_id: u64,

    /// Bids to update
    #[serde(rename = "b", deserialize_with = "deserialize_price_levels")]
    pub bids: Vec<PriceLevel>,

    /// Asks to update
    #[serde(rename = "a", deserialize_with = "deserialize_price_levels")]
    pub asks: Vec<PriceLevel>,
}

/// Binance trade message
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct Trade {
    /// Event type
    #[serde(rename = "e")]
    pub event_type: String,

    /// Event time
    #[serde(rename = "E")]
    pub event_time: u64,

    /// Symbol
    #[serde(rename = "s")]
    pub symbol: String,

    /// Trade ID
    #[serde(rename = "t")]
    pub trade_id: u64,

    /// Price
    #[serde(rename = "p", deserialize_with = "deserialize_decimal")]
    pub price: Decimal,

    /// Quantity
    #[serde(rename = "q", deserialize_with = "deserialize_decimal")]
    pub quantity: Decimal,

    /// Buyer order ID
    #[serde(rename = "b")]
    pub buyer_order_id: u64,

    /// Seller order ID
    #[serde(rename = "a")]
    pub seller_order_id: u64,

    /// Trade time
    #[serde(rename = "T")]
    pub trade_time: u64,

    /// Is buyer maker
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

/// Price level (price, quantity pair)
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: Decimal,
    pub quantity: Decimal,
}

/// Order book snapshot from REST API
#[derive(Debug, Clone, Deserialize)]
pub struct OrderBookSnapshot {
    /// Last update ID
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,

    /// Bids
    #[serde(deserialize_with = "deserialize_price_levels")]
    pub bids: Vec<PriceLevel>,

    /// Asks
    #[serde(deserialize_with = "deserialize_price_levels")]
    pub asks: Vec<PriceLevel>,
}

/// Combined stream message wrapper
#[derive(Debug, Clone, Deserialize)]
pub struct StreamMessage {
    /// Stream name
    pub stream: String,

    /// Data payload
    pub data: serde_json::Value,
}

/// Parsed WebSocket message
#[derive(Debug, Clone)]
pub enum ParsedMessage {
    DepthUpdate(DepthUpdate),
    Trade(Trade),
    Unknown(String),
}

impl ParsedMessage {
    /// Parse a raw WebSocket message
    pub fn parse(raw: &str) -> Result<Self, serde_json::Error> {
        // Try to parse as stream message first (combined streams)
        if let Ok(stream_msg) = serde_json::from_str::<StreamMessage>(raw) {
            return Self::parse_stream_data(&stream_msg.stream, &stream_msg.data);
        }

        // Try direct parsing
        if let Ok(depth) = serde_json::from_str::<DepthUpdate>(raw) {
            if depth.event_type == "depthUpdate" {
                return Ok(ParsedMessage::DepthUpdate(depth));
            }
        }

        if let Ok(trade) = serde_json::from_str::<Trade>(raw) {
            if trade.event_type == "trade" {
                return Ok(ParsedMessage::Trade(trade));
            }
        }

        Ok(ParsedMessage::Unknown(raw.to_string()))
    }

    fn parse_stream_data(
        stream: &str,
        data: &serde_json::Value,
    ) -> Result<Self, serde_json::Error> {
        if stream.contains("depth") {
            let depth: DepthUpdate = serde_json::from_value(data.clone())?;
            Ok(ParsedMessage::DepthUpdate(depth))
        } else if stream.contains("trade") {
            let trade: Trade = serde_json::from_value(data.clone())?;
            Ok(ParsedMessage::Trade(trade))
        } else {
            Ok(ParsedMessage::Unknown(data.to_string()))
        }
    }
}

/// Custom deserializer for Decimal from string
fn deserialize_decimal<'de, D>(deserializer: D) -> Result<Decimal, D::Error>
where
    D: Deserializer<'de>,
{
    let s: &str = Deserialize::deserialize(deserializer)?;
    Decimal::from_str(s).map_err(serde::de::Error::custom)
}

/// Custom deserializer for price levels from array of string pairs
fn deserialize_price_levels<'de, D>(deserializer: D) -> Result<Vec<PriceLevel>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: Vec<Vec<String>> = Deserialize::deserialize(deserializer)?;
    raw.into_iter()
        .map(|pair| {
            if pair.len() != 2 {
                return Err(serde::de::Error::custom("Invalid price level format"));
            }
            Ok(PriceLevel {
                price: Decimal::from_str(&pair[0]).map_err(serde::de::Error::custom)?,
                quantity: Decimal::from_str(&pair[1]).map_err(serde::de::Error::custom)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_depth_update() {
        let raw = r#"{
            "e": "depthUpdate",
            "E": 1672531200000,
            "s": "BTCUSDT",
            "U": 100,
            "u": 105,
            "b": [["50000.00", "1.5"], ["49999.00", "2.0"]],
            "a": [["50001.00", "1.0"], ["50002.00", "0.5"]]
        }"#;

        let msg = ParsedMessage::parse(raw).unwrap();
        if let ParsedMessage::DepthUpdate(depth) = msg {
            assert_eq!(depth.symbol, "BTCUSDT");
            assert_eq!(depth.bids.len(), 2);
            assert_eq!(depth.asks.len(), 2);
            assert_eq!(depth.bids[0].price, Decimal::from_str("50000.00").unwrap());
        } else {
            panic!("Expected DepthUpdate");
        }
    }

    #[test]
    fn test_parse_trade() {
        let raw = r#"{
            "e": "trade",
            "E": 1672531200000,
            "s": "BTCUSDT",
            "t": 12345,
            "p": "50000.50",
            "q": "0.5",
            "b": 111,
            "a": 222,
            "T": 1672531200000,
            "m": false
        }"#;

        let msg = ParsedMessage::parse(raw).unwrap();
        if let ParsedMessage::Trade(trade) = msg {
            assert_eq!(trade.symbol, "BTCUSDT");
            assert_eq!(trade.price, Decimal::from_str("50000.50").unwrap());
            assert!(!trade.is_buyer_maker);
        } else {
            panic!("Expected Trade");
        }
    }
}
