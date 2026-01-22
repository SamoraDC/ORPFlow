#!/bin/bash
set -e

# ORPFlow HFT Paper Trading - Entrypoint Script
# Initializes the environment and starts all services
# Note: No Python in runtime - Rust binary handles everything

echo "========================================"
echo "ORPFlow HFT Paper Trading System"
echo "========================================"
echo ""

# Create necessary directories
mkdir -p /var/log/supervisor
mkdir -p /data

# Display configuration
echo "Configuration:"
echo "  Symbols: ${SYMBOLS:-BTCUSDT,ETHUSDT}"
echo "  Timezone: ${TIMEZONE:-America/Sao_Paulo}"
echo "  Max Position: ${RISK_MAX_POSITION:-1.0}"
echo "  Max Drawdown: ${RISK_MAX_DRAWDOWN:-0.05}"
echo "  Initial Balance: ${INITIAL_BALANCE:-10000}"
echo "  ML Enabled: ${ML_ENABLED:-true}"
echo "  ONNX Model Dir: ${ONNX_MODEL_DIR:-/app/models/onnx}"
echo ""

# Check Binance connectivity
echo "Checking Binance API connectivity..."
if curl -s --connect-timeout 5 https://api.binance.com/api/v3/ping > /dev/null; then
    echo "  Binance API: OK"
else
    echo "  Binance API: WARNING - Unable to connect"
fi
echo ""

# Verify ONNX models exist
echo "Checking ONNX models..."
if [ -d "${ONNX_MODEL_DIR:-/app/models/onnx}" ]; then
    MODEL_COUNT=$(find "${ONNX_MODEL_DIR:-/app/models/onnx}" -name "*.onnx" 2>/dev/null | wc -l)
    echo "  ONNX models found: $MODEL_COUNT"
else
    echo "  ONNX models: WARNING - Directory not found"
fi
echo ""

echo "Starting services..."
echo ""

# Execute the main command (supervisord)
exec "$@"
