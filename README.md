# ORPflow - HFT Paper Trading

> **O**Caml + **R**ust + **P**ython Flow

[![CI](https://github.com/SamoraDC/ORPflow/actions/workflows/ci.yml/badge.svg)](https://github.com/SamoraDC/ORPflow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7)](https://render.com)

A high-frequency paper trading system demonstrating multi-language systems engineering with **OCaml**, **Rust**, and **Python**. Designed as a portfolio project showcasing low-latency market data processing, type-safe risk management, and quantitative trading strategies.

## Features

- **Real-time Market Data** (Rust): WebSocket connection to Binance with order book reconstruction
- **Type-Safe Risk Engine** (OCaml): Position limits, drawdown circuit breakers, P&L calculation
- **Trading Strategy** (Python): Order flow imbalance strategy with volatility adjustment
- **Paper Trading**: Realistic execution simulation with slippage, market impact, and fees
- **Shabbat Pause**: Automatic trading pause from Friday to Saturday sunset
- **Live Dashboard**: Auto-updating README with performance charts via GitHub Actions
- **Cloud Deployment**: Runs 24/6 on Render (starter plan $7/month for always-on)
- **State Recovery**: Checkpointing with warm-up period on restart
- **Safety Systems**: Kill switch, circuit breakers, rate limiting, sanity checks
- **Complementary Data**: Funding rates, open interest, liquidation monitoring
- **Strategy Versioning**: A/B testing support with performance tracking
- **Replay Logging**: Full event logging for debugging and analysis

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Binance   │────▶│    Rust     │────▶│   OCaml     │
│  WebSocket  │     │ Market Data │     │ Risk Engine │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           └───────┬───────────┘
                                   ▼
                          ┌─────────────────┐
                          │     Python      │
                          │ Strategy Engine │
                          │   + REST API    │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │     SQLite      │
                          │    Database     │
                          └─────────────────┘
```

### Why This Tech Stack?

| Component | Language | Reason |
|-----------|----------|--------|
| Market Data | **Rust** | Zero-cost abstractions, memory safety, excellent async runtime |
| Risk Engine | **OCaml** | Algebraic data types prevent invalid states, exhaustive pattern matching |
| Strategy | **Python** | Rapid prototyping, rich data science ecosystem, easy modification |

## Quick Start

### Prerequisites

- Rust 1.75+
- OCaml 5.1+ with opam
- Python 3.11+
- Docker (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/SamoraDC/ORPflow.git
cd ORPflow

# Run setup script (installs all dependencies)
./scripts/setup-dev.sh

# Start all components locally
./scripts/run-local.sh
```

### Using Docker

```bash
docker-compose up --build
```

## Project Structure

```
ORPflow/
├── market-data/          # Rust - WebSocket & order book
│   └── src/
│       ├── websocket/    # Connection management
│       ├── parser/       # Message deserialization
│       ├── orderbook/    # Order book data structure
│       └── publisher/    # IPC to other components
│
├── core/                 # OCaml - Risk engine
│   └── lib/
│       ├── types/        # Domain types (Order, Trade, Position)
│       ├── orderbook/    # Type-safe order book
│       ├── risk/         # Risk validation & limits
│       └── pnl/          # P&L calculation
│
├── strategy/             # Python - Trading strategy
│   └── src/
│       ├── signals/      # Trading strategies
│       ├── features/     # Microstructure features
│       ├── broker/       # Paper broker
│       ├── api/          # REST API (FastAPI)
│       └── storage/      # SQLite persistence
│
├── reports/              # Report generator
│   ├── generate.py       # Chart generation
│   └── templates/        # README templates
│
├── docs/                 # Documentation
│   ├── architecture.md   # System design
│   ├── strategies.md     # Trading logic
│   └── deployment.md     # Deploy guide
│
└── .github/workflows/    # CI/CD
    ├── ci.yml            # Tests on every push
    ├── deploy-render.yml # Deploy to Render
    ├── daily-report.yml  # Update charts daily
    └── health-check.yml  # Monitor every 15min
```

## Trading Strategy

The primary strategy exploits **order book imbalance** - when bid volume significantly exceeds ask volume, buying pressure tends to push prices up.

### Signal Generation

1. Calculate imbalance: `(bid_vol - ask_vol) / (bid_vol + ask_vol)`
2. Apply weighted decay (closer levels matter more)
3. Require persistence (3+ ticks in same direction)
4. Confirm with price momentum
5. Filter by spread and volatility

### Risk Management

- **Position Limits**: Max 1 BTC per symbol
- **Drawdown Circuit Breaker**: Pauses at 5% drawdown
- **Rate Limiting**: Max 60 orders/minute
- **Stop Loss**: 2x ATR below entry

## Robustness Features

### State Management & Recovery

The system implements periodic checkpointing to handle crashes and restarts gracefully:

- **Checkpointing**: Full state saved every 60 seconds
- **Warm-up Period**: 5 minutes + 100 data points before trading
- **Graceful Shutdown**: SIGTERM handler ensures clean state persistence

### Realistic Execution Simulation

Paper trading simulates real-world execution challenges:

- **Fill Probability**: Models queue position for limit orders
- **Market Impact**: Large orders walk through multiple price levels
- **Variable Latency**: Higher latency during volatile periods
- **Partial Fills**: Not all orders fill completely

### Safety Systems

Multiple layers protect against catastrophic bugs:

```
┌─────────────────────────────────────────────────────┐
│                   Kill Switch                        │
│  (Manual emergency stop via API or Telegram)        │
├─────────────────────────────────────────────────────┤
│               Circuit Breaker                        │
│  (Auto-pause on consecutive losses or drawdown)     │
├─────────────────────────────────────────────────────┤
│               Rate Limiter                           │
│  (Hard limit: 5/sec, 60/min, 500/hour)             │
├─────────────────────────────────────────────────────┤
│              Sanity Checks                           │
│  (Price/quantity bounds, symbol validation)         │
└─────────────────────────────────────────────────────┘
```

### Complementary Data

Enhanced signals using additional market data:

- **Funding Rate**: Detects extreme sentiment (>0.1% triggers caution)
- **Open Interest**: Confirms trend strength
- **Liquidations**: Monitors cascade risk

### Strategy Versioning

Track and compare strategy variations:

- Each trade tagged with strategy version
- Shadow mode for A/B testing
- Performance comparison by version

<!-- METRICS_START -->
## Live Performance

*System starting - metrics will appear after first trades*

| Metric | Value |
|--------|-------|
| Total Trades | 0 |
| Win Rate | 0% |
| Total P&L | $0.00 |
| Sharpe Ratio | - |
| Max Drawdown | 0% |

*Charts will be generated daily by GitHub Actions*
<!-- METRICS_END -->

## Deployment

### Deploy to Render

1. Create account at [render.com](https://render.com) (sign up with GitHub)
2. Click **New +** → **Blueprint**
3. Connect this repository
4. Select `render.yaml` (paid, $7/month, always-on) or `render-free.yaml` (free, spins down)
5. Click **Apply**

### Environment Variables (set in Render Dashboard)

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | No | For trade notifications |
| `TELEGRAM_CHAT_ID` | No | Your Telegram chat ID |

### Telegram Setup (Optional)

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → Copy the token
2. Message [@userinfobot](https://t.me/userinfobot) → Copy your ID
3. Start a chat with your bot
4. Add tokens to Render Dashboard → Environment

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/account` | GET | Account state |
| `/positions` | GET | Open positions |
| `/trades` | GET | Trade history |
| `/metrics` | GET | Performance metrics |

## Development

### Running Tests

```bash
# All tests
make test

# Individual components
cd market-data && cargo test
cd core && dune test
cd strategy && pytest
```

### Linting

```bash
make lint
```

### Benchmarks

```bash
cd market-data && cargo bench
```

## Documentation

- [Architecture](docs/architecture.md) - System design and data flow
- [Strategies](docs/strategies.md) - Trading strategy documentation
- [Deployment](docs/deployment.md) - Deploy and configuration guide

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Important Disclaimers

### Educational Purpose Only

This is a **paper trading** system created for educational and portfolio demonstration purposes only.

- **No Real Trading**: This system does not execute real trades or handle real money
- **No API Keys Required**: Uses only public WebSocket streams from Binance
- **Not Financial Advice**: This project is not financial advice and should not be used to make investment decisions

### Performance Disclaimer

- **Simulated Results**: All performance metrics are from paper trading, not real trading
- **No Guarantee**: Past simulated performance does not guarantee future results
- **Market Conditions**: Real markets include factors not fully captured in simulation (liquidity, counterparty risk, exchange downtime)

### Binance API Usage

This project uses Binance's public APIs in accordance with their [Terms of Use](https://www.binance.com/en/terms). No private API access or trading functionality is implemented.

### Data Privacy

- The optional Telegram integration sends only trade summaries, never sensitive data
- No personal information is collected or stored
- All data is stored locally on the deployment server

### Legal

This software is provided "as is" without warranty of any kind. The authors are not responsible for any financial losses or legal issues arising from the use of this software.

---

Built with **O**Caml, **R**ust, and **P**ython - **ORPflow**
