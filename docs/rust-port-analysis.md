# Python to Rust Strategy Engine Port Analysis

## Executive Summary

This document provides a comprehensive analysis of the Python strategy engine codebase to facilitate porting to Rust. It covers all mathematical formulas, algorithms, business logic, and data structures that need to be implemented in the Rust version.

---

## 1. Strategy Logic (`imbalance_strategy.py`)

### 1.1 Configuration Parameters

```rust
// StrategyConfig equivalent in Rust
struct StrategyConfig {
    // Imbalance thresholds
    imbalance_threshold: f64,        // Default: 0.3
    min_confidence: f64,             // Default: 0.6

    // Volatility adjustment
    low_vol_multiplier: f64,         // Default: 1.5
    high_vol_multiplier: f64,        // Default: 0.5
    vol_threshold_low: f64,          // Default: -1.0 (z-score)
    vol_threshold_high: f64,         // Default: 1.0 (z-score)

    // Momentum confirmation
    require_momentum_confirm: bool,   // Default: true
    momentum_threshold: f64,          // Default: 0.0001

    // Imbalance persistence
    persistence_required: i32,        // Default: 3 ticks
    persistence_decay: f64,           // Default: 0.9

    // Position sizing
    base_position_pct: f64,          // Default: 0.1 (10%)

    // Spread filter
    max_spread_bps: f64,             // Default: 10.0 basis points
}
```

### 1.2 Core Algorithm: Signal Evaluation

**Signal Generation Logic:**

```python
# Pseudocode for evaluate() function
def evaluate(features, account_balance, current_position):
    # 1. Basic validation
    if features.imbalance is None or features.mid_price is None:
        return None

    # 2. Spread filter
    if features.spread_bps > max_spread_bps:
        return None  # Too wide

    # 3. Track imbalance persistence
    imbalance_sign = 1 if imbalance > 0 else -1

    if last_sign == imbalance_sign:
        streak += 1
    else:
        streak = 1
        last_sign = imbalance_sign

    # 4. Check imbalance significance
    if abs(imbalance) < imbalance_threshold:
        return None

    # 5. Check persistence requirement
    if streak < persistence_required:
        return None

    # 6. Calculate confidence
    confidence = calculate_confidence(features)
    if confidence < min_confidence:
        return None

    # 7. Momentum confirmation
    if require_momentum_confirm:
        expected_momentum_sign = imbalance_sign
        if momentum * expected_momentum_sign < momentum_threshold:
            return None

    # 8. Imbalance momentum check
    if imbalance_momentum * imbalance_sign < 0:
        return None  # Diverging

    # 9. Calculate position size
    size = calculate_position_size(features, account_balance, current_position)
    if size <= 0:
        return None

    # 10. Generate signal
    side = BUY if imbalance > 0 else SELL
    return Signal(symbol, side, confidence, size, reason, timestamp)
```

### 1.3 Confidence Calculation Algorithm

**Mathematical Formula:**

```python
confidence = 0.0

# 1. Base confidence from imbalance strength (max 0.4)
imbalance_strength = min(abs(imbalance), 1.0)
confidence += 0.4 * imbalance_strength

# 2. Weighted imbalance contribution (max 0.2)
weighted_strength = min(abs(weighted_imbalance), 1.0)
confidence += 0.2 * weighted_strength

# 3. Persistence bonus (max 0.2)
persistence_score = min(streak / 10.0, 1.0)  # Max at 10 ticks
confidence += 0.2 * persistence_score

# 4. Volatility adjustment (±0.1)
if volatility_z < vol_threshold_low:      # Low vol
    confidence += 0.1
elif volatility_z > vol_threshold_high:   # High vol
    confidence -= 0.1

# 5. Imbalance momentum bonus (max 0.1)
if imbalance_momentum * imbalance_sign > 0:
    confidence += 0.1

# Clamp to [0.0, 1.0]
confidence = max(0.0, min(1.0, confidence))
```

**Confidence Components:**
- Imbalance strength: 40%
- Weighted imbalance: 20%
- Persistence: 20%
- Volatility adjustment: ±10%
- Imbalance momentum: 10%

### 1.4 Position Sizing Algorithm

```python
def calculate_position_size(features, account_balance, current_position):
    # 1. Base size calculation
    base_size = account_balance * base_position_pct  # Dollar amount

    # 2. Volatility adjustment
    if volatility_z < vol_threshold_low:
        base_size *= low_vol_multiplier   # 1.5x in low vol
    elif volatility_z > vol_threshold_high:
        base_size *= high_vol_multiplier  # 0.5x in high vol

    # 3. Convert to quantity
    if mid_price == 0:
        return 0.0
    quantity = base_size / mid_price

    # 4. Adjust for existing position
    if current_position != 0:
        imbalance_sign = 1 if imbalance > 0 else -1
        position_sign = 1 if current_position > 0 else -1

        # If adding to position, halve the size
        if imbalance_sign == position_sign:
            quantity *= 0.5

    # 5. Round to 6 decimal places
    return round(quantity, 6)
```

**Position Sizing Rules:**
1. Base size: 10% of account balance (configurable)
2. Low volatility: Multiply by 1.5
3. High volatility: Multiply by 0.5
4. Adding to position: Halve the quantity
5. Precision: 6 decimal places

### 1.5 State Management

**Per-Symbol State Tracking:**
```rust
// State that needs to be maintained per symbol
struct SymbolState {
    imbalance_streak: i32,      // Count of consecutive same-sign imbalances
    last_imbalance_sign: i32,   // 1 or -1
}

// Storage: HashMap<String, SymbolState>
```

---

## 2. Microstructure Features (`microstructure.py`)

### 2.1 Feature Snapshot Data Structure

```rust
struct FeatureSnapshot {
    timestamp: i64,
    symbol: String,

    // Order book features
    mid_price: Option<Decimal>,
    spread_bps: Option<Decimal>,
    imbalance: Option<f64>,
    weighted_imbalance: Option<f64>,

    // Volume features
    bid_depth: Option<Decimal>,
    ask_depth: Option<Decimal>,
    volume_ratio: Option<f64>,

    // Derived features
    volatility: Option<f64>,
    momentum: Option<f64>,
    imbalance_momentum: Option<f64>,

    // Normalized features
    imbalance_z: Option<f64>,
    volatility_z: Option<f64>,
}
```

### 2.2 Rolling Window Configuration

```rust
struct MicrostructureFeatures {
    window_size: usize,          // Default: 100
    volatility_window: usize,    // Default: 20
    momentum_window: usize,      // Default: 10

    // Per-symbol rolling windows
    mid_prices: HashMap<String, VecDeque<f64>>,
    imbalances: HashMap<String, VecDeque<f64>>,
    timestamps: HashMap<String, VecDeque<i64>>,

    // Statistics for normalization
    imbalance_mean: HashMap<String, f64>,
    imbalance_std: HashMap<String, f64>,
    volatility_mean: HashMap<String, f64>,
    volatility_std: HashMap<String, f64>,
}
```

### 2.3 Volatility Calculation

**Algorithm: Annualized Log-Return Volatility**

```python
def calculate_volatility(prices):
    if len(prices) < volatility_window:
        return None

    # Get recent prices
    recent = prices[-volatility_window:]

    # Calculate log returns
    returns = np.diff(np.log(recent))

    if len(returns) == 0:
        return None

    # Standard deviation of returns
    std_dev = np.std(returns)

    # Annualize (252 trading days * 24 hours * 60 minutes)
    annualized = std_dev * sqrt(252 * 24 * 60)

    return annualized
```

**Mathematical Formula:**
```
returns[i] = ln(price[i+1]) - ln(price[i])
volatility = std(returns) * sqrt(N)
where N = 252 * 24 * 60 (annualization factor for minute data)
```

### 2.4 Momentum Calculation

**Price Momentum:**

```python
def calculate_momentum(prices):
    if len(prices) < momentum_window:
        return None

    recent = prices[-momentum_window:]

    if recent[0] == 0:
        return None

    # Percentage change over window
    momentum = (recent[-1] - recent[0]) / recent[0]

    return momentum
```

**Formula:**
```
momentum = (P_now - P_window_start) / P_window_start
```

### 2.5 Imbalance Momentum Calculation

**Algorithm: Linear Regression Slope**

```python
def calculate_imbalance_momentum(imbalances):
    if len(imbalances) < momentum_window:
        return None

    recent = imbalances[-momentum_window:]

    # Create x coordinates (0, 1, 2, ...)
    x = np.arange(len(recent))
    y = np.array(recent)

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    # Calculate correlation
    correlation = np.corrcoef(x, y)[0, 1]

    # Calculate slope
    slope = correlation * (np.std(y) / np.std(x))

    return slope
```

**Mathematical Formula:**
```
correlation = cov(x, y) / (std(x) * std(y))
slope = correlation * (std(y) / std(x))

where:
  x = [0, 1, 2, ..., window_size-1]
  y = imbalance values
```

### 2.6 Feature Normalization (Z-Score)

**Imbalance Normalization:**

```python
def normalize_imbalance(imbalance):
    # Update statistics from rolling window (min 20 samples)
    if len(imbalances) >= 20:
        mean = np.mean(imbalances)
        std = max(np.std(imbalances), 0.001)  # Prevent division by zero

    # Calculate z-score
    z_score = (imbalance - mean) / std

    return z_score
```

**Volatility Normalization:**

```python
def normalize_volatility(volatility):
    # Calculate rolling volatility statistics
    if len(prices) >= volatility_window:
        returns = np.diff(np.log(prices[-volatility_window:]))
        current_vol = np.std(returns)

        # Calculate volatility of volatility
        all_returns = np.diff(np.log(prices))
        window_vols = [
            np.std(all_returns[i:i+volatility_window])
            for i in range(len(all_returns) - volatility_window + 1)
        ]
        vol_mean = current_vol
        vol_std = max(np.std(window_vols), 0.0001)

    # Calculate z-score
    z_score = (volatility - vol_mean) / vol_std

    return z_score
```

### 2.7 Average True Range (ATR)

**Simplified ATR using Mid Prices:**

```python
def get_atr(prices, period=14):
    if len(prices) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(prices)):
        tr = abs(prices[i] - prices[i-1])
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    # Average of last 'period' true ranges
    atr = np.mean(true_ranges[-period:])

    return atr
```

### 2.8 Volume Ratio Calculation

```python
def calculate_volume_ratio(bid_depth, ask_depth):
    if bid_depth and ask_depth and ask_depth > 0:
        volume_ratio = bid_depth / ask_depth
    else:
        volume_ratio = None

    return volume_ratio
```

---

## 3. Broker Logic (`paper_broker.py`)

### 3.1 Fee Structure

```rust
// Binance VIP 0 fee structure
const MAKER_FEE: Decimal = Decimal::from_str("0.001").unwrap();  // 0.1%
const TAKER_FEE: Decimal = Decimal::from_str("0.001").unwrap();  // 0.1%

fn calculate_fee(price: Decimal, quantity: Decimal, is_maker: bool) -> Decimal {
    let fee_rate = if is_maker { MAKER_FEE } else { TAKER_FEE };
    price * quantity * fee_rate
}
```

### 3.2 Order ID Generation

```python
def generate_order_id():
    counter += 1
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    return f"ORD-{timestamp}-{counter}"

def generate_trade_id():
    uuid_hex = uuid.uuid4().hex[:12]
    return f"TRD-{uuid_hex}"
```

### 3.3 Risk Check Algorithm

**Local Risk Check (Fallback):**

```python
def local_risk_check(symbol, side, quantity):
    current_pos = positions.get(symbol, 0)

    # Calculate new position
    if side == BUY:
        new_qty = current_pos + quantity
    else:
        new_qty = current_pos - quantity

    # Check against max position
    if abs(new_qty) > max_position_size:
        allowed = max_position_size - abs(current_pos)

        if allowed <= 0:
            return (False, "Position limit reached", None)

        return (True, "Adjusted for position limit", allowed)

    return (True, None, None)
```

**Risk Check Response:**
```rust
struct RiskCheckResult {
    approved: bool,
    reason: Option<String>,
    adjusted_qty: Option<Decimal>,
}
```

### 3.4 Market Order Execution with Slippage

```python
def execute_market_order(order, current_price, slippage_bps=5):
    # 1. Apply slippage
    slippage = current_price * slippage_bps / 10000

    if order.side == BUY:
        fill_price = current_price + slippage
    else:
        fill_price = current_price - slippage

    # 2. Calculate fee
    fee = calculate_fee(fill_price, quantity, is_maker=False)

    # 3. Calculate P&L if closing position
    pnl = None
    if position exists:
        is_closing = (
            (position.qty > 0 and side == SELL) or
            (position.qty < 0 and side == BUY)
        )

        if is_closing:
            close_qty = min(abs(position.qty), order.qty)

            if position.qty > 0:
                pnl = (fill_price - position.entry_price) * close_qty - fee
            else:
                pnl = (position.entry_price - fill_price) * close_qty - fee

    # 4. Create trade
    trade = Trade(id, order_id, symbol, side, fill_price, quantity, fee, pnl)

    # 5. Update order status
    order.status = FILLED
    order.filled_quantity = quantity
    order.avg_fill_price = fill_price

    # 6. Update position
    update_position(symbol, side, quantity, fill_price, pnl)

    # 7. Update account balance
    cost = fill_price * quantity + fee

    if side == BUY:
        balance -= cost
    else:
        balance += cost - (fee * 2)  # fee already deducted

    if pnl:
        total_pnl += pnl
        total_trades += 1

    return trade
```

**Slippage Calculation:**
```
slippage_amount = current_price * slippage_bps / 10000

BUY:  fill_price = current_price + slippage_amount
SELL: fill_price = current_price - slippage_amount
```

**Default Slippage:** 5 basis points (0.05%)

### 3.5 Position Update Logic

```python
def update_position(symbol, side, quantity, price, pnl):
    position = positions.get(symbol)

    trade_qty = quantity if side == BUY else -quantity

    if position is None:
        # New position
        position = Position(
            symbol=symbol,
            quantity=trade_qty,
            entry_price=price
        )
    else:
        current_qty = position.quantity
        new_qty = current_qty + trade_qty

        if new_qty == 0:
            # Position closed
            position.quantity = 0
            position.realized_pnl += pnl or 0

        elif (current_qty > 0 and new_qty > 0) or (current_qty < 0 and new_qty < 0):
            # Adding to position - calculate new average price
            total_cost = position.entry_price * abs(current_qty) + price * quantity
            position.entry_price = total_cost / abs(new_qty)
            position.quantity = new_qty

        else:
            # Position flipped
            position.quantity = new_qty
            position.entry_price = price
            if pnl:
                position.realized_pnl += pnl

    positions[symbol] = position
```

**Position Update Cases:**
1. **New Position:** Set quantity and entry price directly
2. **Position Closed:** Set quantity to 0, add PnL to realized
3. **Adding to Position:** Calculate weighted average entry price
4. **Position Flipped:** Set new quantity and entry price, record PnL

**Average Entry Price Calculation:**
```
total_cost = old_entry_price * abs(old_qty) + fill_price * new_qty
new_entry_price = total_cost / abs(total_qty)
```

### 3.6 Account Equity Calculation

```python
def update_account_equity():
    unrealized_pnl = sum(position.unrealized_pnl for position in positions)
    equity = balance + unrealized_pnl
```

---

## 4. Scheduler Logic (`scheduler.py`)

### 4.1 Shabbat Calculation Configuration

```rust
struct ShabbatScheduler {
    latitude: f64,        // Default: -23.5505 (São Paulo)
    longitude: f64,       // Default: -46.6333
    timezone: String,     // Default: "America/Sao_Paulo"
    buffer_minutes: i32,  // Default: 18 minutes before sunset
}
```

### 4.2 Sunset Calculation

**Uses Astral library for astronomical calculations:**

```python
def get_friday_sunset(reference_time):
    # 1. Find current or next Friday
    days_until_friday = (4 - now.weekday()) % 7

    if days_until_friday == 0 and now.hour >= 12:
        friday = now.date()
    else:
        friday = now.date() + timedelta(days=days_until_friday)

    # 2. Calculate sunset for that Friday
    sun_times = sun(location.observer, date=friday, tzinfo=tz)

    # 3. Apply buffer (start pause 18 minutes before)
    return sun_times["sunset"] - buffer
```

**Saturday Sunset:**
```python
def get_saturday_sunset(reference_time):
    days_until_saturday = (5 - now.weekday()) % 7
    saturday = now.date() + timedelta(days=days_until_saturday)

    sun_times = sun(location.observer, date=saturday, tzinfo=tz)
    return sun_times["sunset"]
```

### 4.3 Shabbat Detection Algorithm

```python
def is_shabbat(reference_time=None):
    now = reference_time or datetime.now(tz)

    friday_sunset = get_friday_sunset(now)
    saturday_sunset = get_saturday_sunset(now)

    # Handle Sunday edge case
    if now.weekday() == 6:  # Sunday
        last_saturday = now - timedelta(days=1)
        last_saturday_sunset = get_saturday_sunset(last_saturday - timedelta(days=6))
        if now < last_saturday_sunset:
            return True
        return False

    # Check if in Shabbat window
    return friday_sunset <= now <= saturday_sunset
```

**Shabbat Period:**
- Start: Friday sunset - 18 minutes
- End: Saturday sunset

### 4.4 Schedule Event Calculation

```python
def next_pause_time(reference_time):
    friday_sunset = get_friday_sunset(reference_time)

    if now >= friday_sunset:
        # Get next Friday
        next_friday = now + timedelta(days=(7 - now.weekday() + 4) % 7 + 1)
        return get_friday_sunset(next_friday)

    return friday_sunset

def next_resume_time(reference_time):
    saturday_sunset = get_saturday_sunset(reference_time)

    if now >= saturday_sunset:
        # Get next Saturday
        next_saturday = now + timedelta(days=(7 - now.weekday() + 5) % 7 + 1)
        return get_saturday_sunset(next_saturday)

    return saturday_sunset

def next_event(reference_time):
    if is_shabbat(reference_time):
        return next_resume_time(reference_time)
    else:
        return next_pause_time(reference_time)
```

---

## 5. Database/Storage Logic (`database.py`)

### 5.1 Database Schema

**Tables:**

1. **trades**
```sql
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price TEXT NOT NULL,
    quantity TEXT NOT NULL,
    fee TEXT NOT NULL,
    fee_asset TEXT NOT NULL,
    pnl TEXT,
    timestamp DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
```

2. **orders**
```sql
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price TEXT,
    status TEXT NOT NULL,
    filled_quantity TEXT NOT NULL,
    avg_fill_price TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);
```

3. **positions**
```sql
CREATE TABLE positions (
    symbol TEXT PRIMARY KEY,
    quantity TEXT NOT NULL,
    entry_price TEXT NOT NULL,
    unrealized_pnl TEXT NOT NULL,
    realized_pnl TEXT NOT NULL,
    updated_at DATETIME NOT NULL
);
```

4. **account_snapshots**
```sql
CREATE TABLE account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance TEXT NOT NULL,
    equity TEXT NOT NULL,
    total_pnl TEXT NOT NULL,
    win_rate REAL NOT NULL,
    total_trades INTEGER NOT NULL,
    positions_json TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

5. **daily_stats**
```sql
CREATE TABLE daily_stats (
    date DATE PRIMARY KEY,
    starting_balance TEXT NOT NULL,
    ending_balance TEXT NOT NULL,
    pnl TEXT NOT NULL,
    trades_count INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,
    max_drawdown TEXT NOT NULL,
    sharpe_ratio REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 5.2 Data Type Storage

**Decimal Storage:**
- All Decimal types are stored as TEXT (string representation)
- Prevents floating-point precision errors
- Must be parsed back to Decimal on retrieval

**DateTime Storage:**
- Stored as ISO 8601 format strings
- UTC timezone
- Format: "YYYY-MM-DDTHH:MM:SS.ffffff"

**Enum Storage:**
- Stored as TEXT (string value)
- Side: "buy" or "sell"
- OrderType: "market" or "limit"
- OrderStatus: "pending", "filled", "partially_filled", "cancelled", "rejected"

### 5.3 Key Database Operations

**Save Trade:**
```rust
async fn save_trade(trade: &Trade) -> Result<()> {
    // INSERT INTO trades with all fields
    // Convert Decimal to string
    // Convert Side enum to string
    // Format timestamp as ISO 8601
}
```

**Save/Update Order:**
```rust
async fn save_order(order: &Order) -> Result<()> {
    // INSERT OR REPLACE into orders
    // Handles both creation and updates
}
```

**Save/Update Position:**
```rust
async fn save_position(position: &Position) -> Result<()> {
    // INSERT OR REPLACE into positions
    // Symbol is primary key
}
```

**Get Trade Count Today:**
```sql
SELECT COUNT(*) FROM trades
WHERE DATE(timestamp) = CURRENT_DATE
```

**Get Trades with Filters:**
```sql
SELECT * FROM trades
WHERE 1=1
  AND symbol = ? (optional)
  AND timestamp >= ? (optional)
  AND timestamp <= ? (optional)
ORDER BY timestamp DESC
LIMIT ?
```

---

## 6. Data Models (`models.py`)

### 6.1 Core Enums

```rust
enum Side {
    BUY,
    SELL,
}

enum OrderType {
    MARKET,
    LIMIT,
}

enum OrderStatus {
    PENDING,
    FILLED,
    PARTIALLY_FILLED,
    CANCELLED,
    REJECTED,
}
```

### 6.2 Data Structures

**PriceLevel:**
```rust
struct PriceLevel {
    price: Decimal,
    quantity: Decimal,
}
```

**OrderBookState:**
```rust
struct OrderBookState {
    symbol: String,
    timestamp: i64,
    last_update_id: i64,
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
    mid_price: Option<Decimal>,
    spread_bps: Option<Decimal>,
    imbalance: Option<Decimal>,
    weighted_imbalance: Option<Decimal>,
}
```

**Signal:**
```rust
struct Signal {
    symbol: String,
    side: Side,
    confidence: f64,           // Range: [0.0, 1.0]
    suggested_size: Decimal,
    reason: String,
    timestamp: DateTime<Utc>,
}
```

**Order:**
```rust
struct Order {
    id: String,
    symbol: String,
    side: Side,
    order_type: OrderType,
    quantity: Decimal,
    price: Option<Decimal>,
    status: OrderStatus,
    filled_quantity: Decimal,
    avg_fill_price: Option<Decimal>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}
```

**Trade:**
```rust
struct Trade {
    id: String,
    order_id: String,
    symbol: String,
    side: Side,
    price: Decimal,
    quantity: Decimal,
    fee: Decimal,
    fee_asset: String,         // Default: "USDT"
    timestamp: DateTime<Utc>,
    pnl: Option<Decimal>,
}
```

**Position:**
```rust
struct Position {
    symbol: String,
    quantity: Decimal,         // Positive = long, Negative = short
    entry_price: Decimal,
    unrealized_pnl: Decimal,
    realized_pnl: Decimal,
    updated_at: DateTime<Utc>,
}
```

**Account:**
```rust
struct Account {
    balance: Decimal,
    equity: Decimal,
    positions: Vec<Position>,
    initial_balance: Decimal,
    total_pnl: Decimal,
    win_rate: f64,
    total_trades: i32,
    created_at: DateTime<Utc>,
}
```

---

## 7. Mathematical Formulas Summary

### 7.1 Order Book Imbalance

```
imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
Range: [-1.0, 1.0]
  > 0: More buying pressure
  < 0: More selling pressure
```

### 7.2 Weighted Imbalance

```
weighted_imbalance = Σ(bid_qty * weight) - Σ(ask_qty * weight)
                     ──────────────────────────────────────────
                     Σ(bid_qty * weight) + Σ(ask_qty * weight)

weight = 1 / distance_from_mid
```

### 7.3 Spread in Basis Points

```
spread_bps = (best_ask - best_bid) / mid_price * 10000
mid_price = (best_bid + best_ask) / 2
```

### 7.4 Volatility (Annualized)

```
returns[i] = ln(price[i+1] / price[i])
volatility = std(returns) * sqrt(252 * 24 * 60)
```

### 7.5 Momentum

```
momentum = (P_current - P_start) / P_start
```

### 7.6 Imbalance Momentum (Linear Regression)

```
slope = corr(x, y) * (std(y) / std(x))
where x = [0, 1, 2, ...], y = imbalance values
```

### 7.7 Z-Score Normalization

```
z = (value - mean) / std
```

### 7.8 Slippage

```
slippage = price * bps / 10000
fill_price = price ± slippage
```

### 7.9 Trading Fee

```
fee = price * quantity * fee_rate
fee_rate = 0.001 (0.1%)
```

### 7.10 Position P&L

**Long Position Closing:**
```
pnl = (exit_price - entry_price) * quantity - fee
```

**Short Position Closing:**
```
pnl = (entry_price - exit_price) * quantity - fee
```

### 7.11 Average Entry Price

```
new_avg = (old_price * old_qty + new_price * new_qty) / total_qty
```

---

## 8. Critical Business Rules

### 8.1 Signal Generation Rules

1. **Imbalance must exceed threshold** (default: 0.3)
2. **Spread must be within limit** (default: 10 bps)
3. **Imbalance must persist** for N ticks (default: 3)
4. **Confidence must exceed minimum** (default: 0.6)
5. **Momentum must confirm** (if enabled)
6. **Imbalance momentum must align** with imbalance direction

### 8.2 Position Sizing Rules

1. **Base size:** 10% of account balance (in dollars)
2. **Volatility adjustments:**
   - Low vol (z < -1.0): Multiply by 1.5
   - High vol (z > 1.0): Multiply by 0.5
3. **Adding to position:** Halve the quantity
4. **Precision:** Round to 6 decimal places

### 8.3 Risk Management Rules

1. **Maximum position size** per symbol (configurable)
2. **Daily trade limit** (configurable)
3. **Position limit enforcement** before order submission
4. **Quantity adjustment** if exceeding limits

### 8.4 Shabbat Trading Rules

1. **Pause trading:** Friday sunset - 18 minutes
2. **Resume trading:** Saturday sunset
3. **Timezone-aware:** Based on geographic location
4. **Astronomical calculation:** Actual sunset times, not fixed hours

### 8.5 Order Execution Rules

1. **Market orders only** (currently)
2. **Slippage:** 5 basis points default
3. **Fee structure:** 0.1% (taker fee)
4. **P&L calculation:** Only when closing positions
5. **Position updates:** Immediate after trade execution

---

## 9. Dependencies and Libraries to Replace

### Python → Rust Equivalents

| Python Library | Rust Crate | Purpose |
|---------------|------------|---------|
| `numpy` | `ndarray` | Array operations, statistics |
| `pydantic` | `serde` | Data validation and serialization |
| `structlog` | `tracing` | Structured logging |
| `aiosqlite` | `sqlx` or `rusqlite` | Async SQLite |
| `httpx` | `reqwest` | HTTP client |
| `decimal` | `rust_decimal` | Precise decimal arithmetic |
| `astral` | `sun-times` or custom | Astronomical calculations |
| `pytz` | `chrono-tz` | Timezone handling |
| `datetime` | `chrono` | Date/time operations |

---

## 10. Performance Considerations for Rust

### 10.1 Precision Requirements

- Use `rust_decimal::Decimal` for all financial calculations
- Use `f64` for statistical calculations (volatility, correlation)
- Round quantities to 6 decimal places
- Round prices to symbol-specific precision

### 10.2 Memory Management

- Use `VecDeque` for rolling windows (efficient push/pop)
- Pre-allocate collections when size is known
- Use `HashMap` for symbol-based state tracking
- Consider memory pooling for frequently allocated objects

### 10.3 Async Operations

- Database operations should be async
- HTTP requests (risk checks) should be async
- Order execution should be async
- Use tokio runtime

### 10.4 Data Validation

- Validate all inputs at API boundaries
- Use type system to enforce invariants
- Validate enum conversions from strings
- Check for division by zero in calculations

---

## 11. Testing Requirements

### 11.1 Unit Tests Needed

1. **Strategy Logic:**
   - Confidence calculation
   - Position sizing
   - Signal generation
   - Persistence tracking

2. **Microstructure Features:**
   - Volatility calculation
   - Momentum calculation
   - Imbalance momentum
   - Normalization

3. **Broker Logic:**
   - Fee calculation
   - Slippage application
   - P&L calculation
   - Position updates

4. **Scheduler:**
   - Sunset calculation
   - Shabbat detection
   - Event timing

5. **Database:**
   - CRUD operations
   - Data type conversions
   - Query filtering

### 11.2 Integration Tests Needed

1. End-to-end signal generation
2. Order submission and execution flow
3. Position lifecycle (open → add → close)
4. Database persistence
5. Risk check integration

### 11.3 Edge Cases to Test

1. Division by zero (prices, volumes)
2. Empty rolling windows
3. Position flips (long → short)
4. Exact position closes
5. Shabbat week transitions (Sunday edge case)
6. Negative spreads (crossed book)
7. Zero volumes
8. Decimal precision limits

---

## 12. Configuration Parameters Summary

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `imbalance_threshold` | f64 | 0.3 | Min imbalance for signal |
| `min_confidence` | f64 | 0.6 | Min confidence for signal |
| `low_vol_multiplier` | f64 | 1.5 | Size mult in low vol |
| `high_vol_multiplier` | f64 | 0.5 | Size mult in high vol |
| `vol_threshold_low` | f64 | -1.0 | Low vol z-score |
| `vol_threshold_high` | f64 | 1.0 | High vol z-score |
| `momentum_threshold` | f64 | 0.0001 | Min momentum |
| `persistence_required` | i32 | 3 | Ticks to persist |
| `base_position_pct` | f64 | 0.1 | % of balance |
| `max_spread_bps` | f64 | 10.0 | Max spread to trade |
| `window_size` | usize | 100 | Main rolling window |
| `volatility_window` | usize | 20 | Vol calculation window |
| `momentum_window` | usize | 10 | Momentum window |
| `slippage_bps` | i32 | 5 | Market order slippage |
| `shabbat_buffer` | i32 | 18 | Minutes before sunset |

---

## 13. State Management Requirements

### 13.1 Per-Symbol State

```rust
struct SymbolState {
    // Strategy state
    imbalance_streak: i32,
    last_imbalance_sign: i32,

    // Feature state
    mid_prices: VecDeque<f64>,
    imbalances: VecDeque<f64>,
    timestamps: VecDeque<i64>,

    // Statistics
    imbalance_mean: f64,
    imbalance_std: f64,
    volatility_mean: f64,
    volatility_std: f64,

    // Position
    position: Option<Position>,
}
```

### 13.2 Global State

```rust
struct StrategyEngine {
    config: StrategyConfig,
    symbols: HashMap<String, SymbolState>,
    account: Account,
    scheduler: ShabbatScheduler,
    db: Database,
    order_counter: AtomicI32,
}
```

---

## 14. Implementation Priority

### Phase 1: Core Data Structures
1. Implement all enums (Side, OrderType, OrderStatus)
2. Implement all data models
3. Implement Decimal conversions
4. Set up database schema

### Phase 2: Microstructure Features
1. Rolling window management
2. Volatility calculation
3. Momentum calculations
4. Feature normalization
5. Unit tests

### Phase 3: Strategy Logic
1. Confidence calculation
2. Position sizing
3. Signal generation
4. Persistence tracking
5. Unit tests

### Phase 4: Broker Logic
1. Order management
2. Trade execution with slippage
3. Position updates
4. P&L calculation
5. Risk checks
6. Unit tests

### Phase 5: Scheduler
1. Sunset calculations
2. Shabbat detection
3. Event timing
4. Unit tests

### Phase 6: Integration
1. Connect all components
2. Database persistence
3. End-to-end tests
4. Performance optimization

---

## Conclusion

This document provides a comprehensive reference for porting the Python strategy engine to Rust. All mathematical formulas, algorithms, business rules, and data structures have been documented with precise specifications to ensure accurate implementation in Rust.
