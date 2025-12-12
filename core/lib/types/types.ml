(** Core types for QuantumFlow HFT

    This module defines the fundamental domain types using OCaml's algebraic
    data types to ensure correctness by construction. Invalid states are
    made unrepresentable through the type system.
*)

(** Decimal representation for prices and quantities
    We use a scaled integer representation for exact decimal arithmetic *)
module Decimal = struct
  type t = {
    value: int64;     (** Scaled value *)
    scale: int;       (** Number of decimal places *)
  } [@@deriving yojson, show, eq]

  let zero = { value = 0L; scale = 8 }

  let of_float ?(scale=8) f =
    let multiplier = Float.pow 10.0 (Float.of_int scale) in
    { value = Int64.of_float (f *. multiplier); scale }

  let to_float t =
    let divisor = Float.pow 10.0 (Float.of_int t.scale) in
    Int64.to_float t.value /. divisor

  let of_string ?(scale=8) s =
    of_float ~scale (Float.of_string s)

  let to_string t =
    Printf.sprintf "%.*f" t.scale (to_float t)

  let compare a b =
    (* Normalize to same scale before comparing *)
    let a_normalized = Int64.to_float a.value /. Float.pow 10.0 (Float.of_int a.scale) in
    let b_normalized = Int64.to_float b.value /. Float.pow 10.0 (Float.of_int b.scale) in
    Float.compare a_normalized b_normalized

  let ( + ) a b =
    if a.scale = b.scale then
      { value = Int64.add a.value b.value; scale = a.scale }
    else
      of_float ~scale:(max a.scale b.scale) (to_float a +. to_float b)

  let ( - ) a b =
    if a.scale = b.scale then
      { value = Int64.sub a.value b.value; scale = a.scale }
    else
      of_float ~scale:(max a.scale b.scale) (to_float a -. to_float b)

  let ( * ) a b =
    of_float ~scale:(max a.scale b.scale) (to_float a *. to_float b)

  let ( / ) a b =
    of_float ~scale:(max a.scale b.scale) (to_float a /. to_float b)

  let ( > ) a b = compare a b > 0
  let ( < ) a b = compare a b < 0
  let ( >= ) a b = compare a b >= 0
  let ( <= ) a b = compare a b <= 0
  let ( = ) a b = compare a b = 0

  let abs t = { t with value = Int64.abs t.value }
  let neg t = { t with value = Int64.neg t.value }
  let is_zero t = Int64.equal t.value 0L
  let is_positive t = (Int64.compare t.value 0L) > 0
  let is_negative t = (Int64.compare t.value 0L) < 0
end

(** Symbol representing a trading pair *)
module Symbol = struct
  type t = string [@@deriving yojson, show, eq]

  let of_string s = String.uppercase_ascii s
  let to_string t = t
end

(** Side of a trade or order *)
type side =
  | Buy
  | Sell
[@@deriving yojson, show, eq]

let opposite_side = function
  | Buy -> Sell
  | Sell -> Buy

(** Order type *)
type order_type =
  | Market
  | Limit of Decimal.t  (** Limit price *)
[@@deriving yojson, show, eq]

(** Time in force for orders *)
type time_in_force =
  | GTC  (** Good till cancelled *)
  | IOC  (** Immediate or cancel *)
  | FOK  (** Fill or kill *)
[@@deriving yojson, show, eq]

(** Order status *)
type order_status =
  | Pending
  | PartiallyFilled of Decimal.t  (** Filled quantity *)
  | Filled
  | Cancelled
  | Rejected of string  (** Rejection reason *)
[@@deriving yojson, show, eq]

(** Unique order identifier *)
module OrderId = struct
  type t = string [@@deriving yojson, show, eq, ord]

  let counter = ref 0

  let generate () =
    incr counter;
    Printf.sprintf "ORD-%d-%d" (int_of_float (Unix.gettimeofday () *. 1000.0)) !counter
end

(** An order in the system *)
type order = {
  id: OrderId.t;
  symbol: Symbol.t;
  side: side;
  order_type: order_type;
  quantity: Decimal.t;
  time_in_force: time_in_force;
  status: order_status;
  created_at: float;  (** Unix timestamp *)
  updated_at: float;
} [@@deriving yojson, show]

(** A trade execution *)
type trade = {
  trade_id: string;
  order_id: OrderId.t;
  symbol: Symbol.t;
  side: side;
  price: Decimal.t;
  quantity: Decimal.t;
  fee: Decimal.t;
  fee_asset: string;
  timestamp: float;
} [@@deriving yojson, show]

(** Position in a single symbol *)
type position = {
  symbol: Symbol.t;
  quantity: Decimal.t;  (** Positive for long, negative for short *)
  entry_price: Decimal.t;
  unrealized_pnl: Decimal.t;
  realized_pnl: Decimal.t;
  updated_at: float;
} [@@deriving yojson, show]

(** Account state *)
type account = {
  balance: Decimal.t;         (** Available balance in quote currency *)
  equity: Decimal.t;          (** Balance + unrealized P&L *)
  positions: position list;
  initial_balance: Decimal.t; (** Starting balance for P&L calculation *)
  created_at: float;
} [@@deriving yojson, show]

(** Price level in the order book *)
type price_level = {
  price: Decimal.t;
  quantity: Decimal.t;
} [@@deriving yojson, show]

(** Order book state received from market data *)
type orderbook_state = {
  symbol: Symbol.t;
  timestamp: int64;
  last_update_id: int64;
  bids: price_level list;
  asks: price_level list;
  mid_price: Decimal.t option;
  spread_bps: Decimal.t option;
  imbalance: Decimal.t option;
  weighted_imbalance: Decimal.t option;
} [@@deriving yojson, show]

(** Trading signal from strategy *)
type signal = {
  symbol: Symbol.t;
  side: side;
  confidence: float;      (** 0.0 to 1.0 *)
  suggested_size: Decimal.t;
  reason: string;
  timestamp: float;
} [@@deriving yojson, show]
