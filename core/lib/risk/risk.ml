(** Risk Management Engine

    Validates orders against risk limits before execution.
    Implements circuit breakers and position limits.
*)

open Types

(** Risk configuration *)
type config = {
  max_position_size: Decimal.t;     (** Max position per symbol *)
  max_total_exposure: Decimal.t;    (** Max total exposure across all positions *)
  max_drawdown_pct: float;          (** Max drawdown as percentage of initial balance *)
  max_loss_per_trade: Decimal.t;    (** Max loss on a single trade *)
  max_orders_per_minute: int;       (** Rate limiting *)
  min_balance_threshold: Decimal.t; (** Minimum balance to keep trading *)
} [@@deriving yojson, show]

let default_config = {
  max_position_size = Decimal.of_float 1.0;
  max_total_exposure = Decimal.of_float 5.0;
  max_drawdown_pct = 0.05;
  max_loss_per_trade = Decimal.of_float 100.0;
  max_orders_per_minute = 60;
  min_balance_threshold = Decimal.of_float 100.0;
}

(** Risk check result *)
type check_result =
  | Approved
  | Rejected of string
  | RequiresAdjustment of { original: Decimal.t; adjusted: Decimal.t; reason: string }
[@@deriving yojson, show]

(** Risk engine state *)
type t = {
  config: config;
  orders_last_minute: int;
  last_order_time: float;
  circuit_breaker_active: bool;
  circuit_breaker_reason: string option;
}

let create ?(config=default_config) () = {
  config;
  orders_last_minute = 0;
  last_order_time = 0.0;
  circuit_breaker_active = false;
  circuit_breaker_reason = None;
}

(** Check if circuit breaker is active *)
let is_circuit_breaker_active t = t.circuit_breaker_active

(** Activate circuit breaker *)
let activate_circuit_breaker reason t =
  { t with circuit_breaker_active = true; circuit_breaker_reason = Some reason }

(** Deactivate circuit breaker *)
let deactivate_circuit_breaker t =
  { t with circuit_breaker_active = false; circuit_breaker_reason = None }

(** Check position size limit *)
let check_position_size ~symbol ~current_position ~order_qty ~order_side config =
  let new_position = match order_side with
    | Buy -> Decimal.(current_position + order_qty)
    | Sell -> Decimal.(current_position - order_qty)
  in
  let abs_position = Decimal.abs new_position in
  if Decimal.(abs_position > config.max_position_size) then
    let allowed = Decimal.(config.max_position_size - Decimal.abs current_position) in
    if Decimal.(allowed > zero) then
      RequiresAdjustment {
        original = order_qty;
        adjusted = allowed;
        reason = Printf.sprintf "Position limit: max %.4f, reducing order to %.4f"
          (Decimal.to_float config.max_position_size)
          (Decimal.to_float allowed)
      }
    else
      Rejected (Printf.sprintf "Position limit reached for %s" symbol)
  else
    Approved

(** Check total exposure *)
let check_total_exposure ~positions ~order_value config =
  let total_exposure = List.fold_left (fun acc pos ->
    Decimal.(acc + Decimal.abs pos.quantity)
  ) Decimal.zero positions in
  let new_exposure = Decimal.(total_exposure + order_value) in
  if Decimal.(new_exposure > config.max_total_exposure) then
    Rejected (Printf.sprintf "Total exposure would exceed limit: %.4f > %.4f"
      (Decimal.to_float new_exposure)
      (Decimal.to_float config.max_total_exposure))
  else
    Approved

(** Check drawdown *)
let check_drawdown account config =
  let drawdown = Decimal.((account.initial_balance - account.equity) / account.initial_balance) in
  let drawdown_pct = Decimal.to_float drawdown in
  if drawdown_pct > config.max_drawdown_pct then
    Rejected (Printf.sprintf "Drawdown limit exceeded: %.2f%% > %.2f%%"
      (drawdown_pct *. 100.0)
      (config.max_drawdown_pct *. 100.0))
  else
    Approved

(** Check minimum balance *)
let check_min_balance account config =
  if Decimal.(account.balance < config.min_balance_threshold) then
    Rejected (Printf.sprintf "Balance below minimum threshold: %.2f < %.2f"
      (Decimal.to_float account.balance)
      (Decimal.to_float config.min_balance_threshold))
  else
    Approved

(** Check rate limit *)
let check_rate_limit t =
  if t.orders_last_minute >= t.config.max_orders_per_minute then
    Rejected "Rate limit exceeded"
  else
    Approved

(** Update rate limit counter *)
let update_rate_limit t =
  let now = Unix.gettimeofday () in
  if now -. t.last_order_time > 60.0 then
    { t with orders_last_minute = 1; last_order_time = now }
  else
    { t with orders_last_minute = t.orders_last_minute + 1; last_order_time = now }

(** Main risk check function *)
let check_order ~(order:order) ~account ~positions ~current_position t =
  (* Check circuit breaker first *)
  if t.circuit_breaker_active then
    Rejected (Printf.sprintf "Circuit breaker active: %s"
      (Option.value t.circuit_breaker_reason ~default:"unknown"))
  else
    (* Run all checks *)
    let checks = [
      (fun () -> check_rate_limit t);
      (fun () -> check_min_balance account t.config);
      (fun () -> check_drawdown account t.config);
      (fun () -> check_position_size
        ~symbol:order.symbol
        ~current_position
        ~order_qty:order.quantity
        ~order_side:order.side
        t.config);
    ] in
    let rec run_checks = function
      | [] -> Approved
      | check :: rest ->
        match check () with
        | Approved -> run_checks rest
        | other -> other
    in
    run_checks checks

(** Calculate maximum safe order size given current state *)
let max_safe_order_size ~symbol ~side ~account ~current_position t =
  let position_room = Decimal.(t.config.max_position_size - Decimal.abs current_position) in
  let balance_room = Decimal.(account.balance * of_float 0.95) in  (* Keep 5% buffer *)
  let open Decimal in
  if position_room < zero then zero
  else if position_room < balance_room then position_room
  else balance_room
