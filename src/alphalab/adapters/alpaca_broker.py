"""Alpaca order-execution adapter — paper by default, structurally safe.

This adapter is the one place in the system that can move an account. Every
safety property it owns is structural rather than advisory, because a comment
that says "do not trade live" has never stopped a single order:

* **Paper is the only reachable default.** The constructor obtains its base URL
  through :meth:`AlpacaSettings.trading_base_url`, which *raises*
  :class:`LiveTradingNotAcknowledgedError` when the configured endpoint is live
  and ``acknowledge_live`` was not passed. Reaching live therefore requires two
  deliberate acts (env var *and* flag); it cannot happen by inherited shell
  state.
* **Idempotency.** Every order carries a deterministic ``client_order_id``
  derived from the strategy, date, ticker and side. Re-running a cycle produces
  the *same* ids, so Alpaca rejects the duplicates (HTTP 409/422). The adapter
  treats that rejection as success, which turns an accidental double invocation
  into a no-op instead of double exposure.
* **Kill switch.** A :class:`~alphalab.adapters.kill_switch.KillSwitch` is read
  immediately before *every* POST. If engaged, no request is made and the
  attempt is reported as skipped.
* **Account-flag and market-clock gating.** Blocked accounts and a closed
  market both short-circuit emission with an actionable message rather than a
  rejected or dangling order.

Long-only, market, ``time_in_force=day`` is the only shape this adapter will
send. Shorting is already impossible upstream (the planner refuses it); this
adapter does not reintroduce the option.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol

from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.domain.execution import OrderIntent
from alphalab.domain.types import Side, Ticker

__all__ = [
    "AccountSnapshot",
    "AlpacaBroker",
    "BrokerOrder",
    "BrokerPosition",
    "MarketClock",
    "OrderOutcome",
    "OrderStatus",
    "WriteTransport",
]

#: Alpaca returns these when a client_order_id already exists; both are treated
#: as idempotent success rather than failure.
_DUPLICATE_STATUS = frozenset({409, 422})
_OK_STATUS = frozenset({200, 201})


class WriteTransport(Protocol):
    """HTTP transport that can both read and write.

    Extends the project's read-only :class:`~alphalab.adapters.edgar.Transport`
    shape with a ``post`` so the broker can submit orders. Kept a Protocol so
    tests inject a double and no unit test ever touches the network.
    """

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse: ...
    def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, object]
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class AccountSnapshot:
    """The account fields that gate or size execution."""

    equity: float
    buying_power: float
    trading_blocked: bool
    account_blocked: bool

    @property
    def is_blocked(self) -> bool:
        """Whether any flag forbids sending orders."""
        return self.trading_blocked or self.account_blocked


@dataclass(frozen=True)
class BrokerPosition:
    """One long position as reported by Alpaca."""

    ticker: Ticker
    shares: float
    price: float
    avg_entry_price: float | None = None


@dataclass(frozen=True)
class BrokerOrder:
    """One working (open/unfilled) order as reported by Alpaca."""

    ticker: Ticker
    side: Side
    shares: float
    client_order_id: str
    broker_order_id: str | None = None


@dataclass(frozen=True)
class MarketClock:
    """Whether the market is open right now, per Alpaca's clock endpoint."""

    is_open: bool
    timestamp: datetime | None = None
    next_open: datetime | None = None
    next_close: datetime | None = None


class OrderStatus(StrEnum):
    """Terminal classification of a single submission attempt."""

    SUBMITTED = "submitted"
    DUPLICATE = "duplicate"
    HALTED = "halted"
    FAILED = "failed"


@dataclass(frozen=True)
class OrderOutcome:
    """What happened to one intent, with the broker id when there is one."""

    intent: OrderIntent
    status: OrderStatus
    client_order_id: str
    broker_order_id: str | None = None
    detail: str | None = None

    @property
    def reached_broker(self) -> bool:
        """Whether the order now exists at the broker (new or pre-existing)."""
        return self.status in (OrderStatus.SUBMITTED, OrderStatus.DUPLICATE)


class AlpacaBroker:
    """Reads account state and submits long-only market orders to Alpaca."""

    def __init__(
        self,
        settings: AlpacaSettings,
        *,
        transport: WriteTransport,
        kill_switch: KillSwitch | None = None,
        acknowledge_live: bool = False,
    ) -> None:
        # Obtaining the base URL is what enforces the paper-only default: for a
        # live endpoint without acknowledgement this raises before any field is
        # stored, so a live broker cannot even be constructed by accident.
        self._base_url = settings.trading_base_url(acknowledge_live=acknowledge_live)
        self._settings = settings
        self._transport = transport
        self._kill_switch = kill_switch or KillSwitch()

    # -- reads ------------------------------------------------------------

    def account(self) -> AccountSnapshot:
        """Fetch the account snapshot (equity, buying power, block flags)."""
        payload = self._get("/v2/account")
        return AccountSnapshot(
            equity=_as_float(payload.get("equity")),
            buying_power=_as_float(payload.get("buying_power")),
            trading_blocked=bool(payload.get("trading_blocked")),
            account_blocked=bool(payload.get("account_blocked")),
        )

    def positions(self) -> dict[Ticker, BrokerPosition]:
        """Fetch current positions keyed by ticker."""
        payload = self._get_list("/v2/positions")
        out: dict[Ticker, BrokerPosition] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            if not isinstance(symbol, str):
                continue
            ticker = Ticker(symbol)
            avg_entry = _as_float(row.get("avg_entry_price"))
            out[ticker] = BrokerPosition(
                ticker=ticker,
                shares=_as_float(row.get("qty")),
                price=_as_float(row.get("current_price")),
                avg_entry_price=avg_entry if avg_entry > 0.0 else None,
            )
        return out

    def clock(self) -> MarketClock:
        """Fetch the market clock (whether the market is currently open)."""
        payload = self._get("/v2/clock")
        return MarketClock(
            is_open=bool(payload.get("is_open")),
            timestamp=_as_datetime(payload.get("timestamp")),
            next_open=_as_datetime(payload.get("next_open")),
            next_close=_as_datetime(payload.get("next_close")),
        )

    def open_orders(self) -> tuple[BrokerOrder, ...]:
        """Fetch working orders from ``/v2/orders?status=open``."""
        rows = self._get_list("/v2/orders?status=open")
        out: list[BrokerOrder] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            side_raw = str(row.get("side", "")).lower()
            if not isinstance(symbol, str) or side_raw not in ("buy", "sell"):
                continue
            qty = _as_float(row.get("qty"))
            filled = _as_float(row.get("filled_qty"))
            remaining = max(0.0, qty - filled) if qty > 0.0 else 0.0
            if remaining <= 0.0:
                continue
            coid = row.get("client_order_id")
            boid = row.get("id")
            out.append(
                BrokerOrder(
                    ticker=Ticker(symbol),
                    side=Side.BUY if side_raw == "buy" else Side.SELL,
                    shares=remaining,
                    client_order_id=coid if isinstance(coid, str) else "",
                    broker_order_id=boid if isinstance(boid, str) else None,
                )
            )
        return tuple(out)

    def open_order_ids(self) -> frozenset[str]:
        """Return the set of ``client_order_id`` values already at the broker.

        Lets a caller see which intents are already live before submitting, a
        second idempotency guard on top of Alpaca's own duplicate rejection.
        """
        rows = self._get_list("/v2/orders?status=all")
        ids = {
            row["client_order_id"]
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("client_order_id"), str)
        }
        return frozenset(ids)

    # -- writes -----------------------------------------------------------

    def submit(self, intent: OrderIntent, *, strategy: str, as_of: date) -> OrderOutcome:
        """Submit one long market order, idempotently and under the kill switch.

        The kill switch is read here, immediately before the POST, so flipping
        it takes effect on the very next order. A blocked switch means no
        request is made at all.
        """
        coid = intent.client_order_id(strategy=strategy, as_of=as_of)
        if self._kill_switch.engaged():
            return OrderOutcome(
                intent=intent,
                status=OrderStatus.HALTED,
                client_order_id=coid,
                detail=self._kill_switch.state().reason(),
            )
        if intent.side is not Side.BUY and intent.side is not Side.SELL:  # pragma: no cover
            raise ValueError(f"unsupported side {intent.side!r}")
        body = _order_body(intent, coid)
        response = self._transport.post(
            f"{self._base_url}/v2/orders",
            headers=self._settings.auth_headers(),
            json=body,
        )
        return _classify(intent, coid, response)

    # -- transport helpers ------------------------------------------------

    def _get(self, path: str) -> dict[str, object]:
        response = self._transport.get(
            f"{self._base_url}{path}", headers=self._settings.auth_headers()
        )
        if response.status_code not in _OK_STATUS:
            raise RuntimeError(f"Alpaca GET {path} failed (status={response.status_code})")
        return response.json if isinstance(response.json, dict) else {}

    def _get_list(self, path: str) -> list[object]:
        response = self._transport.get(
            f"{self._base_url}{path}", headers=self._settings.auth_headers()
        )
        if response.status_code not in _OK_STATUS:
            raise RuntimeError(f"Alpaca GET {path} failed (status={response.status_code})")
        return response.json if isinstance(response.json, list) else []


def _order_body(intent: OrderIntent, client_order_id: str) -> dict[str, object]:
    """Build the Alpaca order payload: long-only, market, day.

    ``qty`` is sent as an integer string because the planner emits whole shares
    by default; Alpaca accepts a decimal string, and formatting it ourselves
    keeps the wire value explicit and testable.
    """
    return {
        "symbol": str(intent.ticker),
        "qty": _format_qty(intent.shares),
        "side": intent.side.value,
        "type": "market",
        "time_in_force": "day",
        "client_order_id": client_order_id,
    }


def _format_qty(shares: float) -> str:
    """Render a share count without a trailing ``.0`` when it is whole."""
    if shares == int(shares):
        return str(int(shares))
    return f"{round(shares, 4):.4f}".rstrip("0").rstrip(".")


def _classify(intent: OrderIntent, coid: str, response: HttpResponse) -> OrderOutcome:
    """Map an HTTP response to an :class:`OrderOutcome`.

    A duplicate (409/422) is success: the deterministic id already exists, so
    the desired state is already true. Anything else non-2xx is a failure with
    the broker's message preserved for the audit log.
    """
    if response.status_code in _OK_STATUS:
        body = response.json if isinstance(response.json, dict) else {}
        broker_id = body.get("id")
        return OrderOutcome(
            intent=intent,
            status=OrderStatus.SUBMITTED,
            client_order_id=coid,
            broker_order_id=broker_id if isinstance(broker_id, str) else None,
        )
    if response.status_code in _DUPLICATE_STATUS:
        return OrderOutcome(
            intent=intent,
            status=OrderStatus.DUPLICATE,
            client_order_id=coid,
            detail=_message(response),
        )
    return OrderOutcome(
        intent=intent,
        status=OrderStatus.FAILED,
        client_order_id=coid,
        detail=f"status={response.status_code}: {_message(response)}",
    )


def _message(response: HttpResponse) -> str:
    """Extract Alpaca's error message, falling back to the raw payload."""
    body = response.json
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, str):
            return message
        return json.dumps(body, sort_keys=True)
    return str(body)


def _as_float(value: object) -> float:
    """Parse Alpaca's string-encoded numerics; absent/garbage becomes 0.0."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _as_datetime(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp from Alpaca; return None if absent/invalid."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
