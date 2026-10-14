"""Unit tests for the Alpaca execution broker.

Every test uses a transport double; none touches the network. The focus is the
broker's non-negotiable safety properties: paper-only default, idempotent
treatment of duplicate orders, the kill switch, and that no POST is issued when
emission is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pytest

from alphalab.adapters.alpaca_broker import (
    AlpacaBroker,
    OrderStatus,
    WriteTransport,
)
from alphalab.adapters.alpaca_config import (
    AlpacaEndpoint,
    AlpacaSettings,
    LiveTradingNotAcknowledgedError,
)
from alphalab.adapters.edgar import HttpResponse
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.domain.execution import OrderIntent
from alphalab.domain.types import Side, Ticker

_AS_OF = date(2024, 1, 2)
_STRATEGY = "rank_ensemble"


def _paper_settings() -> AlpacaSettings:
    return AlpacaSettings(
        key_id="paper-key",
        secret_key="super-secret-value",
        endpoint=AlpacaEndpoint.PAPER,
    )


def _live_settings() -> AlpacaSettings:
    return AlpacaSettings(
        key_id="live-key",
        secret_key="super-secret-value",
        endpoint=AlpacaEndpoint.LIVE,
    )


def _buy(ticker: str = "AAPL", shares: float = 10.0) -> OrderIntent:
    return OrderIntent(ticker=Ticker(ticker), side=Side.BUY, shares=shares, reference_price=100.0)


@dataclass
class FakeTransport:
    """A WriteTransport double recording every call and replaying scripted responses."""

    get_responses: dict[str, HttpResponse] = field(default_factory=dict)
    post_response: HttpResponse = field(
        default_factory=lambda: HttpResponse(status_code=200, json={"id": "srv-1"})
    )
    posts: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    gets: list[str] = field(default_factory=list)

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        self.gets.append(url)
        for suffix, response in self.get_responses.items():
            if url.endswith(suffix):
                return response
        return HttpResponse(status_code=200, json={})

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> HttpResponse:
        self.posts.append((url, json))
        return self.post_response


def _broker(
    transport: WriteTransport,
    *,
    settings: AlpacaSettings | None = None,
    kill_switch: KillSwitch | None = None,
    acknowledge_live: bool = False,
) -> AlpacaBroker:
    return AlpacaBroker(
        settings or _paper_settings(),
        transport=transport,
        kill_switch=kill_switch,
        acknowledge_live=acknowledge_live,
    )


# -- paper-only default ---------------------------------------------------


def test_live_endpoint_without_acknowledge_is_rejected() -> None:
    # Arrange
    transport = FakeTransport()

    # Act / Assert — constructing against live without acknowledgement raises.
    with pytest.raises(LiveTradingNotAcknowledgedError):
        _broker(transport, settings=_live_settings())


def test_live_endpoint_with_acknowledge_is_allowed() -> None:
    # Arrange
    transport = FakeTransport()

    # Act — explicit acknowledgement is the only way to reach live.
    broker = _broker(transport, settings=_live_settings(), acknowledge_live=True)

    # Assert — it constructed and targets the live host.
    broker.clock()
    assert transport.gets
    assert transport.gets[0].startswith("https://api.alpaca.markets")


def test_paper_is_the_default_host() -> None:
    # Arrange
    transport = FakeTransport()

    # Act
    broker = _broker(transport)
    broker.clock()

    # Assert
    assert transport.gets[0].startswith("https://paper-api.alpaca.markets")


# -- reads ----------------------------------------------------------------


def test_account_parses_flags_and_numerics() -> None:
    # Arrange
    transport = FakeTransport(
        get_responses={
            "/v2/account": HttpResponse(
                status_code=200,
                json={
                    "equity": "100000.5",
                    "buying_power": "200000",
                    "trading_blocked": False,
                    "account_blocked": False,
                },
            )
        }
    )

    # Act
    account = _broker(transport).account()

    # Assert
    assert account.equity == pytest.approx(100000.5)
    assert account.buying_power == pytest.approx(200000.0)
    assert not account.is_blocked


def test_positions_keyed_by_ticker() -> None:
    # Arrange
    transport = FakeTransport(
        get_responses={
            "/v2/positions": HttpResponse(
                status_code=200,
                json=[  # type: ignore[arg-type] - Alpaca returns an array here
                    {"symbol": "AAPL", "qty": "10", "current_price": "190.5"},
                ],
            )
        }
    )

    # Act
    positions = _broker(transport).positions()

    # Assert
    assert positions[Ticker("AAPL")].shares == pytest.approx(10.0)
    assert positions[Ticker("AAPL")].price == pytest.approx(190.5)


def test_open_order_ids_collects_client_order_ids() -> None:
    # Arrange
    transport = FakeTransport(
        get_responses={
            "/v2/orders?status=all": HttpResponse(
                status_code=200,
                json=[  # type: ignore[arg-type]
                    {"client_order_id": "rank_ensemble-2024-01-02-AAPL-buy"},
                    {"client_order_id": "rank_ensemble-2024-01-02-MSFT-buy"},
                ],
            )
        }
    )

    # Act
    ids = _broker(transport).open_order_ids()

    # Assert
    assert "rank_ensemble-2024-01-02-AAPL-buy" in ids
    assert len(ids) == 2


# -- idempotency ----------------------------------------------------------


def test_submit_uses_deterministic_client_order_id() -> None:
    # Arrange
    transport = FakeTransport()
    broker = _broker(transport)
    intent = _buy()

    # Act
    broker.submit(intent, strategy=_STRATEGY, as_of=_AS_OF)

    # Assert — the id is the planner's deterministic one, enabling idempotency.
    _, body = transport.posts[0]
    assert body["client_order_id"] == "rank_ensemble-2024-01-02-AAPL-buy"
    assert body["type"] == "market"
    assert body["time_in_force"] == "day"
    assert body["side"] == "buy"


@pytest.mark.parametrize("status", [409, 422])
def test_duplicate_order_is_treated_as_idempotent_success(status: int) -> None:
    # Arrange — Alpaca says the client_order_id already exists.
    transport = FakeTransport(
        post_response=HttpResponse(
            status_code=status, json={"message": "client_order_id must be unique"}
        )
    )
    broker = _broker(transport)

    # Act
    outcome = broker.submit(_buy(), strategy=_STRATEGY, as_of=_AS_OF)

    # Assert — a duplicate is success, not failure; the state is already true.
    assert outcome.status is OrderStatus.DUPLICATE
    assert outcome.reached_broker


def test_running_the_same_submission_twice_does_not_double_expose() -> None:
    # Arrange — first POST creates, second returns the duplicate rejection.
    class _OnceThenDuplicate:
        def __init__(self) -> None:
            self.calls = 0
            self.posts: list[dict[str, object]] = []

        def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
            return HttpResponse(status_code=200, json={})

        def post(
            self, url: str, *, headers: dict[str, str], json: dict[str, object]
        ) -> HttpResponse:
            self.calls += 1
            self.posts.append(json)
            if self.calls == 1:
                return HttpResponse(status_code=200, json={"id": "srv-1"})
            return HttpResponse(status_code=422, json={"message": "duplicate"})

    transport = _OnceThenDuplicate()
    broker = _broker(transport)
    intent = _buy()

    # Act — submit the identical intent twice.
    first = broker.submit(intent, strategy=_STRATEGY, as_of=_AS_OF)
    second = broker.submit(intent, strategy=_STRATEGY, as_of=_AS_OF)

    # Assert — both carry the same id; the second is an idempotent no-op.
    assert first.status is OrderStatus.SUBMITTED
    assert second.status is OrderStatus.DUPLICATE
    assert transport.posts[0]["client_order_id"] == transport.posts[1]["client_order_id"]


def test_server_error_is_a_failure_not_a_duplicate() -> None:
    # Arrange
    transport = FakeTransport(post_response=HttpResponse(status_code=500, json={"message": "boom"}))

    # Act
    outcome = _broker(transport).submit(_buy(), strategy=_STRATEGY, as_of=_AS_OF)

    # Assert
    assert outcome.status is OrderStatus.FAILED
    assert not outcome.reached_broker


# -- kill switch ----------------------------------------------------------


def test_kill_switch_file_blocks_every_post(tmp_path: object) -> None:
    # Arrange — a present halt file engages the switch.
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    halt = tmp_path / "HALT"
    halt.write_text("stop", encoding="utf-8")
    transport = FakeTransport()
    broker = _broker(transport, kill_switch=KillSwitch(halt_path=halt, env={}))

    # Act
    outcome = broker.submit(_buy(), strategy=_STRATEGY, as_of=_AS_OF)

    # Assert — zero POSTs, outcome reports the halt.
    assert transport.posts == []
    assert outcome.status is OrderStatus.HALTED
    assert not outcome.reached_broker


def test_kill_switch_env_var_blocks_every_post(tmp_path: object) -> None:
    # Arrange — env var engaged, no halt file.
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    transport = FakeTransport()
    switch = KillSwitch(halt_path=tmp_path / "HALT", env={"ALPHALAB_PAPER_HALT": "1"})
    broker = _broker(transport, kill_switch=switch)

    # Act
    outcome = broker.submit(_buy(), strategy=_STRATEGY, as_of=_AS_OF)

    # Assert
    assert transport.posts == []
    assert outcome.status is OrderStatus.HALTED


def test_kill_switch_clear_allows_the_post(tmp_path: object) -> None:
    # Arrange
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    transport = FakeTransport()
    switch = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = _broker(transport, kill_switch=switch)

    # Act
    outcome = broker.submit(_buy(), strategy=_STRATEGY, as_of=_AS_OF)

    # Assert
    assert len(transport.posts) == 1
    assert outcome.status is OrderStatus.SUBMITTED


# -- secret safety --------------------------------------------------------


def test_secret_never_appears_in_outcome_or_exception() -> None:
    # Arrange — a failing POST whose message we render into the outcome.
    secret = "super-secret-value"
    transport = FakeTransport(
        post_response=HttpResponse(status_code=400, json={"message": "bad request"})
    )
    broker = _broker(transport)

    # Act
    outcome = broker.submit(_buy(), strategy=_STRATEGY, as_of=_AS_OF)

    # Assert — nothing the broker surfaces leaks the secret.
    assert secret not in (outcome.detail or "")
    assert secret not in repr(outcome)


# -- defensive branches ---------------------------------------------------


def test_account_get_failure_raises_runtime_error() -> None:
    # Arrange
    transport = FakeTransport(get_responses={"/v2/account": HttpResponse(status_code=500, json={})})

    # Act / Assert
    with pytest.raises(RuntimeError, match="GET /v2/account failed"):
        _broker(transport).account()


def test_positions_skips_malformed_rows() -> None:
    # Arrange — a non-dict row and a dict missing its symbol are both ignored.
    transport = FakeTransport(
        get_responses={
            "/v2/positions": HttpResponse(
                status_code=200,
                json=[  # type: ignore[arg-type]
                    "garbage",
                    {"qty": "5"},
                    {"symbol": "AAPL", "qty": "10", "current_price": "190.5"},
                ],
            )
        }
    )

    # Act
    positions = _broker(transport).positions()

    # Assert — only the well-formed row survives.
    assert list(positions) == [Ticker("AAPL")]


def test_failed_order_without_message_falls_back_to_payload() -> None:
    # Arrange — no "message" key, so the raw payload is serialised instead.
    transport = FakeTransport(post_response=HttpResponse(status_code=400, json={"code": 42}))

    # Act
    outcome = _broker(transport).submit(_buy(), strategy=_STRATEGY, as_of=_AS_OF)

    # Assert
    assert outcome.status is OrderStatus.FAILED
    assert "42" in (outcome.detail or "")


def test_fractional_quantity_is_formatted_as_repr() -> None:
    # Arrange — a fractional share count should not be truncated to an int.
    transport = FakeTransport()
    intent = OrderIntent(ticker=Ticker("AAPL"), side=Side.BUY, shares=1.5, reference_price=100.0)

    # Act
    _broker(transport).submit(intent, strategy=_STRATEGY, as_of=_AS_OF)

    # Assert
    _, body = transport.posts[0]
    assert body["qty"] == "1.5"
