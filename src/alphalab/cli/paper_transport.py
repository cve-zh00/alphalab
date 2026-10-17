"""Concrete httpx transport for the paper broker, isolated from the logic.

The broker and runner speak only to the
:class:`~alphalab.adapters.alpaca_broker.WriteTransport` protocol, so this is
the single file that actually opens a socket. Keeping it apart means every unit
test injects a double and the network lives in exactly one, obvious place.
"""

from __future__ import annotations

import httpx

from alphalab.adapters.edgar import HttpResponse

__all__ = ["HttpxWriteTransport"]


class HttpxWriteTransport:
    """A ``WriteTransport`` backed by a single httpx client."""

    def __init__(self, *, timeout: float = 30.0, retries: int = 3) -> None:
        transport = httpx.HTTPTransport(retries=retries)
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        response = self._client.get(url, headers=headers)
        return HttpResponse(status_code=response.status_code, json=_body(response))  # type: ignore[arg-type]

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> HttpResponse:
        response = self._client.post(url, headers=headers, json=json)
        return HttpResponse(status_code=response.status_code, json=_body(response))  # type: ignore[arg-type]


def _body(response: httpx.Response) -> object:
    """Parse a JSON body, tolerating empty/non-JSON responses.

    Returns the parsed value unchanged — a ``dict`` for account/clock/order
    endpoints and a ``list`` for positions/orders — so the broker can branch on
    the shape it expects. Alpaca's position and order endpoints return arrays,
    which the read-only :class:`HttpResponse` field is too narrow to type; the
    established project pattern is to pass them through with a type-ignore.
    """
    try:
        parsed = response.json()
    except ValueError:
        return {}
    return parsed
