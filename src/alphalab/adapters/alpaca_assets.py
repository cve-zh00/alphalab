"""Alpaca assets adapter: the US-equity universe *including delisted names*.

Why this adapter exists
-----------------------
Every measurement in this project carries one declared debt: **survivorship
bias**. The price source used so far (Yahoo) does not serve delisted names, so
a point-in-time universe could never include companies that later vanished —
and that absence was shown to be the entire apparent edge of the strategy.

Alpaca's ``GET /v2/assets`` endpoint answers that gap directly: with
``status=inactive`` it returns assets that are **no longer tradable**, i.e.
delisted. Pulling both the active and inactive sets lets a backtest reconstruct
a universe that does not silently drop the losers, which is the single most
valuable thing this adapter can provide.

Design
------
* The one HTTP call sits behind the shared
  :class:`~alphalab.adapters.edgar.Transport` protocol, so tests use a double
  and never touch the network — the same pattern as the EDGAR and CIK adapters.
* Authentication goes exclusively through
  :meth:`AlpacaSettings.auth_headers`; the secret is never read, logged, or
  placed in an error message here.
* Reading assets places no orders, so the paper trading base URL is used
  **without** ``acknowledge_live``.
* Results are modelled as frozen value objects; the active/inactive counts are
  exposed explicitly because *that ratio is the magnitude of the survivorship
  bias* the project was carrying.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import Clock, Transport, _SystemClock
from alphalab.domain.types import Ticker

__all__ = [
    "AlpacaAsset",
    "AlpacaAssetsClient",
    "AssetStatus",
    "UniverseSnapshot",
]

_US_EQUITY = "us_equity"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Alpaca asset status strings. ``active`` means currently tradable; ``inactive``
#: is the delisted / no-longer-tradable set this project needs for de-biasing.
AssetStatus = str
_ACTIVE = "active"
_INACTIVE = "inactive"


@dataclass(frozen=True)
class AlpacaAsset:
    """One asset as reported by Alpaca, reduced to the fields that matter.

    ``status`` is kept verbatim from the API (``"active"`` / ``"inactive"``);
    :attr:`is_delisted` interprets it so callers do not hard-code the string.
    """

    symbol: str
    name: str
    exchange: str
    asset_class: str
    status: str
    tradable: bool
    fractionable: bool

    @property
    def is_delisted(self) -> bool:
        """Whether this asset is no longer tradable (delisted)."""
        return self.status == _INACTIVE

    @property
    def ticker(self) -> Ticker:
        """The symbol as a domain :data:`Ticker`."""
        return Ticker(self.symbol)


@dataclass(frozen=True)
class UniverseSnapshot:
    """A point-in-time view of the US-equity universe with delisted names kept.

    The whole point of this adapter is that :attr:`inactive` is non-empty: those
    are the companies a survivorship-biased source would have dropped.
    """

    active: tuple[AlpacaAsset, ...]
    inactive: tuple[AlpacaAsset, ...]

    @property
    def all_assets(self) -> tuple[AlpacaAsset, ...]:
        """Active and inactive assets together — the de-biased universe."""
        return self.active + self.inactive

    @property
    def active_count(self) -> int:
        """Number of currently tradable US equities."""
        return len(self.active)

    @property
    def inactive_count(self) -> int:
        """Number of delisted US equities — the survivors' missing counterpart."""
        return len(self.inactive)

    @property
    def survivorship_ratio(self) -> float:
        """Inactive-to-active ratio: the magnitude of the survivorship bias.

        Returns ``0.0`` when there are no active assets, avoiding a division by
        zero rather than reporting a misleading infinity.
        """
        if not self.active:
            return 0.0
        return self.inactive_count / self.active_count


class AlpacaAssetsClient:
    """Fetches the US-equity universe, active and delisted, from Alpaca.

    The transport is injected (same protocol as the EDGAR adapter) so the
    network is never touched in unit tests.
    """

    def __init__(
        self,
        settings: AlpacaSettings,
        *,
        transport: Transport,
        clock: Clock | None = None,
        max_retries: int = 5,
        base_backoff: float = 0.5,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._clock = clock or _SystemClock()
        self._max_retries = max_retries
        self._base_backoff = base_backoff

    def fetch_universe(self) -> UniverseSnapshot:
        """Return the US-equity universe split into active and delisted sets."""
        active = self.fetch_assets(status=_ACTIVE)
        inactive = self.fetch_assets(status=_INACTIVE)
        return UniverseSnapshot(active=active, inactive=inactive)

    def fetch_assets(self, *, status: str) -> tuple[AlpacaAsset, ...]:
        """Fetch US equities for one ``status`` (``"active"`` or ``"inactive"``)."""
        url = (
            f"{self._settings.trading_base_url()}/v2/assets"
            f"?status={status}&asset_class={_US_EQUITY}"
        )
        payload = self._get_with_retry(url)
        return tuple(self._parse_asset(row) for row in payload if self._is_us_equity(row))

    def _get_with_retry(self, url: str) -> list[dict[str, object]]:
        headers = self._settings.auth_headers()
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            response = self._transport.get(url, headers=headers)
            if response.status_code == 200:
                return _as_list(response.json)
            last_status = response.status_code
            if response.status_code not in _RETRYABLE_STATUS:
                break
            if attempt < self._max_retries:
                self._clock.sleep(self._base_backoff * (2**attempt))
        raise RuntimeError(f"Alpaca assets request failed (status={last_status})")

    @staticmethod
    def _is_us_equity(row: object) -> bool:
        return isinstance(row, dict) and row.get("class") == _US_EQUITY

    @staticmethod
    def _parse_asset(row: dict[str, object]) -> AlpacaAsset:
        return AlpacaAsset(
            symbol=_as_str(row.get("symbol")),
            name=_as_str(row.get("name")),
            exchange=_as_str(row.get("exchange")),
            asset_class=_as_str(row.get("class")),
            status=_as_str(row.get("status")),
            tradable=bool(row.get("tradable", False)),
            fractionable=bool(row.get("fractionable", False)),
        )


def _as_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""
