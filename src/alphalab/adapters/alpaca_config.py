"""Alpaca connection settings, with paper trading as the only default.

Secrets discipline
------------------
Credentials arrive exclusively from the environment (prefix
``ALPHALAB_ALPACA_``) or an untracked ``.env``. Nothing is hardcoded, and the
secret is wrapped in :class:`~pydantic.SecretStr` so it does not leak into
logs, tracebacks or ``repr`` output — this project's observability layer
redacts secrets, and a bare string would defeat it.

Why live trading is gated rather than configurable
--------------------------------------------------
Alpaca is a brokerage: the same credentials that read market data can place
orders against real money. As of this writing the system has **no demonstrated
edge**, and that is a measured statement, not modesty:

* The momentum ensemble's out-of-sample information ratio on a de-biased
  universe was **-0.10** — it did not beat the universe it traded.
* The turbulence risk overlay failed to beat random abstention at the same
  exposure (82nd percentile of the control, where 95th was required).
* The agentic layer has **zero settled predictions**, so its calibration is
  entirely unmeasured.

Routing orders to a live account on that evidence would be indefensible.
Accordingly :class:`AlpacaSettings` defaults to the paper endpoint, and
reaching the live endpoint requires setting ``ALPHALAB_ALPACA_ENDPOINT=live``
*and* passing an explicit acknowledgement flag at the call site. Two
independent deliberate acts, so it cannot happen by typo or inherited shell
state.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DATA_BASE_URL",
    "AlpacaEndpoint",
    "AlpacaSettings",
    "LiveTradingNotAcknowledgedError",
]

#: Market data host. Shared by paper and live accounts: reading prices carries
#: no execution risk, so it is not gated.
DATA_BASE_URL = "https://data.alpaca.markets"

_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL = "https://api.alpaca.markets"


class AlpacaEndpoint(StrEnum):
    """Which trading environment to address."""

    PAPER = "paper"
    LIVE = "live"

    @property
    def base_url(self) -> str:
        """Trading API host for this environment."""
        return _LIVE_URL if self is AlpacaEndpoint.LIVE else _PAPER_URL

    @property
    def is_live(self) -> bool:
        """Whether this environment can move real money."""
        return self is AlpacaEndpoint.LIVE


class LiveTradingNotAcknowledgedError(RuntimeError):
    """Raised when the live endpoint is requested without acknowledgement.

    Deliberately an error rather than a silent downgrade to paper. Quietly
    redirecting a live request would be worse: the caller would believe orders
    were live when they were not, which is its own category of disaster.
    """


class AlpacaSettings(BaseSettings):
    """Credentials and endpoint selection, sourced from the environment.

    Set ``ALPHALAB_ALPACA_KEY_ID`` and ``ALPHALAB_ALPACA_SECRET_KEY``. The
    endpoint defaults to paper; see the module docstring for why live access
    requires a second, explicit act.
    """

    model_config = SettingsConfigDict(
        env_prefix="ALPHALAB_ALPACA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    key_id: str = Field(min_length=1)
    secret_key: SecretStr
    endpoint: AlpacaEndpoint = AlpacaEndpoint.PAPER

    #: Requests per minute. Alpaca's free tier allows 200; the default leaves
    #: headroom rather than riding the limit and risking throttled retries.
    rate_limit_per_minute: int = Field(default=150, gt=0)

    def auth_headers(self) -> dict[str, str]:
        """Headers Alpaca expects for authentication.

        The secret is unwrapped only here, at the moment of use, keeping it out
        of every intermediate structure that might be logged.
        """
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key.get_secret_value(),
        }

    def trading_base_url(self, *, acknowledge_live: bool = False) -> str:
        """Trading API host, refusing live access without acknowledgement.

        Args:
            acknowledge_live: Must be ``True`` to obtain the live host. The
                parameter is keyword-only and defaults to ``False`` so that
                live access is impossible to reach by accident.

        Raises:
            LiveTradingNotAcknowledgedError: If the configured endpoint is live
                and ``acknowledge_live`` was not set.
        """
        if self.endpoint.is_live and not acknowledge_live:
            raise LiveTradingNotAcknowledgedError(
                "endpoint is set to live but acknowledge_live was not passed. "
                "Live orders move real money, and this system has no "
                "demonstrated edge: out-of-sample information ratio was -0.10 "
                "and no prediction has been settled yet. Use paper."
            )
        return self.endpoint.base_url

    def data_base_url(self) -> str:
        """Market data host. Ungated: reading prices places no orders."""
        return DATA_BASE_URL
