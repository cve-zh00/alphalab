"""Feature declarations and an immutable registry.

Every feature must declare its availability lag — the gap between the period
a datum refers to and the earliest date it could be publicly known (e.g. the
~45-90 day lag between a fiscal quarter end and its SEC filing). Requesting a
feature that was never registered is a hard error, so no feature can be used
without an audited lag.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.pit.errors import UnregisteredFeatureError


class FeatureSpec(BaseModel):
    """Immutable declaration of one feature and its availability lag.

    Attributes:
        name: Unique feature identifier.
        availability_lag_days: Days after ``as_of`` before the datum is
            publicly knowable. Must be non-negative.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    availability_lag_days: int = Field(ge=0)

    def earliest_knowable(self, as_of: date) -> date:
        """Return the earliest date this feature could be known for ``as_of``."""
        return as_of + timedelta(days=self.availability_lag_days)


class FeatureRegistry:
    """Read-only collection of :class:`FeatureSpec` keyed by name.

    The registry is populated once at construction and exposes only read
    access thereafter, so the set of usable features cannot drift at runtime.
    """

    __slots__ = ("_specs",)

    def __init__(self, specs: Iterable[FeatureSpec]) -> None:
        mapping: dict[str, FeatureSpec] = {}
        for spec in specs:
            if spec.name in mapping:
                raise ValueError(f"duplicate feature registration: {spec.name!r}")
            mapping[spec.name] = spec
        self._specs: MappingProxyType[str, FeatureSpec] = MappingProxyType(mapping)

    def get(self, name: str) -> FeatureSpec:
        """Return the spec for ``name`` or raise :class:`UnregisteredFeatureError`."""
        try:
            return self._specs[name]
        except KeyError as exc:
            raise UnregisteredFeatureError(
                f"feature {name!r} is not registered; declare its availability lag before use"
            ) from exc

    def names(self) -> frozenset[str]:
        """Return the registered feature names as an immutable set."""
        return frozenset(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs
