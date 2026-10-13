"""Dividend-discount (Gordon growth) valuation operator.

For a stable dividend payer the fair value per share is the next dividend
divided by the spread between the required return and the perpetual dividend
growth:

    ``P0 = D1 / (r - g)``  where ``D1 = D0 * (1 + g)``.

As with the DCF terminal value, ``g`` must be strictly below ``r`` or the
series diverges; we raise :class:`GrowthExceedsDiscountError` rather than
return nonsense.

Pure: no I/O, no state.
"""

from __future__ import annotations

from alphalab.domain.compute.errors import (
    GrowthExceedsDiscountError,
    InvalidValuationInputError,
)
from alphalab.domain.compute.provenance import ProvenancedValue, combine
from alphalab.domain.compute.thresholds import MIN_GROWTH_DISCOUNT_SPREAD

_DDM_VERSION = "1.0.0"


def ddm_value_per_share(
    *,
    dividend_per_share: float,
    required_return: float,
    dividend_growth: float,
) -> ProvenancedValue:
    """Gordon-growth fair value per share.

    Args:
        dividend_per_share: Trailing dividend ``D0`` per share (>= 0).
        required_return: Required return on equity ``r`` (fraction).
        dividend_growth: Perpetual dividend growth ``g`` (fraction).

    Returns:
        A provenanced value per share in ``USD_per_share`` units.

    Raises:
        GrowthExceedsDiscountError: if ``g`` is not strictly below ``r``.
        InvalidValuationInputError: on a negative dividend.
    """
    if dividend_per_share < 0.0:
        raise InvalidValuationInputError("dividend_per_share must be non-negative")
    if required_return - dividend_growth < MIN_GROWTH_DISCOUNT_SPREAD:
        raise GrowthExceedsDiscountError(
            f"dividend growth g={dividend_growth:.6f} must be below required "
            f"return r={required_return:.6f} by at least "
            f"{MIN_GROWTH_DISCOUNT_SPREAD:.6f}"
        )

    next_dividend = dividend_per_share * (1.0 + dividend_growth)
    value_per_share = next_dividend / (required_return - dividend_growth)

    return combine(
        value=value_per_share,
        operator="ddm_value_per_share",
        version=_DDM_VERSION,
        unit="USD_per_share",
        parents=(),
        extra_inputs={
            "dividend_per_share": dividend_per_share,
            "required_return": required_return,
            "dividend_growth": dividend_growth,
        },
    )
