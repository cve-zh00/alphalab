"""Domain-level exceptions for the compute (valuation) layer.

Pure signalling types. A valuation that is mathematically ill-posed (for
example a terminal growth rate that meets or exceeds the discount rate) must
fail loudly with one of these rather than silently return an absurd finite
number. No I/O.
"""

from __future__ import annotations


class ComputeError(Exception):
    """Base class for all compute-layer domain errors."""


class GrowthExceedsDiscountError(ComputeError):
    """Raised when ``g >= r`` so the Gordon/terminal value diverges.

    Used by both the DCF terminal value and the DDM: as the growth rate
    approaches the discount rate the geometric series diverges, so any finite
    output would be meaningless. We refuse rather than return nonsense.
    """


class InvalidValuationInputError(ComputeError):
    """Raised when an input is structurally invalid for the operator.

    Examples: a non-positive discount rate, an empty peer set for a multiple,
    or a negative share count.
    """
