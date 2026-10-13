"""Point-in-time (PIT) data correctness primitives.

This package makes look-ahead *inexpressible by construction*: records carry
both the period they refer to and the date they became knowable, features
must declare an availability lag, and the only filter exposed withholds
anything not yet knowable at a given knowledge date.
"""

from __future__ import annotations

from alphalab.domain.pit.errors import (
    LeakageError,
    PITError,
    UnregisteredFeatureError,
)
from alphalab.domain.pit.features import FeatureRegistry, FeatureSpec
from alphalab.domain.pit.filter import visible_records
from alphalab.domain.pit.records import KnowledgeDate, PITRecord
from alphalab.domain.pit.resolve import resolve_feature

__all__ = [
    "FeatureRegistry",
    "FeatureSpec",
    "KnowledgeDate",
    "LeakageError",
    "PITError",
    "PITRecord",
    "UnregisteredFeatureError",
    "resolve_feature",
    "visible_records",
]
