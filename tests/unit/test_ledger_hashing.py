"""Unit tests for canonical spec hashing (:mod:`alphalab.domain.ledger.hashing`)."""

from __future__ import annotations

import pytest

from alphalab.domain.ledger.hashing import canonicalize, spec_hash


class TestKeyOrderIndependence:
    def test_reordered_keys_hash_identically(self) -> None:
        # Arrange
        a = {"alpha": 1, "beta": 2, "gamma": 3}
        b = {"gamma": 3, "alpha": 1, "beta": 2}

        # Act / Assert
        assert spec_hash(a) == spec_hash(b)

    def test_nested_reordering_hashes_identically(self) -> None:
        # Arrange
        a = {"outer": {"x": 1, "y": 2}, "name": "mr"}
        b = {"name": "mr", "outer": {"y": 2, "x": 1}}

        # Act / Assert
        assert spec_hash(a) == spec_hash(b)


class TestNumericNormalisation:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ({"a": 1}, {"a": 1.0}),  # int vs float
            ({"a": 1}, {"a": True}),  # int vs bool
            ({"a": 0}, {"a": False}),  # int vs bool
            ({"a": 1}, {"a": "1"}),  # int vs str
            ({"a": 1.0}, {"a": "1.0"}),  # float vs str
        ],
    )
    def test_distinct_scalar_types_do_not_collide(
        self, left: dict[str, object], right: dict[str, object]
    ) -> None:
        # Act / Assert
        assert spec_hash(left) != spec_hash(right)

    def test_equal_floats_hash_identically(self) -> None:
        # Arrange: different literal forms of the same float value.
        a = {"x": 1000.0}
        b = {"x": 1e3}

        # Act / Assert
        assert spec_hash(a) == spec_hash(b)

    def test_negative_zero_hashes_as_zero(self) -> None:
        # Arrange: -0.0 == 0.0, so they must hash identically.
        # Act / Assert
        assert spec_hash({"x": -0.0}) == spec_hash({"x": 0.0})


class TestSequencesVsMappings:
    def test_sequence_order_is_significant(self) -> None:
        # Act / Assert
        assert spec_hash({"xs": [1, 2, 3]}) != spec_hash({"xs": [3, 2, 1]})

    def test_list_and_mapping_do_not_alias(self) -> None:
        # Act / Assert
        assert spec_hash({"x": [1, 2]}) != spec_hash({"x": {"0": 1, "1": 2}})


class TestCanonicalize:
    def test_is_deterministic_string(self) -> None:
        # Arrange
        spec = {"b": 2, "a": 1}

        # Act
        first = canonicalize(spec)
        second = canonicalize(spec)

        # Assert
        assert first == second
        assert isinstance(first, str)

    def test_digest_is_64_hex_chars(self) -> None:
        # Act
        digest = spec_hash({"a": 1})

        # Assert
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


def test_unsupported_scalar_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(TypeError):
        spec_hash({"bad": object()})
