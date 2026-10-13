"""Unit tests for :mod:`alphalab.domain.compute.provenance`.

Covers leaf/derived construction, the composable derivation chain
(``leaves`` and ``tree``), immutability, the ``provenanced`` decorator helper,
and the conservative knowledge-date merge.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from alphalab.domain.compute.provenance import (
    ProvenanceDepthError,
    ProvenancedValue,
    combine,
    leaf,
    provenanced,
)
from alphalab.domain.compute.thresholds import MAX_PROVENANCE_DEPTH


class TestLeaf:
    def test_leaf_has_no_parents(self) -> None:
        # Arrange / Act
        node = leaf(value=10.0, name="price", unit="USD")

        # Assert
        assert node.is_leaf()
        assert node.parents == ()
        assert node.inputs == (("price", 10.0),)
        assert node.leaves() == (node,)

    def test_leaf_carries_pit_metadata(self) -> None:
        kd = date(2024, 3, 31)
        node = leaf(value=1.0, name="x", unit="USD", dataset_hash="abc", knowledge_date=kd)
        assert node.dataset_hash == "abc"
        assert node.knowledge_date == kd


class TestImmutability:
    def test_cannot_reassign_value(self) -> None:
        node = leaf(value=1.0, name="x", unit="USD")
        with pytest.raises(ValidationError):
            node.value = 2.0  # type: ignore[misc]

    def test_cannot_reassign_parents(self) -> None:
        node = leaf(value=1.0, name="x", unit="USD")
        with pytest.raises(ValidationError):
            node.parents = ()  # type: ignore[misc]

    def test_requires_non_empty_operator(self) -> None:
        with pytest.raises(ValidationError):
            ProvenancedValue(
                value=1.0,
                operator="",
                operator_version="1",
                inputs=(),
                unit="USD",
            )


class TestDerivationChain:
    def test_leaves_reach_all_raw_inputs(self) -> None:
        # Arrange: (a + b) composed with c -> two levels.
        a = leaf(value=2.0, name="a", unit="USD")
        b = leaf(value=3.0, name="b", unit="USD")
        c = leaf(value=4.0, name="c", unit="USD")

        @provenanced(operator="add", version="1", unit="USD")
        def add(*, x: float, y: float) -> float:
            return x + y

        ab = add(x=a, y=b)
        abc = add(x=ab, y=c)

        # Act
        leaf_inputs = {node.inputs[0][0] for node in abc.leaves()}

        # Assert: the final number enumerates every leaf.
        assert abc.value == pytest.approx(9.0)
        assert leaf_inputs == {"a", "b", "c"}

    def test_tree_is_nested_and_readable(self) -> None:
        a = leaf(value=2.0, name="a", unit="USD")
        b = leaf(value=3.0, name="b", unit="USD")

        @provenanced(operator="add", version="7", unit="USD")
        def add(*, x: float, y: float) -> float:
            return x + y

        tree = add(x=a, y=b).tree()

        assert tree["operator"] == "add"
        assert tree["operator_version"] == "7"
        assert tree["value"] == pytest.approx(5.0)
        assert len(tree["parents"]) == 2
        assert tree["parents"][0]["operator"] == "leaf"

    def test_shared_subexpression_dedupes_leaves(self) -> None:
        # Arrange: a feeds two branches; it must appear once in leaves().
        a = leaf(value=5.0, name="a", unit="USD")
        b = leaf(value=1.0, name="b", unit="USD")

        @provenanced(operator="op", version="1", unit="USD")
        def op(*, x: float, y: float) -> float:
            return x + y

        left = op(x=a, y=b)
        right = op(x=a, y=b)
        root = op(x=left, y=right)

        # Act
        leaf_names = sorted(node.inputs[0][0] for node in root.leaves())

        # Assert: 'a' deduplicated by identity, 'b' leaves are distinct objects.
        assert leaf_names.count("a") == 1


class TestProvenancedDecorator:
    def test_records_scalar_and_provenanced_inputs(self) -> None:
        # Arrange
        base = leaf(value=10.0, name="base", unit="USD")

        @provenanced(operator="scale", version="1", unit="USD")
        def scale(*, amount: float, factor: float) -> float:
            return amount * factor

        # Act: 'amount' is provenanced (a parent), 'factor' is a scalar.
        result = scale(amount=base, factor=2.0)

        # Assert
        assert result.value == pytest.approx(20.0)
        assert len(result.parents) == 1
        assert dict(result.inputs) == {"amount": 10.0, "factor": 2.0}

    def test_merges_conservative_knowledge_date(self) -> None:
        # Arrange: the derived value cannot be known before the latest parent.
        early = leaf(value=1.0, name="e", unit="USD", knowledge_date=date(2024, 1, 1))
        late = leaf(value=2.0, name="l", unit="USD", knowledge_date=date(2024, 6, 30))

        @provenanced(operator="add", version="1", unit="USD")
        def add(*, x: float, y: float) -> float:
            return x + y

        # Act
        result = add(x=early, y=late)

        # Assert
        assert result.knowledge_date == date(2024, 6, 30)


class TestDepthGuard:
    def _build_deep_chain(self) -> ProvenancedValue:
        # A linear chain deeper than MAX_PROVENANCE_DEPTH guarantees the guard
        # trips before any natural termination.
        node = leaf(value=1.0, name="root", unit="USD")
        for _ in range(MAX_PROVENANCE_DEPTH + 2):
            node = combine(
                value=node.value,
                operator="chain",
                version="1",
                unit="USD",
                parents=(node,),
            )
        return node

    def test_leaves_raises_on_excessive_depth(self) -> None:
        deep = self._build_deep_chain()
        with pytest.raises(ProvenanceDepthError):
            deep.leaves()

    def test_tree_raises_on_excessive_depth(self) -> None:
        deep = self._build_deep_chain()
        with pytest.raises(ProvenanceDepthError):
            deep.tree()


class TestCombine:
    def test_combine_attaches_parents_and_extra_inputs(self) -> None:
        p = leaf(value=4.0, name="p", unit="USD")
        node = combine(
            value=8.0,
            operator="double",
            version="1",
            unit="USD",
            parents=(p,),
            extra_inputs={"k": 2.0},
        )
        assert node.value == pytest.approx(8.0)
        assert node.parents == (p,)
        assert ("k", 2.0) in node.inputs
