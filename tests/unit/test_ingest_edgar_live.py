"""Unit tests for the fundamentals extraction over the EDGAR adapter."""

from __future__ import annotations

from datetime import date

from alphalab.adapters.edgar import EdgarClient, EdgarSettings, HttpResponse
from alphalab.adapters.edgar_live import (
    CANONICAL_CONCEPTS,
    FundamentalField,
    extract_fundamentals,
)
from alphalab.domain.types import Ticker


class _StubTransport:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(200, self._payload)


class _FakeClock:
    def monotonic(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        return None


def _usd(concept: str, entries: list[dict[str, object]]) -> dict[str, object]:
    return {concept: {"units": {"USD": entries}}}


def _client(payload: dict[str, object]) -> EdgarClient:
    settings = EdgarSettings(user_agent="alphalab-test t@example.com")
    return EdgarClient(settings, transport=_StubTransport(payload), clock=_FakeClock())


def _facts(*concepts: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for c in concepts:
        merged.update(c)
    return {"facts": {"us-gaap": merged}}


class TestConceptMapping:
    def test_maps_raw_concept_to_canonical_field(self) -> None:
        # Arrange — NetIncomeLoss is the sole concept for NET_INCOME.
        payload = _facts(
            _usd(
                "NetIncomeLoss",
                [{"start": "2023-01-01", "end": "2023-12-31", "val": 100.0, "filed": "2024-02-01"}],
            )
        )

        # Act
        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        # Assert — the record is keyed by the canonical field name.
        net = [r for r in result.records if r.field == FundamentalField.NET_INCOME]
        assert len(net) == 1
        assert net[0].value == 100.0
        assert net[0].as_of == date(2023, 12, 31)
        assert net[0].observed_at == date(2024, 2, 1)

    def test_uses_first_available_fallback_concept(self) -> None:
        # Arrange — only the *second*-priority revenue concept is present.
        payload = _facts(
            _usd(
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                [{"start": "2023-01-01", "end": "2023-12-31", "val": 500.0, "filed": "2024-02-01"}],
            )
        )

        # Act
        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        # Assert — fallback resolved to the canonical REVENUE field.
        rev = [r for r in result.records if r.field == FundamentalField.REVENUE]
        assert [r.value for r in rev] == [500.0]

    def test_prefers_higher_priority_concept_when_both_present(self) -> None:
        # Arrange — both revenue concepts present; ``Revenues`` wins.
        payload = _facts(
            _usd("Revenues", [{"start": "2023-01-01", "end": "2023-12-31", "val": 900.0, "filed": "2024-02-01"}]),
            _usd(
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                [{"start": "2023-01-01", "end": "2023-12-31", "val": 500.0, "filed": "2024-02-01"}],
            ),
        )

        # Act
        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        # Assert — only the preferred concept's value is emitted for REVENUE.
        rev = [r for r in result.records if r.field == FundamentalField.REVENUE]
        assert [r.value for r in rev] == [900.0]


class TestAsFiledPreservation:
    def test_keeps_all_filings_for_the_same_period(self) -> None:
        # Arrange — a restatement: two filings for the same period end.
        payload = _facts(
            _usd(
                "NetIncomeLoss",
                [
                    {"start": "2023-01-01", "end": "2023-12-31", "val": 100.0, "filed": "2024-02-01"},
                    {"start": "2023-01-01", "end": "2023-12-31", "val": 80.0, "filed": "2024-08-01"},
                ],
            )
        )

        # Act
        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        # Assert — both as-filed records survive; nothing is deduplicated.
        net = [r for r in result.records if r.field == FundamentalField.NET_INCOME]
        by_filed = {r.observed_at: r.value for r in net}
        assert by_filed == {
            date(2024, 2, 1): 100.0,
            date(2024, 8, 1): 80.0,
        }


class TestAnnualFlowsOnly:
    """A flow must come from a full year, never from the latest quarter.

    EDGAR carries a filer's 10-Q facts as well as its 10-K ones, and the
    quarterly facts are filed *more recently*. Resolving a flow by "most
    recently filed" therefore picks a three-month figure for a model that
    expects twelve, understating it roughly fourfold — which on real data
    pushed every valuation an order of magnitude below the traded price.
    """

    def test_quarterly_flow_is_ignored_in_favour_of_the_annual_one(self) -> None:
        # Arrange — the quarter is filed later than the full year, so a
        # "latest filing wins" rule would wrongly prefer it.
        payload = _facts(
            _usd(
                "Revenues",
                [
                    {
                        "start": "2023-01-01",
                        "end": "2023-12-31",
                        "val": 4000.0,
                        "filed": "2024-02-01",
                    },
                    {
                        "start": "2024-01-01",
                        "end": "2024-03-31",
                        "val": 1000.0,
                        "filed": "2024-05-01",
                    },
                ],
            )
        )

        # Act
        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        # Assert — only the annual figure survives.
        rev = [r for r in result.records if r.field == FundamentalField.REVENUE]
        assert [r.value for r in rev] == [4000.0]

    def test_a_flow_with_only_quarters_is_missing_not_backfilled(self) -> None:
        """A stated absence costs one name; a 4x understatement poisons everything."""
        payload = _facts(
            _usd(
                "Revenues",
                [
                    {
                        "start": "2024-01-01",
                        "end": "2024-03-31",
                        "val": 1000.0,
                        "filed": "2024-05-01",
                    }
                ],
            )
        )

        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        assert FundamentalField.REVENUE in result.missing_fields
        assert not [r for r in result.records if r.field == FundamentalField.REVENUE]

    def test_falls_back_to_a_concept_that_does_have_the_annual_figure(self) -> None:
        """Preference ranks concepts; it must not override having usable data.

        A filer may tag its preferred concept only quarterly while the fallback
        carries the annual figure. Choosing the preferred concept first and only
        then demanding an annual period would report the field missing when it
        is right there under the other name.
        """
        payload = _facts(
            _usd(
                "Revenues",
                [
                    {
                        "start": "2024-01-01",
                        "end": "2024-03-31",
                        "val": 1000.0,
                        "filed": "2024-05-01",
                    }
                ],
            ),
            _usd(
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                [
                    {
                        "start": "2023-01-01",
                        "end": "2023-12-31",
                        "val": 4000.0,
                        "filed": "2024-02-01",
                    }
                ],
            ),
        )

        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        rev = [r for r in result.records if r.field == FundamentalField.REVENUE]
        assert [r.value for r in rev] == [4000.0]
        assert (
            result.resolved_concepts[FundamentalField.REVENUE]
            == "RevenueFromContractWithCustomerExcludingAssessedTax"
        )

    def test_a_balance_sheet_stock_has_no_duration_and_is_kept(self) -> None:
        """Instants carry no ``start``; the annual rule must not swallow them."""
        payload = _facts(_usd("Assets", [{"end": "2024-03-31", "val": 7.0, "filed": "2024-05-01"}]))

        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        assets = [r for r in result.records if r.field == FundamentalField.ASSETS]
        assert [r.value for r in assets] == [7.0]
        assert assets[0].period_days is None

    def test_a_52_week_fiscal_year_still_counts_as_annual(self) -> None:
        """364 days is a full year for a 52-week filer, not a long quarter."""
        payload = _facts(
            _usd(
                "Revenues",
                [
                    {
                        "start": "2023-01-02",
                        "end": "2023-12-31",
                        "val": 4000.0,
                        "filed": "2024-02-01",
                    }
                ],
            )
        )

        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        rev = [r for r in result.records if r.field == FundamentalField.REVENUE]
        assert [r.value for r in rev] == [4000.0]
        assert rev[0].period_days == 363


class TestMissingData:
    def test_absent_concept_is_simply_missing_not_zero(self) -> None:
        # Arrange — only net income; everything else is absent.
        payload = _facts(
            _usd(
                "NetIncomeLoss",
                [{"start": "2023-01-01", "end": "2023-12-31", "val": 100.0, "filed": "2024-02-01"}],
            )
        )

        # Act
        result = extract_fundamentals(_client(payload), Ticker("ACME"), cik="1")

        # Assert — REVENUE is reported as a missing field, never fabricated.
        fields_present = {r.field for r in result.records}
        assert FundamentalField.REVENUE not in fields_present
        assert FundamentalField.REVENUE in result.missing_fields
        # No fabricated zero sneaked in.
        assert all(
            r.value != 0.0 or r.field == FundamentalField.NET_INCOME is False
            for r in result.records
        )

    def test_every_canonical_field_has_a_concept_list(self) -> None:
        # Guard: the fallback table covers all declared canonical fields.
        assert set(CANONICAL_CONCEPTS) == set(FundamentalField)
        for concepts in CANONICAL_CONCEPTS.values():
            assert concepts, "each canonical field needs at least one XBRL concept"
