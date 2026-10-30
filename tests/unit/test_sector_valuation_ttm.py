"""Sector valuation operators, TTM synthesis, Piotroski/YoY and buyback helpers."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from alphalab.adapters.edgar_live import FundamentalField, synthesize_ttm
from alphalab.adapters.pit_store import ParquetPITStore
from alphalab.app.ingest import (
    ValuationAssumptions,
    _build_estimates,
    _build_pit_dossier,
    _compute_piotroski_and_yoy,
    _resolve_net_buyback_yield,
)
from alphalab.domain.compute import (
    InvalidValuationInputError,
    ffo_implied_value,
    residual_income_value_per_share,
)
from alphalab.domain.compute.sectors import SECTOR_PROFILES, GicsSector
from alphalab.domain.pit import KnowledgeDate, PITRecord
from alphalab.domain.types import Ticker

T = Ticker("ACME")


def _rec(field: str, value: float, end: date, days: int | None, filed: date | None = None):
    return PITRecord(
        ticker=T,
        field=field,
        value=value,
        as_of=end,
        observed_at=filed or end + timedelta(days=30),
        source="test",
        period_start=None if days is None else end - timedelta(days=days),
    )


class TestResidualIncome:
    def test_zero_excess_return_equals_book(self) -> None:
        v = residual_income_value_per_share(
            book_value_per_share=10.0, eps=1.0, cost_of_equity=0.10, terminal_growth=0.02
        )
        assert v.value == pytest.approx(10.0)

    def test_high_roe_above_book(self) -> None:
        v = residual_income_value_per_share(
            book_value_per_share=10.0, eps=2.0, cost_of_equity=0.09, terminal_growth=0.02
        )
        assert v.value > 10.0

    @pytest.mark.parametrize("bv", [0.0, -5.0])
    def test_nonpositive_book_rejected(self, bv: float) -> None:
        with pytest.raises(InvalidValuationInputError):
            residual_income_value_per_share(
                book_value_per_share=bv, eps=1.0, cost_of_equity=0.09, terminal_growth=0.02
            )

    def test_ke_not_above_g_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError):
            residual_income_value_per_share(
                book_value_per_share=10.0, eps=1.0, cost_of_equity=0.02, terminal_growth=0.03
            )


def test_ffo_implied_value_uses_median() -> None:
    v = ffo_implied_value(peer_ffo=(14.0, 16.5, 19.0), ffo_per_share=2.0)
    assert v.value == pytest.approx(33.0)
    with pytest.raises(InvalidValuationInputError):
        ffo_implied_value(peer_ffo=(), ffo_per_share=2.0)


def test_real_estate_profile_has_ffo() -> None:
    assert SECTOR_PROFILES[GicsSector.REAL_ESTATE].peer_ffo == (14.0, 16.5, 19.0)
    assert SECTOR_PROFILES[GicsSector.MARKET_DEFAULT].peer_ffo == ()


def _assump(sector: GicsSector) -> ValuationAssumptions:
    return ValuationAssumptions(
        cost_of_equity=0.09,
        cost_of_debt=0.04,
        tax_rate=0.21,
        terminal_growth=0.02,
        dividend_growth=0.03,
        projection_years=5,
        peer_pe=(15.0, 18.0, 20.0),
        peer_pb=(1.2, 1.5, 1.9),
        peer_ffo=(14.0, 16.5, 19.0),
        sector=sector,
    )


F = FundamentalField
_VALUES = {
    F.STOCKHOLDERS_EQUITY: 1000.0,
    F.SHARES_OUTSTANDING: 100.0,
    F.EPS_DILUTED: 1.5,
    F.NET_INCOME: 150.0,
    F.DEPRECIATION_AMORTIZATION: 50.0,
    F.OPERATING_CASH_FLOW: 200.0,
    F.CAPEX: 20.0,
    F.DIVIDEND_PER_SHARE: 0.8,
}


def test_financials_route_uses_residual_income_no_dcf() -> None:
    _, used, omitted = _build_estimates(_VALUES, _assump(GicsSector.FINANCIALS), current_price=20.0)
    assert "residual_income" in used and "dcf" not in used
    assert "multiples_pe" in used and "multiples_pb" in used
    assert any(o.startswith("dcf") for o in omitted)


def test_real_estate_route_uses_ffo_not_pe() -> None:
    est, used, _ = _build_estimates(_VALUES, _assump(GicsSector.REAL_ESTATE), current_price=20.0)
    assert "multiples_ffo" in used and "multiples_pe" not in used
    ffo = next(e for e in est if e.operator == "ffo_implied_value")
    assert ffo.value == pytest.approx(16.5 * 2.0)


def test_real_estate_ffo_omitted_without_da() -> None:
    vals = {k: v for k, v in _VALUES.items() if k is not F.DEPRECIATION_AMORTIZATION}
    _, used, omitted = _build_estimates(vals, _assump(GicsSector.REAL_ESTATE), current_price=20.0)
    assert "multiples_ffo" not in used
    assert any("multiples_ffo" in o for o in omitted)


def test_buyback_raises_ddm_value() -> None:
    base, _, _ = _build_estimates(_VALUES, _assump(GicsSector.MARKET_DEFAULT), current_price=20.0)
    bb, _, _ = _build_estimates(
        _VALUES, _assump(GicsSector.MARKET_DEFAULT), current_price=20.0, buyback_yield=0.02
    )
    d0 = next(e for e in base if e.operator == "ddm_value_per_share")
    d1 = next(e for e in bb if e.operator == "ddm_value_per_share")
    assert d1.value > d0.value


def _store(tmp_path: Path, recs: list[PITRecord]) -> ParquetPITStore:
    store = ParquetPITStore(root=tmp_path)
    store.write(recs)
    return store


def test_net_buyback_yield_annualised_and_clamped(tmp_path: Path) -> None:
    recs = [
        _rec("shares_outstanding", 100.0, date(2021, 12, 31), None),
        _rec("shares_outstanding", 97.0, date(2022, 12, 31), None),
        _rec("shares_outstanding", 94.0, date(2023, 12, 31), None),
    ]
    y = _resolve_net_buyback_yield(_store(tmp_path, recs), T, KnowledgeDate(date(2024, 6, 1)))
    assert 0.025 < y < 0.035


def test_net_buyback_yield_dilution_is_zero(tmp_path: Path) -> None:
    recs = [
        _rec("shares_outstanding", 100.0, date(2022, 12, 31), None),
        _rec("shares_outstanding", 110.0, date(2023, 12, 31), None),
    ]
    assert (
        _resolve_net_buyback_yield(_store(tmp_path, recs), T, KnowledgeDate(date(2024, 6, 1)))
        == 0.0
    )


def _annual_records(end: date, scale: dict[str, float]) -> list[PITRecord]:
    flows = {"revenue", "net_income", "operating_income", "operating_cash_flow"}
    return [_rec(f, v, end, 365 if f in flows else None) for f, v in scale.items()]


_GOOD_NEW = {
    "revenue": 1200.0,
    "net_income": 150.0,
    "operating_income": 240.0,
    "operating_cash_flow": 200.0,
    "assets": 1000.0,
    "long_term_debt": 100.0,
    "cash": 200.0,
    "liabilities": 500.0,
    "shares_outstanding": 95.0,
}
_OLD = {
    "revenue": 1000.0,
    "net_income": 80.0,
    "operating_income": 150.0,
    "operating_cash_flow": 120.0,
    "assets": 1000.0,
    "long_term_debt": 200.0,
    "cash": 100.0,
    "liabilities": 600.0,
    "shares_outstanding": 100.0,
}


def test_piotroski_strong_and_dossier_notes_have_no_digits(tmp_path: Path) -> None:
    recs = _annual_records(date(2022, 12, 31), _OLD)
    recs += _annual_records(date(2023, 12, 31), _GOOD_NEW)
    kd = KnowledgeDate(date(2024, 6, 1))
    q = _compute_piotroski_and_yoy(_store(tmp_path, recs), T, kd)
    assert q is not None and q.piotroski == 9
    assert q.margin_delta is not None and q.margin_delta > 0
    notes = _build_pit_dossier(
        values=_VALUES,
        assumptions=_assump(GicsSector.MARKET_DEFAULT),
        current_price=float("nan"),
        pit_cagr=None,
        estimates=[],
        quality=q,
    )
    joined = " ".join(notes)
    assert "Piotroski financial-strength regime is strong" in joined
    assert "operating margin expanded" in joined
    assert not any(ch.isdigit() for ch in joined.replace("10-K", ""))


def test_piotroski_none_when_too_few_signals(tmp_path: Path) -> None:
    recs = [
        _rec("revenue", 1000.0, date(2022, 12, 31), 365),
        _rec("revenue", 1100.0, date(2023, 12, 31), 365),
    ]
    assert (
        _compute_piotroski_and_yoy(_store(tmp_path, recs), T, KnowledgeDate(date(2024, 6, 1)))
        is None
    )


def _quarters(first_end: date, n: int, base: float = 10.0) -> list[PITRecord]:
    out = []
    end = first_end
    for i in range(n):
        out.append(_rec("NetIncomeLoss", base + i, end, 91))
        end = end + timedelta(days=91)
    return out


class TestTtm:
    def test_four_quarters_after_annual_sum(self) -> None:
        annual = _rec("NetIncomeLoss", 100.0, date(2022, 12, 31), 365)
        qs = _quarters(date(2023, 3, 31), 4)
        ttm = synthesize_ttm([annual, *qs])
        assert ttm is not None
        assert ttm.value == pytest.approx(10 + 11 + 12 + 13)
        assert ttm.period_days == 365
        assert ttm.as_of == qs[-1].as_of
        assert ttm.observed_at == max(q.observed_at for q in qs)

    def test_three_quarters_no_ttm(self) -> None:
        annual = _rec("NetIncomeLoss", 100.0, date(2022, 12, 31), 365)
        assert synthesize_ttm([annual, *_quarters(date(2023, 3, 31), 3)]) is None

    def test_gap_no_ttm(self) -> None:
        annual = _rec("NetIncomeLoss", 100.0, date(2022, 12, 31), 365)
        qs = _quarters(date(2023, 3, 31), 4)
        gapped = [qs[0], qs[1], qs[3], _rec("NetIncomeLoss", 9.0, date(2024, 6, 30), 91)]
        assert synthesize_ttm([annual, *gapped]) is None

    def test_quarters_before_annual_end_ignored(self) -> None:
        annual = _rec("NetIncomeLoss", 100.0, date(2023, 12, 31), 365)
        assert synthesize_ttm([annual, *_quarters(date(2023, 3, 31), 4)]) is None
