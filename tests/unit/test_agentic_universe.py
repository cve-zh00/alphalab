"""Unit tests for the liquidity-only candidate universe (no network).

Every test here uses in-memory doubles. The two expensive dependencies — the
Alpaca bars client and the SEC CIK resolver — arrive through narrow callables,
so the whole selection can be exercised deterministically without a socket.

The behaviours pinned here are the ones that make the universe *honest*:

* a name below the price or dollar-volume floor is excluded **and counted**;
* a name whose bars carry no volume is excluded, never assumed liquid;
* a name with no price data at all is counted in its own bucket;
* a name the SEC map does not know is counted separately, not silently dropped;
* the cap is deterministic — same input, same output, byte for byte;
* the three exclusion reasons are reported separately, never merged.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from alphalab.app.agentic_universe import (
    DEFAULT_CANDIDATE_CAP,
    CandidateUniverse,
    build_candidate_universe,
)
from alphalab.domain.backtest.types import PriceBar
from alphalab.domain.types import Ticker

_AS_OF = date(2024, 1, 15)


def _series(
    *,
    close: float,
    volume: float | None,
    days: int = 60,
    end: date = _AS_OF,
) -> tuple[PriceBar, ...]:
    """A flat ``days``-long daily series ending on ``end``.

    Flat prices keep the fixtures readable: liquidity depends only on the level
    of ``close`` and ``volume``, not on their path, so a constant line is the
    least surprising way to place a name above or below a threshold.
    """
    return tuple(
        PriceBar(on=end - timedelta(days=days - 1 - i), close=close, volume=volume)
        for i in range(days)
    )


class _StubBars:
    """Returns a fixed panel, ignoring the requested window.

    Records the ``end`` it was asked for so a test can assert the lag applied
    by the caller matches the measured HTTP-403 workaround.
    """

    def __init__(self, panel: dict[Ticker, tuple[PriceBar, ...]]) -> None:
        self._panel = panel
        self.requested_ends: list[date] = []
        self.requested_symbols: list[tuple[Ticker, ...]] = []

    def daily_bars(
        self,
        symbols: tuple[Ticker, ...] | list[Ticker],
        *,
        start: date,
        end: date,
    ) -> dict[Ticker, tuple[PriceBar, ...]]:
        self.requested_ends.append(end)
        self.requested_symbols.append(tuple(symbols))
        return {t: self._panel[t] for t in symbols if t in self._panel}


class _StubCik:
    """Resolves only the tickers in ``known``; raises KeyError otherwise.

    Mirrors ``CikResolver.resolve``, which raises rather than returning ``None``
    when the SEC map has no entry for a symbol.
    """

    def __init__(self, known: dict[Ticker, str]) -> None:
        self._known = known

    def resolve(self, ticker: Ticker) -> str:
        try:
            return self._known[ticker]
        except KeyError as exc:
            raise KeyError(f"no CIK for {ticker!r}") from exc


def _build(
    panel: dict[Ticker, tuple[PriceBar, ...]],
    requested: list[Ticker],
    known_ciks: dict[Ticker, str],
    *,
    cap: int = DEFAULT_CANDIDATE_CAP,
) -> tuple[CandidateUniverse, _StubBars, _StubCik]:
    bars = _StubBars(panel)
    cik = _StubCik(known_ciks)
    universe = build_candidate_universe(
        requested=requested,
        bars=bars,
        cik=cik,
        as_of=_AS_OF,
        cap=cap,
    )
    return universe, bars, cik


class TestLiquidityExclusion:
    def test_illiquid_by_low_price_is_excluded_and_counted(self) -> None:
        # Arrange — ABLE is liquid; PENY trades under the $5 floor.
        able = Ticker("ABLE")
        peny = Ticker("PENY")
        panel = {
            able: _series(close=50.0, volume=1_000_000.0),
            peny: _series(close=0.50, volume=1_000_000.0),
        }
        ciks = {able: "0000000001", peny: "0000000002"}

        # Act
        universe, _, _ = _build(panel, [able, peny], ciks)

        # Assert — the penny stock is gone and recorded under liquidity.
        assert able in universe.candidates
        assert peny not in universe.candidates
        assert peny in universe.excluded_illiquid
        assert universe.excluded_illiquid_count == 1

    def test_illiquid_by_thin_volume_is_excluded_and_counted(self) -> None:
        # Arrange — THIN clears the price floor but trades far too little value.
        rich = Ticker("RICH")
        thin = Ticker("THIN")
        panel = {
            rich: _series(close=50.0, volume=1_000_000.0),
            thin: _series(close=50.0, volume=10.0),
        }
        ciks = {rich: "0000000001", thin: "0000000002"}

        # Act
        universe, _, _ = _build(panel, [rich, thin], ciks)

        # Assert
        assert rich in universe.candidates
        assert thin in universe.excluded_illiquid
        assert universe.excluded_illiquid_count == 1

    def test_absent_volume_is_excluded_not_assumed_liquid(self) -> None:
        # Arrange — NOVL has a healthy price but every bar lacks volume.
        rich = Ticker("RICH")
        novl = Ticker("NOVL")
        panel = {
            rich: _series(close=50.0, volume=1_000_000.0),
            novl: _series(close=50.0, volume=None),
        }
        ciks = {rich: "0000000001", novl: "0000000002"}

        # Act
        universe, _, _ = _build(panel, [rich, novl], ciks)

        # Assert — absent data is treated as not-tradable, counted as illiquid.
        assert novl not in universe.candidates
        assert novl in universe.excluded_illiquid
        assert universe.excluded_illiquid_count == 1


class TestMissingData:
    def test_no_price_data_is_counted_in_its_own_bucket(self) -> None:
        # Arrange — GONE was requested but Alpaca returns nothing for it.
        rich = Ticker("RICH")
        gone = Ticker("GONE")
        panel = {rich: _series(close=50.0, volume=1_000_000.0)}
        ciks = {rich: "0000000001", gone: "0000000002"}

        # Act
        universe, _, _ = _build(panel, [rich, gone], ciks)

        # Assert — missing price is its own category, distinct from illiquid.
        assert gone in universe.excluded_no_price
        assert gone not in universe.excluded_illiquid
        assert universe.excluded_no_price_count == 1

    def test_missing_cik_is_counted_separately_not_silenced(self) -> None:
        # Arrange — both liquid, but NOID is absent from the SEC map.
        rich = Ticker("RICH")
        noid = Ticker("NOID")
        panel = {
            rich: _series(close=50.0, volume=1_000_000.0),
            noid: _series(close=60.0, volume=2_000_000.0),
        }
        ciks = {rich: "0000000001"}

        # Act
        universe, _, _ = _build(panel, [rich, noid], ciks)

        # Assert — a liquid name without a CIK is not a candidate; it is counted.
        assert noid not in universe.candidates
        assert noid in universe.excluded_no_cik
        assert universe.excluded_no_cik_count == 1
        assert rich in universe.candidates

    def test_cik_is_attached_to_each_candidate(self) -> None:
        # Arrange
        rich = Ticker("RICH")
        panel = {rich: _series(close=50.0, volume=1_000_000.0)}
        ciks = {rich: "0000012345"}

        # Act
        universe, _, _ = _build(panel, [rich], ciks)

        # Assert — the resolved CIK rides along for the EDGAR step downstream.
        assert universe.cik_for(rich) == "0000012345"


class TestDeterministicCap:
    def _liquid_panel(
        self, n: int
    ) -> tuple[list[Ticker], dict[Ticker, tuple[PriceBar, ...]], dict[Ticker, str]]:
        tickers = [Ticker(f"T{i:03d}") for i in range(n)]
        # Give each a distinct dollar-volume so the ranking is total and the
        # tie-break never has to fire; higher index -> higher traded value.
        panel = {
            t: _series(close=50.0, volume=100_000.0 + i * 1_000.0) for i, t in enumerate(tickers)
        }
        ciks = {t: str(i + 1).zfill(10) for i, t in enumerate(tickers)}
        return tickers, panel, ciks

    def test_cap_limits_the_candidate_count(self) -> None:
        # Arrange — 50 liquid names, cap of 40.
        tickers, panel, ciks = self._liquid_panel(50)

        # Act
        universe, _, _ = _build(panel, tickers, ciks, cap=40)

        # Assert
        assert len(universe.candidates) == 40

    def test_cap_is_reproducible_byte_for_byte(self) -> None:
        # Arrange
        tickers, panel, ciks = self._liquid_panel(50)

        # Act — build twice from identical inputs.
        first, _, _ = _build(panel, tickers, ciks, cap=40)
        second, _, _ = _build(panel, list(tickers), ciks, cap=40)

        # Assert — identical candidate order and identical overflow bucket.
        assert first.candidates == second.candidates
        assert first.capped_out == second.capped_out

    def test_cap_keeps_the_most_liquid_names(self) -> None:
        # Arrange — index i has dollar-volume increasing in i; the top 40 are
        # the 40 highest-indexed names.
        tickers, panel, ciks = self._liquid_panel(50)

        # Act
        universe, _, _ = _build(panel, tickers, ciks, cap=40)

        # Assert — the 10 thinnest (lowest index) are the ones dropped.
        dropped = set(universe.capped_out)
        assert dropped == {Ticker(f"T{i:03d}") for i in range(10)}
        assert universe.capped_out_count == 10

    def test_cap_overflow_is_not_an_exclusion_category(self) -> None:
        # Arrange
        tickers, panel, ciks = self._liquid_panel(50)

        # Act
        universe, _, _ = _build(panel, tickers, ciks, cap=40)

        # Assert — cap overflow is reported, but it is not a liquidity failure.
        assert set(universe.capped_out).isdisjoint(universe.excluded_illiquid)

    def test_candidates_are_sorted_for_a_stable_report(self) -> None:
        # Arrange — within the kept set, the order on disk is alphabetical so a
        # report diff reflects membership changes, not ranking noise.
        tickers, panel, ciks = self._liquid_panel(45)

        # Act
        universe, _, _ = _build(panel, tickers, ciks, cap=40)

        # Assert
        assert list(universe.candidates) == sorted(universe.candidates)


class TestEndLag:
    def test_bars_are_requested_with_a_one_day_lag(self) -> None:
        # Arrange — the measured HTTP-403 workaround asks for ``as_of - 1``.
        rich = Ticker("RICH")
        panel = {rich: _series(close=50.0, volume=1_000_000.0)}
        ciks = {rich: "0000000001"}

        # Act
        _, bars, _ = _build(panel, [rich], ciks)

        # Assert — no request reaches for the as-of day itself.
        assert bars.requested_ends
        assert all(end < _AS_OF for end in bars.requested_ends)


class TestReportIntegrity:
    def test_counts_sum_to_the_requested_total(self) -> None:
        # Arrange — one of each: liquid+cik, illiquid, no-price, no-cik.
        good = Ticker("GOOD")
        peny = Ticker("PENY")
        gone = Ticker("GONE")
        noid = Ticker("NOID")
        panel = {
            good: _series(close=50.0, volume=1_000_000.0),
            peny: _series(close=0.50, volume=1_000_000.0),
            noid: _series(close=50.0, volume=1_000_000.0),
        }
        ciks = {good: "0000000001", peny: "0000000002", gone: "0000000003"}
        requested = [good, peny, gone, noid]

        # Act
        universe, _, _ = _build(panel, requested, ciks)

        # Assert — every requested name is accounted for exactly once across the
        # candidate list, the three exclusion buckets and the cap overflow.
        accounted = (
            len(universe.candidates)
            + universe.excluded_illiquid_count
            + universe.excluded_no_price_count
            + universe.excluded_no_cik_count
            + universe.capped_out_count
        )
        assert accounted == len(requested)
        assert universe.requested_count == len(requested)

    def test_universe_is_frozen(self) -> None:
        # Arrange
        rich = Ticker("RICH")
        panel = {rich: _series(close=50.0, volume=1_000_000.0)}
        ciks = {rich: "0000000001"}
        universe, _, _ = _build(panel, [rich], ciks)

        # Act / Assert — the audit record cannot be mutated after the fact.
        with pytest.raises(ValidationError):
            universe.candidates = ()  # type: ignore[misc]

    def test_duplicate_requests_are_deduplicated(self) -> None:
        # Arrange — the same ticker twice must not be double-counted.
        rich = Ticker("RICH")
        panel = {rich: _series(close=50.0, volume=1_000_000.0)}
        ciks = {rich: "0000000001"}

        # Act
        universe, _, _ = _build(panel, [rich, rich], ciks)

        # Assert
        assert universe.requested_count == 1
        assert list(universe.candidates) == [rich]

    def test_empty_request_yields_empty_universe(self) -> None:
        # Arrange / Act
        universe, _, _ = _build({}, [], {})

        # Assert
        assert universe.candidates == ()
        assert universe.requested_count == 0

    def test_invalid_cap_is_rejected(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="cap"):
            _build({}, [], {}, cap=0)
