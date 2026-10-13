"""Required demo: 50 random strategies must count as 50 trials, not 1.

This test encodes the core research insight. An autonomous agent that loops
over strategy ideas inflates the trial count silently; the Deflated Sharpe
Ratio's ``SR0`` depends on that count being honest. We register 50 random
strategies — deliberately seeding duplicate specs under permuted key orders —
and assert the ledger reports 50 *trials* (executions), while separately
exposing the smaller number of unique *specifications*. Under-counting here
would let the DSR certify pure noise.
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from alphalab.adapters.ledger_sqlite import SqliteTrialLedger
from alphalab.domain.ledger import TrialMetrics, TrialRecord


def _random_spec(rng: random.Random) -> dict[str, object]:
    return {
        "lookback": rng.randint(5, 200),
        "threshold": round(rng.uniform(0.1, 3.0), 3),
        "use_vol_target": rng.choice([True, False]),
        "universe": rng.choice(["sp500", "nasdaq100", "russell2000"]),
    }


def _permute_keys(spec: dict[str, object], rng: random.Random) -> dict[str, object]:
    items = list(spec.items())
    rng.shuffle(items)
    return dict(items)


def test_fifty_random_strategies_count_as_fifty_trials(tmp_path: Path) -> None:
    # Arrange
    rng = random.Random(1234)
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    n_trials = 50

    # Build 50 specs: 40 random, plus 10 that are duplicates of earlier specs
    # re-submitted with their keys permuted (same spec_hash, different trial).
    specs: list[dict[str, object]] = [_random_spec(rng) for _ in range(40)]
    for i in range(10):
        donor = specs[i]
        specs.append(_permute_keys(donor, rng))
    assert len(specs) == n_trials

    with SqliteTrialLedger(tmp_path / "demo.db") as ledger:
        # Act: register every strategy as its own trial.
        for i, spec in enumerate(specs):
            record = TrialRecord.create(
                spec=spec,
                dataset_hash="demo-dataset",
                family="agent_loop",
                window_start=date(2015, 1, 1),
                window_end=date(2020, 1, 1),
                params=spec,
                metrics=TrialMetrics(
                    sharpe_gross=round(rng.uniform(-1.0, 3.0), 3),
                    sharpe_net=round(rng.uniform(-1.5, 2.5), 3),
                    return_gross=round(rng.uniform(-0.3, 0.6), 3),
                    return_net=round(rng.uniform(-0.4, 0.5), 3),
                    max_drawdown=-round(rng.uniform(0.0, 0.5), 3),
                ),
                recorded_at=base_time + timedelta(seconds=i),
                llm_version="frontier-2026-09",
                llm_cutoff=date(2026, 9, 1),
            )
            ledger.append(record)

        # Assert: 50 trials, NOT 1 — the honest count that feeds the DSR.
        assert ledger.trial_count() == 50
        assert ledger.trial_count("agent_loop") == 50

        # The duplicated-under-permutation specs share a spec_hash, so the
        # number of unique specifications is strictly smaller than 50 but the
        # trial count is unaffected. This is the distinction that matters.
        unique_specs = ledger.distinct_spec_count("agent_loop")
        assert unique_specs == 40
        assert unique_specs < ledger.trial_count("agent_loop")

        # Every trial is retrievable and each has a distinct trial_id.
        all_trials = list(ledger)
        assert len({t.trial_id for t in all_trials}) == 50
