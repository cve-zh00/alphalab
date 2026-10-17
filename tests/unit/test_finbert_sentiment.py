import json
import re

from alphalab.adapters.finbert_sentiment import (
    FinBertSentimentScorer,
    SentimentSummary,
    lexicon_score,
)
from alphalab.cli.commands.autopilot import _apply_sentiment_penalty
from alphalab.domain.types import Ticker


def test_lexicon_scoring():
    assert lexicon_score("Company beats estimates, raises guidance") > 0
    assert lexicon_score("Company misses, files for bankruptcy") <= -0.9
    assert lexicon_score("Company holds annual meeting") == 0.0
    assert lexicon_score("does not beat estimates") < 0


def test_empty_and_never_raises(tmp_path):
    s = FinBertSentimentScorer(cache_path=tmp_path / "c.json")
    assert s.score_documents([]).count == 0
    assert s.score_documents([None, 3]).count == 0  # type: ignore[list-item]


def test_cache_roundtrip_and_corrupt(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    s = FinBertSentimentScorer(cache_path=p)
    r = s.score_documents(["upgrade and record profit"])
    assert r.positive_count == 1 and r.net_score > 0
    assert len(json.loads(p.read_text())) == 1
    s2 = FinBertSentimentScorer(cache_path=p)
    assert s2.score_documents(["upgrade and record profit"]).net_score == r.net_score


def test_regime_note_no_digits():
    for summ in (
        SentimentSummary(),
        SentimentSummary(net_score=0.5, positive_count=2, count=2, worst_score=0.4),
        SentimentSummary(net_score=-0.5, negative_count=2, count=2, worst_score=-0.5),
        SentimentSummary(net_score=-0.9, negative_count=1, count=1, worst_score=-0.9),
        SentimentSummary(net_score=0.0, neutral_count=1, count=1),
    ):
        assert not re.search(r"\d", summ.regime_note())


def test_prescreen_penalty():
    a, b = Ticker("AAA"), Ticker("BBB")
    scored = [(2.0, 0, a), (1.0, -1, b)]
    out = _apply_sentiment_penalty(scored, lambda t: -0.8 if t == a else 0.1, limit=4)
    assert out[0][2] == b and out[1][0] == 0.5

    def boom(_t):
        raise RuntimeError

    assert _apply_sentiment_penalty(scored, boom, limit=4)[0][2] == a
