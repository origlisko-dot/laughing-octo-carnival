from datetime import datetime, timedelta, timezone

import numpy as np

from ml_trading.data.providers.synthetic import SyntheticProvider
from ml_trading.events.classify import (
    Catalyst,
    CatalystType,
    RuleBasedClassifier,
    catalyst_from_llm_json,
)
from ml_trading.events.impact import ImpactModel, catalyst_features, is_fresh, measure_reaction
from ml_trading.events.ingest import EdgarClient, NewsItem, dedup_items, link_tickers

TS = datetime(2025, 3, 3, 14, 30, tzinfo=timezone.utc)


def item(headline: str, **kw) -> NewsItem:
    base = dict(id=kw.pop("id", headline[:8]), ts=TS, source="finnhub", headline=headline)
    return NewsItem(**{**base, **kw})


def test_rule_classifier_catalyst_types() -> None:
    clf = RuleBasedClassifier()
    cases = {
        "MegaCorp to acquire SmallCo for $2B in all-cash deal": (CatalystType.MERGER_ACQUISITION, 1),
        "BioPharma receives FDA approval for lead drug": (CatalystType.REGULATORY, 1),
        "BioPharma receives complete response letter from FDA": (CatalystType.REGULATORY, -1),
        "TechCo announces strategic alliance with CloudInc": (CatalystType.PARTNERSHIP, 1),
        "RetailCo cuts guidance amid weak demand": (CatalystType.GUIDANCE, -1),
        "Chipmaker faces class action lawsuit over disclosures": (CatalystType.LEGAL, -1),
        "Weather remains sunny in the midwest": (CatalystType.OTHER, 0),
    }
    for headline, (expected_type, expected_dir) in cases.items():
        cat = clf.classify(item(headline))
        assert cat.type is expected_type, headline
        assert cat.direction == expected_dir, headline


def test_dedup_and_entity_linking() -> None:
    a = item("MegaCorp to acquire SmallCo for $2B", id="a")
    b = item("MegaCorp to acquire SmallCo for $2B  ", id="b")  # near-duplicate
    c = item("Unrelated story about markets", id="c")
    out = dedup_items([a, b, c])
    assert [i.id for i in out] == ["a", "c"]

    linked = link_tickers(item("MegaCorp beats estimates"), {"MegaCorp": "MEGA"})
    assert linked.tickers == ["MEGA"]


def test_edgar_atom_parsing() -> None:
    xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>8-K - ACME CORP (0000123456)</title>
        <updated>2025-03-03T14:31:00-04:00</updated>
        <link href="https://www.sec.gov/Archives/acme-8k.htm"/>
      </entry>
    </feed>"""
    items = EdgarClient.parse_atom(xml)
    assert len(items) == 1
    assert items[0].form_type == "8-K"
    assert "ACME" in items[0].headline
    assert items[0].url.endswith("acme-8k.htm")


def test_llm_json_mapping_defensive() -> None:
    it = item("TechCo announces breakthrough")
    good = catalyst_from_llm_json(
        it, {"type": "tech_breakthrough", "direction": 1, "strength": 0.8, "rationale": "x"}
    )
    assert good.type is CatalystType.TECH_BREAKTHROUGH and good.strength == 0.8

    bad = catalyst_from_llm_json(it, {"type": "nonsense", "direction": 9, "strength": 7.5})
    assert bad.type is CatalystType.OTHER
    assert bad.direction == 0
    assert bad.strength == 1.0  # clipped


def test_measure_reaction_detects_planted_jump() -> None:
    bars = SyntheticProvider(seed=11).fetch("TEST", "5m", "2025-03-03", "2025-03-05")
    event_ts = bars["ts"][len(bars) // 2]
    # plant a +5% jump with elevated noise after the event
    import polars as pl

    jumped = bars.with_columns(
        pl.when(pl.col("ts") >= event_ts)
        .then(pl.col("close") * 1.05)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    r = measure_reaction(jumped, event_ts, window_bars=12)
    assert r is not None
    assert r.abnormal_return > 0.03
    assert r.abnormal_vol_ratio > 0.0


def test_impact_model_learns_type_effect() -> None:
    rng = np.random.default_rng(0)
    clf = RuleBasedClassifier()
    cats, reacts = [], []
    from ml_trading.events.impact import EventReaction

    for i in range(400):
        if rng.random() < 0.5:
            c = clf.classify(item(f"Corp{i} to acquire Target{i}", id=f"m{i}"))
            reacts.append(EventReaction(abnormal_return=0.04 + rng.normal(0, 0.01),
                                        abnormal_vol_ratio=3.0 + rng.normal(0, 0.3)))
        else:
            c = Catalyst(item=item(f"Corp{i} misc note", id=f"o{i}"), type=CatalystType.OTHER,
                         direction=0, strength=0.0)
            reacts.append(EventReaction(abnormal_return=rng.normal(0, 0.005),
                                        abnormal_vol_ratio=1.0 + rng.normal(0, 0.1)))
        cats.append(c)

    model = ImpactModel().fit(cats, reacts)
    ma = clf.classify(item("BigCo to acquire LittleCo", id="new1"))
    other = Catalyst(item=item("BigCo misc note", id="new2"), type=CatalystType.OTHER,
                     direction=0, strength=0.0)
    exp_ret_ma, exp_vol_ma = model.predict(ma)
    exp_ret_o, exp_vol_o = model.predict(other)
    assert exp_ret_ma > exp_ret_o
    assert exp_vol_ma > 2.0 > exp_vol_o
    assert model.event_signal(ma) > 0.0
    assert model.event_signal(other) == 0.0


def test_freshness_gate() -> None:
    cat = Catalyst(item=item("x"), type=CatalystType.OTHER, direction=0, strength=0.0)
    assert is_fresh(cat, TS + timedelta(minutes=30))
    assert not is_fresh(cat, TS + timedelta(hours=3))


def test_catalyst_feature_vector_shape() -> None:
    cat = RuleBasedClassifier().classify(item("MegaCorp to acquire SmallCo"))
    assert catalyst_features(cat).shape == (len(CatalystType) + 5,)
