"""Catalyst classification ladder.

Level 1: rule-based keyword classifier (always available, fully transparent).
Level 2: FinBERT sentiment (optional `events-nlp` extra) refining direction.
Level 3: LLM-over-API with a structured prompt (bring your own client), for
         nuanced catalysts; its JSON output maps onto the same `Catalyst` type.

Every level emits the same structure so downstream code doesn't care which
level produced the classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ml_trading.events.ingest import NewsItem

try:
    from transformers import pipeline as hf_pipeline
except ImportError:  # optional extra
    hf_pipeline = None


class CatalystType(StrEnum):
    MERGER_ACQUISITION = "m&a"
    PARTNERSHIP = "partnership"
    TECH_BREAKTHROUGH = "tech_breakthrough"
    REGULATORY = "regulatory"
    GUIDANCE = "guidance"
    LEGAL = "legal"
    OTHER = "other"


@dataclass
class Catalyst:
    item: NewsItem
    type: CatalystType
    direction: int  # +1 bullish, -1 bearish, 0 unclear
    strength: float  # 0..1
    rationale: str = ""


_PATTERNS: list[tuple[CatalystType, int, float, str]] = [
    (CatalystType.MERGER_ACQUISITION, 1, 0.9, r"\b(acquir\w+|merger|takeover|buyout|to acquire)\b"),
    (CatalystType.MERGER_ACQUISITION, 1, 0.7, r"\b(m&a|tender offer|all-cash deal)\b"),
    (
        CatalystType.PARTNERSHIP, 1, 0.6,
        r"\b(partnership|collaborat\w+|joint venture|strategic alliance)\b",
    ),
    (CatalystType.TECH_BREAKTHROUGH, 1, 0.7, r"\b(breakthrough|patent granted|milestone|first-in-class)\b"),
    (CatalystType.REGULATORY, 1, 0.8, r"\b(fda approv\w+|clearance granted|approval received)\b"),
    (CatalystType.REGULATORY, -1, 0.8, r"\b(fda reject\w+|crl|complete response letter|recall\w*)\b"),
    (CatalystType.GUIDANCE, 1, 0.7, r"\b(raises? (?:full.year |annual )?guidance|beats? estimates)\b"),
    (
        CatalystType.GUIDANCE, -1, 0.7,
        r"\b(cuts? guidance|lowers? guidance|misses? estimates|profit warning)\b",
    ),
    (CatalystType.LEGAL, -1, 0.6, r"\b(lawsuit|class action|investigation|subpoena|fraud)\b"),
]


class RuleBasedClassifier:
    def classify(self, item: NewsItem) -> Catalyst:
        text = f"{item.headline} {item.body}".lower()
        best: tuple[CatalystType, int, float, str] | None = None
        for ctype, direction, strength, pattern in _PATTERNS:
            if re.search(pattern, text):
                if best is None or strength > best[2]:
                    best = (ctype, direction, strength, pattern)
        if best is None:
            return Catalyst(item=item, type=CatalystType.OTHER, direction=0, strength=0.0)
        return Catalyst(
            item=item,
            type=best[0],
            direction=best[1],
            strength=best[2],
            rationale=f"matched /{best[3]}/",
        )


class FinBERTClassifier:
    """Refines a rule-based classification with FinBERT sentiment direction."""

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        if hf_pipeline is None:
            raise RuntimeError(
                "transformers not installed; run: uv pip install 'ml-trading[events-nlp]'"
            )
        self._pipe = hf_pipeline("sentiment-analysis", model=model_name)
        self._rules = RuleBasedClassifier()

    def classify(self, item: NewsItem) -> Catalyst:
        base = self._rules.classify(item)
        res = self._pipe(item.headline[:512])[0]
        label = res["label"].lower()
        score = float(res["score"])
        direction = {"positive": 1, "negative": -1}.get(label, 0)
        if direction != 0:
            base.direction = direction
            base.strength = max(base.strength, score * 0.8)
            base.rationale += f" | finbert={label}:{score:.2f}"
        return base


LLM_PROMPT_TEMPLATE = """You are a financial-catalyst classifier. Given a news item, output strict JSON:
{{"type": one of ["m&a","partnership","tech_breakthrough","regulatory","guidance","legal","other"],
 "direction": -1|0|1, "strength": 0.0-1.0, "rationale": "<one sentence>"}}

Headline: {headline}
Body: {body}
Tickers: {tickers}"""


def catalyst_from_llm_json(item: NewsItem, payload: dict) -> Catalyst:
    """Map a structured LLM response onto a Catalyst (validates fields defensively:
    LLM output is an external boundary)."""
    try:
        ctype = CatalystType(payload["type"])
    except (KeyError, ValueError):
        ctype = CatalystType.OTHER
    direction = payload.get("direction", 0)
    direction = int(direction) if direction in (-1, 0, 1) else 0
    strength = float(payload.get("strength", 0.0))
    return Catalyst(
        item=item,
        type=ctype,
        direction=direction,
        strength=min(max(strength, 0.0), 1.0),
        rationale=str(payload.get("rationale", ""))[:500],
    )
