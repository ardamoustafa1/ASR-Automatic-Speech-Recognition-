# Core engine detecting profanity/abusive language directed at customers or agents.
from __future__ import annotations

"""Turkish Profanity & Abusive Language Detection Engine.

Curated deliberately narrower than a generic Turkish "bad word" list: several
lists circulating for this purpose (including the one in this repo's legacy
Streamlit tool) include ordinary business vocabulary as "profanity" - "mal"
(goods/merchandise), "hasta" (sick/patient - fatal for a healthcare sector
deployment), "top" (top-up), "parasız" (free-of-charge, as in "parasız
kargo"), "hayvan"/animal names used literally, "deli" (used constantly as a
harmless intensifier, "deli gibi çalıştım"). Shipping that list to a bank or
telecom would flag ordinary calls constantly and get the whole feature
disabled by its first angry QA reviewer. Only words/phrases that are
profanity in every realistic contact-center context are included here.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from asr_pro.core.keyword_engine import SegmentInput

# Explicit vulgarity, sexual/anatomical slurs, and their common leetspeak/
# spacing evasions. Word-boundary matched - "am" alone is NOT included
# (it's a syllable fragment in dozens of ordinary Turkish words), only its
# unambiguous vulgar compounds are.
_EXPLICIT_VULGAR = (
    r"sik(?:t[iı]r(?:git)?|erim|ece[kğ]|eyim|sin|im|iş|me|mek|ici?|ilir|ilen|ilmiş)?",
    r"yar+a[kğ](?:[ıi](?:m|n[ıi]|ç[ıi])?)?",
    r"ta[şs]+a[kğ](?:[ıi](?:m|n[ıi]|l[ıi])?)?",
    r"am[ıi]?n[ae]?\s*ko[dy](?:um|ayım|uğum)\w*",
    r"amk|amq|a[@a]m[ıi]na",
    r"orosp[uo](?:[çc]ocu[ğg]u)?\w*|or0spu|orsp\w*",
    r"pi[çc](?:kurusu|ler|lik)?",
    r"g[öo]t(?:veren\w*|o[şs]|herif)",
    r"kahpe|pezevenk|pezo|gavat\w*",
    r"ib+ne\w*",
)

# Aggressive insults/slurs used AS insults (not descriptive/idiomatic use).
# Kept to terms with no common innocent business-call meaning.
_AGGRESSIVE_INSULT = (
    r"şerefsiz|serefsiz",
    r"namussuz",
    r"haysiyetsiz",
    r"ahlaks[ıi]z",
    r"a[şs]a[ğg][ıi]l[ıi]k",
    r"gerizekal[ıi]",
    r"embesil",
    r"dallama",
    r"yavşak|yavsak",
    r"puşt",
)

_ALL_PATTERNS = tuple(
    re.compile(rf"\b(?:{p})\b", re.IGNORECASE) for p in (*_EXPLICIT_VULGAR, *_AGGRESSIVE_INSULT)
)

# Same false-positive shield as compliance_engine: a word appearing inside a
# denial/prohibition ("küfür etmeyin", "böyle konuşmayın") is the AGENT
# correctly de-escalating, not abuse.
_NEGATORS = {"değil", "yok", "yasak", "etmeyin", "kullanmayın", "söylemeyin"}


@dataclass(frozen=True)
class ToxicityResult:
    toxicity_rate: float  # matched words / total words, 0.0-1.0
    matched_terms: tuple[str, ...]  # de-duplicated, order of first appearance
    flagged_segments: tuple[dict, ...]  # {speaker, text, timestamp_sec}
    is_clean: bool


def _is_negated(text_lower: str, match_start: int) -> bool:
    window_start = max(0, text_lower.rfind(" ", 0, match_start) - 30)
    context = text_lower[window_start:match_start]
    return any(neg in context for neg in _NEGATORS)


def analyze_toxicity(segments: Sequence[SegmentInput]) -> ToxicityResult:
    """Scan transcript segments for profanity/abusive language.

    Returns a call-level toxicity rate plus which segments to review - not a
    per-word transcript redaction (compliance/QA needs to read what was
    actually said, not a censored version of it).
    """
    if not segments:
        return ToxicityResult(0.0, (), (), True)

    total_words = 0
    matched_terms: list[str] = []
    flagged: list[dict] = []

    for seg in segments:
        text = seg.text or ""
        if not text.strip():
            continue
        text_lower = text.lower()
        total_words += len(text_lower.split())

        seg_hit = False
        for pattern in _ALL_PATTERNS:
            for m in pattern.finditer(text_lower):
                if _is_negated(text_lower, m.start()):
                    continue
                term = m.group(0)
                if term not in matched_terms:
                    matched_terms.append(term)
                seg_hit = True

        if seg_hit:
            flagged.append(
                {
                    "speaker": seg.speaker,
                    "text": text.strip(),
                    "timestamp_sec": seg.start,
                }
            )

    toxicity_rate = round(len(matched_terms) / total_words, 4) if total_words else 0.0
    return ToxicityResult(
        toxicity_rate=toxicity_rate,
        matched_terms=tuple(matched_terms),
        flagged_segments=tuple(flagged),
        is_clean=not flagged,
    )
