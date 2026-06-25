from __future__ import annotations

"""Keyword & topic detection engine — generalized from ASR swear detection."""


import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

MatchType = Literal["exact", "fuzzy", "regex", "semantic", "phrase"]

TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü@!*]+")

# "zam" should not match "zaman", "zamanda", etc.
KEYWORD_PREFIX_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "zam": ("zaman", "zamanda", "zamanla", "zamanı", "zamanin", "zamanın"),
}

# Turkish semantic synonyms for topic expansion
SEMANTIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "zam": ("fiyat artışı", "fiyat artisi", "tarife yükseltme", "zamlı", "zamli", "ücret artışı"),
    "iptal": ("abonelik iptali", "hattı kapat", "hatti kapat", "sözleşme iptali"),
    "şikayet": ("sikayet", "memnun değilim", "memnun degilim", "kötü hizmet", "kotu hizmet"),
    "iade": ("para iadesi", "geri ödeme", "geri odeme", "refund"),
    "fatura": ("fatura itirazı", "fatura itirazi", "yanlış yansıma", "yanlis yansima", "fatura hatası"),
}


@dataclass(frozen=True)
class SegmentInput:
    start: float
    end: float
    text: str
    speaker: str | None = None
    segment_index: int = 0


@dataclass(frozen=True)
class RuleInput:
    id: str
    name: str
    keywords: tuple[str, ...]
    match_mode: str = "exact"
    fuzzy_threshold: float = 0.85
    case_sensitive: bool = False
    severity: str = "info"
    topic_id: str | None = None
    sector_scope: tuple[str, ...] | None = None


@dataclass(frozen=True)
class KeywordHitResult:
    rule_id: str
    rule_name: str
    keyword: str
    matched_text: str
    match_type: MatchType
    confidence: float
    timestamp_sec: float
    segment_index: int
    speaker: str | None
    context: str
    severity: str
    topic_id: str | None = None


def normalize_token(value: str) -> str:
    value = (value or "").lower().strip()
    value = value.replace("1", "i").replace("!", "i").replace("@", "a").replace("0", "o")
    value = value.replace("*", "")
    value = re.sub(r"(.)\1{2,}", r"\1\1", value)
    return value


def fuzzy_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def is_excluded_hit(keyword: str, token: str) -> bool:
    exclusions = KEYWORD_PREFIX_EXCLUSIONS.get(normalize_token(keyword), ())
    token_norm = normalize_token(token)
    return any(token_norm.startswith(ex) for ex in exclusions)


def expand_keywords(keyword: str, match_mode: str) -> list[str]:
    terms = [keyword]
    if match_mode == "semantic":
        norm = normalize_token(keyword)
        for key, syns in SEMANTIC_SYNONYMS.items():
            if normalize_token(key) == norm:
                terms.extend(syns)
                break
    return terms


def build_match_sets(keywords: Sequence[str], match_mode: str):
    exact_terms: set[str] = set()
    phrase_terms: list[tuple[str, ...]] = []
    regex_patterns: list[re.Pattern] = []

    for kw in keywords:
        for term in expand_keywords(kw, match_mode):
            if match_mode == "regex":
                try:
                    regex_patterns.append(re.compile(term, re.IGNORECASE))
                except re.error:
                    pass
                continue
            normalized = normalize_token(term)
            if not normalized:
                continue
            parts = normalized.split()
            if len(parts) > 1:
                phrase_terms.append(tuple(parts))
            else:
                exact_terms.add(normalized)

    phrase_terms.sort(key=lambda p: len(p), reverse=True)
    return exact_terms, phrase_terms, regex_patterns


def extract_context(text: str, start_idx: int, end_idx: int, window: int = 40) -> str:
    return (text or "")[max(0, start_idx - window): min(len(text), end_idx + window)].strip()


def analyze_segment(
    segment: SegmentInput,
    rules: Sequence[RuleInput],
    sector: str = "omni",
) -> list[KeywordHitResult]:
    hits: list[KeywordHitResult] = []
    text = segment.text or ""
    if not text.strip():
        return hits

    for rule in rules:
        if rule.sector_scope and sector not in rule.sector_scope:
            continue

        keywords = rule.keywords
        exact_terms, phrase_terms, regex_patterns = build_match_sets(keywords, rule.match_mode)
        tokens = [normalize_token(m.group(0)) for m in TOKEN_PATTERN.finditer(text)]
        tokens = [t for t in tokens if t]
        covered: set[int] = set()
        seen_keys: set[tuple[str, str]] = set()

        def add_hit(keyword: str, matched: str, match_type: MatchType, confidence: float):
            key = (rule.id, normalize_token(matched))
            if key in seen_keys:
                return
            seen_keys.add(key)
            hits.append(
                KeywordHitResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    keyword=keyword,
                    matched_text=matched,
                    match_type=match_type,
                    confidence=confidence,
                    timestamp_sec=segment.start,
                    segment_index=segment.segment_index,
                    speaker=segment.speaker,
                    context=text[:120],
                    severity=rule.severity,
                    topic_id=rule.topic_id,
                )
            )

        if rule.match_mode == "regex":
            for pattern in regex_patterns:
                for match in pattern.finditer(text):
                    add_hit(match.group(0), match.group(0), "regex", 1.0)
            continue

        for phrase in phrase_terms:
            plen = len(phrase)
            for idx in range(0, max(0, len(tokens) - plen + 1)):
                if tuple(tokens[idx: idx + plen]) == phrase:
                    matched = " ".join(phrase)
                    add_hit(matched, matched, "phrase", 1.0)
                    covered.update(range(idx, idx + plen))

        for idx, token in enumerate(tokens):
            if idx in covered:
                continue
            for keyword in keywords:
                norm_kw = normalize_token(keyword)
                if not norm_kw:
                    continue
                if is_excluded_hit(keyword, token):
                    continue

                if rule.match_mode in ("exact", "semantic") and token == norm_kw:
                    add_hit(keyword, token, "exact" if rule.match_mode == "exact" else "semantic", 1.0)
                    break

                if rule.match_mode == "fuzzy" and len(norm_kw) >= 3:
                    ratio = fuzzy_ratio(token, norm_kw)
                    if ratio >= rule.fuzzy_threshold:
                        add_hit(keyword, token, "fuzzy", ratio)
                        break

        if rule.match_mode == "semantic":
            lower_text = text.lower()
            for keyword in keywords:
                for synonym in expand_keywords(keyword, "semantic"):
                    if " " in synonym and synonym.lower() in lower_text:
                        add_hit(keyword, synonym, "semantic", 0.9)

    return hits


def analyze_keywords(
    segments: Sequence[SegmentInput],
    rules: Sequence[RuleInput],
    *,
    sector: str = "omni",
) -> list[KeywordHitResult]:
    all_hits: list[KeywordHitResult] = []
    for idx, segment in enumerate(segments):
        seg = SegmentInput(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=segment.speaker,
            segment_index=idx if segment.segment_index == 0 else segment.segment_index,
        )
        all_hits.extend(analyze_segment(seg, rules, sector=sector))
    return all_hits


def evaluate_rule_on_text(text: str, rule: RuleInput) -> list[KeywordHitResult]:
    segment = SegmentInput(start=0.0, end=0.0, text=text, segment_index=0)
    return analyze_segment(segment, [rule])


def hits_to_dict(hits: Sequence[KeywordHitResult]) -> list[dict]:
    return [
        {
            "rule_id": h.rule_id,
            "rule_name": h.rule_name,
            "keyword": h.keyword,
            "matched_text": h.matched_text,
            "match_type": h.match_type,
            "confidence": round(h.confidence, 3),
            "timestamp_sec": h.timestamp_sec,
            "segment_index": h.segment_index,
            "speaker": h.speaker,
            "context": h.context,
            "severity": h.severity,
            "topic_id": h.topic_id,
        }
        for h in hits
    ]
