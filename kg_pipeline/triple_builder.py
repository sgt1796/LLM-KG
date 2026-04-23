"""Module for constructing a simple knowledge graph from subject–relation–object
triples.

The goal of this builder is to go beyond plain co‑occurrence counts and
associate a free‑text relation with each pair of entities extracted from
a sentence.  Because the runtime environment does not provide a heavy
dependency stack for syntactic parsing, the relation extraction here
uses a very lightweight heuristic: the token span between two entity
mentions is taken as the relation.  If no reasonable relation text
exists, a generic string of ``"related_to"`` is used instead.  Multiple
triples occurring across sentences are accumulated and their weights
incremented.

Users of this class should pass in an iterable of ``(sentence, entities)``
pairs, as produced by the NER extractors in this package.  Each
sentence will be examined for every unordered pair of entities and a
triple will be produced.  Singleton entities are still recorded as
nodes but do not contribute any triples.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple
import re
from kg_pipeline.provenance import DocContext, Evidence
from kg_pipeline.label_store import LabelStore

# -----------------------------------------------------------------------------
# Canonical relation patterns and their synonyms
#
# These mappings expand on the original heuristic for relation extraction by
# recognising common biomedical phrasing and rewriting them to a small set of
# canonical predicates.  Patterns are declared at module scope so they can be
# reused by both the relation extractor and the graph builder when checking
# relative positions of verbs.

_CANON_SYNONYMS: Dict[str, List[str]] = {
    "associated with": [
        "is associated with", "associated with", "associated to", "linked to", "linked with",
        "is linked to", "links to", "link to", "link with", "related with", "related to"
    ],
    "causes": [
        "leads to", "results in", "cause of", "causes", "lead to", "result in",
        "affects", "affect", "induces", "induce", "triggers", "trigger", "triggered",
        "increase risk of", "increase risk for", "raises risk of", "raises risk for"
        , "increases the risk of", "increases the risk for", "increases risk of", "increases risk for",
        "increase the risk of", "increase the risk for"
    ],
    "risk for": [
        "risk for", "risk of", "risk factor for", "risk factors for"
    ],
    "interacts with": [
        "interacts with", "interact with", "interacts"
    ],
    "binds": [
        "binds to", "binds with", "binds", "bind to", "bind with", "bind", "binds onto"
    ],
    "regulates": [
        "regulates", "regulate", "controls", "control", "modulates", "modulate"
    ],
    "inhibits": [
        "inhibits", "inhibit", "suppresses", "suppress", "downregulates", "downregulate"
    ],
    "activates": [
        "activates", "activate", "stimulates", "stimulate", "upregulates", "upregulate"
    ],
    "increases": [
        "increases", "increase", "enhances", "enhance", "raises", "raise"
    ],
    "decreases": [
        "decreases", "decrease", "reduces", "reduce", "lowers", "lower"
    ],
    "treats": [
        "treats", "treat", "therapy for", "remedy for", "cures", "cure"
    ],
    "prevents": [
        "prevents", "prevent", "avoids", "avoid", "protects against", "protect against"
    ],
    "predicts": [
        "predicts", "predict"
    ],
    "correlates with": [
        "correlates with", "correlate with", "correlates", "correlate"
    ],
    "promotes": [
        "promotes", "promote", "facilitates", "facilitate"
    ],
    "suppresses": [
        "suppresses", "suppress", "down-regulates", "down-regulate"
    ],
    "mediates": [
        "mediates", "mediate"
    ],
    "expresses": [
        "expresses", "express"
    ],
    "encodes": [
        "encodes", "encode"
    ],
}

_INVERTED_SYNONYMS: Dict[str, List[str]] = {
    "causes": [
        "caused by", "induced by", "resulting from", "results from", "triggered by"
    ],
    "risk for": [
        "risk from", "risk due to", "risk attributable to"
    ],
    "inhibits": [
        "is inhibited by", "inhibited by", "suppressed by", "downregulated by"
    ],
    "activates": [
        "is activated by", "activated by", "stimulated by", "upregulated by"
    ],
    "increases": [
        "is increased by", "increased by", "enhanced by", "raised by"
    ],
    "decreases": [
        "is decreased by", "decreased by", "reduced by", "lowered by"
    ],
    "treats": [
        "is treated with", "treated with", "therapy with"
    ],
}

_JUNK = {"and", "of", "with", "in", "for", "the", "between", "to", "by", "on", "from", "at", "as"}
_ALLOWED_ENUM_TOKENS = {"and", "or"}
_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b")
_NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|never|without|neither|nor|"
    r"fail(?:ed|s|ing)?\s+to|lack(?:ed|s|ing)?(?:\s+of)?|"
    r"did\s+not|does\s+not|do\s+not|cannot|can't)\b",
    re.IGNORECASE,
)
_HEDGE_PATTERN = re.compile(
    r"\b(?:may|might|could|possibly|potentially|suggest(?:s|ed)?|"
    r"appear(?:s|ed)?(?:\s+to)?|seem(?:s|ed)?(?:\s+to)?|likely|unlikely)\b",
    re.IGNORECASE,
)
_CONTEXT_WINDOW = 32
_GUARD_WINDOW = 48


@dataclass(frozen=True)
class RelationPattern:
    canonical: str
    synonym: str
    direction: int
    pattern: re.Pattern[str] = field(compare=False, repr=False)


@dataclass(frozen=True)
class RelationMatch:
    relation: str
    direction: int
    source: str
    match_text: str
    match_span: Tuple[int, int]


@dataclass(frozen=True)
class MentionSpan:
    entity: str
    start: int
    end: int


@dataclass(frozen=True)
class TripletCandidate:
    subject: str
    relation: str
    object: str
    subject_span: Tuple[int, int]
    object_span: Tuple[int, int]
    relation_span: Tuple[int, int]
    relation_source: str


def _compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Compile a token-boundary regex for a relation phrase."""
    parts = [re.escape(part) for part in phrase.split()]
    body = r"\s+".join(parts) if parts else re.escape(phrase)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _build_relation_patterns() -> List[RelationPattern]:
    """Create ordered relation patterns, preferring longer phrases first."""
    patterns: List[RelationPattern] = []
    seen: Set[Tuple[str, str, int]] = set()
    for canonical, synonyms in _CANON_SYNONYMS.items():
        for synonym in synonyms:
            key = (canonical, synonym.casefold(), 1)
            if key in seen:
                continue
            seen.add(key)
            patterns.append(
                RelationPattern(
                    canonical=canonical,
                    synonym=synonym,
                    direction=1,
                    pattern=_compile_phrase_pattern(synonym),
                )
            )
    for canonical, synonyms in _INVERTED_SYNONYMS.items():
        for synonym in synonyms:
            key = (canonical, synonym.casefold(), -1)
            if key in seen:
                continue
            seen.add(key)
            patterns.append(
                RelationPattern(
                    canonical=canonical,
                    synonym=synonym,
                    direction=-1,
                    pattern=_compile_phrase_pattern(synonym),
                )
            )
    patterns.sort(key=lambda spec: len(spec.synonym), reverse=True)
    return patterns


_RELATION_PATTERNS = _build_relation_patterns()


def _has_guard(text: str, start: int, end: int) -> bool:
    """Return True when a nearby negation or hedge should suppress a match."""
    window_start = max(0, start - _GUARD_WINDOW)
    window_end = min(len(text), end + 16)
    local = text[window_start:window_end]
    rel_start = start - window_start
    rel_end = end - window_start
    prefix = local[:rel_start]
    scope = f"{prefix[-_GUARD_WINDOW:]} {local[rel_start:rel_end]} {local[rel_end:rel_end + 16]}"
    if _NEGATION_PATTERN.search(scope):
        return True
    if _HEDGE_PATTERN.search(prefix[-_GUARD_WINDOW:]):
        return True
    return False


def _search_relation_source(text: str, *, base_offset: int, source: str) -> RelationMatch | None:
    """Search one source segment for the best relation match."""
    for spec in _RELATION_PATTERNS:
        match = spec.pattern.search(text)
        if not match:
            continue
        if _has_guard(text, match.start(), match.end()):
            continue
        return RelationMatch(
            relation=spec.canonical,
            direction=spec.direction,
            source=source,
            match_text=match.group(0),
            match_span=(base_offset + match.start(), base_offset + match.end()),
        )
    return None


def _extract_relation(
    span: str,
    sentence: str = "",
    left_ctx: str = "",
    right_ctx: str = "",
    *,
    span_offset: int = 0,
    left_offset: int = 0,
    right_offset: int = 0,
) -> RelationMatch | None:
    """Extract a relation match and its direction from a mention pair context."""
    cleaned = span.strip() if span else ""
    low = cleaned.casefold()

    if low and len(low) <= 80 and not all(tok in _JUNK for tok in low.split()):
        match = _search_relation_source(span, base_offset=span_offset, source="span")
        if match is not None:
            return match

    if left_ctx:
        match = _search_relation_source(left_ctx, base_offset=left_offset, source="left_ctx")
        if match is not None:
            return match

    if right_ctx:
        match = _search_relation_source(right_ctx, base_offset=right_offset, source="right_ctx")
        if match is not None:
            return match

    return None

@dataclass
class TripletKnowledgeGraphBuilder:
    """Builds a simple knowledge graph of subject–relation–object triples.

    Nodes are entity strings and triples connect a subject to an object
    with a relation label.  Weights count the number of times the same
    (subject, relation, object) triple was observed across sentences.
    """
    nodes: Set[str] = field(default_factory=set)
    # triples[(h,r,t)] = {"weight": int, "sources": List[dict]}
    triples: Dict[Tuple[str, str, str], Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(lambda: {"weight": 0, "sources": []})
    )
    _seen_evidence: Set[Tuple] = field(default_factory=set)  # dedupe key 
    label_store: LabelStore | None = None  # NEW: optional shared store for dynamic labels

    def _iter_fallback_occurrences(self, sentence: str, entity: str) -> List[Tuple[int, int]]:
        """Return approximate substring occurrences when boundary matching fails."""
        spans: List[Tuple[int, int]] = []
        sent_low = sentence.casefold()
        ent_low = entity.casefold()
        start = 0
        while True:
            idx = sent_low.find(ent_low, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(entity)))
            start = idx + len(entity)
        return spans

    def _entity_mentions(self, sentence: str, entities: Set[str]) -> List[MentionSpan]:
        """Return every detected mention span for the provided entities."""
        mentions: List[MentionSpan] = []
        seen: Set[Tuple[int, int, str]] = set()
        for ent in sorted(entities, key=lambda value: (-len(value), value.casefold())):
            ent_clean = ent.strip()
            if not ent_clean:
                continue
            pattern = _compile_phrase_pattern(ent_clean)
            matches = list(pattern.finditer(sentence))
            spans = [(match.start(), match.end()) for match in matches]
            if not spans:
                spans = self._iter_fallback_occurrences(sentence, ent_clean)
            for start, end in spans:
                key = (start, end, ent_clean.casefold())
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(MentionSpan(entity=ent_clean, start=start, end=end))
        mentions.sort(key=lambda mention: (mention.start, -(mention.end - mention.start), mention.entity.casefold()))
        return mentions

    def _entity_positions(self, sentence: str, entities: Set[str]) -> List[Tuple[int, str]]:
        """Return all entity start positions, preserving duplicate mentions."""
        return [(mention.start, mention.entity) for mention in self._entity_mentions(sentence, entities)]

    def _entity_token_set(self, mentions: List[MentionSpan]) -> Set[str]:
        """Return lower-cased tokens seen in entity strings."""
        tokens: Set[str] = set()
        for mention in mentions:
            for token in _TOKEN_PATTERN.findall(mention.entity.casefold()):
                tokens.add(token)
        return tokens

    def _should_skip_mention_pair(
        self,
        sentence: str,
        mentions: List[MentionSpan],
        left_idx: int,
        right_idx: int,
        match: RelationMatch,
    ) -> bool:
        """Suppress remote mention pairs that cross other entities or clauses."""
        if right_idx <= left_idx + 1:
            return False
        if match.source != "span":
            return True

        right_mention = mentions[right_idx]
        intermediate_mentions = mentions[left_idx + 1:right_idx]
        first_intermediate = intermediate_mentions[0]
        if match.match_span[0] >= first_intermediate.start:
            return True

        entity_tokens = self._entity_token_set(mentions)
        sub_span = sentence[first_intermediate.end:right_mention.start].casefold()
        words = _TOKEN_PATTERN.findall(sub_span)
        return any(
            (word not in _ALLOWED_ENUM_TOKENS) and (word not in entity_tokens)
            for word in words
        )

    def _extract_triplet_candidates(self, sentence: str, entities: Set[str]) -> List[TripletCandidate]:
        """Return raw triplet candidates with mention span metadata."""
        result: List[TripletCandidate] = []
        if not entities or len(entities) < 2:
            return result

        mentions = self._entity_mentions(sentence, entities)
        if len(mentions) < 2:
            return result

        seen: Set[Tuple[str, str, str, Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = set()
        for left_idx, left_mention in enumerate(mentions):
            for right_idx in range(left_idx + 1, len(mentions)):
                right_mention = mentions[right_idx]
                if left_mention.end > right_mention.start:
                    continue
                if left_mention.entity.casefold() == right_mention.entity.casefold():
                    continue

                span_start = left_mention.end
                span_end = right_mention.start
                match = _extract_relation(
                    sentence[span_start:span_end],
                    sentence=sentence,
                    left_ctx=sentence[max(0, span_start - _CONTEXT_WINDOW):span_start],
                    right_ctx=sentence[span_end:min(len(sentence), span_end + _CONTEXT_WINDOW)],
                    span_offset=span_start,
                    left_offset=max(0, span_start - _CONTEXT_WINDOW),
                    right_offset=span_end,
                )
                if match is None:
                    continue
                if self._should_skip_mention_pair(sentence, mentions, left_idx, right_idx, match):
                    continue

                if match.direction >= 0:
                    subject_mention = left_mention
                    object_mention = right_mention
                else:
                    subject_mention = right_mention
                    object_mention = left_mention

                key = (
                    subject_mention.entity,
                    match.relation,
                    object_mention.entity,
                    (subject_mention.start, subject_mention.end),
                    (object_mention.start, object_mention.end),
                    match.match_span,
                )
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    TripletCandidate(
                        subject=subject_mention.entity,
                        relation=match.relation,
                        object=object_mention.entity,
                        subject_span=(subject_mention.start, subject_mention.end),
                        object_span=(object_mention.start, object_mention.end),
                        relation_span=match.match_span,
                        relation_source=match.source,
                    )
                )
        return result

    def _canonicalise_relation(self, relation: str) -> str | None:
        """Map a detected relation into a persisted canonical label using LabelStore."""
        if not relation or relation == "__SKIP__":
            return None
        if self.label_store is None:
            return relation

        canon = self.label_store.observe(relation)
        if canon and self.label_store.should_emit(canon):
            return canon

        # If not yet promoted (too rare / noisy), skip emitting to avoid label blow-up.
        return None

    def add_sentence(self, sentence: str, entities: Set[str],
                     *, context: DocContext | None = None, sentence_id: int = 0) -> None:
        """Update the internal graph with relational triples for a single sentence.

        This method records each entity as a node and then delegates to
        :meth:`extract_triplets` to compute candidate subject–relation–object
        tuples.  Only canonical relations (as identified by
        ``_extract_relation``) are considered – generic ``related_to`` links
        and junk spans are ignored.  The weight for each triple is
        incremented by one for every occurrence.

        Parameters
        ----------
        sentence : str
            The original sentence text.
        entities : set of str
            The unique entity names found in the sentence.
        """
        # Record every entity as a node, regardless of pairwise relations
        for e in entities:
            self.nodes.add(e)
        # Extract meaningful triples from this sentence.
        # Relation canonicalisation happens here exactly once so LabelStore
        # counts are not inflated by helper calls.
        for candidate in self._extract_triplet_candidates(sentence, entities):
            relation = self._canonicalise_relation(candidate.relation)
            if not relation:
                continue
            key = (candidate.subject, relation, candidate.object)
            if context is None:
                # legacy behavior: just count
                self.triples[key]["weight"] += 1
            else:
                char_span = (
                    min(candidate.subject_span[0], candidate.object_span[0]),
                    max(candidate.subject_span[1], candidate.object_span[1]),
                )

                ev = Evidence(
                    doc_id=context.doc_id,
                    doc_meta=context.doc_meta,
                    chunk_id=context.chunk_id,
                    sentence_id=sentence_id,
                    page=context.page_hint,
                    char_span=char_span,
                    evidence=sentence,
                    confidence=1.0,
                )
                k = ev.dedupe_key()
                if k not in self._seen_evidence:
                    self._seen_evidence.add(k)
                    self.triples[key]["sources"].append({
                        "doc_id": ev.doc_id,
                        "doc_meta": ev.doc_meta,
                        "chunk_id": ev.chunk_id,
                        "sentence_id": ev.sentence_id,
                        "page": ev.page,
                        "char_span": list(ev.char_span),
                        "evidence": ev.evidence,
                        "confidence": ev.confidence,
                    })
                    self.triples[key]["weight"] += 1

    def extract_triplets(self, sentence: str, entities: Set[str]) -> List[Tuple[str, str, str]]:
        """Return a list of relational triples present in a sentence.

        This helper computes subject–relation–object tuples for all
        ordered pairs of distinct entities that co‑occur in the provided
        sentence.  It uses the same heuristics as :meth:`add_sentence`
        to detect canonical relations and avoid spurious links.  Generic
        ``related_to`` relations and spans containing only stopwords are
        omitted from the output.

        Parameters
        ----------
        sentence : str
            The full sentence containing the entity mentions.
        entities : set of str
            The unique entity strings extracted from the sentence.

        Returns
        -------
        list of tuples
            A list of ``(subject, relation, object)`` triplets for which
            a meaningful relation was detected.
        """
        return [
            (candidate.subject, candidate.relation, candidate.object)
            for candidate in self._extract_triplet_candidates(sentence, entities)
        ]

    def build_from_sentences(
        self,
        sentence_entities: Iterable[Tuple[str, Set[str]]],
        *,
        context: DocContext | None = None,
        start_sentence_id: int = 0,
    ) -> None:
        """Populate the graph from an iterable of (sentence, entities) tuples.

        If ``context`` is provided, evidence rows are attached per sentence,
        and triple weights reflect unique evidence occurrences (deduped).
        """
        sid = start_sentence_id
        for sent, ents in sentence_entities:
            if ents:
                self.add_sentence(sent, ents, context=context, sentence_id=sid)
            sid += 1

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        """Convert the graph to a JSON‑serialisable dictionary.

        The returned dictionary has two keys:

        - ``nodes``: a sorted list of unique entity strings
        - ``triples``: a list of dictionaries with keys
          ``subject``, ``relation``, ``object`` and ``weight``.
        """
        nodes_list = sorted(self.nodes)
        triples_list = []
        for (subj, rel, obj), data in sorted(self.triples.items()):
            triples_list.append({
                "subject": subj,
                "relation": rel,
                "object": obj,
                "weight": data.get("weight", 0),
                "sources": data.get("sources", []),  # <- provenance in JSON
            })
        return {"nodes": nodes_list, "triples": triples_list}
