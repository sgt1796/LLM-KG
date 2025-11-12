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
from typing import Dict, Iterable, List, Set, Tuple, Any, Optional
import re
from kg_pipeline.provenance import DocContext, Evidence
from kg_pipeline.label_store import LabelStore

# --- Canonical relation map (now general-domain) ---
_CANON_SYNONYMS: Dict[str, List[str]] = {
    "associated with": [
        "is associated with", "associated with", "associated to", "linked to", "linked with",
        "is linked to", "links to", "link to", "link with", "related with", "related to"
    ],
    "causes": [
        "leads to", "results in", "cause of", "causes", "lead to", "result in",
        "affects", "affect", "induces", "induce", "triggers", "trigger", "triggered",
        "increase risk of", "increase risk for", "raises risk of", "raises risk for",
        "increases the risk of", "increases the risk for", "increases risk of", "increases risk for",
        "increase the risk of", "increase the risk for"
    ],

    # --- General-domain additions ---
    "part_of": [
        "part of", "component of", "member of", "subset of", "belongs to", "included in"
    ],
    "located_in": [
        "located in", "based in", "headquartered in", "situated in"
    ],
    "owns": [
        "owns", "owner of", "acquires", "acquired", "purchases", "purchased", "bought", "buy", "buys"
    ],
    "produces": [
        "produces", "produce", "manufactures", "manufacture", "builds", "build",
        "creates", "create", "makes", "make", "develops", "develop"
    ],
    "uses": [
        "uses", "use", "utilizes", "utilize", "employs", "employ", "adopts", "adopt",
        "leverages", "leverage"
    ],
    "provides": [
        "provides", "provide", "offers", "offer", "supplies", "supply", "delivers", "deliver"
    ],
    "supports": [
        "supports", "support", "compatible with", "works with", "integrates with",
        "integrates", "integration with"
    ],
    "publishes": [
        "publishes", "publish", "releases", "release", "launched", "launches", "launch",
        "announces", "announce", "announced"
    ],
    "authored_by": [
        "authored by", "written by", "by"
    ],
    "founded_by": [
        "founded by", "established by"
    ],
    "founded": [
        "founded", "established", "set up"
    ],
    "partners_with": [
        "partners with", "in partnership with", "collaborates with", "collaborate with",
        "collaborates", "collaborate"
    ],
    "competes_with": [
        "competes with", "rivals", "vs", "versus"
    ],

    "risk for": [
        "risk for", "risk of", "risk factor for", "risk factors for"
    ],
    "interacts with": [
        "interacts with", "interact with", "interacts"
    ],
}

def _flatten_synonyms(canon_map: Dict[str, List[str]]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Create (synonym -> canonical) pairs and a verbish fallback list from a map."""
    pairs: List[Tuple[str, str]] = []
    for _canon, _syns in canon_map.items():
        for _syn in _syns:
            pairs.append((_syn, _canon))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)  # prefer longer matches
    return pairs, list(canon_map.keys())

_SYNONYM_PAIRS, _VERBISH = _flatten_synonyms(_CANON_SYNONYMS)


def _extract_relation(
    span: str,
    sentence: str = "",
    left_ctx: str = "",
    right_ctx: str = "",
    *,
    synonym_pairs: Optional[List[Tuple[str, str]]] = None,
    verbish: Optional[List[str]] = None,
) -> str:
    """Return a canonical relation label if a synonym is detected, else '' to skip."""
    low = span.lower().strip()
    pairs = synonym_pairs if synonym_pairs is not None else _SYNONYM_PAIRS
    verbs = verbish if verbish is not None else _VERBISH

    # direct hit inside the span
    if low:
        for syn, canon in pairs:
            if syn in low:
                return canon

    # soft context fallback (left/right window)
    ctx = f"{left_ctx} {right_ctx}".lower()
    if ctx:
        for syn, canon in pairs:
            if syn in ctx:
                return canon
        for v in verbs:
            if v in ctx:
                return v

    return ""  # __SKIP__


@dataclass
class TripletKnowledgeGraphBuilder:
    nodes: Set[str] = field(default_factory=set)
    triples: Dict[Tuple[str, str, str], Dict[str, Any]] = field(default_factory=lambda: defaultdict(lambda: {"weight": 0, "sources": []}))
    _seen_evidence: Set[Tuple] = field(default_factory=set)  # dedupe key store

    label_store: LabelStore | None = None  # NEW: optional shared store for dynamic labels

    # --- configuration knobs for general pipeline ---
    allow_generic_related: bool = False
    canonical_map_override: Optional[Dict[str, List[str]]] = None
    _syn_pairs: List[Tuple[str, str]] = field(init=False, default_factory=list)
    _verbish: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Compile synonym pairs/verbish list (allowing per-instance override)."""
        if self.canonical_map_override:
            self._syn_pairs, self._verbish = _flatten_synonyms(self.canonical_map_override)
        else:
            self._syn_pairs, self._verbish = _SYNONYM_PAIRS, _VERBISH
    def _commit_label(self, rel: str) -> str:
        """
        If a store is present, fold the relation spelling into the store and
        return the canonical label to use downstream. Otherwise return as-is.
        """
        if not rel:
            return ""
        if self.label_store is None:
            return rel


        canon = self.label_store.observe(rel)

        return canon if self.label_store.should_emit(canon) else rel
    
    def set_relation_catalog(self, canon_map: Dict[str, List[str]]) -> None:
        """Dynamically replace the relation catalog."""
        self.canonical_map_override = canon_map
        self._syn_pairs, self._verbish = _flatten_synonyms(canon_map)


    def _entity_positions(self, sentence: str, entities: Set[str]) -> List[Tuple[int, str]]:
        """Return a list of (start_index, entity) tuples sorted by position.

        This helper lower‑cases the sentence and entity strings for
        approximate matching.  It returns the index of the first
        occurrence of each entity.  Entities that are not found are
        ignored.

        Parameters
        ----------
        sentence : str
            The full sentence containing the entities.
        entities : set of str
            The unique entity names extracted from the sentence.

        Returns
        -------
        list of tuples
            A list of ``(position, entity)`` sorted by position in
            ascending order.  Only entities located in the sentence are
            included.
        """
        pos_list: List[Tuple[int, str]] = []
        sent_low = sentence.lower()
        # Precompile regex patterns for each entity to match whole words.  This
        # prevents partial matching of substrings (e.g. entity "C" matching
        # the letter "c" in "interacts").  If no whole‑word match is found
        # we fall back to a simple substring search to ensure we still
        # capture approximate positions.
        for ent in entities:
            ent_clean = ent.strip()
            if not ent_clean:
                continue
            ent_low = ent_clean.lower()
            # Build a regex that matches the entity as a whole word.  Word
            # boundaries (\b) ensure we don't match substrings inside
            # larger tokens.  Some biomedical entities include hyphens or
            # other punctuation; escaping handles those safely.
            try:
                import re  # local import to avoid a global dependency if unused
                pattern = re.compile(r"\b" + re.escape(ent_low) + r"\b", re.IGNORECASE)
                match = pattern.search(sentence)
                if match:
                    idx = match.start()
                else:
                    # fallback: approximate match by substring search
                    idx = sent_low.find(ent_low)
                    if idx < 0:
                        continue
            except Exception:
                # as a last resort, use substring search
                idx = sent_low.find(ent_low)
                if idx < 0:
                    continue
            pos_list.append((idx, ent_clean))
        # sort by starting index to maintain textual order
        pos_list.sort(key=lambda x: x[0])
        return pos_list

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
        # Extract meaningful triples from this sentence
        triples = self.extract_triplets(sentence, entities)
        for subj, relation, obj in triples:
            key = (subj, relation, obj)
            if context is None:
                # legacy behavior: just count
                self.triples[key]["weight"] += 1
            else:
                # attach one evidence row per triple occurrence (per sentence)
                # best-effort char_span: cover both mentions bounding box
                # (we don't have per-entity spans here; approximate from names)
                s_idx = sentence.lower().find(subj.lower())
                o_idx = sentence.lower().find(obj.lower())
                if s_idx < 0 or o_idx < 0:
                    char_span = (0, 0)
                else:
                    s_end = s_idx + len(subj)
                    o_end = o_idx + len(obj)
                    char_span = (min(s_idx, o_idx), max(s_end, o_end))

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
        result: List[Tuple[str, str, str]] = []
        if not entities or len(entities) < 2:
            return result
        # Determine positions of entities within the sentence
        pos_entities = self._entity_positions(sentence, entities)
        # Fallback: if positions are missing for some entities, we cannot
        # reliably orient subject/object order.  In that case, we refrain
        # from generating any relation-specific triples and return an
        # empty list, delegating to generic co-occurrence if needed.
        if not pos_entities or len(pos_entities) < 2:
            return result
        n = len(pos_entities)
        for i in range(n):
            subj_pos, subj = pos_entities[i]
            for j in range(i + 1, n):
                obj_pos, obj = pos_entities[j]
                # Extract the raw substring between the two entity mentions
                start = subj_pos + len(subj)
                end = obj_pos
                span = sentence[start:end]
                # Capture a small amount of surrounding context to aid in
                # relation extraction when the span is empty or too long
                left_ctx = sentence[max(0, start - 20):start]
                right_ctx = sentence[end:min(len(sentence), end + 20)]
                relation = _extract_relation(
                    span, sentence, left_ctx, right_ctx,
                    synonym_pairs=self._syn_pairs, verbish=self._verbish
                )
                # Skip spans that produce no informative relation
                if relation in ("__SKIP__",""): # for now reserve  "related_to"
                    continue
                
                relation = self._commit_label(relation)
                # Ensure that the canonical relation occurs within the span itself.
                # If it was only detected via context, disregard this pair.
                span_lower = span.lower()
                contains_rel = False
                for syn, canon in _SYNONYM_PAIRS:
                    if canon == relation and syn in span_lower:
                        contains_rel = True
                        break
                if not contains_rel:
                    continue
                # Avoid transitive/enum relations for non‑adjacent entities
                skip_triple = False
                if j > i + 1:
                    # Determine position of the next entity between subject and object
                    inter_pos, inter_ent = pos_entities[i + 1]
                    # Lower‑case span for matching intermediate entity
                    span_lower_full = span.lower()
                    inter_idx = None
                    try:
                        pat = re.compile(r"\b" + re.escape(inter_ent.lower()) + r"\b")
                        m = pat.search(span_lower_full)
                        if m:
                            inter_idx = m.start()
                    except Exception:
                        pass
                    if inter_idx is None:
                        inter_idx = span_lower_full.find(inter_ent.lower())
                    # Collect positions of the relation phrase(s) in the span
                    rel_positions: List[int] = []
                    for syn, canon in _SYNONYM_PAIRS:
                        if canon == relation:
                            search_start = 0
                            while True:
                                idx = span_lower_full.find(syn, search_start)
                                if idx == -1:
                                    break
                                rel_positions.append(idx)
                                search_start = idx + 1
                    if inter_idx is not None and rel_positions:
                        # If any relation occurs after the intermediate entity
                        if any(pos > inter_idx for pos in rel_positions):
                            skip_triple = True
                if skip_triple:
                    continue
                # Additional guard: ensure the subsegment between the intermediate
                # entity and the object contains only simple conjunctions for
                # non‑adjacent entities.  This prevents linking to remote
                # clauses (e.g. "acts as").
                if j > i + 1:
                    # Determine the substring between the end of the first
                    # intermediate entity and the start of the current object
                    inter_start, inter_ent = pos_entities[i + 1]
                    inter_end = inter_start + len(inter_ent)
                    sub_span = sentence[inter_end:obj_pos].lower()
                    try:
                        words = re.findall(r"\b\w+\b", sub_span)
                    except Exception:
                        words = sub_span.split()
                    # In enumeration constructs, other entity names may appear
                    # between the intermediate and the target (e.g., "B, C and D").
                    # We therefore allow known entity tokens as well as simple
                    # conjunctions.  Any extra word outside these sets causes
                    # the triple to be skipped.
                    allowed_conj = {"and", "or"}
                    # Build a set of lowercased entity names for quick lookup
                    entity_tokens = {e.lower() for _, e in pos_entities}
                    if any((w not in allowed_conj) and (w not in entity_tokens) for w in words):
                        continue
                result.append((subj, relation, obj))
        return result

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
        """Convert the graph to a JSON-serialisable dictionary.

        Each triple includes its total weight and a list of supporting
        evidence records (sources).  This keeps full provenance traceable.
        """
        nodes_list = sorted(self.nodes)
        triples_list: List[Dict[str, object]] = []

        for (subj, rel, obj), data in sorted(self.triples.items()):
            triples_list.append({
                "subject": subj,
                "relation": rel,
                "object": obj,
                "weight": data.get("weight", 0),
                "sources": data.get("sources", []),   # <-- provenance evidence
            })

        return {"nodes": nodes_list, "triples": triples_list}

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