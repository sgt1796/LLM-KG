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
from typing import Dict, Iterable, List, Set, Tuple


def _extract_relation(span: str, sentence: str = "", left_ctx: str = "", right_ctx: str = "") -> str:
    if not span:
        cleaned = ""
    else:
        cleaned = span.strip()
    # Normalize case
    low = cleaned.lower()
    # If empty/long, try to fish a nearby predicate; otherwise bail
    if not low or len(low) > 50:
        low = ""

    # Canonicalization table (simple phrase rewrites)
    CANON = {
        "is associated with": "associated with",
        "associated to": "associated with",
        "linked to": "associated with",
        "linked with": "associated with",
        "leads to": "causes",
        "results in": "causes",
        "risk for": "risk for",
        "risk of": "risk for",
        "interacts with": "interacts with",
        "binds to": "binds",
        "regulates": "regulates",
        "inhibits": "inhibits",
        "activates": "activates",
        "increases": "increases",
        "decreases": "decreases",
        "treats": "treats",
        "prevents": "prevents",
    }

    # Pure function-word junk that should not create a triple
    JUNK = {"and","of","with","in","for","the","between","to","by","on","from","at","as"}

    if low in JUNK:
        return "__SKIP__"

    # Try to collapse multiword junk like "and adolescent", "in the"
    tokens = low.split()
    if all(tok in JUNK for tok in tokens):
        return "__SKIP__"

    # Canonicalize known phrases
    for k, v in CANON.items():
        if low == k or low.startswith(k+" "):
            return v

    # If we still have nothing helpful, peek a small context window
    # (cheap heuristic: scan left/right contexts for a verb-like keyword)
    if not low:
        VERBISH = ["inhibits","activates","causes","associated with","predicts","correlates with",
                   "treats","prevents","increases","decreases","binds","regulates","promotes",
                   "suppresses","mediates","expresses","encodes"]
        ctx = " ".join([left_ctx.lower()[-40:], right_ctx.lower()[:40]])
        for v in VERBISH:
            if v in ctx:
                return v
        return "related_to"

    return low

@dataclass
class TripletKnowledgeGraphBuilder:
    """Builds a simple knowledge graph of subject–relation–object triples.

    Nodes are entity strings and triples connect a subject to an object
    with a relation label.  Weights count the number of times the same
    (subject, relation, object) triple was observed across sentences.
    """

    nodes: Set[str] = field(default_factory=set)
    triples: Dict[Tuple[str, str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

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
        for ent in entities:
            # approximate match: find first occurrence ignoring case
            idx = sent_low.find(ent.lower())
            if idx >= 0:
                pos_list.append((idx, ent))
        # sort by starting index to maintain textual order
        pos_list.sort(key=lambda x: x[0])
        return pos_list

    def add_sentence(self, sentence: str, entities: Set[str]) -> None:
        """Add triples for all unordered pairs of entities in a sentence.

        Entities are first sorted by their occurrence in the sentence to
        establish a deterministic subject–object orientation.  For
        every pair of distinct entities, the substring between the two
        mentions is extracted and normalised to serve as the relation.

        Parameters
        ----------
        sentence : str
            The original sentence text.
        entities : set of str
            The unique entity names found in the sentence.
        """
        # Record nodes regardless of whether we find pairs
        for e in entities:
            self.nodes.add(e)
        # Only consider pairs of distinct entities
        if len(entities) < 2:
            return
        # Sort entities by their first occurrence in the sentence
        pos_entities = self._entity_positions(sentence, entities)
        # Fallback: if positions are missing for some entities, simply
        # iterate through all combinations in sorted entity names
        if not pos_entities or len(pos_entities) < 2:
            sorted_ents = sorted(entities)
            for i in range(len(sorted_ents)):
                for j in range(i + 1, len(sorted_ents)):
                    subj, obj = sorted_ents[i], sorted_ents[j]
                    relation = "related_to"
                    self.triples[(subj, relation, obj)] += 1
            return
        # Generate triples for every pair
        n = len(pos_entities)
        for i in range(n):
            subj_pos, subj = pos_entities[i]
            for j in range(i + 1, n):
                obj_pos, obj = pos_entities[j]
                # extract text between the end of subject and start of object
                start = subj_pos + len(subj)
                end = obj_pos
                span = sentence[start:end]
                # light context for verb fallback
                left_ctx = sentence[max(0, start-20):start]
                right_ctx = sentence[end:min(len(sentence), end+20)]
                relation = _extract_relation(span, sentence, left_ctx, right_ctx)
                if relation == "__SKIP__":
                    continue  # don't record junk-only connectors
                self.triples[(subj, relation, obj)] += 1

    def build_from_sentences(self, sentence_entities: Iterable[Tuple[str, Set[str]]]) -> None:
        """Populate the graph from an iterable of (sentence, entities) tuples."""
        for sent, ents in sentence_entities:
            if ents:
                self.add_sentence(sent, ents)

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        """Convert the graph to a JSON‑serialisable dictionary.

        The returned dictionary has two keys:

        - ``nodes``: a sorted list of unique entity strings
        - ``triples``: a list of dictionaries with keys
          ``subject``, ``relation``, ``object`` and ``weight``.
        """
        nodes_list = sorted(self.nodes)
        triples_list = [
            {"subject": subj, "relation": rel, "object": obj, "weight": w}
            for (subj, rel, obj), w in sorted(self.triples.items())
        ]
        return {"nodes": nodes_list, "triples": triples_list}