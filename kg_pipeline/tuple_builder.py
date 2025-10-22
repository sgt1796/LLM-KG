"""Module for extracting tuples of entity with weighted edge of their
co‑occurrences.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class WeightedTupleBuilder:
    """Nodes are entity strings and edges are weighted by co‑occurrence
    counts.  The graph does not model relation types – edges simply
    indicate that two entities appeared in the same sentence.
    """

    nodes: Set[str] = field(default_factory=set)
    edges: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def add_cooccurrence(self, entities: Set[str]) -> None:
        """Add edges for all unordered pairs of entities in a sentence.

        Parameters
        ----------
        entities : set of str
            The entity names found in one sentence.
        """
        ents = list(entities)
        for i in range(len(ents)):
            self.nodes.add(ents[i])
            for j in range(i + 1, len(ents)):
                a, b = sorted([ents[i], ents[j]])  # ensure deterministic ordering
                self.edges[(a, b)] += 1

    def build_from_sentences(self, sentence_entities: Iterable[Tuple[str, Set[str]]]) -> None:
        """Populate the graph from an iterable of (sentence, entities) tuples."""
        for _, ents in sentence_entities:
            if len(ents) >= 2:
                self.add_cooccurrence(ents)
            else:
                # still record singletons as nodes
                for e in ents:
                    self.nodes.add(e)

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        """Convert the graph to a JSON‑serialisable dictionary.

        The returned dict has two keys: ``nodes`` and ``edges``.
        ``nodes`` is a list of unique entity names.  ``edges`` is a list
        of objects with keys ``source``, ``target`` and ``weight``.
        """
        nodes_list = sorted(self.nodes)
        edges_list = [
            {"source": src, "target": dst, "weight": w}
            for (src, dst), w in sorted(self.edges.items())
        ]
        return {"nodes": nodes_list, "edges": edges_list}