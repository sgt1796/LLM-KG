"""Utility for merging and saving triplet‑based knowledge graphs.

This module mirrors the functionality of :mod:`kg_pipeline.kg_merger`
but operates on graphs whose edges are subject–relation–object
triples.  Graphs are stored as dictionaries produced by
``TripletKnowledgeGraphBuilder.to_dict``.  When merging two graphs the
node list is deduplicated and triple weights are summed when the
subject, relation and object match exactly.

Example
-------

>>> merger = TripletGraphMerger(base_graph={"nodes": ["A"], "triples": []})
>>> merger.merge({"nodes": ["B"], "triples": [{"subject": "A", "relation": "rel", "object": "B", "weight": 1}]})
>>> merger.graph
{'nodes': ['A', 'B'], 'triples': [{'subject': 'A', 'relation': 'rel', 'object': 'B', 'weight': 1}]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


class TripletGraphMerger:
    """Merge multiple triplet‑based knowledge graphs.

    Graphs must follow the format returned by
    :meth:`kg_pipeline.triple_builder.TripletKnowledgeGraphBuilder.to_dict`.
    ``nodes`` must be a list of strings and ``triples`` a list of
    dictionaries with ``subject``, ``relation``, ``object`` and ``weight``
    keys.
    """

    def __init__(self, base_graph: Dict[str, List[Dict[str, object]]] | None = None) -> None:
        self.graph = {"nodes": [], "triples": []}
        if base_graph:
            self.merge(base_graph)

    def merge(self, other: Dict[str, List[Dict[str, object]]]) -> None:
        """Merge ``other`` into the current graph.

        Nodes are deduplicated and triple weights are summed if a
        subject–relation–object combination already exists.
        """
        # Merge nodes
        existing_nodes = set(self.graph["nodes"])
        for n in other.get("nodes", []):
            if n not in existing_nodes:
                self.graph["nodes"].append(n)
                existing_nodes.add(n)
        # Merge triples
        triple_index: Dict[Tuple[str, str, str], int] = {}
        for idx, triple in enumerate(self.graph.get("triples", [])):
            key = (triple["subject"], triple["relation"], triple["object"])
            triple_index[key] = idx
        for triple in other.get("triples", []):
            key = (triple.get("subject"), triple.get("relation"), triple.get("object"))
            weight = triple.get("weight", 1)
            if key in triple_index:
                self.graph["triples"][triple_index[key]]["weight"] += weight
            else:
                self.graph["triples"].append({
                    "subject": key[0],
                    "relation": key[1],
                    "object": key[2],
                    "weight": weight,
                })

    @staticmethod
    def load_json(file_path: str | Path) -> Dict[str, List[Dict[str, object]]]:
        """Load a graph dictionary from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(graph: Dict[str, List[Dict[str, object]]], file_path: str | Path) -> None:
        """Save a graph dictionary to a JSON or JSONL file based on extension.

        If ``file_path`` ends with ``.jsonl`` (case-insensitive), the graph is
        written as a JSON Lines file where each triple is one line and the
        nodes are omitted.  Otherwise, the full graph dictionary (with
        ``nodes`` and ``triples`` arrays) is written as a single JSON object.

        Parameters
        ----------
        graph : dict
            The graph dictionary containing at least ``nodes`` and ``triples``.
        file_path : str or pathlib.Path
            The destination file path.  The suffix determines the
            serialisation format.
        """
        path_str = str(file_path)
        if path_str.lower().endswith(".jsonl"):
            # write triples in JSON Lines format
            triples = graph.get("triples", [])
            with open(file_path, "w", encoding="utf-8") as f:
                for trip in triples:
                    # ensure keys order: subject, relation, object, weight
                    rec = {
                        "subject": trip.get("subject"),
                        "relation": trip.get("relation"),
                        "object": trip.get("object"),
                        "weight": trip.get("weight", 1),
                    }
                    json.dump(rec, f, ensure_ascii=False)
                    f.write("\n")
        else:
            # standard JSON output
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)