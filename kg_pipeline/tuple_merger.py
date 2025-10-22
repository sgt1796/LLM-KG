"""Utility for merging and saving WeightedTuple graphs.

This module defines a simple class that can merge two WeightedTuple graphs
 and provides convenience methods for loading from and
saving to JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


class WeightedTupleMerger:
    """Merge multiple WeightedTuple graphs.

    Graphs follow the format returned by
    :meth:`kg_pipeline.tuple_builder.WeightedTupleBuilder.to_dict`.
    ``nodes`` must be a list of strings and ``edges`` a list of
    dictionaries with ``source``, ``target`` and ``weight`` keys.
    """

    def __init__(self, base_graph: Dict[str, List[Dict[str, object]]] | None = None) -> None:
        self.graph = {"nodes": [], "edges": []}
        if base_graph:
            self.merge(base_graph)

    def merge(self, other: Dict[str, List[Dict[str, object]]]) -> None:
        """Merge ``other`` into the current graph.

        Nodes are deduplicated and edge weights are summed if an edge
        already exists.
        """
        # Merge nodes
        existing_nodes = set(self.graph["nodes"])
        for n in other.get("nodes", []):
            if n not in existing_nodes:
                self.graph["nodes"].append(n)
                existing_nodes.add(n)
        # Merge edges
        edge_index: Dict[Tuple[str, str], int] = {}
        for idx, edge in enumerate(self.graph.get("edges", [])):
            edge_index[(edge["source"], edge["target"])] = idx
        for edge in other.get("edges", []):
            key = (edge["source"], edge["target"])
            if key in edge_index:
                # sum weights
                self.graph["edges"][edge_index[key]]["weight"] += edge.get("weight", 1)
            else:
                self.graph["edges"].append({"source": key[0], "target": key[1], "weight": edge.get("weight", 1)})

    @staticmethod
    def load_json(file_path: str | Path) -> Dict[str, List[Dict[str, object]]]:
        """Load a graph dictionary from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(graph: Dict[str, List[Dict[str, object]]], file_path: str | Path) -> None:
        """Save a graph dictionary to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)