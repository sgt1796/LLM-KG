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
from typing import Dict, List, Tuple, Any

class TripletGraphMerger:
    """Merge multiple triplet-based knowledge graphs.

    Graphs must follow the format returned by
    :meth:`kg_pipeline.triple_builder.TripletKnowledgeGraphBuilder.to_dict`
    or the aggregated JSONL/JSON style from `dedupe.py` (h/r/t keys).
    ``nodes`` is a list of strings (or identifiers) and ``triples`` a list of
    dicts with keys: subject|h, relation|r, object|t, weight, and optional sources.
    """

    def __init__(self, base_graph: Dict[str, List[Dict[str, object]]] | None = None) -> None:
        self.graph: Dict[str, Any] = {"nodes": [], "triples": []}
        if base_graph:
            self.merge(base_graph)

    # ---------------- internal helpers ----------------
    @staticmethod
    def _norm_triple_dict(tri: Dict[str, Any]) -> Tuple[str, str, str, float, List[Any], Dict[str, Any]]:
        """Return normalized (s, r, o, w, sources, raw) from a triple-like dict."""
        s = tri.get("subject", tri.get("h"))
        r = tri.get("relation", tri.get("r"))
        o = tri.get("object", tri.get("t"))
        # enforce strings for s,r,o (skip invalid)
        if s is None or o is None:
            raise ValueError("Triple missing subject/h or object/t")
        s = str(s)
        r = "" if r is None else str(r)
        o = str(o)
        # weight
        w_raw = tri.get("weight", 1)
        try:
            w = float(w_raw)
        except Exception:
            w = 1.0
        # sources (optional)
        sources = tri.get("sources") or []
        if not isinstance(sources, list):
            sources = [sources]
        return s, r, o, w, sources, tri

    @staticmethod
    def _sources_key(src: Any) -> str:
        """Stable key for a source record (works for dicts or primitives)."""
        # Most of your sources are dicts; stable JSON string is an easy, safe key.
        try:
            return json.dumps(src, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(src)

    @staticmethod
    def _merge_sources(dst_list: List[Any], add_list: List[Any]) -> List[Any]:
        """Union of sources preserving insertion order."""
        seen = {TripletGraphMerger._sources_key(s) for s in dst_list}
        for s in add_list:
            k = TripletGraphMerger._sources_key(s)
            if k not in seen:
                dst_list.append(s)
                seen.add(k)
        return dst_list

    # ---------------- public API ----------------
    def merge(self, other: Dict[str, List[Dict[str, object]]]) -> None:
        """Merge ``other`` into the current graph.

        - Nodes are deduplicated.
        - Triples are keyed by (subject, relation, object).
        - ``weight`` values are summed.
        - ``sources`` lists are merged and de-duplicated.
        - Accepts both subject/relation/object and h/r/t keys.
        """
        # ---- merge nodes ----
        existing_nodes = set(self.graph.get("nodes", []))
        for n in other.get("nodes", []):
            if n not in existing_nodes:
                self.graph["nodes"].append(n)
                existing_nodes.add(n)

        # ---- index current triples ----
        triples = self.graph.setdefault("triples", [])
        index: Dict[Tuple[str, str, str], int] = {}
        for idx, tri in enumerate(triples):
            try:
                s, r, o, w, srcs, _ = self._norm_triple_dict(tri)
            except ValueError:
                continue
            key = (s, r, o)
            index[key] = idx
            # normalize in-place (also align h/r/t → subject/relation/object)
            tri.setdefault("subject", s); tri.pop("h", None)
            tri.setdefault("relation", r); tri.pop("r", None)
            tri.setdefault("object", o); tri.pop("t", None)
            # ensure numeric weight
            try:
                tri["weight"] = float(tri.get("weight", 1))
            except Exception:
                tri["weight"] = 1.0
            # ensure sources list
            if "sources" in tri:
                if not isinstance(tri["sources"], list):
                    tri["sources"] = [tri["sources"]]
            else:
                tri["sources"] = []

        # ---- merge incoming triples ----
        for tri in other.get("triples", []):
            try:
                s, r, o, w, srcs, raw = self._norm_triple_dict(tri)
            except ValueError:
                continue
            key = (s, r, o)
            if key in index:
                dst = triples[index[key]]
                # sum weights
                try:
                    dst["weight"] = float(dst.get("weight", 1.0)) + w
                except Exception:
                    dst["weight"] = float(w)
                # merge sources
                dst_sources = dst.get("sources", [])
                if not isinstance(dst_sources, list):
                    dst_sources = [dst_sources]
                dst["sources"] = self._merge_sources(dst_sources, srcs)
            else:
                # create a normalized record
                new_tri = {
                    "subject": s,
                    "relation": r,
                    "object": o,
                    "weight": float(w),
                }
                if srcs:
                    new_tri["sources"] = list(srcs)
                triples.append(new_tri)
                index[key] = len(triples) - 1

    @staticmethod
    def load_json(file_path: str | Path) -> Dict[str, List[Dict[str, object]]]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(graph: Dict[str, Any], file_path: str | Path) -> None:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
