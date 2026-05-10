"""Hybrid natural-language retriever for knowledge-graph search.

This module builds reusable text and optional KGE sidecars for a graph JSON
produced by ``main.py``. Runtime search combines alias matching, text
embeddings, relation intent detection, optional graph-embedding completion, and
short path extraction so higher-level callers can answer natural-language
questions with grounded KG context.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .triple_builder import _CANON_SYNONYMS, _INVERTED_SYNONYMS

RRF_K = 60
MIN_TRIPLE_CANDIDATES = 8
TOP_TEXT_CANDIDATES = 64
RELATION_EMBED_THRESHOLD = 0.25


@dataclass
class RelationIntent:
    """Structured relation guess for a natural-language question."""

    relation: str
    direction: int
    source: str
    phrase: str
    score: float
    span: Optional[Tuple[int, int]] = None


@dataclass
class QueryResult:
    """Normalized retrieval payload shared by the app and CLI."""

    focus_nodes: List[Dict[str, Any]]
    triples: List[Dict[str, Any]]
    paths: List[Dict[str, Any]]
    context: str
    debug_scores: Dict[str, Any]


@dataclass
class KGEArtifacts:
    """Optional persisted graph-embedding artifacts."""

    available: bool
    metadata: Dict[str, Any]
    entity_embeddings: Optional[np.ndarray]
    relation_embeddings: Optional[np.ndarray]
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]


@dataclass
class HybridKGRetriever:
    """Reusable hybrid retriever with offline indices and runtime search."""

    graph_path: Path
    cache_dir: Path
    text_embedder: Any
    graph: nx.Graph
    triples: List[Dict[str, Any]]
    triple_records: List[Dict[str, Any]]
    node_records: List[Dict[str, Any]]
    relation_records: List[Dict[str, Any]]
    node_embeddings: np.ndarray
    triple_embeddings: np.ndarray
    relation_embeddings: np.ndarray
    incident_triple_ids: Dict[str, List[int]]
    pair_to_triple_ids: Dict[Tuple[str, str], List[int]]
    alias_lookup: Dict[str, List[str]]
    alias_patterns: List[Tuple[str, re.Pattern[str], List[str]]]
    relation_patterns: List[Tuple[str, re.Pattern[str], str, int]]
    kge: KGEArtifacts

    def query(
        self,
        question: str,
        top_k: int = 10,
        hop_limit: int = 2,
        visible_node_ids: Optional[Iterable[str]] = None,
    ) -> QueryResult:
        """Search the graph with a natural-language question."""

        question = str(question or "").strip()
        if not question:
            raise ValueError("question must not be empty")

        top_k = max(1, int(top_k))
        hop_limit = max(1, int(hop_limit))
        question_norm = _normalize_phrase(question)
        query_vec = _embed_texts(self.text_embedder, [question])[0]

        anchors = self._match_aliases(question_norm)
        relation_intent = self._detect_relation_intent(question_norm, query_vec)

        alias_rank = [item["node_id"] for item in anchors]
        node_sim_rank, node_sim_scores = self._rank_text_items(
            query_vec,
            self.node_embeddings,
            [record["id"] for record in self.node_records],
            limit=max(TOP_TEXT_CANDIDATES, top_k * 8),
        )
        triple_sim_rank, triple_sim_scores = self._rank_text_items(
            query_vec,
            self.triple_embeddings,
            [record["id"] for record in self.triple_records],
            limit=max(TOP_TEXT_CANDIDATES, top_k * 8),
        )
        triple_to_node_rank = _project_triple_rank_to_nodes(triple_sim_rank, self.triple_records)
        kge_rank, kge_scores = self._rank_kge_candidates(anchors, relation_intent, limit=max(48, top_k * 8))

        fused_node_scores = _rrf_fuse(
            {
                "alias": alias_rank,
                "node_text": node_sim_rank,
                "triple_text": triple_to_node_rank,
                "kge": kge_rank,
            }
        )

        seed_nodes = [node_id for node_id, _ in _sorted_score_items(fused_node_scores, limit=max(top_k * 4, 12))]
        if not seed_nodes:
            seed_nodes = node_sim_rank[: max(top_k * 2, 6)]

        candidate_triple_ids, expansion_depth = self._collect_candidate_triples(
            seed_nodes=seed_nodes,
            triple_sim_rank=triple_sim_rank,
            anchors=anchors,
            hop_limit=hop_limit,
            desired=max(top_k * 3, MIN_TRIPLE_CANDIDATES),
        )
        scored_triples = self._score_triples(
            candidate_triple_ids=candidate_triple_ids,
            triple_text_scores=triple_sim_scores,
            fused_node_scores=fused_node_scores,
            relation_intent=relation_intent,
            anchors=anchors,
        )
        top_triples = [payload for _, payload in scored_triples[:top_k]]

        focus_nodes = self._build_focus_nodes(
            top_triples=top_triples,
            fused_node_scores=fused_node_scores,
            node_sim_scores=node_sim_scores,
            visible_node_ids=visible_node_ids,
            top_k=top_k,
        )
        paths = self._build_paths(
            anchors=anchors,
            top_triples=top_triples,
            fused_node_scores=fused_node_scores,
            hop_limit=hop_limit,
            top_k=min(3, top_k),
        )
        context = _build_context(top_triples, paths)

        relation_payload = asdict(relation_intent) if relation_intent else None
        debug_scores = {
            "anchors": anchors,
            "relation_intent": relation_payload,
            "node_rankings": {
                "alias": alias_rank[:10],
                "node_text": node_sim_rank[:10],
                "triple_text_nodes": triple_to_node_rank[:10],
                "kge": kge_rank[:10],
                "fused": _sorted_score_items(fused_node_scores, limit=10),
            },
            "triple_rankings": {
                "triple_text": triple_sim_rank[:10],
                "selected": [item["id"] for item in top_triples],
            },
            "kge": self.kge.metadata,
            "expansion_depth": expansion_depth,
        }
        return QueryResult(
            focus_nodes=focus_nodes,
            triples=top_triples,
            paths=paths,
            context=context,
            debug_scores=debug_scores,
        )

    def _match_aliases(self, question_norm: str) -> List[Dict[str, Any]]:
        matches: Dict[str, Dict[str, Any]] = {}
        for alias, pattern, node_ids in self.alias_patterns:
            for match in pattern.finditer(question_norm):
                for node_id in node_ids:
                    score = len(alias) + len(alias.split()) * 4
                    previous = matches.get(node_id)
                    if previous and previous["score"] >= score:
                        continue
                    matches[node_id] = {
                        "node_id": node_id,
                        "alias": alias,
                        "score": float(score),
                        "span": (match.start(), match.end()),
                    }
        return sorted(matches.values(), key=lambda item: (-item["score"], item["node_id"].casefold()))

    def _detect_relation_intent(self, question_norm: str, query_vec: np.ndarray) -> Optional[RelationIntent]:
        for phrase, pattern, relation, direction in self.relation_patterns:
            match = pattern.search(question_norm)
            if match:
                return RelationIntent(
                    relation=relation,
                    direction=direction,
                    source="lexicon",
                    phrase=phrase,
                    score=float(len(phrase)),
                    span=(match.start(), match.end()),
                )

        if not len(self.relation_records):
            return None

        sims = (self.relation_embeddings @ query_vec.reshape(-1, 1)).ravel()
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < RELATION_EMBED_THRESHOLD:
            return None
        relation = self.relation_records[best]["relation"]
        return RelationIntent(
            relation=relation,
            direction=1,
            source="embedding",
            phrase=relation,
            score=score,
            span=None,
        )

    def _rank_text_items(
        self,
        query_vec: np.ndarray,
        matrix: np.ndarray,
        item_ids: Sequence[str],
        limit: int,
    ) -> Tuple[List[str], Dict[str, float]]:
        if matrix.size == 0:
            return [], {}
        sims = (matrix @ query_vec.reshape(-1, 1)).ravel()
        limit = min(limit, len(sims))
        idx = np.argpartition(-sims, limit - 1)[:limit]
        idx = idx[np.argsort(-sims[idx])]
        ranking = [item_ids[i] for i in idx]
        scores = {item_ids[i]: float(sims[i]) for i in idx}
        return ranking, scores

    def _rank_kge_candidates(
        self,
        anchors: Sequence[Dict[str, Any]],
        relation_intent: Optional[RelationIntent],
        limit: int,
    ) -> Tuple[List[str], Dict[str, float]]:
        if not self.kge.available or not anchors or relation_intent is None:
            return [], {}

        anchor = anchors[0]
        anchor_id = anchor["node_id"]
        relation = relation_intent.relation
        role = _infer_anchor_role(anchor, relation_intent)
        ranking = self._score_rotate_completion(anchor_id=anchor_id, relation=relation, known_role=role, limit=limit)
        return ranking

    def _score_rotate_completion(
        self,
        anchor_id: str,
        relation: str,
        known_role: str,
        limit: int,
    ) -> Tuple[List[str], Dict[str, float]]:
        if (
            self.kge.entity_embeddings is None
            or self.kge.relation_embeddings is None
            or anchor_id not in self.kge.entity_to_id
            or relation not in self.kge.relation_to_id
        ):
            return [], {}

        entity_matrix = self.kge.entity_embeddings
        relation_matrix = self.kge.relation_embeddings
        anchor_vec = entity_matrix[self.kge.entity_to_id[anchor_id]]
        relation_vec = relation_matrix[self.kge.relation_to_id[relation]]

        try:
            scores = _rotate_scores(entity_matrix, anchor_vec, relation_vec, known_role)
        except Exception:
            scores = _fallback_relation_scores(entity_matrix, anchor_vec, relation_vec, known_role)

        if anchor_id in self.kge.entity_to_id:
            scores[self.kge.entity_to_id[anchor_id]] = -np.inf

        item_ids = _invert_mapping(self.kge.entity_to_id)
        limit = min(limit, len(scores))
        idx = np.argpartition(-scores, limit - 1)[:limit]
        idx = idx[np.argsort(-scores[idx])]
        ranking = [item_ids[i] for i in idx]
        score_map = {item_ids[i]: float(scores[i]) for i in idx}
        return ranking, score_map

    def _collect_candidate_triples(
        self,
        seed_nodes: Sequence[str],
        triple_sim_rank: Sequence[str],
        anchors: Sequence[Dict[str, Any]],
        hop_limit: int,
        desired: int,
    ) -> Tuple[List[int], int]:
        candidates: Dict[int, None] = {}

        for triple_id in triple_sim_rank[: max(desired, 12)]:
            candidates[int(triple_id)] = None

        anchor_nodes = [item["node_id"] for item in anchors]
        for node_id in anchor_nodes:
            for triple_id in self.incident_triple_ids.get(node_id, []):
                candidates[triple_id] = None

        if len(anchor_nodes) >= 2:
            for left, right in combinations(anchor_nodes[:3], 2):
                for triple_id in self.pair_to_triple_ids.get(_pair_key(left, right), []):
                    candidates[triple_id] = None

        seen_nodes = set(seed_nodes)
        frontier = [node_id for node_id in seed_nodes if node_id in self.graph]
        used_depth = 0

        for depth in range(1, hop_limit + 1):
            if not frontier:
                break
            next_frontier: List[str] = []
            for node_id in frontier:
                for triple_id in self.incident_triple_ids.get(node_id, []):
                    candidates[triple_id] = None
                    record = self.triple_records[triple_id]
                    for neighbor in (record["subject"], record["object"]):
                        if neighbor not in seen_nodes:
                            seen_nodes.add(neighbor)
                            next_frontier.append(neighbor)
            used_depth = depth
            if depth == 1 and (len(candidates) >= desired or hop_limit == 1):
                break
            frontier = next_frontier

        ordered = sorted(candidates.keys())
        return ordered, used_depth

    def _score_triples(
        self,
        candidate_triple_ids: Sequence[int],
        triple_text_scores: Mapping[str, float],
        fused_node_scores: Mapping[str, float],
        relation_intent: Optional[RelationIntent],
        anchors: Sequence[Dict[str, Any]],
    ) -> List[Tuple[float, Dict[str, Any]]]:
        anchor_nodes = {item["node_id"] for item in anchors}
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        triple_rank_positions = {
            triple_id: rank
            for rank, triple_id in enumerate(triple_text_scores.keys(), start=1)
        }

        for triple_id in candidate_triple_ids:
            record = self.triple_records[triple_id]
            text_score = float(triple_text_scores.get(record["id"], 0.0))
            fused_bonus = max(
                float(fused_node_scores.get(record["subject"], 0.0)),
                float(fused_node_scores.get(record["object"], 0.0)),
            )
            relation_bonus = 0.75 if relation_intent and record["relation"] == relation_intent.relation else 0.0
            anchor_bonus = 0.3 * len(anchor_nodes.intersection({record["subject"], record["object"]}))
            source_bonus = 0.08 * math.log1p(record["source_count"])
            weight_bonus = 0.12 * math.log1p(record["weight"])
            text_rrf = 1.0 / (RRF_K + triple_rank_positions.get(record["id"], 10_000))
            score = text_score + (0.5 * fused_bonus) + relation_bonus + anchor_bonus + source_bonus + weight_bonus + text_rrf

            payload = {
                "id": record["id"],
                "subject": record["subject"],
                "relation": record["relation"],
                "object": record["object"],
                "weight": record["weight"],
                "score": float(score),
                "source_count": record["source_count"],
                "evidence_excerpt": record["evidence_excerpt"],
                "paper": record["paper"],
            }
            ranked.append((score, payload))

        ranked.sort(key=lambda item: (-item[0], item[1]["subject"].casefold(), item[1]["relation"].casefold(), item[1]["object"].casefold()))
        return ranked

    def _build_focus_nodes(
        self,
        top_triples: Sequence[Dict[str, Any]],
        fused_node_scores: Mapping[str, float],
        node_sim_scores: Mapping[str, float],
        visible_node_ids: Optional[Iterable[str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        visible = set(visible_node_ids or [])
        scores: Dict[str, float] = defaultdict(float)
        for triple in top_triples:
            scores[triple["subject"]] += float(triple["score"])
            scores[triple["object"]] += float(triple["score"])

        for node_id, score in fused_node_scores.items():
            scores[node_id] += 0.25 * float(score)
        for node_id, score in node_sim_scores.items():
            scores[node_id] += 0.05 * float(score)

        ordered = [item for item in _sorted_score_items(scores, limit=max(top_k * 3, 12))]
        if visible:
            filtered = [item for item in ordered if item[0] in visible]
            if not filtered:
                filtered = [item for item in _sorted_score_items({k: v for k, v in scores.items() if k in visible}, limit=top_k)]
            ordered = filtered
        return [{"id": node_id, "score": float(score)} for node_id, score in ordered[:top_k]]

    def _build_paths(
        self,
        anchors: Sequence[Dict[str, Any]],
        top_triples: Sequence[Dict[str, Any]],
        fused_node_scores: Mapping[str, float],
        hop_limit: int,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen_paths: set[Tuple[str, ...]] = set()
        anchor_nodes = [item["node_id"] for item in anchors]

        pairs: List[Tuple[str, str]] = []
        if len(anchor_nodes) >= 2:
            pairs.extend(combinations(anchor_nodes[:3], 2))
        elif len(anchor_nodes) == 1:
            anchor = anchor_nodes[0]
            for triple in top_triples[:top_k]:
                for node_id in (triple["subject"], triple["object"]):
                    if node_id != anchor:
                        pairs.append((anchor, node_id))
        else:
            for triple in top_triples[:top_k]:
                pairs.append((triple["subject"], triple["object"]))

        for left, right in pairs:
            if left not in self.graph or right not in self.graph:
                continue
            try:
                path_nodes = nx.shortest_path(self.graph, left, right)
            except nx.NetworkXNoPath:
                continue
            if len(path_nodes) - 1 > hop_limit:
                continue
            key = tuple(path_nodes)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            results.append(self._path_payload(path_nodes, fused_node_scores))
            if len(results) >= top_k:
                break

        return results

    def _path_payload(self, path_nodes: Sequence[str], fused_node_scores: Mapping[str, float]) -> Dict[str, Any]:
        edges: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        score = 0.0

        for left, right in zip(path_nodes, path_nodes[1:]):
            triple_id = self._best_edge_triple(left, right)
            if triple_id is None:
                relation = "related to"
                edge_payload = {"subject": left, "relation": relation, "object": right}
                text_parts.append(f"{left} --[{relation}]--> {right}")
            else:
                record = self.triple_records[triple_id]
                relation = record["relation"]
                edge_payload = {
                    "subject": record["subject"],
                    "relation": relation,
                    "object": record["object"],
                    "triple_id": record["id"],
                }
                text_parts.append(f"{record['subject']} --[{relation}]--> {record['object']}")
                score += 0.35 * record["weight"]
            score += float(fused_node_scores.get(left, 0.0))
            edges.append(edge_payload)

        if path_nodes:
            score += float(fused_node_scores.get(path_nodes[-1], 0.0))

        return {
            "nodes": list(path_nodes),
            "edges": edges,
            "score": float(score),
            "text": " ; ".join(text_parts),
        }

    def _best_edge_triple(self, left: str, right: str) -> Optional[int]:
        triple_ids = self.pair_to_triple_ids.get(_pair_key(left, right), [])
        if not triple_ids:
            return None
        return max(
            triple_ids,
            key=lambda triple_id: (
                self.triple_records[triple_id]["weight"],
                self.triple_records[triple_id]["source_count"],
            ),
        )


def build_index(
    graph_path: str | Path,
    cache_dir: str | Path,
    text_embedder: Any,
    kge_enabled: bool = True,
    *,
    method: str = "hybrid",
    semantic_threshold: float = 0.05,
    structural_threshold: float = 0.05,
) -> Any:
    """Build or load deterministic sidecars for KG retrieval."""

    return _build_index(
        graph_path=graph_path,
        cache_dir=cache_dir,
        text_embedder=text_embedder,
        kge_enabled=kge_enabled,
        method=method,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
    )


def _build_index(
    graph_path: str | Path,
    cache_dir: str | Path,
    text_embedder: Any,
    kge_enabled: bool = True,
    *,
    method: str = "hybrid",
    semantic_threshold: float = 0.05,
    structural_threshold: float = 0.05,
) -> Any:
    """Build or load deterministic sidecars for the selected retriever."""

    graph_path = Path(graph_path)
    method_norm = str(method or "hybrid").strip().casefold()
    if method_norm == "semantic":
        from query_tools.semantic_subgraph_retriever import SemanticSubgraphRetriever

        return SemanticSubgraphRetriever(
            graph_path=graph_path,
            cache_dir=Path(cache_dir) / graph_path.stem,
            text_embedder=text_embedder,
            semantic_threshold=semantic_threshold,
            structural_threshold=structural_threshold,
        )
    if method_norm != "hybrid":
        raise ValueError(f"Unknown retriever method: {method}")

    cache_dir = Path(cache_dir) / graph_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    graph, nodes_from_graph, raw_triples = _load_graph(graph_path)
    triple_records = _build_triple_records(raw_triples)
    incident_triple_ids, pair_to_triple_ids = _build_triple_adjacency(nodes_from_graph, triple_records)
    alias_lookup = _build_alias_lookup(nodes_from_graph)
    node_records = _build_node_records(nodes_from_graph, triple_records, incident_triple_ids, alias_lookup)
    relation_records = _build_relation_records(triple_records)

    node_embeddings = _load_or_compute_matrix(
        cache_dir=cache_dir,
        basename="node_index",
        records=node_records,
        text_embedder=text_embedder,
    )
    triple_embeddings = _load_or_compute_matrix(
        cache_dir=cache_dir,
        basename="triple_index",
        records=triple_records,
        text_embedder=text_embedder,
    )
    relation_embeddings = _load_or_compute_matrix(
        cache_dir=cache_dir,
        basename="relation_index",
        records=relation_records,
        text_embedder=text_embedder,
    )

    _write_json(cache_dir / "aliases.json", alias_lookup)
    kge = _load_or_build_kge(
        cache_dir=cache_dir,
        triple_records=triple_records,
        enabled=kge_enabled,
    )

    return HybridKGRetriever(
        graph_path=graph_path,
        cache_dir=cache_dir,
        text_embedder=text_embedder,
        graph=graph,
        triples=raw_triples,
        triple_records=triple_records,
        node_records=node_records,
        relation_records=relation_records,
        node_embeddings=node_embeddings,
        triple_embeddings=triple_embeddings,
        relation_embeddings=relation_embeddings,
        incident_triple_ids=incident_triple_ids,
        pair_to_triple_ids=pair_to_triple_ids,
        alias_lookup=alias_lookup,
        alias_patterns=_compile_alias_patterns(alias_lookup),
        relation_patterns=_compile_relation_patterns(),
        kge=kge,
    )


def query(
    question: str,
    top_k: int = 10,
    hop_limit: int = 2,
    *,
    retriever: Any,
    visible_node_ids: Optional[Iterable[str]] = None,
) -> QueryResult:
    """Convenience wrapper around a KG retriever's ``query`` method."""

    return retriever.query(question=question, top_k=top_k, hop_limit=hop_limit, visible_node_ids=visible_node_ids)


RAGRetriever = HybridKGRetriever


def _load_graph(path: Path) -> Tuple[nx.Graph, List[str], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    triples = data.get("triples") or data.get("edges") or []
    graph = nx.Graph()

    for node_id in data.get("nodes") or []:
        graph.add_node(str(node_id))

    for tri in triples:
        subject = str(tri.get("subject") or tri.get("h") or "").strip()
        relation = str(tri.get("relation") or tri.get("r") or "").strip()
        obj = str(tri.get("object") or tri.get("t") or "").strip()
        if not subject or not obj:
            continue
        weight = float(tri.get("weight", 1.0) or 1.0)
        graph.add_node(subject)
        graph.add_node(obj)
        if not graph.has_edge(subject, obj):
            graph.add_edge(subject, obj, relation=relation, weight=weight)
    node_ids = sorted((str(node_id) for node_id in graph.nodes()), key=lambda item: item.casefold())
    return graph, node_ids, triples


def _build_triple_records(triples: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, tri in enumerate(triples):
        subject = str(tri.get("subject") or tri.get("h") or "").strip()
        relation = str(tri.get("relation") or tri.get("r") or "").strip()
        obj = str(tri.get("object") or tri.get("t") or "").strip()
        if not subject or not relation or not obj:
            continue

        weight = float(tri.get("weight", 1.0) or 1.0)
        sources = tri.get("sources") or []
        source_count = len(sources)
        evidence_excerpt = ""
        paper = ""
        if sources:
            first = sources[0] or {}
            evidence_excerpt = str(first.get("evidence") or "").strip()
            paper = str((first.get("doc_meta") or {}).get("filename") or "").strip()

        record = {
            "id": str(len(records)),
            "subject": subject,
            "relation": relation,
            "object": obj,
            "weight": weight,
            "source_count": source_count,
            "evidence_excerpt": evidence_excerpt,
            "paper": paper,
            "text": _linearize_triple(subject, relation, obj, weight, sources),
            "original_index": index,
        }
        records.append(record)

    records.sort(
        key=lambda item: (
            item["subject"].casefold(),
            item["relation"].casefold(),
            item["object"].casefold(),
            item["paper"].casefold(),
            item["original_index"],
        )
    )
    for index, record in enumerate(records):
        record["id"] = str(index)
    return records


def _build_triple_adjacency(
    node_ids: Sequence[str],
    triple_records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, str], List[int]]]:
    incident: Dict[str, List[int]] = {node_id: [] for node_id in node_ids}
    pair_to_triples: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for triple_id, record in enumerate(triple_records):
        subject = str(record["subject"])
        obj = str(record["object"])
        incident.setdefault(subject, []).append(triple_id)
        incident.setdefault(obj, []).append(triple_id)
        pair_to_triples[_pair_key(subject, obj)].append(triple_id)
    return incident, dict(pair_to_triples)


def _build_alias_lookup(node_ids: Sequence[str]) -> Dict[str, List[str]]:
    alias_map: Dict[str, set[str]] = defaultdict(set)
    for node_id in node_ids:
        for alias in _generate_aliases(node_id):
            alias_map[alias].add(node_id)
    return {
        alias: sorted(node_set, key=lambda item: item.casefold())
        for alias, node_set in sorted(alias_map.items(), key=lambda item: (-len(item[0]), item[0]))
    }


def _build_node_records(
    node_ids: Sequence[str],
    triple_records: Sequence[Mapping[str, Any]],
    incident_triple_ids: Mapping[str, Sequence[int]],
    alias_lookup: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    aliases_by_node: Dict[str, List[str]] = defaultdict(list)
    for alias, nodes in alias_lookup.items():
        for node_id in nodes:
            aliases_by_node[node_id].append(alias)

    records: List[Dict[str, Any]] = []
    for node_id in node_ids:
        triple_ids = sorted(
            incident_triple_ids.get(node_id, []),
            key=lambda triple_id: (
                -float(triple_records[triple_id]["weight"]),
                str(triple_records[triple_id]["relation"]).casefold(),
                str(triple_records[triple_id]["object"]).casefold(),
            ),
        )
        text = _build_node_text(node_id, triple_ids, triple_records)
        records.append(
            {
                "id": node_id,
                "degree": int(len(triple_ids)),
                "aliases": sorted(set(aliases_by_node.get(node_id, [])), key=lambda item: (-len(item), item)),
                "text": text,
            }
        )
    return records


def _build_relation_records(triple_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen_relations = sorted({str(record["relation"]) for record in triple_records}, key=lambda item: item.casefold())
    records: List[Dict[str, Any]] = []
    for relation in seen_relations:
        synonyms = sorted(
            set(_CANON_SYNONYMS.get(relation, []) + _INVERTED_SYNONYMS.get(relation, [])),
            key=lambda item: (len(item), item.casefold()),
        )
        pieces = [f"Relation: {relation}."]
        if synonyms:
            pieces.append("Synonyms: " + ", ".join(synonyms[:10]) + ".")
        records.append({"relation": relation, "text": " ".join(pieces)})
    return records


def _load_or_compute_matrix(
    cache_dir: Path,
    basename: str,
    records: Sequence[Mapping[str, Any]],
    text_embedder: Any,
) -> np.ndarray:
    records_path = cache_dir / f"{basename}.json"
    matrix_path = cache_dir / f"{basename}.npy"
    normalized_records = [dict(record) for record in records]

    if records_path.exists() and matrix_path.exists():
        cached_records = _read_json(records_path)
        if cached_records == normalized_records:
            return np.load(matrix_path)

    texts = [str(record.get("text") or "") for record in normalized_records]
    matrix = _embed_texts(text_embedder, texts)
    _write_json(records_path, normalized_records)
    np.save(matrix_path, matrix)
    return matrix


def _embed_texts(text_embedder: Any, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="f")
    matrix = np.asarray(text_embedder.get_embedding(list(texts)), dtype="f")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return matrix / norms


def _compile_alias_patterns(alias_lookup: Mapping[str, Sequence[str]]) -> List[Tuple[str, re.Pattern[str], List[str]]]:
    patterns: List[Tuple[str, re.Pattern[str], List[str]]] = []
    for alias, node_ids in alias_lookup.items():
        if not alias:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        patterns.append((alias, pattern, list(node_ids)))
    patterns.sort(key=lambda item: (-len(item[0]), item[0]))
    return patterns


def _compile_relation_patterns() -> List[Tuple[str, re.Pattern[str], str, int]]:
    items: List[Tuple[str, re.Pattern[str], str, int]] = []
    seen: set[Tuple[str, str, int]] = set()
    for relation, phrases in _CANON_SYNONYMS.items():
        for phrase in phrases:
            normalized = _normalize_phrase(phrase)
            key = (normalized, relation, 1)
            if key in seen:
                continue
            seen.add(key)
            items.append((normalized, re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)"), relation, 1))
    for relation, phrases in _INVERTED_SYNONYMS.items():
        for phrase in phrases:
            normalized = _normalize_phrase(phrase)
            key = (normalized, relation, -1)
            if key in seen:
                continue
            seen.add(key)
            items.append((normalized, re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)"), relation, -1))
    items.sort(key=lambda item: (-len(item[0]), item[0]))
    return items


def _load_or_build_kge(cache_dir: Path, triple_records: Sequence[Mapping[str, Any]], enabled: bool) -> KGEArtifacts:
    metadata_path = cache_dir / "kge_metadata.json"
    entity_emb_path = cache_dir / "kge_entity_embeddings.npy"
    relation_emb_path = cache_dir / "kge_relation_embeddings.npy"
    entity_map_path = cache_dir / "kge_entity_to_id.json"
    relation_map_path = cache_dir / "kge_relation_to_id.json"

    if metadata_path.exists() and entity_map_path.exists() and relation_map_path.exists():
        metadata = _read_json(metadata_path)
        if metadata.get("status") == "ready" or not enabled:
            entity_to_id = _read_json(entity_map_path)
            relation_to_id = _read_json(relation_map_path)
            entity_embeddings = np.load(entity_emb_path) if entity_emb_path.exists() else None
            relation_embeddings = np.load(relation_emb_path) if relation_emb_path.exists() else None
            available = bool(entity_embeddings is not None and relation_embeddings is not None and metadata.get("status") == "ready")
            return KGEArtifacts(
                available=available,
                metadata=metadata,
                entity_embeddings=entity_embeddings,
                relation_embeddings=relation_embeddings,
                entity_to_id={str(k): int(v) for k, v in entity_to_id.items()},
                relation_to_id={str(k): int(v) for k, v in relation_to_id.items()},
            )

    if not enabled:
        metadata = {"status": "disabled", "model": "RotatE"}
        _write_json(metadata_path, metadata)
        _write_json(entity_map_path, {})
        _write_json(relation_map_path, {})
        return KGEArtifacts(False, metadata, None, None, {}, {})

    try:
        payload = _train_rotate_embeddings(triple_records)
    except ImportError as exc:
        metadata = {"status": "unavailable", "model": "RotatE", "reason": str(exc)}
        _write_json(metadata_path, metadata)
        _write_json(entity_map_path, {})
        _write_json(relation_map_path, {})
        return KGEArtifacts(False, metadata, None, None, {}, {})
    except Exception as exc:
        metadata = {"status": "failed", "model": "RotatE", "reason": str(exc)}
        _write_json(metadata_path, metadata)
        _write_json(entity_map_path, {})
        _write_json(relation_map_path, {})
        return KGEArtifacts(False, metadata, None, None, {}, {})

    metadata = payload["metadata"]
    entity_embeddings = payload["entity_embeddings"]
    relation_embeddings = payload["relation_embeddings"]
    entity_to_id = payload["entity_to_id"]
    relation_to_id = payload["relation_to_id"]

    _write_json(metadata_path, metadata)
    _write_json(entity_map_path, entity_to_id)
    _write_json(relation_map_path, relation_to_id)
    np.save(entity_emb_path, entity_embeddings)
    np.save(relation_emb_path, relation_embeddings)
    return KGEArtifacts(True, metadata, entity_embeddings, relation_embeddings, entity_to_id, relation_to_id)


def _train_rotate_embeddings(triple_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not triple_records:
        raise RuntimeError("cannot train KGE without triples")

    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("PyKEEN is not installed") from exc

    labeled_triples = np.asarray(
        [
            [str(record["subject"]), str(record["relation"]), str(record["object"])]
            for record in triple_records
        ],
        dtype=str,
    )
    factory = TriplesFactory.from_labeled_triples(labeled_triples)
    result = pipeline(
        training=factory,
        testing=factory,
        validation=factory,
        model="RotatE",
        training_kwargs={"num_epochs": 25, "batch_size": 256},
        random_seed=1234,
        device="cpu",
    )

    model = result.model
    entity_repr = model.entity_representations[0](indices=None).detach().cpu().numpy().astype("f")
    relation_repr = model.relation_representations[0](indices=None).detach().cpu().numpy().astype("f")
    metadata = {
        "status": "ready",
        "model": "RotatE",
        "num_entities": int(entity_repr.shape[0]),
        "num_relations": int(relation_repr.shape[0]),
        "entity_dim": int(entity_repr.shape[1]) if entity_repr.ndim == 2 else 0,
        "relation_dim": int(relation_repr.shape[1]) if relation_repr.ndim == 2 else 0,
    }
    return {
        "metadata": metadata,
        "entity_embeddings": entity_repr,
        "relation_embeddings": relation_repr,
        "entity_to_id": {str(key): int(value) for key, value in factory.entity_to_id.items()},
        "relation_to_id": {str(key): int(value) for key, value in factory.relation_to_id.items()},
    }


def _build_node_text(node_id: str, triple_ids: Sequence[int], triple_records: Sequence[Mapping[str, Any]]) -> str:
    pieces = [f"Entity: {node_id}."]
    if not triple_ids:
        pieces.append("No connected triples in the current graph.")
        return " ".join(pieces)

    for triple_id in triple_ids[:5]:
        record = triple_records[triple_id]
        pieces.append(
            f"Relation: {record['subject']} {record['relation']} {record['object']}."
        )
        if record["evidence_excerpt"]:
            pieces.append(f"Evidence: {record['evidence_excerpt']}")
        if record["paper"]:
            pieces.append(f"Paper: {record['paper']}")
    return " ".join(pieces)


def _linearize_triple(
    subject: str,
    relation: str,
    obj: str,
    weight: float,
    sources: Sequence[Mapping[str, Any]],
) -> str:
    pieces = [f"Triple: {subject} {relation} {obj}.", f"Weight: {weight:g}."]
    for source in sources[:2]:
        evidence = str(source.get("evidence") or "").strip()
        if evidence:
            pieces.append(f"Evidence: {evidence}")
        filename = str((source.get("doc_meta") or {}).get("filename") or "").strip()
        if filename:
            pieces.append(f"Paper: {filename}")
    return " ".join(pieces)


def _project_triple_rank_to_nodes(triple_rank: Sequence[str], triple_records: Sequence[Mapping[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for triple_id in triple_rank:
        record = triple_records[int(triple_id)]
        for node_id in (str(record["subject"]), str(record["object"])):
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)
    return ordered


def _build_context(triples: Sequence[Mapping[str, Any]], paths: Sequence[Mapping[str, Any]]) -> str:
    sections: List[str] = []
    if triples:
        lines = ["Top ranked triples:"]
        for triple in triples[:8]:
            line = (
                f"- {triple['subject']} --[{triple['relation']}]--> {triple['object']} "
                f"(score={float(triple['score']):.3f}, weight={float(triple['weight']):g}, sources={int(triple['source_count'])})"
            )
            lines.append(line)
            if triple.get("evidence_excerpt"):
                lines.append(f"  Evidence: {triple['evidence_excerpt']}")
            if triple.get("paper"):
                lines.append(f"  Paper: {triple['paper']}")
        sections.append("\n".join(lines))

    if paths:
        lines = ["Supporting paths:"]
        for path in paths[:4]:
            lines.append(f"- {path['text']} (score={float(path['score']):.3f})")
        sections.append("\n".join(lines))

    if not sections:
        return "No relevant KG evidence found."
    return "\n\n".join(sections)


def _rrf_fuse(rankings: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings.values():
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (RRF_K + rank)
    return dict(scores)


def _sorted_score_items(score_map: Mapping[str, float], limit: int) -> List[Tuple[str, float]]:
    items = sorted(score_map.items(), key=lambda item: (-float(item[1]), item[0].casefold()))
    return [(key, float(value)) for key, value in items[:limit]]


def _rotate_scores(
    candidate_matrix: np.ndarray,
    anchor_vec: np.ndarray,
    relation_vec: np.ndarray,
    known_role: str,
) -> np.ndarray:
    if anchor_vec.ndim != 1 or relation_vec.ndim != 1:
        raise ValueError("expected flat anchor and relation vectors")

    if candidate_matrix.shape[1] % 2 != 0:
        raise ValueError("entity embedding dimension must be even for RotatE scoring")

    real_dim = candidate_matrix.shape[1] // 2
    entities_real = candidate_matrix[:, :real_dim]
    entities_imag = candidate_matrix[:, real_dim:]
    anchor_real = anchor_vec[:real_dim]
    anchor_imag = anchor_vec[real_dim:]

    if relation_vec.shape[0] == real_dim:
        phase = relation_vec
        relation_real = np.cos(phase)
        relation_imag = np.sin(phase)
    elif relation_vec.shape[0] == candidate_matrix.shape[1]:
        relation_real = relation_vec[:real_dim]
        relation_imag = relation_vec[real_dim:]
    else:
        raise ValueError("unexpected relation embedding shape")

    if known_role == "head":
        base_real = anchor_real * relation_real - anchor_imag * relation_imag
        base_imag = anchor_real * relation_imag + anchor_imag * relation_real
        diff_real = base_real[None, :] - entities_real
        diff_imag = base_imag[None, :] - entities_imag
    else:
        conj_real = relation_real
        conj_imag = -relation_imag
        base_real = anchor_real * conj_real - anchor_imag * conj_imag
        base_imag = anchor_real * conj_imag + anchor_imag * conj_real
        diff_real = base_real[None, :] - entities_real
        diff_imag = base_imag[None, :] - entities_imag

    return -np.linalg.norm(np.concatenate([diff_real, diff_imag], axis=1), axis=1)


def _fallback_relation_scores(
    candidate_matrix: np.ndarray,
    anchor_vec: np.ndarray,
    relation_vec: np.ndarray,
    known_role: str,
) -> np.ndarray:
    rel = relation_vec
    if relation_vec.shape[0] != anchor_vec.shape[0]:
        rel = np.resize(relation_vec, anchor_vec.shape[0])
    composed = anchor_vec + rel if known_role == "head" else anchor_vec - rel
    norms = np.linalg.norm(candidate_matrix, axis=1) * (np.linalg.norm(composed) + 1e-8)
    return (candidate_matrix @ composed) / (norms + 1e-8)


def _infer_anchor_role(anchor: Mapping[str, Any], relation_intent: RelationIntent) -> str:
    span = anchor.get("span")
    rel_span = relation_intent.span
    if not span or not rel_span:
        return "tail"
    anchor_after_relation = int(span[0]) >= int(rel_span[1])
    role = "tail" if anchor_after_relation else "head"
    if relation_intent.direction < 0:
        role = "head" if role == "tail" else "tail"
    return role


def _generate_aliases(node_id: str) -> List[str]:
    aliases = {str(node_id).strip()}
    normalized = _normalize_phrase(node_id)
    if normalized:
        aliases.add(normalized)

    no_punct = re.sub(r"[^0-9a-zA-Z\s]+", " ", str(node_id))
    no_punct = _normalize_phrase(no_punct)
    if no_punct:
        aliases.add(no_punct)

    alnum = re.sub(r"[^0-9a-zA-Z]+", "", str(node_id))
    if alnum:
        aliases.add(alnum.casefold())

    tokens = re.findall(r"[A-Za-z0-9]+", str(node_id))
    if len(tokens) >= 2:
        acronym = "".join(token[0] for token in tokens if token)
        if len(acronym) >= 2:
            aliases.add(acronym.casefold())

    return [alias for alias in aliases if alias]


def _normalize_phrase(text: str) -> str:
    text = re.sub(r"[_/]+", " ", str(text or ""))
    text = re.sub(r"[^0-9a-zA-Z\s-]+", " ", text)
    text = text.casefold().replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _pair_key(left: str, right: str) -> Tuple[str, str]:
    return tuple(sorted((left, right), key=lambda item: item.casefold()))


def _invert_mapping(mapping: Mapping[str, int]) -> List[str]:
    output = [""] * len(mapping)
    for key, value in mapping.items():
        output[int(value)] = str(key)
    return output


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _default_embedder(use_api: str = "openai", model_name: str = "text-embedding-3-small") -> Any:
    from Embedder import Embedder  # imported lazily to keep tests light

    return Embedder(use_api=use_api, model_name=model_name)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="KG retriever")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build cached retrieval sidecars")
    build_parser.add_argument("--graph", required=True, help="Path to graph JSON")
    build_parser.add_argument("--cache-dir", default=".kg_cache", help="Cache directory root")
    build_parser.add_argument("--method", choices=("hybrid", "semantic"), default="hybrid", help="Retriever method")
    build_parser.add_argument("--disable-kge", action="store_true", help="Skip optional RotatE training")
    build_parser.add_argument("--semantic-threshold", type=float, default=0.05, help="Semantic node filter threshold")
    build_parser.add_argument("--structural-threshold", type=float, default=0.05, help="Structural node filter threshold")

    query_parser = subparsers.add_parser("query", help="Query a graph with natural language")
    query_parser.add_argument("--graph", required=True, help="Path to graph JSON")
    query_parser.add_argument("--cache-dir", default=".kg_cache", help="Cache directory root")
    query_parser.add_argument("--question", required=True, help="Natural-language question")
    query_parser.add_argument("--top-k", type=int, default=10, help="Number of triples/nodes to return")
    query_parser.add_argument("--hop-limit", type=int, default=2, help="Maximum path hop count")
    query_parser.add_argument("--method", choices=("hybrid", "semantic"), default="hybrid", help="Retriever method")
    query_parser.add_argument("--disable-kge", action="store_true", help="Skip optional RotatE training")
    query_parser.add_argument("--semantic-threshold", type=float, default=0.05, help="Semantic node filter threshold")
    query_parser.add_argument("--structural-threshold", type=float, default=0.05, help="Structural node filter threshold")

    args = parser.parse_args()
    embedder = _default_embedder()
    retriever = build_index(
        graph_path=args.graph,
        cache_dir=args.cache_dir,
        text_embedder=embedder,
        kge_enabled=not args.disable_kge,
        method=args.method,
        semantic_threshold=args.semantic_threshold,
        structural_threshold=args.structural_threshold,
    )

    if args.command == "build":
        print(
            json.dumps(
                {
                    "graph": str(Path(args.graph)),
                    "cache_dir": str(retriever.cache_dir),
                    "method": getattr(retriever, "method", args.method),
                    "node_count": len(retriever.node_records),
                    "triple_count": len(retriever.triple_records),
                    "kge": retriever.kge.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    result = retriever.query(
        question=args.question,
        top_k=args.top_k,
        hop_limit=args.hop_limit,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(_cli())
