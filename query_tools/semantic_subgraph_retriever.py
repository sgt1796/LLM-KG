"""Semantic subgraph retriever adapted to the LLM-KG graph format.

The implementation follows the ALEQ (Adaptive Locating and Expanding Query)
semantic subgraph query shape from "From biomedical knowledge graph
construction to semantic querying: a comprehensive approach"
(Scientific Reports, 2025; https://www.nature.com/articles/s41598-025-93334-5),
but it deliberately reuses this repository's own KG contracts:

* graph JSON loading from ``kg_pipeline.rag``
* canonical relation aliases from ``kg_pipeline.triple_builder``
* cached node/triple/relation embeddings
* the shared ``QueryResult`` payload consumed by ``kg_rag_app.py``

That keeps this retriever interchangeable with ``HybridKGRetriever`` while
preserving the representative-node, semantic-filter, structural-filter, and
path-verification steps from the semantic subgraph algorithm.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from kg_pipeline.rag import (
    KGEArtifacts,
    QueryResult,
    RelationIntent,
    RELATION_EMBED_THRESHOLD,
    _build_alias_lookup,
    _build_context,
    _build_node_records,
    _build_relation_records,
    _build_triple_adjacency,
    _build_triple_records,
    _compile_alias_patterns,
    _compile_relation_patterns,
    _embed_texts,
    _infer_anchor_role,
    _load_graph,
    _load_or_compute_matrix,
    _normalize_phrase,
    _pair_key,
    _sorted_score_items,
    _write_json,
)


@dataclass
class SemanticDebugScores:
    """Compact debug information for a semantic subgraph query."""

    anchors: List[Dict[str, Any]]
    representative: str
    relation_intent: Optional[Dict[str, Any]]
    candidate_nodes: List[str]
    focus_nodes: List[str]
    semantic_scores: Dict[str, float]
    structural_scores: Dict[str, float]


class SemanticSubgraphRetriever:
    """Retrieve query-relevant local subgraphs around semantic anchor nodes."""

    method = "ALEQ"

    def __init__(
        self,
        graph_path: str | Path,
        text_embedder: Any,
        *,
        cache_dir: str | Path | None = None,
        semantic_threshold: float = 0.05,
        structural_threshold: float = 0.05,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(".kg_cache") / self.graph_path.stem
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.text_embedder = text_embedder
        self.semantic_threshold = float(semantic_threshold)
        self.structural_threshold = float(structural_threshold)

        graph, node_ids, raw_triples = _load_graph(self.graph_path)
        self.graph: nx.Graph = graph
        self.node_ids: List[str] = node_ids
        self.node_index: Dict[str, int] = {node_id: index for index, node_id in enumerate(node_ids)}
        self.triples: List[Dict[str, Any]] = raw_triples
        self.triple_records = _build_triple_records(raw_triples)
        self.incident_triple_ids, self.pair_to_triple_ids = _build_triple_adjacency(
            node_ids,
            self.triple_records,
        )
        self.alias_lookup = _build_alias_lookup(node_ids)
        self.alias_patterns = _compile_alias_patterns(self.alias_lookup)
        self.relation_patterns = _compile_relation_patterns()
        self.node_records = _build_node_records(
            node_ids,
            self.triple_records,
            self.incident_triple_ids,
            self.alias_lookup,
        )
        self.relation_records = _build_relation_records(self.triple_records)

        self.node_embeddings = _load_or_compute_matrix(
            cache_dir=self.cache_dir,
            basename="node_index",
            records=self.node_records,
            text_embedder=text_embedder,
        )
        self.triple_embeddings = _load_or_compute_matrix(
            cache_dir=self.cache_dir,
            basename="triple_index",
            records=self.triple_records,
            text_embedder=text_embedder,
        )
        self.relation_embeddings = _load_or_compute_matrix(
            cache_dir=self.cache_dir,
            basename="relation_index",
            records=self.relation_records,
            text_embedder=text_embedder,
        )
        _write_json(self.cache_dir / "aliases.json", self.alias_lookup)

        if self.graph.number_of_nodes() and self.graph.number_of_edges():
            self.pagerank = nx.pagerank(self.graph, alpha=0.85, weight="weight")
        else:
            equal_rank = 1.0 / max(1, len(self.node_ids))
            self.pagerank = {node_id: equal_rank for node_id in self.node_ids}
        self.neighbor_sets: Dict[str, set[str]] = {
            node_id: set(self.graph.neighbors(node_id)) for node_id in self.node_ids
        }
        self.kge = KGEArtifacts(
            available=False,
            metadata={
                "status": "disabled",
                "model": "semantic-subgraph",
                "reason": "ALEQ uses text and local graph structure only",
            },
            entity_embeddings=None,
            relation_embeddings=None,
            entity_to_id={},
            relation_to_id={},
        )

    def query(
        self,
        question: str,
        top_k: int = 10,
        hop_limit: int = 2,
        visible_node_ids: Optional[Iterable[str]] = None,
    ) -> QueryResult:
        """Search for a semantically relevant anchored subgraph."""

        question = str(question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        if not self.node_ids:
            raise ValueError("Graph contains no nodes")

        top_k = max(1, int(top_k))
        hop_limit = max(1, int(hop_limit))
        visible = set(visible_node_ids) if visible_node_ids is not None else None
        question_norm = _normalize_phrase(question)
        query_vec = _embed_texts(self.text_embedder, [question])[0]

        anchors = self._match_aliases(question_norm)
        relation_intent = self._detect_relation_intent(question_norm, query_vec)
        representative = self._select_representative(anchors, query_vec)

        semantic_scores = self._semantic_similarity(representative, query_vec)
        structural_scores = self._structural_similarity(representative, hop_limit)
        candidate_scores = self._candidate_scores(
            representative=representative,
            anchors=anchors,
            semantic_scores=semantic_scores,
            structural_scores=structural_scores,
            top_k=top_k,
        )
        candidate_nodes = [node_id for node_id, _ in _sorted_score_items(candidate_scores, limit=max(32, top_k * 8))]

        top_triples = self._build_triple_payloads(
            query_vec=query_vec,
            representative=representative,
            candidate_scores=candidate_scores,
            anchors=anchors,
            relation_intent=relation_intent,
            top_k=top_k,
        )
        focus_nodes = self._build_focus_nodes(
            representative=representative,
            anchors=anchors,
            top_triples=top_triples,
            candidate_scores=candidate_scores,
            visible_node_ids=visible,
            top_k=top_k,
        )
        paths = self._build_paths(
            representative=representative,
            anchors=anchors,
            top_triples=top_triples,
            focus_nodes=[node["id"] for node in focus_nodes],
            candidate_scores=candidate_scores,
            hop_limit=hop_limit,
            top_k=min(3, top_k),
        )
        context = _build_context(top_triples, paths)

        relation_payload = asdict(relation_intent) if relation_intent else None
        debug = SemanticDebugScores(
            anchors=anchors,
            representative=representative,
            relation_intent=relation_payload,
            candidate_nodes=candidate_nodes,
            focus_nodes=[node["id"] for node in focus_nodes],
            semantic_scores=dict(_sorted_score_items(semantic_scores, limit=10)),
            structural_scores=dict(_sorted_score_items(structural_scores, limit=10)),
        )
        return QueryResult(
            focus_nodes=focus_nodes,
            triples=top_triples,
            paths=paths,
            context=context,
            debug_scores=asdict(debug),
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

        if self.relation_embeddings.size == 0 or not self.relation_records:
            return None

        sims = (self.relation_embeddings @ query_vec.reshape(-1, 1)).ravel()
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < RELATION_EMBED_THRESHOLD:
            return None
        relation = str(self.relation_records[best]["relation"])
        return RelationIntent(
            relation=relation,
            direction=1,
            source="embedding",
            phrase=relation,
            score=score,
            span=None,
        )

    def _select_representative(self, anchors: Sequence[Mapping[str, Any]], query_vec: np.ndarray) -> str:
        if anchors:
            return max(
                (str(anchor["node_id"]) for anchor in anchors),
                key=lambda node_id: self._influence_score(node_id),
            )

        if self.node_embeddings.size:
            sims = (self.node_embeddings @ query_vec.reshape(-1, 1)).ravel()
            return max(
                self.node_ids,
                key=lambda node_id: (
                    0.75 * float(sims[self._node_index(node_id)])
                    + 0.25 * self._influence_score(node_id)
                ),
            )
        return max(self.node_ids, key=self._influence_score)

    def _semantic_similarity(self, representative: str, query_vec: np.ndarray) -> Dict[str, float]:
        if self.node_embeddings.size == 0:
            return {}

        query_sims = (self.node_embeddings @ query_vec.reshape(-1, 1)).ravel()
        rep_index = self._node_index(representative)
        rep_vec = self.node_embeddings[rep_index]
        rep_sims = (self.node_embeddings @ rep_vec.reshape(-1, 1)).ravel()
        return {
            node_id: float((0.7 * query_sims[index]) + (0.3 * rep_sims[index]))
            for index, node_id in enumerate(self.node_ids)
        }

    def _structural_similarity(self, representative: str, hop_limit: int) -> Dict[str, float]:
        rep_neighbors = self.neighbor_sets.get(representative, set())
        try:
            path_lengths = nx.single_source_shortest_path_length(
                self.graph,
                representative,
                cutoff=hop_limit,
            )
        except nx.NetworkXError:
            path_lengths = {}

        direct_weights = [
            float((self.graph.get_edge_data(representative, neighbor) or {}).get("weight", 1.0) or 1.0)
            for neighbor in rep_neighbors
        ]
        max_direct_weight = max(direct_weights) if direct_weights else 1.0
        scores: Dict[str, float] = {}
        for node_id in self.node_ids:
            neighbors = self.neighbor_sets.get(node_id, set())
            union = rep_neighbors | neighbors
            jaccard = (len(rep_neighbors & neighbors) / len(union)) if union else 0.0

            distance = path_lengths.get(node_id)
            proximity = (1.0 / (float(distance) + 1.0)) if distance is not None else 0.0
            edge_weight = 0.0
            if self.graph.has_edge(representative, node_id):
                raw_weight = float((self.graph.get_edge_data(representative, node_id) or {}).get("weight", 1.0) or 1.0)
                edge_weight = math.log1p(raw_weight) / (math.log1p(max_direct_weight) + 1e-8)

            scores[node_id] = float((0.5 * proximity) + (0.3 * jaccard) + (0.2 * edge_weight))
        return scores

    def _candidate_scores(
        self,
        *,
        representative: str,
        anchors: Sequence[Mapping[str, Any]],
        semantic_scores: Mapping[str, float],
        structural_scores: Mapping[str, float],
        top_k: int,
    ) -> Dict[str, float]:
        anchor_nodes = {str(anchor["node_id"]) for anchor in anchors}
        scores: Dict[str, float] = {}

        for node_id in self.node_ids:
            if node_id == representative:
                continue
            semantic = float(semantic_scores.get(node_id, 0.0))
            structural = float(structural_scores.get(node_id, 0.0))
            keep = (
                semantic >= self.semantic_threshold
                or structural >= self.structural_threshold
                or node_id in anchor_nodes
                or self.graph.has_edge(representative, node_id)
            )
            if not keep:
                continue

            score = (0.65 * semantic) + (0.35 * structural) + (0.03 * self._influence_score(node_id))
            if node_id in anchor_nodes:
                score += 0.25
            scores[node_id] = float(score)

        return scores

    def _build_triple_payloads(
        self,
        *,
        query_vec: np.ndarray,
        representative: str,
        candidate_scores: Mapping[str, float],
        anchors: Sequence[Mapping[str, Any]],
        relation_intent: Optional[RelationIntent],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        scope_nodes = {representative, *candidate_scores.keys(), *(str(anchor["node_id"]) for anchor in anchors)}
        candidate_ids: Dict[int, None] = {}
        for node_id in scope_nodes:
            for triple_id in self.incident_triple_ids.get(node_id, []):
                candidate_ids[int(triple_id)] = None

        triple_text_scores = self._triple_text_scores(query_vec)
        if not candidate_ids:
            return []

        anchor_nodes = {str(anchor["node_id"]) for anchor in anchors}
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for triple_id in candidate_ids:
            record = self.triple_records[triple_id]
            subject = str(record["subject"])
            obj = str(record["object"])
            relation = str(record["relation"])
            text_score = float(triple_text_scores.get(str(record["id"]), 0.0))
            node_bonus = max(
                float(candidate_scores.get(subject, 0.0)),
                float(candidate_scores.get(obj, 0.0)),
            )
            relation_bonus = self._relation_bonus(record, relation_intent, anchors)
            anchor_bonus = 0.25 * len(anchor_nodes.intersection({subject, obj}))
            representative_bonus = 0.2 if representative in {subject, obj} else 0.0
            source_bonus = 0.08 * math.log1p(float(record["source_count"]))
            weight_bonus = 0.12 * math.log1p(float(record["weight"]))
            score = (
                text_score
                + (0.45 * node_bonus)
                + relation_bonus
                + anchor_bonus
                + representative_bonus
                + source_bonus
                + weight_bonus
            )
            payload = {
                "id": str(record["id"]),
                "subject": subject,
                "relation": relation,
                "object": obj,
                "weight": float(record["weight"]),
                "score": float(score),
                "source_count": int(record["source_count"]),
                "evidence_excerpt": str(record["evidence_excerpt"]),
                "paper": str(record["paper"]),
            }
            ranked.append((score, payload))

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1]["subject"].casefold(),
                item[1]["relation"].casefold(),
                item[1]["object"].casefold(),
            )
        )
        return [payload for _, payload in ranked[:top_k]]

    def _triple_text_scores(self, query_vec: np.ndarray) -> Dict[str, float]:
        if self.triple_embeddings.size == 0:
            return {}
        sims = (self.triple_embeddings @ query_vec.reshape(-1, 1)).ravel()
        return {str(self.triple_records[index]["id"]): float(sims[index]) for index in range(len(self.triple_records))}

    def _relation_bonus(
        self,
        record: Mapping[str, Any],
        relation_intent: Optional[RelationIntent],
        anchors: Sequence[Mapping[str, Any]],
    ) -> float:
        if relation_intent is None:
            return 0.0
        if str(record["relation"]) != relation_intent.relation:
            return 0.0

        score = 0.9
        if not anchors:
            return score

        expected_role = _infer_anchor_role(anchors[0], relation_intent)
        actual_roles: List[str] = []
        for anchor in anchors:
            node_id = str(anchor["node_id"])
            if str(record["subject"]) == node_id:
                actual_roles.append("head")
            if str(record["object"]) == node_id:
                actual_roles.append("tail")

        if expected_role in actual_roles:
            score += 0.35
        elif actual_roles:
            score -= 0.2
        return score

    def _build_focus_nodes(
        self,
        *,
        representative: str,
        anchors: Sequence[Mapping[str, Any]],
        top_triples: Sequence[Mapping[str, Any]],
        candidate_scores: Mapping[str, float],
        visible_node_ids: Optional[set[str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = defaultdict(float)
        scores[representative] += 0.4 + self._influence_score(representative)

        for anchor in anchors:
            scores[str(anchor["node_id"])] += 0.5 + float(anchor.get("score", 0.0)) * 0.01

        for triple in top_triples:
            score = float(triple.get("score", 0.0))
            scores[str(triple["subject"])] += score
            scores[str(triple["object"])] += score

        for node_id, score in _sorted_score_items(candidate_scores, limit=max(16, top_k * 3)):
            scores[node_id] += 0.25 * float(score)

        ordered = _sorted_score_items(scores, limit=max(top_k * 3, 12))
        if visible_node_ids:
            filtered = [item for item in ordered if item[0] in visible_node_ids]
            if filtered:
                ordered = filtered
        return [{"id": node_id, "score": float(score)} for node_id, score in ordered[:top_k]]

    def _build_paths(
        self,
        *,
        representative: str,
        anchors: Sequence[Mapping[str, Any]],
        top_triples: Sequence[Mapping[str, Any]],
        focus_nodes: Sequence[str],
        candidate_scores: Mapping[str, float],
        hop_limit: int,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        pairs: List[Tuple[str, str]] = []
        anchor_nodes = [str(anchor["node_id"]) for anchor in anchors]

        if len(anchor_nodes) >= 2:
            for index, left in enumerate(anchor_nodes[:3]):
                for right in anchor_nodes[index + 1 : 3]:
                    pairs.append((left, right))

        for triple in top_triples[: max(top_k, 3)]:
            subject = str(triple["subject"])
            obj = str(triple["object"])
            if representative in {subject, obj}:
                other = obj if subject == representative else subject
                pairs.append((representative, other))
            else:
                pairs.append((subject, obj))

        for node_id in focus_nodes:
            if node_id != representative:
                pairs.append((representative, node_id))

        results: List[Dict[str, Any]] = []
        seen: set[Tuple[str, ...]] = set()
        for left, right in pairs:
            if left not in self.graph or right not in self.graph or left == right:
                continue
            try:
                path_nodes = nx.shortest_path(self.graph, left, right)
            except nx.NetworkXNoPath:
                continue
            if len(path_nodes) - 1 > hop_limit:
                continue
            key = tuple(path_nodes)
            if key in seen:
                continue
            seen.add(key)
            results.append(self._path_payload(path_nodes, candidate_scores))
            if len(results) >= top_k:
                break
        return results

    def _path_payload(self, path_nodes: Sequence[str], candidate_scores: Mapping[str, float]) -> Dict[str, Any]:
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
                edge_payload = {
                    "subject": record["subject"],
                    "relation": record["relation"],
                    "object": record["object"],
                    "triple_id": record["id"],
                }
                text_parts.append(f"{record['subject']} --[{record['relation']}]--> {record['object']}")
                score += 0.35 * float(record["weight"])
            score += float(candidate_scores.get(left, 0.0))
            edges.append(edge_payload)

        if path_nodes:
            score += float(candidate_scores.get(path_nodes[-1], 0.0))

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
                float(self.triple_records[triple_id]["weight"]),
                int(self.triple_records[triple_id]["source_count"]),
            ),
        )

    def _node_index(self, node_id: str) -> int:
        return self.node_index[node_id]

    def _influence_score(self, node_id: str) -> float:
        degree = float(self.graph.degree[node_id]) if node_id in self.graph else 0.0
        pagerank = float(self.pagerank.get(node_id, 0.0))
        return math.log1p(degree) + pagerank
