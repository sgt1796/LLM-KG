"""Natural-language retriever helpers for ALEQ knowledge-graph search.

This module builds reusable text sidecars for graph JSON produced by
``main.py`` and exposes the ALEQ semantic subgraph retriever used by the Flask
app and CLI.
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


def build_index(
    graph_path: str | Path,
    cache_dir: str | Path,
    text_embedder: Any,
    *,
    semantic_threshold: float = 0.05,
    structural_threshold: float = 0.05,
) -> Any:
    """Build or load deterministic sidecars for the ALEQ retriever."""

    from query_tools.semantic_subgraph_retriever import SemanticSubgraphRetriever

    graph_path = Path(graph_path)
    return SemanticSubgraphRetriever(
        graph_path=graph_path,
        cache_dir=Path(cache_dir) / graph_path.stem,
        text_embedder=text_embedder,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
    )


def _build_index(
    graph_path: str | Path,
    cache_dir: str | Path,
    text_embedder: Any,
    *,
    semantic_threshold: float = 0.05,
    structural_threshold: float = 0.05,
) -> Any:
    """Compatibility wrapper for older internal callers."""

    return build_index(
        graph_path=graph_path,
        cache_dir=cache_dir,
        text_embedder=text_embedder,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
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


def _add_or_update_edge(
    graph: nx.Graph,
    subject: str,
    obj: str,
    relation: str,
    weight: float,
    raw_triple: Mapping[str, Any],
) -> None:
    """Add an undirected structural edge while preserving parallel triples."""

    if graph.has_edge(subject, obj):
        data = graph[subject][obj]
        data["weight"] = float(data.get("weight", 0.0) or 0.0) + weight
        relations = data.setdefault("relations", [])
        if relation and relation not in relations:
            relations.append(relation)
        data.setdefault("raw_triples", []).append(dict(raw_triple))
        return

    relations = [relation] if relation else []
    graph.add_edge(
        subject,
        obj,
        relation=relation,
        relations=relations,
        weight=weight,
        raw=dict(raw_triple),
        raw_triples=[dict(raw_triple)],
    )


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
        _add_or_update_edge(graph, subject, obj, relation, weight, tri)
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

    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(node_id))
        if token.casefold() != "s"
    ]
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
    build_parser.add_argument("--semantic-threshold", type=float, default=0.05, help="Semantic node filter threshold for ALEQ")
    build_parser.add_argument("--structural-threshold", type=float, default=0.05, help="Structural node filter threshold for ALEQ")

    query_parser = subparsers.add_parser("query", help="Query a graph with natural language")
    query_parser.add_argument("--graph", required=True, help="Path to graph JSON")
    query_parser.add_argument("--cache-dir", default=".kg_cache", help="Cache directory root")
    query_parser.add_argument("--question", required=True, help="Natural-language question")
    query_parser.add_argument("--top-k", type=int, default=10, help="Number of triples/nodes to return")
    query_parser.add_argument("--hop-limit", type=int, default=2, help="Maximum path hop count")
    query_parser.add_argument("--semantic-threshold", type=float, default=0.05, help="Semantic node filter threshold for ALEQ")
    query_parser.add_argument("--structural-threshold", type=float, default=0.05, help="Structural node filter threshold for ALEQ")

    args = parser.parse_args()
    embedder = _default_embedder()
    retriever = build_index(
        graph_path=args.graph,
        cache_dir=args.cache_dir,
        text_embedder=embedder,
        semantic_threshold=args.semantic_threshold,
        structural_threshold=args.structural_threshold,
    )

    if args.command == "build":
        print(
            json.dumps(
                {
                    "graph": str(Path(args.graph)),
                    "cache_dir": str(retriever.cache_dir),
                    "method": getattr(retriever, "method", "ALEQ"),
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
