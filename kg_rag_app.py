#!/usr/bin/env python3
"""
kg_rag_app.py

A minimal KG-RAG demo server that:

- Loads a graph JSON produced by main.py ({"nodes": [...], "triples": [...]})
- Uses pyvis to render an interactive HTML network
- Adds a query bar on top of the pyvis HTML
- On each query:
    * Embeds the query text using Embedder.py (same model family as node embeddings)
    * Computes cosine similarity against pre-computed node embeddings
    * Returns the top-N most relevant nodes
    * Highlights those nodes in the visualization
    * (Optionally) calls an LLM to answer the question using the local KG neighborhood

This file is designed to be dropped into the existing repo root and run as:

    export OPENAI_API_KEY=...
    python kg_rag_app.py --graph ADHD.json --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple
import sys, subprocess
import re

import numpy as np
from flask import Flask, jsonify, redirect, request, url_for
import networkx as nx
from dotenv import load_dotenv
load_dotenv()

# ---- Import Embedder (your existing class) ---------------------------------

from Embedder import Embedder  # type: ignore
from kg_pipeline.rag import build_index
from llm_utils.LLMClient import OpenAIClient  # type: ignore
from llm_utils.POP import PromptFunction  # type: ignore
# ---- Data structures --------------------------------------------------------


@dataclass
class KGRagState:
    graph: nx.Graph
    retriever: Any
    graph_path: Path
    cache_dir: Path
    incident_triples: Dict[str, List[dict]]
    visible_node_ids: List[str]
    retriever_method: str = "ALEQ"
    llm_enabled: bool = False


EVIDENCE_PREVIEW_CHARS = 240
DEFAULT_GRAPH_ID = "default"


@dataclass
class KGGraphRecord:
    graph_id: str
    graph_path: Path
    cache_dir: Path
    source_name: str
    node_ids: List[str]
    triples: List[dict]
    incident_triples: Dict[str, List[dict]]
    visible_node_ids: List[str]

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def triple_count(self) -> int:
        return len(self.triples)


class KGGraphRegistry:
    """Track uploaded/default graphs plus their lazy retriever states."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        embed_model: str = "text-embedding-3-small",
        embedder_factory: Callable[[str], Any] | None = None,
        semantic_threshold: float = 0.05,
        structural_threshold: float = 0.05,
        llm_enabled: bool | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.upload_dir = self.cache_dir / "uploads"
        self.render_dir = self.cache_dir / "rendered"
        self.embed_model = embed_model
        self.embedder_factory = embedder_factory
        self.semantic_threshold = float(semantic_threshold)
        self.structural_threshold = float(structural_threshold)
        self.llm_enabled = _llm_answer_enabled() if llm_enabled is None else bool(llm_enabled)
        self.records: Dict[str, KGGraphRecord] = {}
        self.states: Dict[str, KGRagState] = {}
        self.default_graph_id: str | None = None

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)

    def register_existing_graph(
        self,
        graph_path: Path,
        *,
        graph_id: str | None = None,
        source_name: str | None = None,
        make_default: bool = True,
    ) -> KGGraphRecord:
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        graph_id = graph_id or _graph_id_for_file(graph_path)
        _graph, node_ids, triples = load_graph(graph_path)
        _validate_graph_parts(node_ids, triples)
        record = KGGraphRecord(
            graph_id=graph_id,
            graph_path=graph_path,
            cache_dir=self.cache_dir,
            source_name=source_name or graph_path.name,
            node_ids=node_ids,
            triples=triples,
            incident_triples=build_incident_triples(node_ids, triples),
            visible_node_ids=[],
        )
        self.records[graph_id] = record
        if make_default or self.default_graph_id is None:
            self.default_graph_id = graph_id
        return record

    def register_upload(self, file_storage: Any) -> KGGraphRecord:
        filename = str(getattr(file_storage, "filename", "") or "uploaded_graph.json")
        payload = file_storage.read()
        if not payload:
            raise ValueError("Uploaded file is empty.")

        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ValueError("Uploaded file must be valid UTF-8 JSON.") from exc

        node_ids, triples = _validate_graph_data(data)
        graph_id = hashlib.sha256(payload).hexdigest()[:24]
        graph_path = self.upload_dir / f"{graph_id}.json"
        if not graph_path.exists():
            with open(graph_path, "wb") as handle:
                handle.write(payload)

        record = KGGraphRecord(
            graph_id=graph_id,
            graph_path=graph_path,
            cache_dir=self.cache_dir,
            source_name=Path(filename).name,
            node_ids=node_ids,
            triples=triples,
            incident_triples=build_incident_triples(node_ids, triples),
            visible_node_ids=[],
        )
        self.records[graph_id] = record
        if self.default_graph_id is None:
            self.default_graph_id = graph_id
        return record

    def get_record(self, graph_id: str | None = None) -> KGGraphRecord:
        resolved = self.resolve_graph_id(graph_id)
        if not resolved or resolved not in self.records:
            raise KeyError("No graph is loaded.")
        return self.records[resolved]

    def resolve_graph_id(self, graph_id: str | None = None) -> str | None:
        if graph_id:
            return str(graph_id)
        return self.default_graph_id

    def get_state(self, graph_id: str | None = None) -> KGRagState:
        record = self.get_record(graph_id)
        key = record.graph_id
        if key not in self.states:
            embedder = self._make_embedder()
            state = build_state(
                graph_path=record.graph_path,
                cache_dir=self.cache_dir,
                embedder=embedder,
                embed_model=self.embed_model,
                semantic_threshold=self.semantic_threshold,
                structural_threshold=self.structural_threshold,
                llm_enabled=self.llm_enabled,
            )
            state.visible_node_ids = list(record.visible_node_ids)
            self.states[key] = state
        return self.states[key]

    def set_visible_node_ids(self, graph_id: str, node_ids: Iterable[str]) -> None:
        record = self.records[graph_id]
        record.visible_node_ids = [str(node_id) for node_id in node_ids]
        for state_graph_id, state in self.states.items():
            if state_graph_id == graph_id:
                state.visible_node_ids = list(record.visible_node_ids)

    def render_path(self, graph_id: str) -> Path:
        return self.render_dir / f"{graph_id}.html"

    def _make_embedder(self) -> Any:
        if self.embedder_factory is not None:
            return self.embedder_factory(self.embed_model)
        return Embedder(use_api="openai", model_name=self.embed_model)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _llm_answer_enabled() -> bool:
    return _env_bool("KG_ENABLE_LLM_ANSWER", True) and bool(os.getenv("OPENAI_API_KEY"))


def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _resolve_llm_enabled(*, enable_llm: bool = False, disable_llm: bool = False) -> bool:
    """Resolve final LLM behavior from CLI flags plus the environment default."""

    if disable_llm:
        return False
    if enable_llm:
        return _llm_available()
    return _llm_answer_enabled()


# ---- Graph loading & embedding ----------------------------------------------


def load_graph(path: Path) -> Tuple[nx.Graph, List[str], List[dict]]:
    """
    Load a KG produced by main.py (triples format).

    Expected format:
        {
          "nodes": [...],               # optional, may be ignored
          "triples": [
             {"subject": "...", "relation": "...", "object": "...", "weight": 1, "sources": [...]},
             ...
          ]
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    triples = data.get("triples") or data.get("edges") or []

    G = nx.Graph()
    for tri in triples:
        h = tri.get("h") or tri.get("subject")
        t = tri.get("t") or tri.get("object")
        r = tri.get("r") or tri.get("relation")
        w = float(tri.get("weight", 1.0) or 1.0)
        if not h or not t:
            continue
        u, v = str(h), str(t)
        if not G.has_node(u):
            G.add_node(u)
        if not G.has_node(v):
            G.add_node(v)
        G.add_edge(u, v, relation=r, weight=w, raw=tri)

    node_ids = list(G.nodes())
    print(f"[kg_rag_app] Loaded graph: nodes={len(node_ids)}, edges={G.number_of_edges()}")
    return G, node_ids, triples


def _triple_subject(triple: dict) -> str:
    return str(triple.get("h") or triple.get("subject") or "").strip()


def _triple_object(triple: dict) -> str:
    return str(triple.get("t") or triple.get("object") or "").strip()


def _triple_relation(triple: dict) -> str:
    return str(triple.get("r") or triple.get("relation") or "").strip()


def _validate_graph_parts(node_ids: List[str], triples: List[dict]) -> None:
    valid = [
        triple
        for triple in triples
        if isinstance(triple, dict) and _triple_subject(triple) and _triple_object(triple)
    ]
    if not node_ids or not valid:
        raise ValueError("Graph JSON must contain at least one valid triple with subject/object nodes.")


def _node_id_from_payload(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("id") or node.get("name") or node.get("label") or "").strip()
    return str(node or "").strip()


def _validate_graph_data(data: Any) -> Tuple[List[str], List[dict]]:
    if not isinstance(data, dict):
        raise ValueError("Graph JSON must be an object with a triples or edges array.")

    raw_triples = data.get("triples") or data.get("edges") or []
    if not isinstance(raw_triples, list):
        raise ValueError("Graph JSON triples/edges must be an array.")

    node_ids: List[str] = []
    seen_nodes: set[str] = set()
    for node in data.get("nodes") or []:
        node_id = _node_id_from_payload(node)
        if node_id and node_id not in seen_nodes:
            seen_nodes.add(node_id)
            node_ids.append(node_id)

    valid_triples: List[dict] = []
    for triple in raw_triples:
        if not isinstance(triple, dict):
            continue
        subject = _triple_subject(triple)
        obj = _triple_object(triple)
        if not subject or not obj:
            continue
        valid_triples.append(triple)
        for node_id in (subject, obj):
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                node_ids.append(node_id)

    _validate_graph_parts(node_ids, valid_triples)
    return node_ids, valid_triples


def _graph_id_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:24]


def build_incident_triples(node_ids, triples):
    incident = {n: [] for n in node_ids}
    for tri in triples:
        h = tri.get("h") or tri.get("subject")
        t = tri.get("t") or tri.get("object")
        if h in incident:
            incident[h].append(tri)
        if t in incident:
            incident[t].append(tri)
    return incident


def build_state(
    graph_path: Path,
    cache_dir: Path,
    *,
    embedder: Any | None = None,
    embed_model: str = "text-embedding-3-small",
    semantic_threshold: float = 0.05,
    structural_threshold: float = 0.05,
    llm_enabled: bool | None = None,
) -> KGRagState:
    """Build app state once for both the UI and agent API routes."""

    if embedder is None:
        embedder = Embedder(use_api="openai", model_name=embed_model)

    retriever = build_index(
        graph_path=graph_path,
        cache_dir=cache_dir,
        text_embedder=embedder,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
    )
    node_ids = [record["id"] for record in retriever.node_records]
    incident_triples = build_incident_triples(node_ids, retriever.triples)
    return KGRagState(
        graph=retriever.graph,
        retriever=retriever,
        retriever_method=getattr(retriever, "method", "ALEQ"),
        graph_path=graph_path,
        cache_dir=cache_dir,
        incident_triples=incident_triples,
        visible_node_ids=[],
        llm_enabled=_llm_answer_enabled() if llm_enabled is None else bool(llm_enabled),
    )


def build_node_text(node, incident_triples, max_triples=5):
    pieces = [node]  # always begin with the label itself
    used = 0
    
    for tri in incident_triples:
        if used >= max_triples:
            break

        h = tri.get("subject") or tri.get("h")
        r = tri.get("relation") or tri.get("r")
        t = tri.get("object") or tri.get("t")

        # Skip inconsistent triples
        if node != h and node != t:
            continue

        # Human readable relation sentence
        if node == h:
            rel_str = f"{node} {r} {t}"
        else:
            rel_str = f"{h} {r} {node}"

        pieces.append(f"Relation: {rel_str}.")

        # Evidence
        sources = tri.get("sources") or []
        if sources:
            ev = sources[0].get("evidence", "")
            fn = (sources[0].get("doc_meta") or {}).get("filename", "")
            if ev:
                pieces.append(f"Evidence: {ev}")
            if fn:
                pieces.append(f"Paper: {fn}")

        used += 1

    return " ".join(pieces)

def ensure_node_embeddings(
    graph_path: Path,
    node_ids: List[str],
    embedder: Embedder,
    cache_dir: Path,
    incident_triples: Dict[str, List[dict]]
) -> np.ndarray:

    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / f"{graph_path.stem}_node_embeddings.npy"
    idx_path = cache_dir / f"{graph_path.stem}_node_ids.json"

    # Try load from cache
    if emb_path.exists() and idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == node_ids:
            print(f"[kg_rag_app] Loading cached node embeddings from {emb_path}")
            return np.load(emb_path)

    # Build descriptive texts
    print("[kg_rag_app] Computing node embeddings via Embedder...")

    texts = [
        build_node_text(n, incident_triples.get(n, []))
        for n in node_ids
    ]

    if not texts:
        raise SystemExit("[kg_rag_app] No nodes found to embed.")

    # Call your existing Embedder
    emb = embedder.get_embedding(texts).astype("f")

    # Normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = emb / norms

    # Save
    np.save(emb_path, emb)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(node_ids, f, ensure_ascii=False, indent=2)

    print(f"[kg_rag_app] Saved embeddings to {emb_path}")
    return emb

# ---- RAG logic --------------------------------------------------------------


def cosine_top_k(
    query_vec: np.ndarray,
    node_embeddings: np.ndarray,
    node_ids: List[str],
    k: int = 15,
) -> List[Tuple[str, float]]:
    """
    Return the top-K nodes by cosine similarity.
    """
    if query_vec.ndim == 1:
        q = query_vec[None, :]
    else:
        q = query_vec
    # assume both q and node_embeddings are normalized
    sims = (node_embeddings @ q.T).ravel()
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]  # sort descending
    return [(node_ids[i], float(sims[i])) for i in idx]


def build_llm_context(triples: List[dict], focus_nodes: List[str], max_triples: int = 40) -> str:
    """
    Build a textual description of the subgraph induced by focus_nodes.
    """
    focus_set = set(focus_nodes)
    selected = []
    for tri in triples:
        h = str(tri.get("h") or tri.get("subject"))
        t = str(tri.get("t") or tri.get("object"))
        if h in focus_set or t in focus_set:
            r = tri.get("r") or tri.get("relation", "")
            selected.append((h, r, t))
            if len(selected) >= max_triples:
                break

    if not selected:
        return "No directly connected triples found for the selected nodes."

    lines = []
    for h, r, t in selected:
        lines.append(f"- ({h}) --[{r}]--> ({t})")
    return "Relevant KG context:\n" + "\n".join(lines)


def call_llm_answer(question: str, context: str) -> str:
    """
    (Optional) Call an LLM to synthesize an answer.

    In this demo environment we avoid actually calling external APIs, so
    this function just returns a stub string. In your real environment,
    you can plug in POP.PromptFunction or a direct OpenAIClient call here.
    """
    # Example stub:
    ai = PromptFunction(
        sys_prompt="You are a helpful assistant that answers questions based on provided KG context.",
        prompt="Context:\n<<<context>>>\n\nQuestion: <<<question>>>\n\nAnswer:",
        client=OpenAIClient(),
    )

    result = ai.execute(
        model="gpt-5.1",
        temperature=0.0,
        context=context,
        question=question,
    )

    # IMPORTANT: return only the LLM answer, not context+answer.
    return result


def maybe_call_llm_answer(question: str, context: str, *, enabled: bool = True) -> str:
    """Best-effort LLM answer generation that can be disabled for deployments."""

    if not enabled or not _llm_available():
        return ""
    try:
        return call_llm_answer(question, context)
    except Exception as exc:
        print(f"[kg_rag_app] WARNING: answer generation failed: {exc}")
        return ""




# ---- Build pyvis HTML + inject query bar ------------------------------------
def extract_nodes_from_pyvis_html(html: str) -> List[str]:
    """
    Extract node IDs from a pyvis-generated HTML file by scanning
    the 'nodes = new vis.DataSet([...])' block.
    """
    m = re.search(r"nodes\s*=\s*new vis\.DataSet\(\s*(\[.*?\])\s*\)", html, re.S)
    if not m:
        raise SystemExit("[kg_rag_app] Could not locate nodes DataSet in HTML.")

    nodes_json = m.group(1)
    # Convert JS → JSON (safe because pyvis outputs JSON-like dicts)
    nodes = json.loads(nodes_json)
    return [n["id"] for n in nodes]


KG_RAG_INJECTION_TEMPLATE = Path(__file__).resolve().parent / "templates" / "kg_rag_injection.tmpl"


def build_kg_rag_injection(graph_id: str | None = None) -> str:
    """Load the HTML/CSS/JS template that turns PyVis into a KG-RAG workbench."""

    try:
        template = KG_RAG_INJECTION_TEMPLATE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(
            f"[kg_rag_app] KG-RAG injection template not found: {KG_RAG_INJECTION_TEMPLATE}"
        ) from exc

    config = (
        "<script type=\"text/javascript\">"
        f"window.KG_RAG_GRAPH_ID = {json.dumps(graph_id)};"
        f"window.KG_RAG_EVIDENCE_PREVIEW_CHARS = {EVIDENCE_PREVIEW_CHARS};"
        "</script>\n"
    )
    return config + template


def build_pyvis_html(
    graph_path: Path,
    height: str = "1000px",
    width: str = "100%",
    *,
    graph_id: str | None = None,
    html_path: Path | None = None,
) -> str:
    """
    Use the existing pyvis_view.py script to generate the base HTML
    (with all the project's filtering/physics settings), then inject
    a KG-RAG query bar and JS hooks to talk to the /query endpoint.

    This keeps behavior consistent with the rest of the project.
    """
    if html_path is None:
        rendered_dir = Path(".kg_cache") / "rendered"
        rendered_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", graph_id or graph_path.stem)
        html_path = rendered_dir / f"{safe_name}.html"
    else:
        html_path.parent.mkdir(parents=True, exist_ok=True)

    # Path to pyvis_view.py in the same repo
    pyvis_view_path = Path(__file__).with_name("pyvis_view.py")

    if not pyvis_view_path.exists():
        raise SystemExit(f"[kg_rag_app] pyvis_view.py not found at {pyvis_view_path}")

    if not html_path.exists():
        cmd = [
            sys.executable,
            str(pyvis_view_path),
            "--input", str(graph_path),
            "--html", str(html_path),
            "--weight", ">=0",
            "--k-core", "0",
            "--max-nodes", "500",
            "--max-edges", "600",
            "--label-top", "20",
            "--physics", "barnesHut",
            "--largest-only",
            "--directed",
            "--select-menu",
            "--filter-menu",
            "--config-ui",
            "--theme", "dark",
            "--cdn-resources", "in_line",
        ]

        try:
            print(f"[kg_rag_app] Running pyvis_view: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"[kg_rag_app] pyvis_view.py failed: {e}") from e

    if not html_path.exists():
        raise SystemExit(f"[kg_rag_app] Expected HTML not found: {html_path}")

    # Load the generated HTML
    html = html_path.read_text(encoding="utf-8")

    injection = build_kg_rag_injection(graph_id)
    if "</body>" in html:
        html = html.replace("</body>", injection + "\n</body>")
    else:
        html = html + injection

    return html


def call_llm_node_explanation(
    question: str,
    global_answer: str,
    node_id: str,
    node_context: str,
) -> str:
    """
    Ask the LLM to explain how a single node contributes
    to the overall answer.
    """
    ai = PromptFunction(
        sys_prompt=(
            "You explain how individual knowledge-graph nodes and edges "
            "contribute to answering a question. Be specific and honest "
            "about how strong or weak the connection is."
        ),
        prompt=(
            "Original question:\n<<<question>>>\n\n"
            "Overall answer based on the full KG context:\n<<<global_answer>>>\n\n"
            "Now focus on this single node: <<<node_id>>>.\n\n"
            "Local KG triples involving this node:\n<<<node_context>>>\n\n"
            "In 2–5 sentences, explain how this node and its incident edges "
            "help support, refine, or challenge the overall answer above. "
            "If the node is only weakly or indirectly related, say that explicitly."
        ),
        client=OpenAIClient(),
    )
    return ai.execute(
        model="gpt-5.1",
        temperature=0.1,
        question=question,
        global_answer=global_answer,
        node_id=node_id,
        node_context=node_context,
    )


def maybe_call_llm_node_explanation(
    question: str,
    global_answer: str,
    node_id: str,
    node_context: str,
    *,
    enabled: bool = True,
) -> str:
    """Best-effort node explanation for hosted deployments."""

    if not enabled or not _llm_available():
        return ""
    try:
        return call_llm_node_explanation(
            question=question,
            global_answer=global_answer,
            node_id=node_id,
            node_context=node_context,
        )
    except Exception as exc:
        print(f"[kg_rag_app] WARNING: node explanation failed: {exc}")
        return ""


def build_highlight_payload(result: Any) -> Dict[str, Any]:
    """Return graph-highlight metadata derived from a retriever result."""

    node_scores: Dict[str, float] = {}
    ranked_node_ids: List[str] = []
    for node in result.focus_nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        node_scores[node_id] = float(node.get("score", 0.0) or 0.0)
        ranked_node_ids.append(node_id)

    evidence_edges: List[Dict[str, Any]] = []
    highlight_node_ids = set(ranked_node_ids)
    for rank, triple in enumerate(result.triples, start=1):
        subject = str(triple.get("subject", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not subject or not obj:
            continue
        relation = str(triple.get("relation", "")).strip()
        highlight_node_ids.update([subject, obj])
        evidence_edges.append(
            {
                "from": subject,
                "to": obj,
                "relation": relation,
                "triple_id": str(triple.get("id", "")),
                "rank": rank,
                "score": float(triple.get("score", 0.0) or 0.0),
                "weight": float(triple.get("weight", 0.0) or 0.0),
                "kind": "triple",
            }
        )

    path_edges: List[Dict[str, Any]] = []
    for path_rank, path in enumerate(result.paths, start=1):
        for edge in path.get("edges", []) or []:
            subject = str(edge.get("subject", "")).strip()
            obj = str(edge.get("object", "")).strip()
            if not subject or not obj:
                continue
            relation = str(edge.get("relation", "")).strip()
            highlight_node_ids.update([subject, obj])
            path_edges.append(
                {
                    "from": subject,
                    "to": obj,
                    "relation": relation,
                    "triple_id": str(edge.get("triple_id", "")),
                    "rank": path_rank,
                    "kind": "path",
                }
            )

    return {
        "node_ids": sorted(highlight_node_ids, key=lambda item: item.casefold()),
        "ranked_node_ids": ranked_node_ids,
        "node_scores": node_scores,
        "edges": evidence_edges,
        "path_edges": path_edges,
    }

# ---- Flask app wiring -------------------------------------------------------


def build_query_payload(
    state: KGRagState,
    *,
    question: str,
    top_n: int,
    hop_limit: int = 2,
    visible_node_ids: List[str] | None = None,
    include_answer: bool = True,
) -> Dict[str, Any]:
    """Run retrieval and return a normalized JSON-ready payload."""

    result = state.retriever.query(
        question=question,
        top_k=top_n,
        hop_limit=hop_limit,
        visible_node_ids=visible_node_ids,
    )
    answer = maybe_call_llm_answer(question, result.context, enabled=include_answer)
    return {
        "nodes": result.focus_nodes,
        "triples": result.triples,
        "paths": result.paths,
        "context": result.context,
        "answer": answer,
        "llm_enabled": bool(include_answer and state.llm_enabled),
        "llm_available": _llm_available(),
        "retriever_method": state.retriever_method,
        "highlight": build_highlight_payload(result),
        "debug_scores": result.debug_scores,
    }


def _truncate_text(text: str, limit: int = EVIDENCE_PREVIEW_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _source_payload(source: Any) -> Dict[str, Any]:
    if not isinstance(source, dict):
        return {"raw": str(source)}
    return {
        "doc_id": source.get("doc_id"),
        "sentence_id": source.get("sentence_id"),
        "char_span": source.get("char_span"),
        "confidence": source.get("confidence"),
        "filename": (source.get("doc_meta") or {}).get("filename"),
        "evidence": source.get("evidence"),
    }


def _format_source(source: Any) -> str:
    payload = _source_payload(source)
    pieces: List[str] = []
    filename = payload.get("filename")
    if filename:
        pieces.append(f"File: {filename}")
    doc_id = payload.get("doc_id")
    if doc_id:
        pieces.append(f"Doc ID: {doc_id}")
    sentence_id = payload.get("sentence_id")
    if sentence_id is not None:
        pieces.append(f"Sentence: {sentence_id}")
    char_span = payload.get("char_span")
    if char_span is not None:
        pieces.append(f"Span: {char_span}")
    confidence = payload.get("confidence")
    if confidence is not None:
        pieces.append(f"Confidence: {confidence}")
    evidence = payload.get("evidence")
    if evidence:
        pieces.append(f"Evidence: {evidence}")
    if pieces:
        return "\n".join(str(piece) for piece in pieces)
    return str(payload.get("raw") or source or "")


def _matching_triples(
    triples: Iterable[dict],
    *,
    subject: str = "",
    obj: str = "",
    relation: str = "",
    node_id: str = "",
) -> List[dict]:
    matches: List[dict] = []
    for triple in triples:
        h = _triple_subject(triple)
        t = _triple_object(triple)
        r = _triple_relation(triple)
        if node_id and node_id not in {h, t}:
            continue
        if subject and obj and {subject, obj} != {h, t}:
            continue
        if relation and r and relation != r:
            continue
        matches.append(triple)
    return matches


def build_provenance_payload(
    record: KGGraphRecord,
    *,
    item_type: str,
    node_id: str = "",
    subject: str = "",
    obj: str = "",
    relation: str = "",
    preview_chars: int = EVIDENCE_PREVIEW_CHARS,
) -> Dict[str, Any]:
    if item_type == "node":
        triples = _matching_triples(record.triples, node_id=node_id)
        title = node_id or "Node"
    elif item_type == "edge":
        triples = _matching_triples(
            record.triples,
            subject=subject,
            obj=obj,
            relation=relation,
        )
        title = f"{subject} --[{relation or 'related to'}]-- {obj}".strip()
    else:
        raise ValueError("item_type must be node or edge")

    lines = [title, f"{len(triples)} connected evidence triple(s)."]
    triple_payloads: List[Dict[str, Any]] = []
    for index, triple in enumerate(triples, start=1):
        h = _triple_subject(triple)
        r = _triple_relation(triple) or "related to"
        t = _triple_object(triple)
        sources = triple.get("sources") or []
        lines.append("")
        lines.append(f"{index}. {h} --[{r}]--> {t}")
        if triple.get("weight") is not None:
            lines.append(f"Weight: {triple.get('weight')}")
        for source in sources:
            formatted = _format_source(source)
            if formatted:
                lines.append(formatted)
        triple_payloads.append(
            {
                "subject": h,
                "relation": r,
                "object": t,
                "weight": triple.get("weight", 1),
                "sources": [_source_payload(source) for source in sources],
            }
        )

    full_text = "\n".join(lines).strip()
    return {
        "graph_id": record.graph_id,
        "type": item_type,
        "title": title,
        "preview": _truncate_text(full_text, preview_chars),
        "preview_chars": int(preview_chars),
        "full_text": full_text,
        "is_truncated": len(re.sub(r"\s+", " ", full_text).strip()) > preview_chars,
        "triples": triple_payloads,
    }


def render_upload_page(error: str = "") -> str:
    error_html = (
        f'<div class="upload-error">{html.escape(error)}</div>'
        if error
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KG-RAG Workbench</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #050816;
      color: #e5e7eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        linear-gradient(rgba(148, 163, 184, 0.09) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.08) 1px, transparent 1px),
        #050816;
      background-size: 36px 36px;
    }}
    main {{
      width: min(520px, calc(100vw - 32px));
      padding: 22px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.94);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 18px;
      color: #94a3b8;
      line-height: 1.5;
    }}
    input[type="file"] {{
      display: block;
      width: 100%;
      min-height: 42px;
      padding: 8px;
      border: 1px solid #334155;
      border-radius: 6px;
      background: #020617;
      color: #e5e7eb;
    }}
    button {{
      width: 100%;
      height: 40px;
      margin-top: 12px;
      border: 1px solid #f59e0b;
      border-radius: 6px;
      background: #facc15;
      color: #111827;
      font-weight: 750;
      cursor: pointer;
    }}
    button:hover {{ background: #fde047; }}
    .upload-error {{
      margin-bottom: 12px;
      padding: 9px 10px;
      border: 1px solid rgba(248, 113, 113, 0.55);
      border-radius: 6px;
      background: rgba(127, 29, 29, 0.35);
      color: #fecaca;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>KG-RAG Workbench</h1>
    <p>Upload a knowledge-graph JSON file to render an interactive graph and query it with ALEQ retrieval.</p>
    {error_html}
    <form method="post" action="/upload" enctype="multipart/form-data">
      <input name="graph" type="file" accept=".json,application/json" required />
      <button type="submit">Upload graph JSON</button>
    </form>
  </main>
</body>
</html>"""


def _registry_from_state(state: KGRagState) -> KGGraphRegistry:
    registry = KGGraphRegistry(
        cache_dir=state.cache_dir,
        llm_enabled=state.llm_enabled,
    )
    node_ids = [record["id"] for record in state.retriever.node_records]
    graph_id = DEFAULT_GRAPH_ID
    registry.records[graph_id] = KGGraphRecord(
        graph_id=graph_id,
        graph_path=state.graph_path,
        cache_dir=state.cache_dir,
        source_name=state.graph_path.name,
        node_ids=node_ids,
        triples=state.retriever.triples,
        incident_triples=state.incident_triples,
        visible_node_ids=state.visible_node_ids,
    )
    registry.default_graph_id = graph_id
    registry.states[graph_id] = state
    return registry


def create_app(state: KGRagState | KGGraphRegistry | None = None) -> Flask:
    if isinstance(state, KGGraphRegistry):
        registry = state
    elif isinstance(state, KGRagState):
        registry = _registry_from_state(state)
    else:
        registry = KGGraphRegistry(cache_dir=Path(os.getenv("KG_CACHE_DIR", ".kg_cache")))

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("KG_UPLOAD_MAX_BYTES", str(100 * 1024 * 1024)))

    @app.route("/")
    def index():
        if registry.default_graph_id and registry.default_graph_id in registry.records:
            return redirect(url_for("graph_page", graph_id=registry.default_graph_id))
        return render_upload_page()

    @app.route("/upload", methods=["POST"])
    def upload_graph():
        file_storage = request.files.get("graph")
        if file_storage is None:
            return render_upload_page("Choose a graph JSON file to upload."), 400
        try:
            record = registry.register_upload(file_storage)
        except ValueError as exc:
            return render_upload_page(str(exc)), 400
        return redirect(url_for("graph_page", graph_id=record.graph_id))

    @app.route("/graph/<graph_id>", methods=["GET"])
    def graph_page(graph_id: str):
        try:
            record = registry.get_record(graph_id)
        except KeyError:
            return render_upload_page("That graph is not loaded in this server session."), 404

        html_text = build_pyvis_html(
            record.graph_path,
            graph_id=record.graph_id,
            html_path=registry.render_path(record.graph_id),
        )
        filtered_ids = extract_nodes_from_pyvis_html(html_text)
        if filtered_ids:
            print(f"[kg_rag_app] Filtered visible nodes for {record.graph_id}: {len(filtered_ids)}")
            registry.set_visible_node_ids(record.graph_id, filtered_ids)
        else:
            print("[kg_rag_app] WARNING: no filtered nodes extracted; keeping full retriever node set")

        return html_text

    def _payload_graph_id(payload: Dict[str, Any]) -> str | None:
        return str(payload.get("graph_id") or payload.get("graphId") or "").strip() or None

    def _state_from_payload(payload: Dict[str, Any]) -> KGRagState:
        graph_id = _payload_graph_id(payload)
        return registry.get_state(graph_id=graph_id)

    @app.route("/healthz", methods=["GET"])
    def healthz():
        graph_id = registry.resolve_graph_id(request.args.get("graph_id"))
        if not graph_id or graph_id not in registry.records:
            return jsonify(
                {
                    "status": "ok",
                    "graph_loaded": False,
                    "graph_count": len(registry.records),
                    "cache_dir": str(registry.cache_dir),
                    "llm_enabled": registry.llm_enabled,
                    "llm_available": _llm_available(),
                }
            )

        record = registry.records[graph_id]
        cached_state = registry.states.get(graph_id)
        return jsonify(
            {
                "status": "ok",
                "graph_loaded": True,
                "graph_id": graph_id,
                "graph_path": str(record.graph_path),
                "cache_dir": str(registry.cache_dir),
                "retriever_method": "ALEQ",
                "llm_enabled": registry.llm_enabled,
                "llm_available": _llm_available(),
                "node_count": record.node_count,
                "triple_count": record.triple_count,
                "visible_node_count": len(record.visible_node_ids),
                "kge": cached_state.retriever.kge.metadata if cached_state is not None else {"status": "not_loaded"},
            }
        )

    @app.route("/query", methods=["POST"])
    def query():
        try:
            payload = request.get_json(force=True) or {}
            question = payload.get("query", "").strip()
            top_n = int(payload.get("top_n", 15))
            hop_limit = int(payload.get("hop_limit", 2))
            query_state = _state_from_payload(payload)
            include_answer = bool(payload.get("include_answer", True)) and query_state.llm_enabled
        except Exception as exc:
            return jsonify({"error": f"Invalid JSON payload: {exc}"}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400

        try:
            record = registry.get_record(_payload_graph_id(payload))
            response_payload = build_query_payload(
                query_state,
                question=question,
                top_n=top_n,
                hop_limit=hop_limit,
                visible_node_ids=record.visible_node_ids or None,
                include_answer=include_answer,
            )
            response_payload["graph_id"] = record.graph_id
        except Exception as e:
            return jsonify({"error": f"Retrieval failed: {e}"}), 500

        return jsonify(response_payload)

    @app.route("/api/search", methods=["POST"])
    def api_search():
        try:
            payload = request.get_json(force=True) or {}
            question = str(payload.get("query") or payload.get("question") or "").strip()
            top_n = int(payload.get("top_k", payload.get("top_n", 10)))
            hop_limit = int(payload.get("hop_limit", 2))
            query_state = _state_from_payload(payload)
            include_answer = bool(payload.get("include_answer", False)) and query_state.llm_enabled
        except Exception as exc:
            return jsonify({"error": f"Invalid JSON payload: {exc}"}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400

        try:
            record = registry.get_record(_payload_graph_id(payload))
            response_payload = build_query_payload(
                query_state,
                question=question,
                top_n=top_n,
                hop_limit=hop_limit,
                visible_node_ids=None,
                include_answer=include_answer,
            )
            response_payload["graph_id"] = record.graph_id
        except Exception as e:
            return jsonify({"error": f"Retrieval failed: {e}"}), 500

        return jsonify(response_payload)

    @app.route("/api/answer", methods=["POST"])
    def api_answer():
        try:
            payload = request.get_json(force=True) or {}
            question = str(payload.get("query") or payload.get("question") or "").strip()
            context = str(payload.get("context") or "").strip()
            top_n = int(payload.get("top_k", payload.get("top_n", 10)))
            hop_limit = int(payload.get("hop_limit", 2))
            query_state = _state_from_payload(payload)
        except Exception as exc:
            return jsonify({"error": f"Invalid JSON payload: {exc}"}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400

        retrieval_payload: Dict[str, Any] | None = None
        if not context:
            try:
                retrieval_payload = build_query_payload(
                    query_state,
                    question=question,
                    top_n=top_n,
                    hop_limit=hop_limit,
                    visible_node_ids=None,
                    include_answer=False,
                )
            except Exception as e:
                return jsonify({"error": f"Retrieval failed: {e}"}), 500
            context = retrieval_payload["context"]

        answer = maybe_call_llm_answer(question, context, enabled=query_state.llm_enabled)
        response_payload = {
            "question": question,
            "context": context,
            "answer": answer,
        }
        if retrieval_payload is not None:
            response_payload["retrieval"] = retrieval_payload
        return jsonify(response_payload)

    @app.route("/node_explain", methods=["POST"])
    def node_explain():
        """
        Explain how a SINGLE node contributes to the overall answer.

        Uses only that node's incident triples + the global answer.
        """
        try:
            payload = request.get_json(force=True) or {}
            question = (payload.get("query") or "").strip()
            node_id = str(payload.get("node_id") or "").strip()
            global_answer = (payload.get("global_answer") or "").strip()
            query_state = _state_from_payload(payload)
            include_explanation = bool(payload.get("include_explanation", True)) and query_state.llm_enabled
        except Exception as exc:
            return jsonify({"error": f"Invalid JSON payload: {exc}"}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400
        if not node_id:
            return jsonify({"error": "Missing node_id."}), 400

        node_triples = query_state.incident_triples.get(node_id, [])
        context = build_llm_context(
            triples=node_triples,
            focus_nodes=[node_id],
            max_triples=20,
        )

        explanation = maybe_call_llm_node_explanation(
            question=question,
            global_answer=global_answer,
            node_id=node_id,
            node_context=context,
            enabled=include_explanation,
        )

        return jsonify({
            "node_id": node_id,
            "context": context,        # local triples
            "explanation": explanation, # per-node explanation
            "llm_enabled": include_explanation,
            "llm_available": _llm_available(),
        })

    @app.route("/api/node-explain", methods=["POST"])
    def api_node_explain():
        return node_explain()

    @app.route("/api/provenance", methods=["POST", "GET"])
    def api_provenance():
        try:
            if request.method == "POST":
                payload = request.get_json(force=True) or {}
            else:
                payload = dict(request.args)
            record = registry.get_record(_payload_graph_id(payload))
            item_type = str(payload.get("type") or payload.get("item_type") or "").strip()
            node_id = str(payload.get("node_id") or payload.get("id") or "").strip()
            subject = str(payload.get("from") or payload.get("subject") or "").strip()
            obj = str(payload.get("to") or payload.get("object") or "").strip()
            relation = str(payload.get("relation") or "").strip()
            preview_chars = int(payload.get("preview_chars", EVIDENCE_PREVIEW_CHARS))
        except Exception as exc:
            return jsonify({"error": f"Invalid provenance request: {exc}"}), 400

        try:
            return jsonify(
                build_provenance_payload(
                    record,
                    item_type=item_type,
                    node_id=node_id,
                    subject=subject,
                    obj=obj,
                    relation=relation,
                    preview_chars=preview_chars,
                )
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
    

    return app


def create_app_from_env(
    embedder_factory: Callable[[str], Any] | None = None,
) -> Flask:
    """Create the Flask app using environment variables for container deploys."""

    graph_path_env = os.getenv("KG_GRAPH_PATH")
    cache_dir = Path(os.getenv("KG_CACHE_DIR", ".kg_cache"))
    embed_model = os.getenv("KG_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    semantic_threshold = float(os.getenv("KG_SEMANTIC_THRESHOLD", "0.05"))
    structural_threshold = float(os.getenv("KG_STRUCTURAL_THRESHOLD", "0.05"))
    llm_enabled = _resolve_llm_enabled()

    registry = KGGraphRegistry(
        cache_dir=cache_dir,
        embed_model=embed_model,
        embedder_factory=embedder_factory,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
        llm_enabled=llm_enabled,
    )

    if graph_path_env:
        graph_path = Path(graph_path_env)
        if not graph_path.exists():
            raise SystemExit(f"[kg_rag_app] Graph file not found: {graph_path}")
        try:
            registry.register_existing_graph(graph_path)
        except ValueError as exc:
            raise SystemExit(f"[kg_rag_app] Invalid graph file {graph_path}: {exc}") from exc

    return create_app(registry)

    
# ---- CLI entrypoint ---------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KG-RAG demo server with pyvis highlighting.")
    ap.add_argument("--graph", type=str, default=None,
                    help="Path to graph JSON produced by main.py (triples format).")
    ap.add_argument("--host", type=str, default=os.getenv("KG_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    ap.add_argument("--cache-dir", type=str, default=os.getenv("KG_CACHE_DIR", ".kg_cache"),
                    help="Directory to cache node embeddings.")
    ap.add_argument(
        "--embed-model",
        type=str,
        default=os.getenv("KG_OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        help="Embedding model name forwarded into Embedder.",
    )
    ap.add_argument(
        "--semantic-threshold",
        type=float,
        default=float(os.getenv("KG_SEMANTIC_THRESHOLD", "0.05")),
        help="Semantic node filter threshold for the ALEQ retriever.",
    )
    ap.add_argument(
        "--structural-threshold",
        type=float,
        default=float(os.getenv("KG_STRUCTURAL_THRESHOLD", "0.05")),
        help="Structural node filter threshold for the ALEQ retriever.",
    )
    llm_group = ap.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM answers and node explanations when OPENAI_API_KEY is set.",
    )
    llm_group.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable LLM answers and node explanations.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    graph_path_env = os.getenv("KG_GRAPH_PATH")
    if args.graph:
        graph_path = Path(args.graph)
    elif graph_path_env:
        graph_path = Path(graph_path_env)
    else:
        graph_path = None

    cache_dir = Path(args.cache_dir)
    llm_enabled = _resolve_llm_enabled(enable_llm=args.enable_llm, disable_llm=args.disable_llm)
    if args.enable_llm and not _llm_available():
        print("[kg_rag_app] WARNING: --enable-llm was set, but OPENAI_API_KEY is not available.")

    registry = KGGraphRegistry(
        cache_dir=cache_dir,
        embed_model=args.embed_model,
        semantic_threshold=args.semantic_threshold,
        structural_threshold=args.structural_threshold,
        llm_enabled=llm_enabled,
    )

    if graph_path is not None:
        if not graph_path.exists():
            raise SystemExit(f"[kg_rag_app] Graph file not found: {graph_path}")
        try:
            registry.register_existing_graph(graph_path)
        except ValueError as exc:
            raise SystemExit(f"[kg_rag_app] Invalid graph file {graph_path}: {exc}") from exc

    app = create_app(registry)
    graph_label = graph_path if graph_path is not None else "upload-first"
    print(
        f"[kg_rag_app] Serving KG-RAG demo on http://{args.host}:{args.port} "
        f"(graph={graph_label}, retriever=ALEQ, llm={'on' if registry.llm_enabled else 'off'})"
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
