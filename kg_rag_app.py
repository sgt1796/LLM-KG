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
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import sys, subprocess
import re

import numpy as np
from flask import Flask, jsonify, request
import networkx as nx
from pyvis.network import Network
from dotenv import load_dotenv
load_dotenv()

# ---- Import Embedder (your existing class) ---------------------------------

from Embedder import Embedder  # type: ignore
from kg_pipeline.rag import RETRIEVER_METHOD_CHOICES, build_index, normalize_retriever_method
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
    retriever_method: str = "normal"
    llm_enabled: bool = False


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
    kge_enabled: bool = True,
    embed_model: str = "text-embedding-3-small",
    retriever_method: str = "normal",
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
        kge_enabled=kge_enabled,
        method=retriever_method,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
    )
    node_ids = [record["id"] for record in retriever.node_records]
    incident_triples = build_incident_triples(node_ids, retriever.triples)
    return KGRagState(
        graph=retriever.graph,
        retriever=retriever,
        retriever_method=getattr(retriever, "method", normalize_retriever_method(retriever_method)),
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


def build_kg_rag_injection() -> str:
    """Load the HTML/CSS/JS template that turns PyVis into a KG-RAG workbench."""

    try:
        return KG_RAG_INJECTION_TEMPLATE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(
            f"[kg_rag_app] KG-RAG injection template not found: {KG_RAG_INJECTION_TEMPLATE}"
        ) from exc


def build_pyvis_html(graph_path: Path, height: str = "1000px", width: str = "100%") -> str:
    """
    Use the existing pyvis_view.py script to generate the base HTML
    (with all the project's filtering/physics settings), then inject
    a KG-RAG query bar and JS hooks to talk to the /query endpoint.

    This keeps behavior consistent with the rest of the project.
    """
    # Where to put the temporary HTML (pyvis_view's output)
    html_path = Path("kg_rag_temp.html")

    # Path to pyvis_view.py in the same repo
    pyvis_view_path = Path(__file__).with_name("pyvis_view.py")

    if not pyvis_view_path.exists():
        raise SystemExit(f"[kg_rag_app] pyvis_view.py not found at {pyvis_view_path}")

    # Build the command for pyvis_view.
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

    # Run pyvis_view to generate the HTML
    try:
        print(f"[kg_rag_app] Running pyvis_view: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"[kg_rag_app] pyvis_view.py failed: {e}") from e

    if not html_path.exists():
        raise SystemExit(f"[kg_rag_app] Expected HTML not found: {html_path}")

    # Load the generated HTML
    html = html_path.read_text(encoding="utf-8")

    injection = build_kg_rag_injection()
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
        "highlight": build_highlight_payload(result),
        "debug_scores": result.debug_scores,
    }


def create_app(state: KGRagState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        html = build_pyvis_html(state.graph_path)
        # restrict node_ids to visible nodes from the HTML
        filtered_ids = extract_nodes_from_pyvis_html(html)
        if filtered_ids:
            print(f"[kg_rag_app] Filtered visible nodes: {len(filtered_ids)}")
            state.visible_node_ids = filtered_ids
        else:
            # this should not happen bc we only filter using --largest-only
            print("[kg_rag_app] WARNING: no filtered nodes extracted; keeping full retriever node set")

        return html

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify(
            {
                "status": "ok",
                "graph_path": str(state.graph_path),
                "cache_dir": str(state.cache_dir),
                "retriever_method": state.retriever_method,
                "llm_enabled": state.llm_enabled,
                "llm_available": _llm_available(),
                "node_count": len(state.retriever.node_records),
                "triple_count": len(state.retriever.triple_records),
                "kge": state.retriever.kge.metadata,
            }
        )

    @app.route("/query", methods=["POST"])
    def query():
        try:
            payload = request.get_json(force=True) or {}
            question = payload.get("query", "").strip()
            top_n = int(payload.get("top_n", 15))
            hop_limit = int(payload.get("hop_limit", 2))
            include_answer = bool(payload.get("include_answer", True)) and state.llm_enabled
        except Exception:
            return jsonify({"error": "Invalid JSON payload."}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400

        try:
            response_payload = build_query_payload(
                state,
                question=question,
                top_n=top_n,
                hop_limit=hop_limit,
                visible_node_ids=state.visible_node_ids or None,
                include_answer=include_answer,
            )
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
            include_answer = bool(payload.get("include_answer", False)) and state.llm_enabled
        except Exception:
            return jsonify({"error": "Invalid JSON payload."}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400

        try:
            response_payload = build_query_payload(
                state,
                question=question,
                top_n=top_n,
                hop_limit=hop_limit,
                visible_node_ids=None,
                include_answer=include_answer,
            )
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
        except Exception:
            return jsonify({"error": "Invalid JSON payload."}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400

        retrieval_payload: Dict[str, Any] | None = None
        if not context:
            try:
                retrieval_payload = build_query_payload(
                    state,
                    question=question,
                    top_n=top_n,
                    hop_limit=hop_limit,
                    visible_node_ids=None,
                    include_answer=False,
                )
            except Exception as e:
                return jsonify({"error": f"Retrieval failed: {e}"}), 500
            context = retrieval_payload["context"]

        answer = maybe_call_llm_answer(question, context, enabled=state.llm_enabled)
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
            include_explanation = bool(payload.get("include_explanation", True)) and state.llm_enabled
        except Exception:
            return jsonify({"error": "Invalid JSON payload."}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400
        if not node_id:
            return jsonify({"error": "Missing node_id."}), 400

        node_triples = state.incident_triples.get(node_id, [])
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
    

    return app


def create_app_from_env(
    embedder_factory: Callable[[str], Any] | None = None,
) -> Flask:
    """Create the Flask app using environment variables for container deploys."""

    graph_path = Path(os.getenv("KG_GRAPH_PATH", "graph.json"))
    cache_dir = Path(os.getenv("KG_CACHE_DIR", ".kg_cache"))
    embed_model = os.getenv("KG_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    kge_enabled = _env_bool("KG_KGE_ENABLED", True)
    retriever_method = normalize_retriever_method(os.getenv("KG_RETRIEVER_METHOD", "normal"))
    semantic_threshold = float(os.getenv("KG_SEMANTIC_THRESHOLD", "0.05"))
    structural_threshold = float(os.getenv("KG_STRUCTURAL_THRESHOLD", "0.05"))
    llm_enabled = _resolve_llm_enabled()

    if not graph_path.exists():
        raise SystemExit(f"[kg_rag_app] Graph file not found: {graph_path}")

    if embedder_factory is None:
        embedder = Embedder(use_api="openai", model_name=embed_model)
    else:
        embedder = embedder_factory(embed_model)

    state = build_state(
        graph_path=graph_path,
        cache_dir=cache_dir,
        embedder=embedder,
        kge_enabled=kge_enabled,
        embed_model=embed_model,
        retriever_method=retriever_method,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
        llm_enabled=llm_enabled,
    )
    return create_app(state)

    
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
        "--disable-kge",
        action="store_true",
        help="Disable optional RotatE/PyKEEN graph-embedding artifacts.",
    )
    ap.add_argument(
        "--embed-model",
        type=str,
        default=os.getenv("KG_OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        help="Embedding model name forwarded into Embedder.",
    )
    ap.add_argument(
        "--retriever-method",
        type=normalize_retriever_method,
        choices=RETRIEVER_METHOD_CHOICES,
        default=normalize_retriever_method(os.getenv("KG_RETRIEVER_METHOD", "normal")),
        metavar="{normal,ALEQ}",
        help=(
            "KG retrieval method: normal is the default project retriever; "
            "ALEQ is Adaptive Locating and Expanding Query."
        ),
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
        graph_path = Path("graph.json")

    if not graph_path.exists():
        raise SystemExit(f"[kg_rag_app] Graph file not found: {graph_path}")

    cache_dir = Path(args.cache_dir)
    llm_enabled = _resolve_llm_enabled(enable_llm=args.enable_llm, disable_llm=args.disable_llm)
    if args.enable_llm and not _llm_available():
        print("[kg_rag_app] WARNING: --enable-llm was set, but OPENAI_API_KEY is not available.")

    state = build_state(
        graph_path=graph_path,
        cache_dir=cache_dir,
        embedder=Embedder(use_api="openai", model_name=args.embed_model),
        kge_enabled=not args.disable_kge,
        embed_model=args.embed_model,
        retriever_method=args.retriever_method,
        semantic_threshold=args.semantic_threshold,
        structural_threshold=args.structural_threshold,
        llm_enabled=llm_enabled,
    )

    app = create_app(state)
    print(
        f"[kg_rag_app] Serving KG-RAG demo on http://{args.host}:{args.port} "
        f"(graph={graph_path}, retriever={state.retriever_method}, llm={'on' if state.llm_enabled else 'off'})"
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
