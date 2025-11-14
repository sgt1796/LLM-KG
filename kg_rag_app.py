
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

In this demo environment, external network calls (OpenAI, HF downloads) are disabled.
To allow local testing of the app wiring, you can set:

    export USE_DUMMY_EMBEDDER=1

which will replace the real Embedder with a deterministic dummy that returns
fixed-size random vectors. In your own environment, simply do NOT set this
variable and real embeddings will be used.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple
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


# ---- Optional: simple dummy embedder for offline testing --------------------


class DummyEmbedder:
    """
    A simple deterministic embedder used ONLY when USE_DUMMY_EMBEDDER=1 is set.

    It uses a fixed random seed so the same input text always maps to the same
    vector, and different texts have different vectors, but the values are
    arbitrary. This allows us to test the full stack (Flask, pyvis, JS) without
    making external API calls.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def get_embedding(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            seed = abs(hash(t)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dim)
            # normalize for cosine similarity
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v.astype("f"))
        return np.vstack(vecs)


def make_embedder() -> DummyEmbedder | Embedder:
    """
    Factory that returns either the real Embedder or a DummyEmbedder,
    depending on environment variables.
    """
    if os.getenv("USE_DUMMY_EMBEDDER") == "1":
        print("[kg_rag_app] Using DummyEmbedder for offline testing.")
        return DummyEmbedder(dim=64)

    # Real embedder: use OpenAI API by default; you can tweak as needed
    print("[kg_rag_app] Using real Embedder (OpenAI API).")
    # You can change use_api/model_name to Jina or local HF model if you prefer.
    return Embedder(use_api="openai", model_name=None)


# ---- Data structures --------------------------------------------------------


@dataclass
class KGRagState:
    graph: nx.Graph
    node_ids: List[str]
    node_embeddings: np.ndarray  # shape: (N, D), assumed L2-normalized
    embedder: DummyEmbedder | Embedder
    triples: List[dict]
    graph_path: Path
    cache_dir: Path


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


def ensure_node_embeddings(
    graph_path: Path,
    node_ids: List[str],
    embedder: DummyEmbedder | Embedder,
    cache_dir: Path,
) -> np.ndarray:
    """
    Compute (or load cached) node embeddings.

    We embed the node label itself for now. You can improve this by
    embedding richer descriptions (e.g., concatenating neighbor labels,
    relation types, evidence snippets, etc.).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / f"{graph_path.stem}_node_embeddings.npy"
    idx_path = cache_dir / f"{graph_path.stem}_node_ids.json"

    if emb_path.exists() and idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == node_ids:
            print(f"[kg_rag_app] Loading cached node embeddings from {emb_path}")
            emb = np.load(emb_path)
            return emb

    # (Re-)compute embeddings
    print("[kg_rag_app] Computing node embeddings via Embedder...")
    texts = node_ids  # simple: each node is just its label
    if not texts:
        raise SystemExit("[kg_rag_app] No nodes found to embed.")
    emb = embedder.get_embedding(texts).astype("f")

    # Normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = emb / norms

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
    return (
        "LLM answer stub.\n\n"
        "In your real environment, replace `call_llm_answer` in kg_rag_app.py\n"
        "with a call to POP.PromptFunction or your preferred LLM client,\n"
        "passing the question and the `context` string shown above."
    )


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
        #"--filter-menu",
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

    # Inject the query bar just before </body>
    injection = r"""
<!-- KG-RAG query panel injection -->
<div id="kg-rag-panel" style="
    position: fixed;
    top: 10px; left: 10px;
    z-index: 9999;
    background: rgba(0,0,0,0.75);
    padding: 10px;
    border-radius: 8px;
    color: #fff;
    max-width: 420px;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
">
  <div style="font-weight: 600; margin-bottom: 6px;">
    KG-RAG Demo
  </div>
  <div style="display:flex; gap:6px; margin-bottom:6px;">
    <input id="kg-query-input" type="text" placeholder="Ask a question about the graph..."
           style="flex:1; padding:4px 6px; border-radius:4px; border:0; font-size:13px;" />
    <button id="kg-query-btn" style="
        padding:4px 10px;
        border-radius:999px;
        border:0;
        background:#ffcc00;
        color:#000;
        font-weight:600;
        cursor:pointer;
    ">Ask</button>
  </div>
  <div id="kg-query-status" style="font-size:11px; opacity:0.8; margin-bottom:4px;"></div>
  <div id="kg-answer" style="
        max-height: 200px;
        overflow:auto;
        padding:6px 8px;
        border-radius:4px;
        background:rgba(0,0,0,0.4);
        font-size:12px;
        white-space:pre-wrap;
  "></div>
</div>

<script type="text/javascript">
(function(){
  // We assume pyvis created 'network', 'nodes', 'edges' variables.
  // If not yet defined, retry a bit later.
  function ensureVisObjects(cb, retries){
    retries = retries || 20;
    if (typeof network !== 'undefined' && typeof nodes !== 'undefined') {
      cb();
      return;
    }
    if (retries <= 0) return;
    setTimeout(function(){ ensureVisObjects(cb, retries-1); }, 200);
  }

  function highlightNodes(nodeIds){
    if (!Array.isArray(nodeIds) || !nodeIds.length) return;
    // Select the nodes and their connected edges (pyvis/vis.js highlights them)
    network.unselectAll();
    network.selectNodes(nodeIds, true);
    // Focus on the first node
    network.focus(nodeIds[0], {scale: 1.5, animation: true});
  }

  function setStatus(msg){
    var el = document.getElementById("kg-query-status");
    if (el) el.textContent = msg || "";
  }

  function setAnswer(msg){
    var el = document.getElementById("kg-answer");
    if (el) el.textContent = msg || "";
  }

  function attachHandlers(){
    var btn = document.getElementById("kg-query-btn");
    var input = document.getElementById("kg-query-input");
    if (!btn || !input) return;

    function sendQuery(){
      var q = input.value.trim();
      if (!q) return;
      setStatus("Querying KG-RAG backend...");
      setAnswer("");
      fetch("/query", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query: q, top_n: 15})
      })
      .then(function(res){ return res.json(); })
      .then(function(data){
        if (data.error){
          setStatus("Error: " + data.error);
          return;
        }
        setStatus("Top nodes highlighted based on semantic similarity.");
        if (Array.isArray(data.nodes)){
          var ids = data.nodes.map(function(n){ return n.id; });
          highlightNodes(ids);
        }
        if (data.answer){
          setAnswer(data.answer);
        } else if (data.context){
          setAnswer(data.context);
        }
      })
      .catch(function(err){
        console.error(err);
        setStatus("Request failed; see console for details.");
      });
    }

    btn.addEventListener("click", sendQuery);
    input.addEventListener("keydown", function(ev){
      if (ev.key === "Enter") sendQuery();
    });
  }

  ensureVisObjects(attachHandlers, 30);
})();
</script>
"""

    if "</body>" in html:
        html = html.replace("</body>", injection + "\n</body>")
    else:
        html = html + injection

    return html



# ---- Flask app wiring -------------------------------------------------------


def create_app(state: KGRagState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        html = build_pyvis_html(state.graph_path)
        # restrict node_ids to visible nodes from the HTML
        filtered_ids = extract_nodes_from_pyvis_html(html)
        if filtered_ids:
            print(f"[kg_rag_app] Filtered visible nodes: {len(filtered_ids)}")
            state.node_ids = filtered_ids

            state.node_embeddings = ensure_node_embeddings(
            state.graph_path,
            state.node_ids,
            state.embedder,
            state.cache_dir,
        )
        else:
            # this should not happen bc we only filter using --largest-only
            print("[kg_rag_app] WARNING: no filtered nodes extracted; keeping original full node_ids")

        return html

    @app.route("/query", methods=["POST"])
    def query():
        if state.node_embeddings is None:
            return jsonify({
                "error": "Embeddings not ready yet. Please reload the page."
            }), 500
        try:
            payload = request.get_json(force=True) or {}
            question = payload.get("query", "").strip()
            top_n = int(payload.get("top_n", 15))
        except Exception:
            return jsonify({"error": "Invalid JSON payload."}), 400

        if not question:
            return jsonify({"error": "Empty query."}), 400

        # Embed query
        try:
            q_vec = state.embedder.get_embedding([question]).astype("f")
            # normalize
            norms = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-8
            q_vec = q_vec / norms
        except Exception as e:
            return jsonify({"error": f"Embedding failed: {e}"}), 500

        # Retrieve top nodes
        top_nodes = cosine_top_k(q_vec[0], state.node_embeddings, state.node_ids, k=top_n)
        node_payload = [{"id": nid, "score": score} for nid, score in top_nodes]
        focus_ids = [nid for nid, _ in top_nodes]

        # Build KG context and (optionally) LLM answer
        context = build_llm_context(state.triples, focus_ids, max_triples=40)
        answer = call_llm_answer(question, context)

        return jsonify({
            "nodes": node_payload,
            "context": context,
            "answer": answer,
        })

    return app


# ---- CLI entrypoint ---------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KG-RAG demo server with pyvis highlighting.")
    ap.add_argument("--graph", type=str, default=None,
                    help="Path to graph JSON produced by main.py (triples format).")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--cache-dir", type=str, default=".kg_cache",
                    help="Directory to cache node embeddings.")
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

    G, node_ids, triples = load_graph(graph_path)
    embedder = make_embedder()
    cache_dir = Path(args.cache_dir)

    state = KGRagState(
        graph=G,
        node_ids=node_ids,
        node_embeddings=None, # updated during create_app
        embedder=embedder,
        triples=triples,
        graph_path=graph_path,
        cache_dir=cache_dir,
    )

    app = create_app(state)
    print(f"[kg_rag_app] Serving KG-RAG demo on http://{args.host}:{args.port}  (graph={graph_path})")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
