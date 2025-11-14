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
from llm_utils.LLMClient import OpenAIClient  # type: ignore
from llm_utils.POP import PromptFunction  # type: ignore
# ---- Data structures --------------------------------------------------------


@dataclass
class KGRagState:
    graph: nx.Graph
    node_ids: List[str]
    node_embeddings: np.ndarray  # shape: (N, D), assumed L2-normalized
    embedder: Embedder
    triples: List[dict]
    graph_path: Path
    cache_dir: Path
    incident_triples: Dict[str, List[dict]]


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
  max-width: 440px;
  min-width: 220px;
  min-height: 80px;
  resize: both;
  overflow: auto;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.25);
  cursor: move;
">
  <div id="kg-rag-panel-header" style="font-weight: 600; margin-bottom: 6px; cursor: move; user-select: none;">
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

  <!-- Global answer area -->
  <div id="kg-global-answer-box" style="
        margin-bottom:6px;
        padding:6px 8px;
        border-radius:4px;
        background:rgba(0,0,0,0.35);
        font-size:12px;
        max-height:120px;
        overflow:auto;
  ">
    <div style="font-size:11px; opacity:0.8; margin-bottom:2px;">
      Overall answer (top-N context)
    </div>
    <div id="kg-global-answer" style="white-space:pre-wrap;"></div>
  </div>

  <!-- Node pager for switching among top-N nodes -->
  <div id="kg-node-pager" style="
        display:none;
        align-items:center;
        gap:6px;
        margin-bottom:4px;
        font-size:11px;
  ">
    <button id="kg-prev-node" style="
        padding:2px 6px;
        border-radius:999px;
        border:0;
        background:#444;
        color:#fff;
        cursor:pointer;
    ">◀</button>
    <span id="kg-node-label" style="flex:1; opacity:0.9;"></span>
    <button id="kg-next-node" style="
        padding:2px 6px;
        border-radius:999px;
        border:0;
        background:#444;
        color:#fff;
        cursor:pointer;
    ">▶</button>
  </div>

  <div id="kg-answer" style="
            max-height: 220px;
            overflow:auto;
            padding:6px 8px;
            border-radius:4px;
            background:rgba(0,0,0,0.4);
            font-size:12px;
    ">
        <div style="font-size:11px; opacity:0.8; margin-bottom:2px;">
        Context triples
        </div>
        <pre id="kg-context" style="
            margin:0 0 4px 0;
            font-family:inherit;
            white-space:pre-wrap;
        "></pre>

        <div style="
            font-size:11px;
            opacity:0.8;
            margin-top:4px;
            margin-bottom:2px;
            border-top:1px solid rgba(255,255,255,0.2);
            padding-top:4px;
        ">
        AI explanation
        </div>
        <div id="kg-ai-answer" style="white-space:pre-wrap;"></div>
    </div>
    </div>
    <pre id="kg-context" style="
          margin:0 0 4px 0;
          font-family:inherit;
          white-space:pre-wrap;
    "></pre>

    <div style="
          font-size:11px;
          opacity:0.8;
          margin-top:4px;
          margin-bottom:2px;
          border-top:1px solid rgba(255,255,255,0.2);
          padding-top:4px;
    ">
      AI explanation
    </div>
    <div id="kg-ai-answer" style="white-space:pre-wrap;"></div>
  </div>
</div>

<script type="text/javascript">
// --- Draggable and resizable panel ---
(function(){
  var panel = document.getElementById('kg-rag-panel');
  var header = document.getElementById('kg-rag-panel-header');
  var isDragging = false, dragOffsetX = 0, dragOffsetY = 0;
  if (panel && header) {
    header.addEventListener('mousedown', function(e) {
      isDragging = true;
      dragOffsetX = e.clientX - panel.offsetLeft;
      dragOffsetY = e.clientY - panel.offsetTop;
      document.body.style.userSelect = 'none';
    });
    document.addEventListener('mousemove', function(e) {
      if (isDragging) {
        panel.style.left = (e.clientX - dragOffsetX) + 'px';
        panel.style.top = (e.clientY - dragOffsetY) + 'px';
      }
    });
    document.addEventListener('mouseup', function(e) {
      isDragging = false;
      document.body.style.userSelect = '';
    });
  }
  // State for current query results
  var kgResult = {
    nodes: [],              // [{id, score}, ...]
    currentIndex: 0,
    highlightedEdgeIds: [], // edges we modified
    edgeOriginalStyles: {}, // id -> {color, width}
    lastQuery: "",          // remember current question
    globalAnswer: ""        // overall answer from top-N context
  };


  function ensureVisObjects(cb, retries){
    retries = retries || 20;
    if (typeof network !== 'undefined' && typeof nodes !== 'undefined' && typeof edges !== 'undefined') {
      cb();
      return;
    }
    if (retries <= 0) return;
    setTimeout(function(){ ensureVisObjects(cb, retries-1); }, 200);
  }

  function setStatus(msg){
    var el = document.getElementById("kg-query-status");
    if (el) el.textContent = msg || "";
  }

  function setAnswer(ctx, ans){
    var ctxEl = document.getElementById("kg-context");
    var ansEl = document.getElementById("kg-ai-answer");
    if (ctxEl) ctxEl.textContent = ctx || "";
    if (ansEl) ansEl.textContent = ans || "";
  }
  function setGlobalAnswer(ans){
    var el = document.getElementById("kg-global-answer");
    if (el) el.textContent = ans || "";
  }
  function updateNodePager(){
    var pager = document.getElementById("kg-node-pager");
    var label = document.getElementById("kg-node-label");
    if (!pager || !label) return;

    if (!kgResult.nodes.length){
      pager.style.display = "none";
      label.textContent = "";
      return;
    }

    pager.style.display = "flex";
    var idx = kgResult.currentIndex;
    if (idx < 0) idx = 0;
    if (idx >= kgResult.nodes.length) idx = kgResult.nodes.length - 1;
    var node = kgResult.nodes[idx];
    label.textContent = (idx+1) + "/" + kgResult.nodes.length + "  " + node.id + "  (score=" + node.score.toFixed(3) + ")";
  }

  function resetHighlightedEdges(){
    if (!kgResult.highlightedEdgeIds.length) return;
    var updates = [];
    kgResult.highlightedEdgeIds.forEach(function(eid){
      var orig = kgResult.edgeOriginalStyles[eid];
      if (!orig) return;
      var u = { id: eid };
      if (orig.color !== undefined) u.color = orig.color;
      if (orig.width !== undefined) u.width = orig.width;
      updates.push(u);
    });
    if (updates.length) edges.update(updates);
    kgResult.highlightedEdgeIds = [];
  }

  function highlightEdgesBetweenSelected(selectedIds){
    var selectedSet = {};
    selectedIds.forEach(function(id){ selectedSet[id] = true; });

    var es = edges.get();
    var updates = [];

    es.forEach(function(e){
      var from = e.from, to = e.to;
      if (selectedSet[from] && selectedSet[to]){
        // store original once
        if (!kgResult.edgeOriginalStyles[e.id]){
          kgResult.edgeOriginalStyles[e.id] = {
            color: e.color || undefined,
            width: e.width || undefined
          };
        }
        updates.push({
          id: e.id,
          color: { color: "#ffcc00" },
          width: (e.width || 1) + 2
        });
        kgResult.highlightedEdgeIds.push(e.id);
      }
    });

    if (updates.length) edges.update(updates);
  }

  function applyHighlightForCurrentResult(){
    if (!kgResult.nodes.length) return;

    // reset edge styles from previous query
    resetHighlightedEdges();

    var ids = kgResult.nodes.map(function(n){ return n.id; });
    if (!ids.length) return;

    // Select all result nodes (vis.js will already emphasize neighbors)
    network.unselectAll();
    network.selectNodes(ids, true);

    // Focus on the "current" node in pager
    var idx = kgResult.currentIndex;
    if (idx < 0) idx = 0;
    if (idx >= kgResult.nodes.length) idx = kgResult.nodes.length - 1;
    kgResult.currentIndex = idx;
    var currentId = kgResult.nodes[idx].id;
    network.focus(currentId, { scale: 1.6, animation: true });

    // Explicitly boost edges connecting nodes in the top-N set
    highlightEdgesBetweenSelected(ids);

    updateNodePager();
  }
  
  function requestNodeExplain(){
    if (!kgResult.nodes.length) return;

    var idx = kgResult.currentIndex;
    if (idx < 0 || idx >= kgResult.nodes.length) return;

    var node = kgResult.nodes[idx];
    var q = kgResult.lastQuery || "";
    if (!q) {
      // If for some reason we lost the query, try reading from the input
      var input = document.getElementById("kg-query-input");
      if (input) q = input.value.trim();
    }
    if (!q) return;

    setStatus('Explaining why node "' + node.id + '" is relevant...');

    fetch("/node_explain", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        query: q,
        node_id: node.id,
        global_answer: kgResult.globalAnswer   // NEW
      })
    })
    .then(function(res){ return res.json(); })
    .then(function(data){
      if (data.error){
        setStatus("Error: " + data.error);
        return;
      }
      setStatus('Node "' + node.id + '" explanation loaded.');
      // Node area: local triples + node contribution explanation
      setAnswer(data.context || "", data.explanation || "");
    })
    .catch(function(err){
      console.error(err);
      setStatus("Node explanation failed; see console for details.");
    });
  }

  function showNodeAt(index){
    if (!kgResult.nodes.length) return;
    if (index < 0) index = 0;
    if (index >= kgResult.nodes.length) index = kgResult.nodes.length - 1;
    kgResult.currentIndex = index;
    applyHighlightForCurrentResult();
    requestNodeExplain();   // now update context+answer for this single node
  }

  function attachHandlers(){
    var btn = document.getElementById("kg-query-btn");
    var input = document.getElementById("kg-query-input");
    var prevBtn = document.getElementById("kg-prev-node");
    var nextBtn = document.getElementById("kg-next-node");

    if (!btn || !input) return;

    if (prevBtn){
      prevBtn.addEventListener("click", function(){
        showNodeAt(kgResult.currentIndex - 1);
      });
    }
    if (nextBtn){
      nextBtn.addEventListener("click", function(){
        showNodeAt(kgResult.currentIndex + 1);
      });
    }

    function sendQuery(){
      var q = input.value.trim();
      if (!q) return;
      setStatus("Querying KG-RAG backend...");
      setAnswer("", "");          // clear context + answer
      // clear previous state
      kgResult.nodes = [];
      kgResult.lastQuery = q;     // remember question
      resetHighlightedEdges();
      updateNodePager();

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

        // Save and show the global answer based on full top-N context
        kgResult.globalAnswer  = data.answer  || "";
        kgResult.globalContext = data.context || "";
        setGlobalAnswer(kgResult.globalAnswer);

        // (Optional) show the global context in the per-node area before any node is selected
        if (data.context || data.answer){
          setAnswer(data.context || "", data.answer || "");
        }
        if (Array.isArray(data.nodes) && data.nodes.length){
          kgResult.nodes = data.nodes;
          kgResult.currentIndex = 0;
          applyHighlightForCurrentResult();
          requestNodeExplain();  // this will now explain node 1 in terms of globalAnswer
        } else {
          setStatus("No matching nodes in visible graph.");
        }

        if (data.context || data.answer){
          setAnswer(data.context || "", data.answer || "");
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
        client='openai',
    )
    return ai.execute(
        model="gpt-5.1",
        temperature=0.1,
        question=question,
        global_answer=global_answer,
        node_id=node_id,
        node_context=node_context,
    )

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
                                       graph_path=state.graph_path,
                                        node_ids=state.node_ids,
                                        embedder=state.embedder,
                                        cache_dir=state.cache_dir,
                                        incident_triples=state.incident_triples
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

        explanation = call_llm_node_explanation(
            question=question,
            global_answer=global_answer,
            node_id=node_id,
            node_context=context,
        )

        return jsonify({
            "node_id": node_id,
            "context": context,        # local triples
            "explanation": explanation # per-node explanation
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
    incident_triples = build_incident_triples(node_ids, triples)
    
    embedder = Embedder(use_api="openai", model_name="text-embedding-3-small")
    cache_dir = Path(args.cache_dir)

    state = KGRagState(
        graph=G,
        node_ids=node_ids,
        node_embeddings=None, # updated during create_app
        embedder=embedder,
        triples=triples,
        graph_path=graph_path,
        cache_dir=cache_dir,
        incident_triples=incident_triples
    )

    app = create_app(state)
    print(f"[kg_rag_app] Serving KG-RAG demo on http://{args.host}:{args.port}  (graph={graph_path})")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
