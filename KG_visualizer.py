#!/usr/bin/env python3
"""
KG_visualizer.py — cleaner, not-messy graph view.

Usage (defaults are sane):
  python KG_visualizer.py
  python KG_visualizer.py --input graph.json --out graph_viz.png --min-weight 2 \
      --max-nodes 300 --label-top 20 --k-core 0 --largest-only

Tips:
- Increase --k-core to 3/4/5 to focus on the dense heart.
- Reduce --max-nodes if the plot is still too busy.
- Use --label-top to control how many labels are shown.
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import networkx as nx
from collections import Counter, defaultdict
import numpy as np

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--input", default="graph.json", help="Path to graph JSON")
ap.add_argument("--out", default="graph_viz.png", help="Output PNG path")
ap.add_argument("--svg", default=None, help="Optional SVG output path")
ap.add_argument("--min-weight", type=float, default=2.0, help="Keep edges with weight >= this")
ap.add_argument("--largest-only", action="store_true", help="Keep only largest connected component")
ap.add_argument("--k-core", type=int, default=0, help="If >0, take k-core of the (sub)graph")
ap.add_argument("--max-nodes", type=int, default=300, help="Cap node count by degree to avoid clutter")
ap.add_argument("--label-top", type=int, default=20, help="Label top-K nodes by degree")
ap.add_argument("--seed", type=int, default=42, help="Layout seed")
ap.add_argument("--highlight-edges-top", type=int, default=20, help="Highlight top-K strongest edges")
ap.add_argument("--edge-labels-top", type=int, default=10, help="Show edge labels for top-K strongest edges")
ap.add_argument("--layout", choices=[
    "bipartite", "circular", 
    "kamada_kawai", "planar", "random", "rescale", 
    "shell", "spring", "spectral", "spiral", "multipartite"
], default="spring",
help="Graph layout algorithm: arf, bipartite, bfs, circular, forceatlas2, kamada_kawai, planar, random, rescale, shell, spring, spectral, spiral, multipartite")
args = ap.parse_args()

in_path = Path(args.input)
if not in_path.exists():
    raise SystemExit(f"Missing input: {in_path}")

# ------------- Robust loader -------------
def load_graph(p: Path):
    """Load a knowledge graph from JSON or JSONL.

    This loader supports several formats:

    * D3-style graphs with ``nodes`` and ``edges``/``links`` arrays.
    * Triplet graphs with a top-level ``triples`` array (``subject``,
      ``relation``, ``object``, ``weight`` fields).
    * Deduplicated JSONL triples produced by ``dedupe.py`` (one JSON
      object per line with keys ``h``/``r``/``t`` and ``weight``).

    Returns: nodes, edges, rel_counts, rel_bag
      - nodes: [{id: ...}, ...]
      - edges: [(u, v, w), ...]
      - rel_counts: Counter over all relation strings
      - rel_bag: dict[(u,v)] -> Counter of relation strings for that pair
    """
    rel_counts = Counter()
    rel_bag = defaultdict(Counter)
    # If the path has a .jsonl extension, treat each line as a triple record.
    if p.suffix.lower() == ".jsonl":
        nodes, edges = {}, []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # dedupe writes h,r,t
                h = rec.get("h") or rec.get("subject")
                t = rec.get("t") or rec.get("object")
                r = rec.get("r") or rec.get("relation", "")
                w = rec.get("weight", 1)
                if h is None or t is None:
                    continue
                u = str(h)
                v = str(t)
                try:
                    w = float(w)
                except Exception:
                    w = 1.0
                edges.append((u, v, w))
                nodes[u] = {}; nodes[v] = {}
                if r:
                    rel_counts[r] += w
                    rel_bag[(u, v)][r] += w
        node_list = [{"id": n, **attrs} for n, attrs in nodes.items()]
        return node_list, edges, rel_counts, rel_bag

    # Otherwise parse as JSON object/list
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    rel_counts = Counter()

    # Triples format
    if isinstance(data, dict) and "triples" in data:
        triples = data.get("triples", [])
        nodes, edges = {}, []
        for tri in triples:
            # Support both aggregated and raw triples
            h = tri.get("h") or tri.get("subject")
            r = tri.get("r") or tri.get("relation", "")
            t = tri.get("t") or tri.get("object")
            w = tri.get("weight", 1)
            if h is None or t is None:
                continue
            u = str(h)
            v = str(t)
            try:
                w_val = float(w)
            except Exception:
                w_val = 1.0
            edges.append((u, v, w_val))
            nodes[u], nodes[v] = {}, {}
            if r:
                rel_counts[r] += w_val
                rel_bag[(u, v)][r] += w_val
        node_list = [{"id": n, **attrs} for n, attrs in nodes.items()]
        return node_list, edges, rel_counts, rel_bag

    # D3 or list-of-edges format
    nodes, edges_raw = [], []
    if isinstance(data, dict):
        nodes = data.get("nodes", data.get("Vertices", []))
        # prefer edges/links arrays if present
        edges_raw = data.get("edges", data.get("links", data.get("Edges", [])))
    elif isinstance(data, list):
        edges_raw = data
    else:
        raise ValueError("Unsupported JSON schema")

    norm_edges = []
    node_ids = set()
    def _nid(n):
        if isinstance(n, dict):
            return str(n.get("id", n.get("name", n.get("label"))))
        return str(n)
    for n in nodes:
        try:
            node_ids.add(_nid(n))
        except Exception:
            pass
    for e in edges_raw:
        if isinstance(e, dict):
            u = e.get("source", e.get("from", e.get("u")))
            v = e.get("target", e.get("to", e.get("v")))
            w = e.get("weight", 1.0)
            if u is None or v is None:
                continue
            u, v = str(u), str(v)
            node_ids.update([u, v])
            try:
                w = float(w)
            except Exception:
                w = 1.0
            norm_edges.append((u, v, w))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            u, v = str(e[0]), str(e[1])
            try:
                w = float(e[2]) if len(e) >= 3 else 1.0
            except Exception:
                w = 1.0
            node_ids.update([u, v])
            norm_edges.append((u, v, w))
    if not nodes:
        nodes = [{"id": n} for n in node_ids]
    return nodes, norm_edges, rel_counts

# Load graph; returns nodes, edges and relation counts (if present)
nodes, edges, rel_counts, rel_bag = load_graph(in_path)

# ------------- Build filtered graph -------------
G = nx.Graph()
# add nodes (keeping any attributes present)
for n in nodes:
    if isinstance(n, dict):
        nid = str(n.get("id", n.get("name", n.get("label"))))
        if nid is None:
            continue
        attrs = {k: v for k, v in n.items() if k not in ("id", "name", "label")}
        G.add_node(nid, **attrs)
    else:
        G.add_node(str(n))

# add edges with threshold
for u, v, w in edges:
    if float(w) >= args.min_weight:
        lab = None
        bag = rel_bag.get((u, v)) or rel_bag.get((v, u))  # handle undirected symmetry
        if bag:
            lab = bag.most_common(1)[0][0]
        G.add_edge(u, v, weight=float(w), label=lab)

# Remove isolates after thresholding
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

if G.number_of_nodes() == 0:
    raise SystemExit("Graph is empty after filtering. Try lowering --min-weight.")

# Largest connected component
if args.largest_only and nx.number_connected_components(G) > 1:
    big = max(nx.connected_components(G), key=len)
    G = G.subgraph(big).copy()

# Optional k-core
if args.k_core and args.k_core > 0:
    try:
        H = nx.k_core(G, k=args.k_core)
        if H.number_of_nodes() > 0:
            G = H
    except Exception:
        pass

# Cap to max-nodes by keeping highest degree nodes and induced subgraph
if G.number_of_nodes() > args.max_nodes:
    degs = dict(G.degree())
    keep = [n for n, _ in sorted(degs.items(), key=lambda x: x[1], reverse=True)[:args.max_nodes]]
    G = G.subgraph(keep).copy()

# ------------- Visual encodings -------------
deg = dict(G.degree())
max_deg = max(deg.values()) if deg else 1
# Node size: scaled by sqrt(degree) for smoother spread, and capped by max size
node_size = []
for n in G.nodes():
    d = deg.get(n, 0)
    sz = 64 * math.sqrt(d / max_deg)  # base size 64
    node_size.append(sz)


# Edge widths: log(1+weight) for dynamic range
def edge_w(e):
    w = G.edges[e].get("weight", 1.0)
    return 0.1 + math.log1p(w)
edge_widths = [edge_w(e) for e in G.edges()]
# normalize edge widths corresponding to node size
edge_widths_median = median(edge_widths)
edge_widths = [ ew / edge_widths_median for ew in edge_widths ]

# Try to color by communities if the subgraph is small enough
color_map = None
try:
    if G.number_of_nodes() <= 1200:  # keep it fast
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        # Map each node to an integer community id
        comm_id = {}
        for i, C in enumerate(comms):
            for n in C:
                comm_id[n] = i
        color_map = [comm_id.get(n, 0) for n in G.nodes()]
except Exception:
    color_map = None

# ------------- Layout -------------
# Spring layout is usually the cleanest general-purpose option.
if args.layout == "spring":
    k = 1.0 / max(1, G.number_of_nodes())**0.5
    pos = nx.spring_layout(G, seed=args.seed, k=k*1.8)
elif args.layout == "circular":
    pos = nx.circular_layout(G)
elif args.layout == "kamada_kawai":
    pos = nx.kamada_kawai_layout(G)
elif args.layout == "bipartite":
    pos = nx.bipartite_layout(G, nodes=[n for n in G.nodes() if G.degree(n) > 0])
elif args.layout == "shell":
    pos = nx.shell_layout(G)
elif args.layout == "spectral":
    pos = nx.spectral_layout(G)
elif args.layout == "random":
    pos = nx.random_layout(G)
elif args.layout == "planar":
    pos = nx.planar_layout(G)
elif args.layout == "rescale":
    pos = nx.rescale_layout(G)
elif args.layout == "spiral":
    pos = nx.spiral_layout(G)
elif args.layout == "multipartite":
    pos = nx.multipartite_layout(G)
else:
    pos = nx.spring_layout(G, seed=args.seed)  # default to spring layout


nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=color_map, alpha=0.9)
# ------------- Draw -------------
plt.figure(figsize=(10, 8), dpi=300)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35, edge_color="gray")
if color_map is None:
    nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.9)
else:
    # letting matplotlib pick default colors; no legend to avoid clutter
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=color_map, alpha=0.9)

# Highlight top-K strongest edges
TOP_EDGES = args.highlight_edges_top
sorted_edges = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 1), reverse=True)[:TOP_EDGES]
highlight_edges = [(u, v) for u, v, _ in sorted_edges]
highlight_edges_widths = [edge_w((u, v)) for u, v in highlight_edges]
highlight_edges_widths = [ ew / edge_widths_median for ew in highlight_edges_widths ]
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=highlight_edges,
    width=highlight_edges_widths,
    edge_color="red",
    alpha=0.8,
)
# Edge labels for highlighted edges
TOP_EDGE_LABELS = args.edge_labels_top
edge_labels = nx.get_edge_attributes(G, "label")
lbl_subset = {}
for u, v in highlight_edges[:TOP_EDGE_LABELS]:
    lab = edge_labels.get((u, v)) or edge_labels.get((v, u))
    if lab:
        lbl_subset[(u, v)] = lab
if lbl_subset:
    nx.draw_networkx_edge_labels(G, pos, edge_labels=lbl_subset, font_size=7, rotate=False, font_color="blue",
                                 bbox=dict(facecolor="yellow", alpha=0.5, boxstyle="round,pad=0.1"))


# Labels only for top-K hubs
label_top = max(0, int(args.label_top))
if label_top > 0:
    topK = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:label_top]
    lbls = {n: n for n, _ in topK}

    # radial nudge away from graph centroid
    xy = np.array(list(pos.values()))
    ctr = xy.mean(axis=0)
    pos_lbl = {}
    for n in lbls:
        v = np.array(pos[n]) - ctr
        nv = v / (np.linalg.norm(v) + 1e-9)
        pos_lbl[n] = (pos[n][0] + 0.02 * nv[0], pos[n][1] + 0.02 * nv[1])

    text_items = nx.draw_networkx_labels(
        G, pos_lbl, labels=lbls, font_size=6, font_color="black",
        bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.2")
    )
    for t in text_items.values():
        t.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])

plt.title(
    f"Knowledge Graph (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, min_w={args.min_weight}, "
    f"{'k-core='+str(args.k_core)+', ' if args.k_core else ''}"
    f"{'largest only' if args.largest_only else 'all comps'})"
)
plt.axis("off")
plt.tight_layout()

out_png = Path(args.out)
plt.savefig(out_png)
if args.svg:
    plt.savefig(Path(args.svg))
plt.close()

print(f"Saved: {out_png}" + (f" and {args.svg}" if args.svg else ""))

# ------------ Relation statistics ------------
if rel_counts:
    # Sort relations by total weight descending
    import collections
    total = sum(rel_counts.values()) or 1
    print("Relation distribution (top 10 by total weight):")
    for rel, cnt in rel_counts.most_common(10):
        perc = cnt / total * 100.0
        print(f"  {rel!r}: {cnt:.0f} ({perc:.1f}%)")
