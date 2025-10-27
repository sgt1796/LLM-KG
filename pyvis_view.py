#!/usr/bin/env python3
import argparse, json, math, ast
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import numpy as np
import networkx as nx
from pyvis.network import Network

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="Interactive KG viewer (pyvis)")
ap.add_argument("--input", default="graph.json", help="Path to graph JSON/JSONL")
ap.add_argument("--html", default="graph_viz.html", help="Output HTML")
ap.add_argument("--weight", type=str, default=">=2.0", 
                help="Filter edges by weight. Examples: '>n', '>=n', '<n', '<=n', '=n', or 'n-m' for a range.")
ap.add_argument("--largest-only", action="store_true", help="Keep only largest connected component")
ap.add_argument("--k-core", type=int, default=0, help="If >0, take k-core")
ap.add_argument("--max-nodes", type=int, default=300, help="Cap node count by degree")
ap.add_argument("--max-edges", type=int, default=300, help="Cap edge count by weight, after node filtering")
ap.add_argument("--label-top", type=int, default=20, help="Label top-K nodes by degree")
ap.add_argument("--physics", choices=["barnesHut","forceAtlas2Based","repulsion","hierarchicalRepulsion","forceAtlas2","false","random"],
                default="barnesHut", help="pyvis physics solver")
ap.add_argument("--height", default="1000px")
ap.add_argument("--width", default="100%")
args = ap.parse_args()

# ---------- Loader (supports nodes/edges, triples, JSONL) ----------
def load_graph(p: Path):
    rel_counts = Counter()
    rel_bag = defaultdict(Counter)          # (u,v) -> Counter(rel -> w)
    edge_sources = defaultdict(set)          # (u,v) -> {source,...}
    node_sources = defaultdict(set)          # node -> {source,...}
    if p.suffix.lower() == ".jsonl":
        nodes, edges = {}, []
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            h = rec.get("h") or rec.get("subject")
            t = rec.get("t") or rec.get("object")
            r = rec.get("r") or rec.get("relation", "")
            w = rec.get("weight", 1)
            srcs = rec.get("sources", []) or []
            if h is None or t is None: continue
            u, v = str(h), str(t)
            try: w = float(w)
            except Exception: w = 1.0
            edges.append((u, v, w))
            nodes[u] = {}; nodes[v] = {}
            if r:
                rel_counts[r] += w
                rel_bag[(u, v)][r] += w
            if srcs:
                for s in srcs:
                    edge_sources[(u, v)].add(str(s))
                    node_sources[u].add(str(s)); node_sources[v].add(str(s))
        node_list = [{"id": n, **attrs} for n, attrs in nodes.items()]
        return node_list, edges, rel_counts, rel_bag, edge_sources, node_sources

    data = json.load(open(p, encoding="utf-8"))

    # Triples format
    if isinstance(data, dict) and "triples" in data:
        nodes, edges = {}, []
        for tri in data.get("triples", []):
            h = tri.get("h") or tri.get("subject")
            r = tri.get("r") or tri.get("relation", "")
            t = tri.get("t") or tri.get("object")
            w = tri.get("weight", 1)
            srcs = tri.get("sources", []) or []
            if h is None or t is None: continue
            u, v = str(h), str(t)
            try: w = float(w)
            except Exception: w = 1.0
            edges.append((u, v, w))
            nodes[u] = {}; nodes[v] = {}
            if r:
                rel_counts[r] += w
                rel_bag[(u, v)][r] += w
            if srcs:
                for s in srcs:
                    edge_sources[(u, v)].add(str(s))
                    node_sources[u].add(str(s)); node_sources[v].add(str(s))
        node_list = [{"id": n, **attrs} for n, attrs in nodes.items()]
        return node_list, edges, rel_counts, rel_bag, edge_sources, node_sources

    # D3-style / list-of-edges
    nodes, edges_raw = [], []
    if isinstance(data, dict):
        nodes = data.get("nodes", data.get("Vertices", []))
        edges_raw = data.get("edges", data.get("links", data.get("Edges", [])))
    elif isinstance(data, list):
        edges_raw = data
    else:
        raise ValueError("Unsupported JSON schema")

    def _nid(n):
        if isinstance(n, dict): return str(n.get("id", n.get("name", n.get("label"))))
        return str(n)

    node_ids = set()
    for n in nodes:
        try: node_ids.add(_nid(n))
        except Exception: pass

    norm_edges = []
    for e in edges_raw:
        if isinstance(e, dict):
            u = e.get("source", e.get("from", e.get("u")))
            v = e.get("target", e.get("to", e.get("v")))
            w = e.get("weight", 1.0)
            if u is None or v is None: continue
            try: w = float(w)
            except Exception: w = 1.0
            norm_edges.append((str(u), str(v), w))
            node_ids.update([str(u), str(v)])
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            u, v = str(e[0]), str(e[1])
            try: w = float(e[2]) if len(e) >= 3 else 1.0
            except Exception: w = 1.0
            norm_edges.append((u, v, w))
            node_ids.update([u, v])

    if not nodes:
        nodes = [{"id": n} for n in node_ids]
    return nodes, norm_edges, rel_counts, rel_bag, edge_sources, node_sources

# ---------- Build + filter ----------
in_path = Path(args.input)
if not in_path.exists():
    raise SystemExit(f"Missing input: {in_path}")

nodes, edges, rel_counts, rel_bag, edge_sources, node_sources = load_graph(in_path)

G = nx.Graph()
for n in nodes:
    if isinstance(n, dict):
        nid = str(n.get("id", n.get("name", n.get("label"))))
        if nid is None: continue
        attrs = {k: v for k, v in n.items() if k not in ("id","name","label")}
        G.add_node(nid, **attrs)
    else:
        G.add_node(str(n))

# Replace min-weight and max-weight with a single weight parameter
import re
weight_arg = args.weight.strip()
range_match = re.match(r"^(\d+(\.\d+)?)-(\d+(\.\d+)?)$", weight_arg)

if range_match:
    weight_filter = lambda w: float(range_match.group(1)) <= w <= float(range_match.group(3))
elif weight_arg.startswith(">="):
    threshold = float(weight_arg[2:])
    weight_filter = lambda w: w >= threshold
elif weight_arg.startswith(">"):
    threshold = float(weight_arg[1:])
    weight_filter = lambda w: w > threshold
elif weight_arg.startswith("<="):
    threshold = float(weight_arg[2:])
    weight_filter = lambda w: w <= threshold
elif weight_arg.startswith("<"):
    threshold = float(weight_arg[1:])
    weight_filter = lambda w: w < threshold
elif weight_arg.startswith("="):
    threshold = float(weight_arg[1:])
    weight_filter = lambda w: w == threshold
else:
    raise ValueError("Invalid weight filter. Use '>n', '>=n', '<n', '<=n', '=n', or 'n-m' for a range.")

# Filter edges based on the weight condition
filtered_edges = []
for u, v, w in edges:
    if weight_filter(float(w)):
        # keep the strongest relation label for hover title
        bag = rel_bag.get((u, v)) or rel_bag.get((v, u))
        label = bag.most_common(1)[0][0] if bag else ""
        filtered_edges.append((u, v, float(w), label))

# Apply max-edges limit
if args.max_edges > 0 and len(filtered_edges) > args.max_edges:
    filtered_edges = sorted(filtered_edges, key=lambda x: x[2], reverse=True)[:args.max_edges]

def _parse_source(x):
    """Return either a dict (rich meta) or a plain string. Never raises."""
    # already structured
    if isinstance(x, dict):
        return x
    if isinstance(x, (list, tuple, set)):
        # take first element if someone nested it accidentally
        x = next(iter(x), "") if x else ""
    # stringify
    s = str(x).strip()
    if not s:
        return ""
    # Handle double-encoded JSON string (e.g., a JSON blob stored as a quoted string)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        try:
            inner = json.loads(s)
            if isinstance(inner, str) and inner.startswith("{") and inner.endswith("}"):
                try:
                    return json.loads(inner)
                except Exception:
                    pass
            if isinstance(inner, str):
                s = inner
        except Exception:
            pass
    # Try strict JSON first
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            # Try Python-literal (handles single quotes, None, True/False)
            try:
                obj = ast.literal_eval(s)
                return obj if isinstance(obj, dict) else s
            except Exception:
                return s
    return s

def _clip(v, lim=100):
        # Clip a string to a maximum length, adding "..." if truncated.
        v = str(v)
        return v if len(v) <= lim else v[: lim - 3] + "..."

def _fmt_one_source(s) -> str:
    src = _parse_source(s)
    if isinstance(src, dict):
        fn   = src.get("doc_meta", {}).get("filename", "")
        did  = src.get("doc_id")
        sid  = src.get("sentence_id")
        span = src.get("char_span")
        conf = src.get("confidence")
        ev   = (src.get("evidence") or "")

        # normalize evidence to a single logical line
        ev = " ".join(ev.split())

        lines = []
        if fn:   lines.append(f"[file]: {_clip(fn)}")
        if did:  lines.append(f"[doc_id]: {_clip(did)}")
        if sid is not None:  lines.append(f"[sentence_id]: {_clip(sid)}")
        if span is not None: lines.append(f"[char_span]: {_clip(span)}")
        if conf is not None: lines.append(f"[confidence]: {_clip(conf)}")
        if ev:
            lines.append("[evidence]:")
            lines.append(f"    {_clip(ev)}")

        return "\n".join(lines) if lines else "<source>"

    return _clip(str(src))

def _normalize_sources(srcs):
    """Flatten containers into a simple list (dicts or strings)."""
    if not srcs:
        return []
    if isinstance(srcs, (list, tuple, set)):
        out = []
        for x in srcs:
            if isinstance(x, (list, tuple, set)):
                out.extend(list(x))
            else:
                out.append(x)
        return out
    return [srcs]

def _fmt_sources(sources, limit=5) -> str:
    """Group multiple sources by (doc_id, filename) to reduce repetition."""
    if not sources:
        return ""
    grouped = {}
    for s in sources:
        src = _parse_source(s)
        if not isinstance(src, dict):
            key = ("<unknown>", "<unknown>")
            grouped.setdefault(key, []).append(str(src))
            continue
        fn = src.get("doc_meta", {}).get("filename", "")
        did = src.get("doc_id", "<unknown>")
        key = (did, fn)
        grouped.setdefault(key, []).append(src)

    lines = []
    for (did, fn), group in list(grouped.items())[:limit]:
        lines.append(f"[doc_id]: {did}  \n[file]: {fn}")
        for g in group:
            ev = g.get("evidence") or ""
            if ev:
                ev = " ".join(ev.split())
                ev = ev[:97] + "..." if len(ev) > 100 else ev
                lines.append(f"> [evidence]: {ev}")

    return "\n".join(lines)

# Add edges to the graph (with provenance in hover)
for u, v, w, label in filtered_edges:
    srcs = edge_sources.get((u, v)) or edge_sources.get((v, u)) or set()
    src_txt = _fmt_sources(srcs, limit=6)
    # Store sources on the edge so later passes can see them
    G.add_edge(
        u, v,
        weight=float(w),
        label=label,
        sources=list(srcs),                 # <-- keep raw list
        title=(f"{label or 'related_to'} | w={w}"
               + (f"\n\n—— sources ——\n{src_txt}" if src_txt else ""))
    )

# prune isolates
G.remove_nodes_from(list(nx.isolates(G)))
if G.number_of_nodes() == 0:
    raise SystemExit("Graph is empty after filtering. Lower --min-weight or raise--max-weight.")

# largest CC
if args.largest_only and nx.number_connected_components(G) > 1:
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

# k-core
if args.k_core and args.k_core > 0:
    try:
        H = nx.k_core(G, k=args.k_core)
        if H.number_of_nodes() > 0:
            G = H
    except Exception:
        pass

# cap by degree
if G.number_of_nodes() > args.max_nodes:
    degs = dict(G.degree())
    keep = [n for n,_ in sorted(degs.items(), key=lambda x: x[1], reverse=True)[:args.max_nodes]]
    G = G.subgraph(keep).copy()

# ----- Node visuals (size by sqrt(degree), labels for top-K by degree) -----
deg = dict(G.degree())
max_deg = max(deg.values()) if deg else 1
labels = set([n for n,_ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:max(0,args.label_top)]])

# community -> group for pyvis coloring
try:
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(G)) if G.number_of_nodes() <= 5000 else []
    comm_map = {}
    for i, c in enumerate(comms):
        for n in c: comm_map[n] = i
except Exception:
    comm_map = {}

for n in G.nodes():
    d = deg.get(n, 0)
    size = 10 + 25 * math.sqrt(d / max_deg)  # 10..35-ish
    G.nodes[n]["value"] = size  # affects node size
    src_txt = _fmt_sources(node_sources.get(n), limit=6)
    G.nodes[n]["title"] = (f"{n} | deg={d}" + (f"\n\n—— sources ——\n{src_txt}" if src_txt else ""))
    G.nodes[n]["label"] = n if n in labels else ""  # label only top-K
    if n in comm_map:
        G.nodes[n]["group"] = comm_map[n]

# ----- Edge widths from weight -----
def w2width(w):
    # gentle log scale
    import math
    return 1 + math.log1p(w)


for u, v, data in G.edges(data=True):
    data["width"] = w2width(data.get("weight", 1.0))



# ----- Render with pyvis -----
# Build pyvis network
net = Network(height=args.height, width=args.width, directed=False, notebook=False)
net.barnes_hut()  # default; can be overridden below

if args.physics == "false":
    opts = {"physics": {"enabled": False}}
else:
    # build a valid JSON object, then dump to string
    opts = {
        "physics": {
            "solver": args.physics,
            "stabilization": {"iterations": 150},
            "barnesHut": {
                "gravitationalConstant": -8000,
                "springLength": 120,
                "springConstant": 0.02
            }
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "multiselect": True,
            "dragNodes": True
        }
    }

net.set_options(json.dumps(opts))

net.from_nx(G, edge_scaling=True)
# Avoid notebook template path issues by writing HTML directly
net.write_html(args.html, open_browser=False)  # or: net.save_graph(args.html)
print(f"Saved: {args.html}  | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
