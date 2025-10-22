#!/usr/bin/env python3
"""
Clean + merge a huge D3-style graph:
- nodes: ["name", ...] or [{"id"/"name"/"label": ...}, ...]
- edges: {"source": <idx|str|obj>, "target": <idx|str|obj>, "weight": int?}

Outputs JSONL with merged, normalized edges:
{"h": "...", "r": "link", "t": "...", "weight": W, "sources": [...]}

Designed to stream 4GB+ without loading all edges into RAM at once (counts only).
"""

import argparse, sys, unicodedata, collections, os, json, math
from pathlib import Path
import regex as re
import ijson, orjson

"""
This script cleans and merges knowledge graphs that may be encoded either
as simple D3‑style co‑occurrence networks (``nodes``/``edges``) or as
lists of subject–relation–object triples (``triples``).  The original
implementation assumed undirected ``source``/``target`` edges.  This
version adds support for triple graphs and gracefully degrades when
heavy NLP dependencies like spaCy are unavailable.  It normalises
entity names using Unicode cleanup and simple punctuation removal, and
aggregates identical triples, summing their weights.  When spaCy is
installed, it will be used for lightweight tokenisation and lemmatisation,
otherwise a fallback normaliser is applied.
"""

try:
    import spacy  # optional dependency
except Exception:
    spacy = None

# ---------- Config ----------
STOP_TOK = {
    "the","a","an","of","and","to","for","in","on","with","by","at",
    "as","from","that","which","be","is","are","was","were","or"
}
# keep hyphen, underscore, slash which are useful in biomedical names
PUNC_RE  = re.compile(r"[^\p{L}\p{N}\s\-_/]")
SPACE_RE = re.compile(r"\s+")
CTRL_RE  = re.compile(r"[\p{C}]")
ONLY_PUNC_RE = re.compile(r"^[\p{P}\p{S}\s]+$")
QUOTE_FIX = { "\u2018":"'", "\u2019":"'", "\u201C":'"', "\u201D":'"' }

def norm_relation(s: str) -> str:
    s = base_cleanup(s)
    if not s:
        return ""
    # drop stop words similar to entities
    toks = [t for t in s.split() if t not in STOP_TOK]
    s = " ".join(toks).strip()
    # cheap guards: get rid of ultra-short or purely function-word relations
    if not s or len(s) <= 2:
        return ""
    return s

def build_nlp(model: str):
    """
    Build a lightweight spaCy pipeline if the package is available.  If
    spaCy is not installed, this returns ``None`` and downstream
    normalisation will fall back to simple string cleanup.  We disable
    the NER, parser and text categoriser to keep the overhead low.
    """
    if spacy is None:
        return None
    try:
        return spacy.load(model, disable=["ner", "parser", "textcat"])
    except Exception:
        return None

def fix_quotes(s: str) -> str:
    return "".join(QUOTE_FIX.get(ch, ch) for ch in s)

def base_cleanup(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = fix_quotes(s).lower()
    # strip control chars
    s = CTRL_RE.sub(" ", s)
    # drop non-word punc (keep -, _, / which are important in bio)
    s = PUNC_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s

def spacy_norm_entity(nlp, text: str) -> str:
    """
    Normalise an entity using spaCy tokenisation when available.  It
    removes stop words, collapses repeated punctuation and strips
    extraneous characters.  If spaCy is not provided, this falls back
    to the basic cleanup.
    """
    s = base_cleanup(text)
    if not s or ONLY_PUNC_RE.match(s):
        return ""
    if nlp is None or (" " not in s and len(s) <= 3):
        return s
    doc = nlp.make_doc(s)
    toks = [t.text for t in doc if t.text not in STOP_TOK]
    s = " ".join(toks).strip()
    s = re.sub(r"[-_]{2,}", "-", s)
    s = SPACE_RE.sub(" ", s).strip(" -_")
    if not s or ONLY_PUNC_RE.match(s):
        return ""
    return s

def resolve_node_name(raw, node_index_to_name, name_normalizer):
    """
    Resolve a raw node identifier into a normalised entity name.  The
    input can be an integer index into a ``nodes`` array, a numeric
    string, a dict with common identifier keys, or a plain string.
    Returns a normalised string or the empty string if resolution fails.
    """
    if isinstance(raw, int):
        name = node_index_to_name.get(raw)
        return name_normalizer(name) if name else ""
    if isinstance(raw, str) and raw.isdigit():
        idx = int(raw)
        name = node_index_to_name.get(idx)
        if name:
            return name_normalizer(name)
    if isinstance(raw, dict):
        for k in ["name", "label", "id", "title", "text"]:
            if k in raw and raw[k]:
                return name_normalizer(raw[k])
        return name_normalizer(json.dumps(raw, ensure_ascii=False))
    if isinstance(raw, str):
        return name_normalizer(raw)
    return ""

def sniff_nodes_array_path(json_path):
    """
    Return an ijson path like ``nodes.item`` if a ``nodes`` array is
    present in the JSON document.  This function scans only shallow
    prefixes to avoid expensive parsing of large files.
    """
    with open(json_path, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if event == "start_array" and prefix.lower().endswith("nodes"):
                return f"{prefix}.item"
            if prefix.count(".") > 3:
                break
    return None

def iter_all_objects_with_keys(json_path, required_lower_keys):
    """
    Stream every object in the JSON document (anywhere) that contains
    all required keys, case‑insensitively.  This utility allows us to
    find edge or triple records even if they are nested in unusual
    locations.  It first attempts to iterate over items at the root
    level, and falls back to a generic streaming parse if necessary.
    """
    required_lower_keys = set(required_lower_keys)
    with open(json_path, "rb") as f:
        for obj in ijson.items(f, "item"):
            if isinstance(obj, dict):
                low = {k.lower(): v for k, v in obj.items()}
                if required_lower_keys.issubset(low.keys()):
                    yield obj
    with open(json_path, "rb") as f:
        parser = ijson.parse(f)
        stack = []
        cur_key = None
        for prefix, event, value in parser:
            if event == "start_map":
                stack.append({})
            elif event == "map_key":
                cur_key = value
            elif event in ("string", "number", "boolean", "null"):
                if stack and cur_key is not None:
                    stack[-1][cur_key] = value
            elif event == "end_map":
                obj = stack.pop()
                low = {k.lower(): v for k, v in obj.items()}
                if required_lower_keys.issubset(low.keys()):
                    yield obj
                if stack:
                    pass

def detect_triple_graph(input_path: str) -> bool:
    """
    Quickly determine whether the input JSON contains a top‑level
    ``triples`` array.  Returns ``True`` if a ``triples`` array is
    detected; otherwise returns ``False`` so that we fall back to
    handling traditional D3 ``edges``.
    """
    with open(input_path, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if event == "start_array" and prefix.lower().endswith("triples"):
                return True
            if prefix.count(".") > 3:
                break
    return False

def aggregate_triple_graph(
    input_path: str,
    output_jsonl: str,
    spacy_model: str,
    min_support: int = 2,
    verbose_every: int = 500000,
    write_stats: bool = True,
):
    """
    Aggregate a graph of subject–relation–object triples.  This reads
    each triple, normalises the subject and object names, and sums
    weights for identical triples.  It supports streaming large files
    using the same object‑with‑required‑keys detection as for edges.

    Parameters
    ----------
    input_path : str
        Path to the input JSON containing a ``triples`` array or any
        objects with keys ``subject`` and ``object``.
    output_jsonl : str
        Path to write the deduplicated triples in JSONL format.
    spacy_model : str
        Name of the spaCy model to use for normalisation, if available.
    min_support : int
        Minimum weight threshold for retaining a triple in the output.
    verbose_every : int
        How often to emit progress messages based on the number of
        triples processed.
    write_stats : bool
        Whether to write a companion ``.stats.json`` file with
        counters and related statistics.
    """
    nlp = build_nlp(spacy_model)
    ent_cache = {}
    def norm_ent(x: str) -> str:
        if x in ent_cache:
            return ent_cache[x]
        y = spacy_norm_entity(nlp, x) if x is not None else ""
        ent_cache[x] = y
        return y

    stats = collections.Counter()
    counts = collections.Counter()

    required_keys = {"subject", "object"}
    triple_iter = iter_all_objects_with_keys(input_path, required_keys)
    processed = 0
    for obj in triple_iter:
        raw_s = next((obj.get(k) for k in obj.keys() if k.lower() == "subject"), None)
        raw_t = next((obj.get(k) for k in obj.keys() if k.lower() == "object"), None)
        raw_r = next((obj.get(k) for k in obj.keys() if k.lower() == "relation"), None)
        raw_w = obj.get("weight", 1)
        h = norm_ent(raw_s)
        t = norm_ent(raw_t)
        r = norm_relation(raw_r) if raw_r is not None else ""
        if not r:
            r = "related_to"
        if not h or not t:
            stats["triple_dropped_unresolvable"] += 1
            continue
        try:
            w = int(raw_w) if raw_w is not None else 1
        except Exception:
            w = 1
        counts[(h, r, t)] += w
        stats["triples_seen"] += 1
        if r == "related_to":
            stats["rel_related_to"] += 1
        else:
            stats["rel_other"] += 1
        processed += 1
        if verbose_every and processed % verbose_every == 0:
            print(f"[pass] triples_seen={processed:,} unique_triples={len(counts):,}", file=sys.stderr)

    # Determine output format based on file extension: JSONL vs JSON
    output_is_jsonl = str(output_jsonl).lower().endswith(".jsonl")
    final_kept = 0
    if output_is_jsonl:
        # Write deduplicated triples as JSON Lines
        with open(output_jsonl, "wb") as out:
            for (h, r, t), w in counts.items():
                if w < min_support:
                    stats["filtered_min_support"] += 1
                    continue
                rec = {"h": h, "r": r, "t": t, "weight": int(w), "sources": []}
                out.write(orjson.dumps(rec) + b"\n")
                final_kept += 1
        stats["final_kept"] = final_kept
        stats["unique_triples_after_agg"] = len(counts)
        if write_stats:
            with open(Path(output_jsonl).with_suffix(".stats.json"), "w", encoding="utf-8") as s:
                json.dump(stats, s, ensure_ascii=False, indent=2)
    else:
        # Build a graph dictionary with nodes and triples arrays
        triples_out = []
        nodes_set = set()
        for (h, r, t), w in counts.items():
            if w < min_support:
                stats["filtered_min_support"] += 1
                continue
            triples_out.append({"subject": h, "relation": r, "object": t, "weight": int(w)})
            nodes_set.add(h)
            nodes_set.add(t)
            final_kept += 1
        graph_out = {"nodes": sorted(nodes_set), "triples": triples_out}
        stats["final_kept"] = final_kept
        stats["unique_triples_after_agg"] = len(counts)
        # Write JSON file
        with open(output_jsonl, "w", encoding="utf-8") as f:
            json.dump(graph_out, f, ensure_ascii=False, indent=2)
        if write_stats:
            with open(Path(output_jsonl).with_suffix(".stats.json"), "w", encoding="utf-8") as s:
                json.dump(stats, s, ensure_ascii=False, indent=2)

def aggregate_d3_graph(
    input_path: str,
    output_jsonl: str,
    spacy_model: str,
    min_support: int = 2,
    max_sources: int = 16,
    verbose_every: int = 500000,
    write_stats: bool = True,
):
    """
    Aggregate a traditional D3‑style co‑occurrence graph.  Nodes may be
    provided in a ``nodes`` array, and edges as objects with ``source``
    and ``target`` fields.  This function normalises node names and
    collapses duplicate edges, summing their weights.  It retains
    compatibility with the original script while sharing common
    utilities with the triple aggregator.
    """
    # Determine output format based on extension: .jsonl => JSON Lines, otherwise JSON
    output_is_jsonl = str(output_jsonl).lower().endswith(".jsonl")

    nlp = build_nlp(spacy_model)
    ent_cache = {}
    def norm_ent(x: str) -> str:
        if x in ent_cache:
            return ent_cache[x]
        y = spacy_norm_entity(nlp, x) if x is not None else ""
        ent_cache[x] = y
        return y

    stats = collections.Counter()
    node_index_to_name = {}

    nodes_path = sniff_nodes_array_path(input_path)
    if nodes_path:
        with open(input_path, "rb") as f:
            for i, node in enumerate(ijson.items(f, nodes_path), 0):
                if isinstance(node, str):
                    node_index_to_name[i] = node
                elif isinstance(node, dict):
                    name = None
                    for k in ["name", "label", "id", "title", "text"]:
                        if k in node and node[k]:
                            name = node[k]; break
                    if name is None:
                        name = json.dumps(node, ensure_ascii=False)
                    node_index_to_name[i] = name
                else:
                    node_index_to_name[i] = str(node)
                stats["nodes_seen"] += 1

    sid_keys = {"source_id", "doc_id", "sid", "source_doc", "paper"}
    Edge = collections.namedtuple("E", "h r t")
    counts = collections.Counter()
    sources = {}

    required_edge_keys = {"source", "target"}
    seen_any_edges = False

    with open(input_path, "rb") as f:
        parser = ijson.parse(f)
        cur = None; key = None
        for prefix, event, value in parser:
            if event == "start_map":
                cur = {}
            elif event == "map_key":
                key = value
            elif event in ("string", "number", "boolean", "null"):
                if cur is not None and key is not None:
                    cur[key] = value
            elif event == "end_map":
                if cur:
                    low = {k.lower(): v for k, v in cur.items()}
                    if required_edge_keys.issubset(low.keys()):
                        seen_any_edges = True
                        raw_s = next((cur.get(k) for k in cur.keys() if k.lower() == "source"), None)
                        raw_t = next((cur.get(k) for k in cur.keys() if k.lower() == "target"), None)
                        h = resolve_node_name(raw_s, node_index_to_name, norm_ent)
                        t = resolve_node_name(raw_t, node_index_to_name, norm_ent)
                        if not h or not t:
                            stats["edge_dropped_unresolvable"] += 1
                            cur = None
                            continue
                        r = "link"
                        e = Edge(h, r, t)
                        counts[e] += int(cur.get("weight", 1) or 1)
                        sid = None
                        for kk in cur.keys():
                            if kk.lower() in sid_keys:
                                sid = cur[kk]; break
                        if sid:
                            sset = sources.get(e)
                            if sset is None:
                                sset = set(); sources[e] = sset
                            if len(sset) < max_sources:
                                sset.add(str(sid))
                        stats["edges_seen"] += 1
                        if verbose_every and stats["edges_seen"] % verbose_every == 0:
                            print(f"[pass] edges_seen={stats['edges_seen']:,} unique_edges={len(counts):,}", file=sys.stderr)
                cur = None

    if not seen_any_edges:
        # If no edges found, write empty output depending on format
        if output_is_jsonl:
            open(output_jsonl, "wb").close()
        else:
            with open(output_jsonl, "w", encoding="utf-8") as f:
                json.dump({"nodes": [], "edges": []}, f, ensure_ascii=False, indent=2)
        if write_stats:
            with open(Path(output_jsonl).with_suffix(".stats.json"), "w", encoding="utf-8") as s:
                json.dump({"error": "no_edges_found", "hint": "Your edges may be under unusual keys; ensure objects have 'source' and 'target'."}, s, ensure_ascii=False, indent=2)
        return

    # Compute degree counts for filtering and output
    # We count degrees only for edges that meet min_support
    deg = collections.Counter()
    for e, w in counts.items():
        if w < min_support:
            continue
        deg[e.h] += 1; deg[e.t] += 1

    LEAF_MAX_W = max(2, min_support)
    final_kept = 0

    if output_is_jsonl:
        # Write JSONL using two-pass approach to handle leaf trimming
        tmp_path = Path(output_jsonl).with_suffix(".tmp.jsonl")
        with open(tmp_path, "wb") as out:
            for e, w in counts.items():
                if w < min_support:
                    stats["filtered_min_support"] += 1
                    continue
                srcs = sorted(list(sources.get(e, ())))[:max_sources]
                out.write(orjson.dumps({"h": e.h, "r": e.r, "t": e.t, "weight": int(w), "sources": srcs}) + b"\n")
        with open(tmp_path, "rb") as inp, open(output_jsonl, "wb") as out:
            for line in inp:
                rec = orjson.loads(line)
                if deg[rec["h"]] == 1 and deg[rec["t"]] == 1 and rec["weight"] <= LEAF_MAX_W:
                    stats["filtered_leaf_lowweight"] += 1
                    continue
                out.write(orjson.dumps(rec) + b"\n")
                final_kept += 1
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        stats["unique_edges_after_agg"] = len(counts)
        stats["final_kept"] = final_kept
        stats["nodes_indexed"] = len(node_index_to_name)
        if write_stats:
            with open(Path(output_jsonl).with_suffix(".stats.json"), "w", encoding="utf-8") as s:
                json.dump(stats, s, ensure_ascii=False, indent=2)
    else:
        # Build JSON graph with nodes and edges arrays
        edges_out = []
        nodes_set = set()
        for e, w in counts.items():
            if w < min_support:
                stats["filtered_min_support"] += 1
                continue
            if deg[e.h] == 1 and deg[e.t] == 1 and w <= LEAF_MAX_W:
                stats["filtered_leaf_lowweight"] += 1
                continue
            rec = {
                "source": e.h,
                "target": e.t,
                "weight": int(w),
                "sources": sorted(list(sources.get(e, ())))[:max_sources]
            }
            edges_out.append(rec)
            nodes_set.add(e.h)
            nodes_set.add(e.t)
            final_kept += 1
        graph_out = {"nodes": sorted(nodes_set), "edges": edges_out}
        stats["unique_edges_after_agg"] = len(counts)
        stats["final_kept"] = final_kept
        stats["nodes_indexed"] = len(node_index_to_name)
        # Write JSON graph
        with open(output_jsonl, "w", encoding="utf-8") as f:
            json.dump(graph_out, f, ensure_ascii=False, indent=2)
        if write_stats:
            with open(Path(output_jsonl).with_suffix(".stats.json"), "w", encoding="utf-8") as s:
                json.dump(stats, s, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the big D3 JSON")
    ap.add_argument("--output", required=True, help="Output path (.json for a graph dictionary, .jsonl for JSON Lines)")
    ap.add_argument("--spacy-model", default="en_core_web_sm",
                    help="e.g., en_core_web_sm | en_core_web_md | en_core_sci_sm")
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--verbose-every", type=int, default=500000)
    args = ap.parse_args()

    # Decide whether this is a triple graph or a D3 graph based on the presence of a top-level
    # "triples" array.  Triple graphs are aggregated differently.
    if detect_triple_graph(args.input):
        aggregate_triple_graph(
            input_path=args.input,
            output_jsonl=args.output,
            spacy_model=args.spacy_model,
            min_support=args.min_support,
            verbose_every=args.verbose_every,
        )
    else:
        aggregate_d3_graph(
            input_path=args.input,
            output_jsonl=args.output,
            spacy_model=args.spacy_model,
            min_support=args.min_support,
            verbose_every=args.verbose_every,
        )

if __name__ == "__main__":
    main()
