import json, orjson, sys
inp, out = "edges_clean.jsonl", "graph.json"
edges = []
with open(inp, "rb") as f:
    for line in f:
        if not line.strip(): continue
        rec = orjson.loads(line)
        edges.append({
            "source": rec["h"],
            "target": rec["t"],
            "weight": rec.get("weight", 1)
        })
json.dump({"nodes": [], "edges": edges}, open(out, "w", encoding="utf-8"))
print(f"Wrote {len(edges):,} edges to {out}")