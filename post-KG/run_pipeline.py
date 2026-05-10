"""Run the post-KG hypothesis discovery and review pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_kg.miner import candidates_to_dicts, discover_hypotheses, load_graph, render_markdown
from post_kg.reviewer import render_review_markdown, review_candidates, reviewed_to_dicts


def _split_focus(values: list[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        terms.extend(part.strip() for part in value.split(",") if part.strip())
    return terms


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine and review post-KG hypothesis candidates.")
    parser.add_argument("--graph", required=True, help="Graph JSON produced by main.py")
    parser.add_argument("--focus", action="append", default=[], help="Focus term. Repeat or comma-separate terms.")
    parser.add_argument("--top", type=int, default=20, help="Number of raw hypotheses to mine")
    parser.add_argument("--out-dir", default="post-KG/outputs", help="Directory for reports and JSON")
    parser.add_argument("--prefix", default=None, help="Output filename prefix; defaults to graph stem + focus terms")
    parser.add_argument("--focus-mode", choices=["all", "any"], default="all")
    parser.add_argument("--focus-scope", choices=["path", "evidence"], default="path")
    parser.add_argument("--max-degree", type=int, default=80)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    graph_path = Path(args.graph)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_terms = _split_focus(args.focus)
    focus_slug = "_".join(term.replace(" ", "-") for term in focus_terms) or "all"
    prefix = args.prefix or f"{graph_path.stem}_{focus_slug}"

    graph = load_graph(graph_path)
    candidates = discover_hypotheses(
        graph,
        focus_terms=focus_terms,
        focus_mode=args.focus_mode,
        focus_scope=args.focus_scope,
        top_k=args.top,
        max_degree=args.max_degree,
        min_score=args.min_score,
    )
    candidate_dicts = candidates_to_dicts(candidates)
    reviewed = review_candidates(candidate_dicts)

    raw_md = out_dir / f"{prefix}_raw.md"
    raw_json = out_dir / f"{prefix}_raw.json"
    reviewed_md = out_dir / f"{prefix}_reviewed.md"
    reviewed_json = out_dir / f"{prefix}_reviewed.json"

    raw_md.write_text(render_markdown(candidates, graph_name=graph_path.name), encoding="utf-8")
    raw_json.write_text(json.dumps(candidate_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    reviewed_md.write_text(
        render_review_markdown(reviewed, title=f"Reviewed Hypotheses from {graph_path.name}"),
        encoding="utf-8",
    )
    reviewed_json.write_text(json.dumps(reviewed_to_dicts(reviewed), ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "raw_count": len(candidate_dicts),
                "reviewed_count": len(reviewed),
                "raw_markdown": str(raw_md),
                "raw_json": str(raw_json),
                "reviewed_markdown": str(reviewed_md),
                "reviewed_json": str(reviewed_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
