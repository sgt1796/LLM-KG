"""Simple usage example for SemanticSubgraphRetriever.

This script demonstrates how to instantiate the semantic subgraph
retriever and run a natural‑language query against a knowledge graph
stored in a JSON file.  The graph format is the same as that produced
by `main.py` in the LLM‑KG project: a dictionary containing a
`"triples"` key with a list of subject–relation–object entries.  Each
entry may optionally specify a weight and evidence sources.  The
retriever will build an in‑memory graph, compute node embeddings via
the supplied `Embedder` class, and return the top candidate nodes,
triples, and supporting paths.

Usage from the repository root:

    python query_tools/sample_usage.py --graph PATH_TO_GRAPH.json --question "Your query"

You must set the `OPENAI_API_KEY` environment variable if using the
default `Embedder` implementation, which calls OpenAI for embeddings.
Alternatively, modify the script to use a locally cached embedding
model.
"""

import argparse
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Embedder import Embedder  # type: ignore

from query_tools.semantic_subgraph_retriever import SemanticSubgraphRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a semantic subgraph query")
    parser.add_argument("--graph", required=True, help="Path to graph JSON file")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--hop-limit", type=int, default=2, help="Maximum hop radius for paths")
    parser.add_argument("--cache-dir", default=".kg_cache", help="Cache directory root")
    parser.add_argument("--semantic-threshold", type=float, default=0.05, help="Semantic node filter threshold")
    parser.add_argument("--structural-threshold", type=float, default=0.05, help="Structural node filter threshold")
    args = parser.parse_args()

    embedder = Embedder(use_api="openai", model_name="text-embedding-3-small")
    graph_path = Path(args.graph)
    retriever = SemanticSubgraphRetriever(
        graph_path,
        embedder,
        cache_dir=Path(args.cache_dir) / graph_path.stem,
        semantic_threshold=args.semantic_threshold,
        structural_threshold=args.structural_threshold,
    )
    result = retriever.query(args.question, top_k=args.top_k, hop_limit=args.hop_limit)
    pprint(result.focus_nodes)
    pprint(result.triples)
    pprint(result.paths)
    print("Context:\n" + result.context)


if __name__ == "__main__":
    main()
