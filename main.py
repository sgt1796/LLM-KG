"""Command-line interface for the KG extraction bot.

This script can process either a single PDF or a directory of PDFs.
It aggregates entities from all PDFs into ONE knowledge graph, and can
optionally merge that graph with an existing graph before saving.

Usage
-----

python main.py --pdf path/to/document_or_dir --output graph.json [--merge existing_graph.json] [--summary]

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from kg_pipeline import (
    DataAcquisition,
    DocumentDigestor,
    NERExtractor,
    SpacyNER,
    LLMNER,
    WeightedTupleBuilder,
    WeightedTupleMerger,
    TripletKnowledgeGraphBuilder,
    TripletGraphMerger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a simple knowledge graph from PDF(s).")
    parser.add_argument("--pdf", type=str, required=True,
                        help="Path to a PDF file or a directory containing PDF files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output graph JSON")
    parser.add_argument("--merge", type=str, default=None,
                        help="Optional path to an existing graph JSON to merge with")
    parser.add_argument("--summary", action="store_true",
                        help="Print a short summary of each document")
    parser.add_argument("--ner", type=str, choices=["simple", "spacy", "llm"], default="simple",
                        help="Named Entity Recognition method to use")
    return parser.parse_args()


def collect_pdfs(path: Path) -> List[Path]:
    """Return a sorted list of PDF paths. Accepts a single file or a directory."""
    if path.is_dir():
        # Case-insensitive .pdf match; deterministic order
        pdfs = sorted([p for p in path.glob("**/*") if p.is_file() and p.suffix.lower() == ".pdf"])
    else:
        pdfs = [path] if path.suffix.lower() == ".pdf" else []
    return pdfs


def main() -> None:
    args = parse_args()
    pdf_root = Path(args.pdf)
    out_path = Path(args.output)
    merge_path: Optional[Path] = Path(args.merge) if args.merge else None

    pdf_paths = collect_pdfs(pdf_root)
    if not pdf_paths:
        print(f"[ERROR] No PDFs found at: {pdf_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(pdf_paths)} PDF(s). Building one combined graph...")

    # One global triplet builder to accumulate triples across ALL PDFs
    kg_builder = TripletKnowledgeGraphBuilder()

    # Choose your NER implementation.  Use the simple regex extractor by default
    # because spaCy may not be available in the runtime environment.  To
    # switch to spaCy-based extraction, replace NERExtractor() with
    # SpacyNER().
    ner = None
    if args.ner == "simple":
        ner = NERExtractor()
    elif args.ner == "spacy":
        ner = SpacyNER()
    elif args.ner == "llm":
        ner = LLMNER()

    for i, pdf_path in enumerate(pdf_paths, start=1):
        try:
            print(f"[{i}/{len(pdf_paths)}] Reading PDF: {pdf_path} ...")
            # Data acquisition
            da = DataAcquisition(pdf_path)
            text = da.read()
            print(f"    Extracted {len(text)} characters.")

            # Optional summary
            if args.summary:
                digestor = DocumentDigestor(max_sentences=5)
                summary = digestor.digest(text)
                print("    Summary:")
                print("    " + "\n    ".join(summary.splitlines()))

            # Entity extraction (returns iterable of (sentence, set(entities)))
            # When using the simple NER, call .extract(); for spaCy wrapper
            # call .extract(..., mode="sentences") for compatibility.
            # Extract entities grouped by sentence.  The simple heuristic
            # extractor exposes an ``extract`` method returning
            # ``List[Tuple[str, Set[str]]]``.  The spaCy wrapper accepts a
            # ``mode`` parameter to request sentence‑level output.
            if isinstance(ner, NERExtractor):
                sentence_entities = ner.extract(text)
            else:
                # type: ignore[attr-defined] – SpacyNER has an ``extract`` method
                sentence_entities = ner.extract(text, mode="sentences")  # type: ignore
                #print(f"    [DEBUG] Extracted sentence entities: {sentence_entities}")
            print(f"    Found {len(sentence_entities)} sentence(s) with entities.")

            # Accumulate into the global triplet builder
            kg_builder.build_from_sentences(sentence_entities)  # type: ignore[arg-type]

        except Exception as e:
            # Keep going even if one file fails
            print(f"    [WARN] Skipping {pdf_path} due to error: {e}", file=sys.stderr)
            continue

    # Build final graph dict from the global builder
    graph_dict = kg_builder.to_dict()
    print(f"\nConstructed combined graph with {len(graph_dict['nodes'])} nodes and {len(graph_dict['triples'])} triples.")

    # Merge if requested
    if merge_path and merge_path.exists():
        print(f"Merging combined graph with existing graph: {merge_path}")
        merger = TripletGraphMerger(base_graph=TripletGraphMerger.load_json(merge_path))
        merger.merge(graph_dict)
        final_graph = merger.graph
    else:
        final_graph = graph_dict

    # Save
    TripletGraphMerger.save_json(final_graph, out_path)
    print(f"Graph saved to {out_path}")


if __name__ == "__main__":
    main()
