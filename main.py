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
    DocumentChunker,
)
from kg_pipeline.provenance import compute_doc_id, DocContext

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
    parser.add_argument("--ner", type=str, choices=["simple", "spacy", "openai", "ollama"], default="simple",
                        help="Named Entity Recognition method to use")
    parser.add_argument(
                        "--chunking",
                        type=str,
                        choices=["none", "sections", "abstract_discussion"],
                        default="none",
                        help="Document chunking method to use before NER."
                    )
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

    chunker = DocumentChunker()

    # Choose your NER implementation.  Use the simple regex extractor by default
    # because spaCy may not be available in the runtime environment.  To
    # switch to spaCy-based extraction, replace NERExtractor() with
    # SpacyNER().
    ner = None
    if args.ner == "simple":
        ner = NERExtractor()
    elif args.ner == "spacy":
        ner = SpacyNER()
    elif args.ner == "openai":
        ner = LLMNER(client="openai", model="gpt-5-nano", temperature=1) # gpt 5 doesn't support temperature 0
    elif args.ner == "ollama":
        ner = LLMNER(client="ollama", model="minstral", temperature=0.0)

    interrupted = False

    for i, pdf_path in enumerate(pdf_paths, start=1):
        try:
            doc_id = compute_doc_id(pdf_path)
            da = DataAcquisition(pdf_path)
            doc_meta = {"filename": Path(pdf_path).name}

            print(f"[{i}/{len(pdf_paths)}] Reading PDF: {pdf_path} ...")
            text = da.read()
            print(f"    Extracted {len(text)} characters.")

            if args.chunking == "sections":
                chunks = chunker.chunk_by_sections(text)
            elif args.chunking == "abstract_discussion":
                chunks = chunker.chunk_abstract_discussion(text)
            else:
                chunks = chunker.no_chunk(text)

            for j, chunk_text in enumerate(chunks, start=1):
                try:
                    print(f"    Processing chunk {j}/{len(chunks)} with {len(chunk_text)} characters.")
                    if args.summary:
                        summary = DocumentDigestor(max_sentences=1).digest(chunk_text)
                        print("    Summary:")
                        print("    " + "\n    ".join(summary.splitlines()))

                    if isinstance(ner, NERExtractor):
                        sentence_entities = ner.extract(chunk_text)
                    else:
                        sentence_entities = ner.extract(chunk_text, mode="sentences")  # type: ignore[attr-defined]

                    print(f"    Found {len(sentence_entities)} sentence(s) with entities.")
                    ctx = DocContext(doc_id=doc_id, doc_meta=doc_meta, chunk_id=j-1, page_hint=None)
                    kg_builder.build_from_sentences(sentence_entities, context=ctx, start_sentence_id=0)  # type: ignore[arg-type]

                except KeyboardInterrupt:
                    print("    [INFO] Interrupted by user. Saving progress.", file=sys.stderr)
                    interrupted = True
                    break  # break chunk loop

            if interrupted:
                break  # break file loop

        except Exception as e:
            # Normal per-file errors (KeyboardInterrupt won’t be caught here)
            print(f"    [WARN] Skipping {pdf_path} due to error: {e}", file=sys.stderr)
            continue

    # ---- Save whatever we have ----
    graph_dict = kg_builder.to_dict()
    print(f"\nConstructed combined graph with {len(graph_dict['nodes'])} nodes and {len(graph_dict.get('triples', []))} triples.")

    if merge_path and merge_path.exists():
        print(f"Merging combined graph with existing graph: {merge_path}")
        merger = TripletGraphMerger(base_graph=TripletGraphMerger.load_json(merge_path))
        merger.merge(graph_dict)
        final_graph = merger.graph
    else:
        final_graph = graph_dict

    # Save
    if interrupted:
        out_path = out_path.with_name(out_path.stem + "_partial" + out_path.suffix)
        print(f"[INFO] Interrupted run; saving to {out_path}")
    TripletGraphMerger.save_json(final_graph, out_path)
    print(f"Graph saved to {out_path}")


if __name__ == "__main__":
    main()
