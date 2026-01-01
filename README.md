# LLM-KG: Biomedical Knowledge Graph Extraction + KG-RAG

This repo turns NCBI search results into a triple-based knowledge graph,
then adds optional deduplication, visualization, and a Flask KG-RAG demo.
It is designed for biomedical PDFs and supports both heuristic and LLM
entity extraction.

## What this project does

- Download PDFs from a CSV/TSV export that includes PMCID or DOI.
- Convert PDFs to text via Poppler `pdftotext`.
- Optionally chunk documents by sections or abstract/discussion.
- Extract entities via regex, spaCy, or LLM (OpenAI/Ollama via POP).
- Build subject-relation-object triples with canonicalized relations.
- Store evidence and provenance per triple.
- Merge graphs and optionally dedupe/normalize large runs.
- Visualize graphs with PyVis and query via a KG-RAG Flask app.

## End-to-end workflow

The example runner is `workflow_kg_extraction.py` (edit paths and
uncomment steps to match your dataset). Typical flow:

1) `fetch_ncbi.py` downloads PDFs from a CSV/TSV list.
2) `sample_papers.sh` optionally samples a smaller subset.
3) `main.py` builds a triple KG JSON from PDFs.
4) `dedupe.py` normalizes and merges large graphs (optional).
5) `pyvis_view.py` generates an interactive HTML view.
6) `kg_rag_app.py` serves a KG-RAG web UI with embeddings + query.

## Quickstart

Install dependencies and Poppler:

```bash
pip install -r requirements.txt
# Ubuntu/Debian:
sudo apt-get install poppler-utils
```

Download papers from an NCBI CSV/TSV export:

```bash
python fetch_ncbi.py --csv dataset1-ADHDMeSHTe-set.csv --out papers --resume
```

Extract a knowledge graph:

```bash
python main.py --pdf papers --output graph_llm.json --ner ollama --chunking sections
```

Visualize with PyVis:

```bash
python pyvis_view.py --input graph_llm.json --html graph_llm.html \
  --weight ">=0" --k-core 0 --max-nodes 500 --max-edges 600 \
  --label-top 20 --physics barnesHut --largest-only --directed --filter-menu
```

Run the KG-RAG Flask app:

```bash
python kg_rag_app.py --graph graph_llm.json --host 0.0.0.0 --port 5000
```

## Main extraction CLI

`main.py` accepts a single PDF or a directory:

```bash
python main.py --pdf papers --output graph.json \
  --ner simple|spacy|openai|ollama \
  --chunking none|sections|abstract_discussion
```

Notes:
- `--ner openai` and the KG-RAG app require `OPENAI_API_KEY`.
- `--ner ollama` requires a local Ollama server running.
- The dynamic relation label store lives at `kg_pipeline/.kg_cache/labels.json`.

## Graph format

Output is JSON with nodes and triples:

```json
{
  "nodes": ["Entity A", "Entity B"],
  "triples": [
    {
      "subject": "Entity A",
      "relation": "associated with",
      "object": "Entity B",
      "weight": 3,
      "sources": [
        {
          "doc_id": "abcd1234ef567890",
          "doc_meta": {"filename": "paper.pdf"},
          "chunk_id": 0,
          "sentence_id": 4,
          "page": null,
          "char_span": [12, 56],
          "evidence": "Entity A is associated with Entity B ...",
          "confidence": 1.0
        }
      ]
    }
  ]
}
```

## Project structure (core pieces)

```
.
├── fetch_ncbi.py              # Download PDFs from NCBI/DOI lists
├── main.py                    # KG extraction CLI
├── workflow_kg_extraction.py  # Example end-to-end runner
├── kg_pipeline/               # Chunking, NER, triple builder, provenance
├── dedupe.py                  # Normalize and merge large graphs
├── pyvis_view.py              # Interactive HTML visualization
├── kg_rag_app.py              # Flask KG-RAG demo app
├── Embedder.py                # Embedding wrapper (OpenAI/Jina/local)
└── llm_utils/                 # POP + LLM client adapters
```

## Notes on embeddings and RAG

`kg_rag_app.py` builds node embeddings using `Embedder.py` (OpenAI by
default). It caches embeddings under `.kg_cache/` and uses cosine
similarity to highlight the most relevant nodes for a query, with
optional LLM explanations.
