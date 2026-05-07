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
- Visualize graphs with PyVis and query via a hybrid KG retriever + Flask app.

## Recent changes

- `LLMNER` now uses a two-stage pipeline for LLM extraction.
- Stage 1 verifies exact surface mentions per sentence only, with a grounding rule that each mention must appear verbatim in the source sentence.
- Stage 2 filters, types, and normalizes verified mentions with a compact biomedical schema: `DISEASE`, `DRUG`, `GENE_PROTEIN`, `PATHWAY`, `CELL_TYPE`, `MEASUREMENT`, `OTHER`.
- LLM prompts now use sentence IDs instead of asking the model to echo full sentence text, which reduces output tokens and malformed JSON risk.
- LLM runs are now sent in small sentence batches instead of whole section-sized payloads.
- Surface mentions are preserved for downstream triple extraction; canonical names are kept alongside them in structured LLM output.
- The configured LLM model is now forwarded correctly into `PromptFunction.execute`, and the Ollama default typo was corrected from `minstral` to `mistral:7b`.
- `kg_pipeline/rag.py` is now a reusable hybrid retriever instead of a placeholder.
- Hybrid retrieval now combines alias/entity linking, node-text embeddings, triple-text embeddings, relation-intent matching, reciprocal-rank fusion, and short path extraction.
- `kg_rag_app.py` now uses the retriever for `/query`, so app responses are grounded in ranked triples and paths rather than node-only cosine similarity.
- The retriever writes per-graph cache sidecars under `.kg_cache/<graph_stem>/` for node, triple, relation, alias, and optional KGE artifacts.
- Optional RotatE-style KGE support is implemented through `PyKEEN`; when `PyKEEN` is unavailable, retrieval cleanly falls back to text-only search.

## End-to-end workflow

The example runner is `workflow_kg_extraction.py` (edit paths and
uncomment steps to match your dataset). Typical flow:

1) `fetch_ncbi.py` downloads PDFs from a CSV/TSV list.
2) `sample_papers.sh` optionally samples a smaller subset.
3) `main.py` builds a triple KG JSON from PDFs.
4) `dedupe.py` normalizes and merges large graphs (optional).
5) `pyvis_view.py` generates an interactive HTML view.
6) `python -m kg_pipeline.rag build` creates cached retrieval sidecars for a graph.
7) `kg_rag_app.py` serves a KG-RAG web UI backed by the hybrid retriever.

## Quickstart

Install dependencies and Poppler:

```bash
pip install -r requirements.txt
# Ubuntu/Debian:
sudo apt-get install poppler-utils
```

Optional dependency for structural KG embeddings:

```bash
pip install pykeen
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

Build retriever caches from the command line:

```bash
python -m kg_pipeline.rag build --graph graph_llm.json --cache-dir .kg_cache
```

Query the retriever directly from the command line:

```bash
python -m kg_pipeline.rag query --graph graph_llm.json \
  --question "What causes ASD?" --top-k 10 --hop-limit 2
```

If you want to skip optional KGE training:

```bash
python -m kg_pipeline.rag build --graph graph_llm.json --disable-kge
```

## Docker Deployment

The repo now includes a production-oriented `Dockerfile`,
`docker-compose.yml`, `gunicorn.conf.py`, and `wsgi.py`.

1. Copy `.env.example` to `.env` and set:
   - `OPENAI_API_KEY`
   - `KG_GRAPH_HOST_PATH`
   - `KG_CACHE_HOST_PATH`
2. Start the service:

```bash
docker compose up --build
```

Default URLs:

- Web UI: `http://localhost:8000/`
- Health check: `http://localhost:8000/healthz`
- Agent search API: `POST http://localhost:8000/api/search`
- Agent answer API: `POST http://localhost:8000/api/answer`

Example agent search request:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What causes ASD?\",\"top_k\":5}"
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
- `--ner ollama` now defaults to `mistral:7b`.
- LLM-based NER now runs sentence-first, in small batches, and uses a proposer-plus-verifier flow rather than one large section-level extraction call.
- The LLM NER path keeps builder-facing entity sets as verbatim surface mentions so relation extraction in `kg_pipeline/triple_builder.py` can still match spans in the original sentence text.
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
├── kg_pipeline/rag.py         # Hybrid retriever + CLI
├── Embedder.py                # Embedding wrapper (OpenAI/Jina/local)
└── llm_utils/                 # POP + LLM client adapters
```

## Notes on embeddings and RAG

The retriever in `kg_pipeline/rag.py` uses `Embedder.py` for three text
indices:

- node descriptions
- triple-plus-evidence descriptions
- relation descriptions

At query time it:

1. normalizes the question
2. anchors entities through alias matching
3. detects relation intent from the canonical relation lexicon already used by `kg_pipeline/triple_builder.py`
4. ranks node and triple candidates with text embeddings
5. optionally adds RotatE-style structural candidates when `PyKEEN` artifacts are available
6. fuses rankings with reciprocal-rank fusion
7. expands short graph neighborhoods and reranks triples and paths for answer context

`kg_rag_app.py` now calls this retriever and returns ranked `nodes`,
`triples`, `paths`, `context`, and an optional LLM `answer`. The
browser still highlights nodes, but the retrieval unit is now the triple
and the short path rather than a node-only embedding hit.

Cache files are written under `.kg_cache/<graph_stem>/`. These include
JSON sidecars for aliases and record metadata, `.npy` matrices for text
embeddings, and optional KGE artifacts when `PyKEEN` is installed.

The retrieval design is informed by the KGE survey *Knowledge Graph
Embedding: An Overview* (Ge, Wang, Wang, and Kuo, 2023; local copy:
`2309.12501v1.pdf`), especially the idea that structural KG embeddings
and textual representations complement each other for downstream search
and question-answering tasks.

## Tests

Run the full test suite with:

```bash
python -m unittest discover -s tests
```
