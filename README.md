# LLM Knowledge Graph Extraction Bot (Skeleton)

This project provides a lightweight skeleton for building a bot that
automatically extracts simple knowledge graphs from scientific papers.
It is designed to be easy to extend and integrates with existing code
from the `POP` and `GPT_embedding` repositories.

## Overview

The bot performs the following steps:

1. **Data acquisition** – converts input PDF files into plain text using
   the `pdftotext` command‐line utility that ships with Poppler.  This
   approach avoids heavy dependencies and works offline.
2. **Digest** – produces a rudimentary summary by selecting the first
   few sentences of the document.  You can replace this with an LLM
   summariser via the `PromptFunction` class imported from the
   `POP` project when API keys are available.
3. **Named entity recognition (NER)** – uses simple heuristics
   (capitalisation, acronyms and alphanumeric patterns) to identify
   candidate entities in each sentence.  This avoids pulling in
   external NLP libraries such as spaCy or NLTK, which are not
   available in the runtime environment.
4. **Knowledge graph construction** – builds a graph of
   subject–relation–object triples.  For every pair of entities that
   co‑occur within a sentence, the substring between their mentions is
   extracted and normalised to serve as a relation label.  If no
   meaningful relation text is present, a generic ``"related_to"`` label
   is used.  Triples are weighted by the number of times they are
   observed across sentences.
5. **Merging and saving** – allows merging a newly extracted triple
   graph with an existing graph on disk and saving the result as JSON.

The repository includes placeholder code for chunking and RAG
construction – these parts are intentionally left as stubs for future
work.

## Usage

Run the example extraction on a PDF:

```bash
python main.py --pdf ../2509.00140v1.pdf --output graph.json
```

This will create a JSON file containing the extracted nodes and
triples.  If you supply the `--merge` flag with an existing graph, the
graphs will be merged before saving.

## Project structure

```
llm_kg_bot/
├── kg_pipeline/
│   ├── __init__.py
│   ├── data_acquisition.py   # PDF to text conversion
│   ├── digestor.py           # Simple summarisation
│   ├── ner.py                # Heuristic entity extraction
│   ├── kg_builder.py         # Build undirected co‑occurrence graphs (legacy)
│   ├── kg_merger.py          # Merge/saving utilities for co‑occurrence graphs
│   ├── triple_builder.py     # Build subject–relation–object KGs
│   └── triple_merger.py      # Merge/saving utilities for triple graphs
├── llm_utils/
│   ├── __init__.py
│   ├── LLMClient.py          # Copied from POP
│   └── POP.py                # Copied from POP
├── main.py                   # Demonstration script
└── README.md
```

## Extending this skeleton

- **Summarisation** – replace the simple digestor with calls to
  `PromptFunction` from `llm_utils.POP` when API keys are available.
- **NER** – integrate a biomedical NER model (e.g. SciSpacy) once the
  execution environment can install additional packages.
- **Knowledge graph schema** – refine the graph construction to
  capture typed relations rather than simple co‑occurrence.
- **Chunking and RAG** – implement chunking of long documents and
  build a retrieval‑augmented generation system using FAISS and
  embedding techniques from `GPT_embedding`.  Stubs have been
  provided for these components.
