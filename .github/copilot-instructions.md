# Copilot Instructions for LLM-KG

## Project Overview
- **Purpose:** Extracts simple knowledge graphs (KGs) from scientific PDFs using lightweight, easily extensible Python code.
- **Main workflow:**
  1. **PDF to text:** Uses `pdftotext` (Poppler) via `kg_pipeline/data_acquisition.py`.
  2. **Summarisation:** Rudimentary summary via `kg_pipeline/digestor.py` (replaceable with LLM summarisation using `llm_utils/POP.py`).
  3. **NER:** Heuristic entity extraction (`kg_pipeline/ner_simple.py`), with optional spaCy-based NER (`kg_pipeline/ner.py`).
  4. **KG construction:** Subject–relation–object triples via `kg_pipeline/triple_builder.py`.
  5. **Merging:** Combine new and existing graphs (`kg_pipeline/triple_merger.py`).

## Key Files & Directories
- `main.py`: Entry point for PDF-to-KG pipeline.
- `dedupe.py`: Cleans/aggregates graphs, supports spaCy model selection.
- `KG_visualizer.py`, `pyvis_view.py`, `visualizer.R`: Graph visualisation (Python, HTML, R).
- `llm_utils/POP.py`: PromptFunction class for LLM-based summarisation and prompt engineering.
- `files/`, `papers_ADHD/`, `papers_SUD/`, `sampled_papers/`: Data directories.

## Developer Workflows
- **Extract KG:** `python main.py --pdf <input> --output <graph.json> [--merge <existing.json>] [--summary]`
- **Deduplicate/Clean:** `python dedupe.py --input <in.json> --output <out.json> --spacy-model en_core_sci_sm --min-support 3`
- **Visualise:** `python KG_visualizer.py --input <graph.json> [options]`
- **Fetch PDFs:** `python fetch_ncbi.py --csv <file> --out <dir> --n <N>`

## Conventions & Patterns
- **Heuristic-first:** Default NER and summarisation are heuristic; LLM/spaCy upgrades are optional and modular.
- **Extensibility:** Stubs for chunking, RAG, and advanced NER are present; see `kg_pipeline/` and `llm_utils/`.
- **Data flow:** Each stage outputs JSON; merging and cleaning are explicit steps.
- **No heavy dependencies:** Avoids spaCy/NLTK unless explicitly requested.
- **PromptFunction:** For LLM tasks, use `PromptFunction` in `llm_utils/POP.py` (see docstrings for usage).

## Integration Points
- **LLM/Prompting:** Integrate LLMs via `llm_utils/POP.py` (requires API keys in `.env`).
- **NER:** Use `kg_pipeline/ner.py` for spaCy-based NER (auto-selects best available model).
- **Visualisation:** Outputs can be visualised in Python (pyvis), HTML, or R.

## Examples
- Extract and merge: `python main.py --pdf sampled_papers --output graph_ADHD_SUD.json --merge graph.json`
- Clean: `python dedupe.py --input graph_ADHD_SUD.json --output graph_clean.json --spacy-model en_core_sci_sm --min-support 3`
- Visualise: `python KG_visualizer.py --k-core 3 --max-nodes 300 --input graph_clean.json`

## Tips
- Check `note.txt` for sample command sequences and workflow notes.
- See `README.md` for high-level architecture and extension points.
- All scripts support `--help` for CLI options.
