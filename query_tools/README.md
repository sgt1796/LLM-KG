# Semantic Subgraph Query Tools

`SemanticSubgraphRetriever` is the LLM-KG graph query retriever. It keeps the
semantic subgraph workflow from the biomedical KG querying paper, but now uses
this repository's own graph loader, canonical relation lexicon, alias
generation, cached embeddings, and `QueryResult` payload.

This mode is ALEQ-inspired: ALEQ means Adaptive Locating and Expanding Query,
from "From biomedical knowledge graph construction to semantic querying: a
comprehensive approach" (Scientific Reports, 2025):
https://www.nature.com/articles/s41598-025-93334-5.

ALEQ is the only query algorithm exposed by the shared CLI and Flask app. It
reuses shared graph loading, relation lexicon, aliases, cached text embeddings,
and response shape.

## Use From The Shared CLI

Build or query with ALEQ:

```bash
python -m kg_pipeline.rag build --graph graph_llm.json

python -m kg_pipeline.rag query --graph graph_llm.json \
  --question "What causes ASD?" \
  --top-k 10 \
  --hop-limit 2
```

The ALEQ retriever writes the same text embedding sidecars under
`.kg_cache/<graph_stem>/`. It does not train RotatE/KGE artifacts.

## Use In The Flask App

Start the app with a graph or upload one from the opening page:

```bash
python kg_rag_app.py --graph graph_llm.json
```

The app uses ALEQ for every query. `KG_SEMANTIC_THRESHOLD` and
`KG_STRUCTURAL_THRESHOLD` tune the ALEQ semantic and structural filters.

The `/query`, `/api/search`, and `/api/answer` routes keep the same response
shape.

## Direct Example

```bash
python query_tools/sample_usage.py --graph graph_llm.json \
  --question "What causes ASD?"
```

## Notes

- Relation intent uses the project canonical relation vocabulary from
  `kg_pipeline/triple_builder.py`, including inverted phrases such as
  "treated with".
- Semantic filtering uses node text built from incident triples and evidence,
  not only raw node labels.
- Structural filtering combines local path proximity, neighbor overlap, and
  direct edge weight so sparse biomedical KGs still return useful direct
  neighbors.
