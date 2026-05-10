# Semantic Subgraph Query Tools

`SemanticSubgraphRetriever` is an alternative retriever for the LLM-KG graph
format. It keeps the semantic subgraph workflow from the biomedical KG querying
paper, but now uses this repository's own graph loader, canonical relation
lexicon, alias generation, cached embeddings, and `QueryResult` payload.

This mode is ALEQ-inspired: ALEQ means Adaptive Locating and Expanding Query,
from "From biomedical knowledge graph construction to semantic querying: a
comprehensive approach" (Scientific Reports, 2025):
https://www.nature.com/articles/s41598-025-93334-5.

`ALEQ` is separate from the default `normal` retriever. The `normal` method
is the original project retriever that fuses alias matches, text embedding
rankings, relation intent, optional KGE completion, and path extraction.
The `ALEQ` method does not get fused into `normal`; it reuses shared graph
loading, relation lexicon, aliases, cached text embeddings, and response shape.

## Use From The Shared CLI

Build or query with the ALEQ method:

```bash
python -m kg_pipeline.rag build --graph graph_llm.json --method ALEQ

python -m kg_pipeline.rag query --graph graph_llm.json \
  --method ALEQ \
  --question "What causes ASD?" \
  --top-k 10 \
  --hop-limit 2
```

The ALEQ retriever writes the same text embedding sidecars under
`.kg_cache/<graph_stem>/` as the normal retriever. It does not train RotatE/KGE
artifacts.

## Use In The Flask App

CLI:

```bash
python kg_rag_app.py --graph graph_llm.json --retriever-method ALEQ
```

Environment:

```bash
KG_RETRIEVER_METHOD=ALEQ
KG_SEMANTIC_THRESHOLD=0.05
KG_STRUCTURAL_THRESHOLD=0.05
```

The `/query`, `/api/search`, and `/api/answer` routes keep the same response
shape as the normal retriever.

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
