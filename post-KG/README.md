# Post-KG Hypothesis Pipeline

This folder contains the work that happens after `main.py` has already built a
knowledge graph.

For agent-facing workflow instructions, see `SKILL.md`.

The pipeline has two stages:

1. **Mine raw candidates** from short KG paths: `source -> bridge -> outcome`.
2. **Review candidates** with a skeptical triage pass that rewrites, classifies,
   prioritizes, or rejects them.

The goal is not to prove a hypothesis. The goal is to turn graph structure plus
evidence sentences into a short list of study ideas worth manual review.

## One-command pipeline

From the repo root:

```powershell
python post-KG/run_pipeline.py --graph GABA_graph.json --focus GABA --focus ASD --top 20
```

Outputs are written to `post-KG/outputs/`:

- `*_raw.md`: raw KG path hypotheses
- `*_raw.json`: raw structured candidates
- `*_reviewed.md`: rewritten and prioritized study ideas
- `*_reviewed.json`: structured review data for an agent or downstream app

## Stage 1: Mine

```powershell
python post-KG/mine_hypotheses.py --graph GABA_graph.json --focus GABA --focus ASD --top 10 --output post-KG/outputs/GABA_raw.md
```

The miner:

1. Loads triples from the graph JSON.
2. Scans two-step paths: `source -> bridge -> outcome`.
3. Keeps paths matching your focus terms.
4. Filters generic nodes and very common bridge nodes.
5. Scores mechanism, evidence, novelty, testability, and specificity.

## Stage 2: Review

```powershell
python post-KG/review_hypotheses.py --input post-KG/outputs/GABA_graph_GABA_ASD_raw.json --output post-KG/outputs/GABA_reviewed.md
```

The reviewer:

1. Treats the KG output as a candidate, not a fact.
2. Rejects obvious artifacts such as author names or study-group labels.
3. Rewrites awkward graph direction into a more biological framing.
4. Classifies each idea as intervention, biomarker, mechanism, subtype, or artifact.
5. Adds concrete next actions, web-search queries, and an agent task block for
   online evidence review.

## Where an LLM agent fits

The current reviewer is deterministic and local. It does not call the internet.
An LLM agent can use the reviewed JSON as its task queue:

1. Open the cited evidence sentences and papers.
2. Decide whether the relation direction is biologically sensible.
3. Search whether the rewritten hypothesis is already known.
4. Draft a study protocol with model, assay, endpoints, and falsification test.
5. Feed rejected artifacts back into the miner filters.

Use the agent after the deterministic review, not before it. The graph is good
at generating candidate space; the agent is better at scientific criticism.

For web evidence, prefer this tool order:

1. Use POP-agent's `post_kg_evidence_tasks` tool to load reviewed hypotheses.
2. Use `perplexity_search` to find papers, PubMed records, reviews, and URLs.
3. Snapshot only the useful URLs returned by search.
4. Do not rely on `perplexity_web_snapshot` until that tool is implemented; in
   the current POP-agent code it is a stub.

Example agent tool call:

```json
{
  "path": "post-KG/outputs/GABA_graph_GABA_ASD_reviewed.json",
  "limit": 3,
  "priority": "high"
}
```
