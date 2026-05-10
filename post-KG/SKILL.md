---
name: post-kg-hypothesis-audit
description: Use when turning an LLM-KG graph into study-ready biomedical hypotheses, reviewing post-KG reports, or auditing hypothesis evidence with POP-agent web tools such as post_kg_evidence_tasks, perplexity_search, and jina_web_snapshot.
---

# Post-KG Hypothesis Audit

Use this workflow after `main.py` has already produced a KG JSON such as
`GABA_graph.json`. The goal is to generate candidate hypotheses, triage them,
then use an internet-capable agent to audit external evidence.

## Core Rule

Keep responsibilities split:

- **LLM-KG/post-KG**: deterministic local mining, scoring, rewriting, and task packaging.
- **POP-agent**: online evidence audit, novelty check, source inspection, and final recommendation.

Do not make the deterministic post-KG pipeline depend on live internet access.

## Inputs

Required:

- A graph JSON produced by `main.py`.

Optional:

- Focus terms, for example `GABA`, `ASD`, `KCC2`, `NKCC1`, `CBD`.
- Existing reviewed output under `post-KG/outputs/*_reviewed.json`.

## Local Post-KG Pipeline

From the repo root, run:

```powershell
python post-KG/run_pipeline.py --graph GABA_graph.json --focus GABA --focus ASD --top 20
```

Expected outputs:

- `post-KG/outputs/*_raw.md`
- `post-KG/outputs/*_raw.json`
- `post-KG/outputs/*_reviewed.md`
- `post-KG/outputs/*_reviewed.json`

If the user asks for a broader scan, use:

```powershell
python post-KG/run_pipeline.py --graph GABA_graph.json --focus GABA --focus-mode any --top 30
```

If evidence text should satisfy focus matching, use:

```powershell
python post-KG/run_pipeline.py --graph GABA_graph.json --focus GABA --focus ASD --focus-scope evidence --top 30
```

## What The Local Review Means

Treat `*_reviewed.md` as a triage report, not truth.

The local reviewer:

1. Rewrites awkward graph paths into clearer biological hypotheses.
2. Flags artifacts such as author names, study-group labels, and negated evidence.
3. Classifies candidates as `intervention`, `biomarker`, `mechanism`, `subtype`, or `artifact`.
4. Adds `web_queries` and an `agent_task` block to the reviewed JSON.

The local score is only a prioritization hint. External evidence can overturn it.

## POP-Agent Evidence Audit

When POP-agent is available, start with its task-loader tool:

```json
{
  "path": "post-KG/outputs/GABA_graph_GABA_ASD_reviewed.json",
  "limit": 3,
  "priority": "high"
}
```

Tool order:

1. Call `post_kg_evidence_tasks` to load reviewed tasks.
2. For each task, run `perplexity_search` on the provided `web_queries`.
3. Snapshot only promising URLs with `jina_web_snapshot`.
4. Do not use `perplexity_web_snapshot` until implemented; it is currently a stub.
5. Return a final audited recommendation.

## Evidence Audit Checklist

For each hypothesis, the agent should answer:

- Does each evidence sentence actually support the KG edge?
- Is the relation direction biologically sensible?
- Are there independent sources supporting the rewritten hypothesis?
- Is the literature already established, mixed, underexplored, or contradictory?
- Is the hypothesis experimentally testable?
- Should the candidate advance, hold, or be rejected?

## Required Agent Output

Use this compact schema:

```json
{
  "hypothesis": "...",
  "evidence_support": "supported | mixed | unsupported | artifact",
  "source_count": 0,
  "best_sources": [
    {
      "title": "...",
      "url": "...",
      "pmid": "optional",
      "why_it_matters": "..."
    }
  ],
  "relation_direction_audit": "...",
  "novelty_assessment": "known | underexplored | contradictory | artifact",
  "study_design": "...",
  "falsification_test": "...",
  "final_recommendation": "advance | hold | reject"
}
```

## Decision Guidance

Advance when:

- The KG evidence is directionally plausible.
- External sources independently support the mechanism or bridge.
- The bridge is measurable or perturbable.

Hold when:

- Evidence is plausible but thin.
- Relation direction is unclear.
- The hypothesis needs manual paper inspection.

Reject when:

- Nodes are extraction artifacts.
- Evidence is negated or says the opposite.
- The hypothesis is only an author citation, control label, dosage fragment, or group label.

## Feedback Loop

When the agent rejects a candidate because of a recurring artifact:

1. Add the bad node or pattern to `post-KG/post_kg/miner.py` or `reviewer.py`.
2. Regenerate the reports with `post-KG/run_pipeline.py`.
3. Re-run `python -m unittest discover -s post-KG/tests`.

This loop improves the KG hypothesis quality over time.
