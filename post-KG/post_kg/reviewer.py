"""Second-stage review for raw KG hypothesis candidates.

The miner is intentionally generous: it surfaces paths that might be useful.
The reviewer is intentionally skeptical: it rewrites, classifies, prioritizes,
or rejects those paths before they become study ideas.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


INTERVENTION_TERMS = re.compile(
    r"\b(?:drug|therapy|treatment|stimulation|tdcs|atdcs|cbd|arbaclofen|bumetanide|"
    r"cannabidiol|agonist|antagonist|inhibitor|modulator)\b",
    re.I,
)
BIOMARKER_TERMS = re.compile(
    r"\b(?:gaba\+?|glutamate|receptor|transporter|kcc2|nkcc1|tspo|cyfip1|"
    r"biomarker|concentration|expression|level|levels|ratio|mrs|pet|eeg|meg)\b",
    re.I,
)
GENERIC_OR_ARTIFACT = re.compile(
    r"^(?:td|nt|sal|saline|vehicle|placebo|sham|control|controls|kim|stagg|gaetz)$",
    re.I,
)
NEGATION = re.compile(
    r"\b(?:no|not|never|without|does not|do not|did not|no evidence|does not support|failed to)\b",
    re.I,
)


@dataclass(frozen=True)
class ReviewedHypothesis:
    """A reviewed hypothesis with decision, rewrite, and study plan."""

    decision: str
    priority: str
    category: str
    priority_score: float
    rewritten_hypothesis: str
    study_design: str
    measurements: List[str]
    concerns: List[str]
    next_actions: List[str]
    web_queries: List[str]
    agent_task: Dict[str, Any]
    raw_hypothesis: str
    raw_score: float
    kg_path: List[Dict[str, Any]]


def review_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[ReviewedHypothesis]:
    """Review raw hypothesis candidates and return prioritized study ideas."""

    reviewed = [_review_one(candidate) for candidate in candidates]
    return sorted(reviewed, key=lambda item: (-item.priority_score, item.decision, item.raw_hypothesis.casefold()))


def render_review_markdown(items: Sequence[ReviewedHypothesis], *, title: str = "Reviewed Hypotheses") -> str:
    """Render reviewed candidates as a Markdown research triage report."""

    lines = [
        f"# {title}",
        "",
        "Review method:",
        "1. Read each raw KG path as a candidate, not as a conclusion.",
        "2. Reject obvious artifacts such as author names, control labels, or unsupported negated claims.",
        "3. Rewrite awkward graph direction into a biologically plausible study hypothesis.",
        "4. Classify the idea as intervention, biomarker, mechanism, subtype, or artifact.",
        "5. Assign a priority, concrete next actions, and web-audit queries for an agent.",
        "",
    ]

    if not items:
        lines.append("No reviewed hypotheses were produced.")
        return "\n".join(lines)

    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                f"## {index}. {item.rewritten_hypothesis}",
                "",
                f"Decision: {item.decision}",
                f"Priority: {item.priority} ({item.priority_score:.3f})",
                f"Category: {item.category}",
                f"Raw: {item.raw_hypothesis} (score={item.raw_score:.3f})",
                "",
                f"Study design: {item.study_design}",
                "",
                "Measurements:",
            ]
        )
        lines.extend(f"- {measurement}" for measurement in item.measurements)
        lines.append("")
        lines.append("KG path:")
        for edge in item.kg_path:
            lines.append(
                f"- {edge['subject']} --[{edge['relation']}]--> {edge['object']} "
                f"(weight={float(edge.get('weight', 0)):g}, sources={int(edge.get('source_count', 0))})"
            )
            if edge.get("evidence_excerpt"):
                lines.append(f"  Evidence: {edge['evidence_excerpt']}")
            if edge.get("paper"):
                lines.append(f"  Paper: {edge['paper']}")
        lines.append("")
        lines.append("Concerns:")
        lines.extend(f"- {concern}" for concern in item.concerns)
        lines.append("")
        lines.append("Next actions:")
        lines.extend(f"- {action}" for action in item.next_actions)
        lines.append("")
        lines.append("Agent web audit:")
        lines.extend(f"- Search: {query}" for query in item.web_queries)
        lines.append("- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def reviewed_to_dicts(items: Sequence[ReviewedHypothesis]) -> List[Dict[str, Any]]:
    """Convert reviewed hypotheses to JSON-serializable dictionaries."""

    return [asdict(item) for item in items]


def _review_one(candidate: Mapping[str, Any]) -> ReviewedHypothesis:
    path = [dict(edge) for edge in candidate.get("path", [])]
    first = path[0] if path else {}
    second = path[1] if len(path) > 1 else {}
    source = str(first.get("subject") or "")
    bridge = str(first.get("object") or "")
    outcome = str(second.get("object") or "")
    raw_score = float(candidate.get("score") or 0.0)
    components = dict(candidate.get("components") or {})
    text = " ".join([source, bridge, outcome, _evidence(first), _evidence(second)])

    category = _category(source, bridge, outcome, text)
    concerns = _concerns(source, bridge, outcome, first, second, components)
    decision = "advance"
    if category == "artifact" or any("artifact" in concern.casefold() for concern in concerns):
        decision = "reject"
    elif any("negated" in concern.casefold() for concern in concerns):
        decision = "needs manual evidence check"
    elif raw_score < 0.65:
        decision = "hold"

    priority_score = _priority_score(raw_score, category, concerns, components)
    priority = _priority_label(priority_score, decision)
    rewritten = _rewrite(source, bridge, outcome, category)
    study_design = _study_design(source, bridge, outcome, category)
    measurements = _measurements(source, bridge, outcome, category)
    next_actions = _next_actions(decision, category)
    web_queries = _web_queries(source, bridge, outcome, rewritten, first, second)
    agent_task = _agent_task(decision, category, rewritten, path, concerns, web_queries)

    return ReviewedHypothesis(
        decision=decision,
        priority=priority,
        category=category,
        priority_score=priority_score,
        rewritten_hypothesis=rewritten,
        study_design=study_design,
        measurements=measurements,
        concerns=concerns,
        next_actions=next_actions,
        web_queries=web_queries,
        agent_task=agent_task,
        raw_hypothesis=str(candidate.get("hypothesis") or ""),
        raw_score=raw_score,
        kg_path=path,
    )


def _category(source: str, bridge: str, outcome: str, text: str) -> str:
    nodes = [source, bridge, outcome]
    if any(_is_artifact(node) for node in nodes):
        return "artifact"
    if INTERVENTION_TERMS.search(source) or INTERVENTION_TERMS.search(bridge):
        return "intervention"
    if BIOMARKER_TERMS.search(bridge) or BIOMARKER_TERMS.search(outcome):
        return "biomarker"
    if re.search(r"\b(?:asd|autism)\b", text, re.I) and re.search(r"\b(?:subtype|severity|phenotype|symptom)\b", text, re.I):
        return "subtype"
    return "mechanism"


def _concerns(
    source: str,
    bridge: str,
    outcome: str,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    components: Mapping[str, Any],
) -> List[str]:
    concerns: List[str] = []
    for node in (source, bridge, outcome):
        if _is_artifact(node):
            concerns.append(f"{node} looks like an artifact or study-group label.")
    if _looks_person_name(outcome):
        concerns.append(f"{outcome} looks like an author name rather than a biomedical endpoint.")
    if NEGATION.search(_evidence(first)) or NEGATION.search(_evidence(second)):
        concerns.append("One evidence sentence appears negated or explicitly unsupported.")
    if float(components.get("evidence") or 0.0) < 0.45:
        concerns.append("Evidence support is thin; inspect the source paper before prioritizing.")
    if float(components.get("testability") or 0.0) < 0.5:
        concerns.append("The path is not obviously measurable or perturbable.")
    if not concerns:
        concerns.append("No automatic red flags; still verify extraction quality manually.")
    return concerns


def _priority_score(
    raw_score: float,
    category: str,
    concerns: Sequence[str],
    components: Mapping[str, Any],
) -> float:
    score = raw_score
    if category in {"intervention", "biomarker"}:
        score += 0.08
    if category == "artifact":
        score -= 0.55
    if any("negated" in concern.casefold() or "unsupported" in concern.casefold() for concern in concerns):
        score -= 0.18
    if any("artifact" in concern.casefold() or "author name" in concern.casefold() for concern in concerns):
        score -= 0.30
    if any("thin" in concern.casefold() for concern in concerns):
        score -= 0.08
    score += 0.08 * float(components.get("testability") or 0.0)
    return max(0.0, min(1.0, score))


def _priority_label(score: float, decision: str) -> str:
    if decision == "reject":
        return "rejected"
    if score >= 0.78:
        return "high"
    if score >= 0.62:
        return "medium"
    return "low"


def _rewrite(source: str, bridge: str, outcome: str, category: str) -> str:
    if category == "artifact":
        return f"Reject or manually inspect the path involving {source}, {bridge}, and {outcome}."
    if category == "intervention" and re.search(r"\b(?:asd|autism)\b", source, re.I):
        return f"{bridge} response may reveal altered {outcome} biology in ASD."
    if category == "intervention":
        return f"{source} may affect ASD-relevant biology through {bridge}."
    if category == "biomarker":
        return f"{bridge} may be a measurable bridge between {source} and {outcome}."
    if category == "subtype":
        return f"{bridge} may define a biologically distinct {source} subgroup linked to {outcome}."
    return f"{bridge} may mediate the relationship between {source} and {outcome}."


def _study_design(source: str, bridge: str, outcome: str, category: str) -> str:
    if category == "intervention":
        return (
            f"Compare {bridge}-responsive and non-responsive samples, then test whether {outcome} "
            f"or ASD-relevant phenotypes change with the intervention."
        )
    if category == "biomarker":
        return (
            f"Measure {bridge} in independent ASD and control cohorts, then model whether it explains "
            f"variation between {source} and {outcome}."
        )
    if category == "artifact":
        return "Do not design a study yet; first verify that all nodes are valid biomedical concepts."
    return (
        f"Perturb or stratify by {bridge}, then measure whether the {source} to {outcome} relationship changes."
    )


def _measurements(source: str, bridge: str, outcome: str, category: str) -> List[str]:
    measurements = [f"Primary bridge measure: {bridge}", f"Endpoint measure: {outcome}"]
    if category == "intervention":
        measurements.append(f"Exposure or response measure for {source if INTERVENTION_TERMS.search(source) else bridge}")
    if category == "biomarker":
        measurements.append("Replication in an independent cohort or model system")
    measurements.append("Manual evidence audit of both KG edges")
    return measurements


def _next_actions(decision: str, category: str) -> List[str]:
    if decision == "reject":
        return [
            "Add the offending node to the miner artifact filters if it recurs.",
            "Do not advance this candidate until the extracted entities are corrected.",
        ]
    actions = [
        "Open the source evidence sentences and verify that the relation direction is correct.",
        "Search for direct literature on the rewritten hypothesis to estimate novelty.",
        "Convert the idea into an experimental contrast with controls and measurable endpoints.",
    ]
    if category in {"intervention", "biomarker"}:
        actions.append("Prioritize this for a short manual literature review because it is relatively testable.")
    return actions


def _web_queries(
    source: str,
    bridge: str,
    outcome: str,
    rewritten: str,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> List[str]:
    queries = [
        f'"{rewritten}"',
        f'"{source}" "{bridge}" "{outcome}" autism',
        f'"{source}" "{bridge}" "{str(first.get("relation") or "")}"',
        f'"{bridge}" "{outcome}" "{str(second.get("relation") or "")}"',
    ]
    for edge in (first, second):
        paper = str(edge.get("paper") or "").strip()
        if paper:
            queries.append(f'"{_paper_title(paper)}"')
    return _unique_nonempty(queries, limit=6)


def _agent_task(
    decision: str,
    category: str,
    rewritten: str,
    path: Sequence[Mapping[str, Any]],
    concerns: Sequence[str],
    web_queries: Sequence[str],
) -> Dict[str, Any]:
    return {
        "objective": "Audit online evidence for one reviewed KG hypothesis.",
        "hypothesis": rewritten,
        "decision_before_web": decision,
        "category": category,
        "recommended_tools": [
            "perplexity_search",
            "jina_web_snapshot or another implemented URL snapshot tool",
        ],
        "avoid_tools": [
            "perplexity_web_snapshot until it is implemented; the current class is a stub",
        ],
        "web_queries": list(web_queries),
        "kg_path": [dict(edge) for edge in path],
        "concerns_to_check": list(concerns),
        "required_agent_output": {
            "evidence_support": "supported | mixed | unsupported | artifact",
            "source_count": "number of independent external sources checked",
            "best_sources": "titles/URLs/PMIDs when available",
            "relation_direction_audit": "whether each KG edge direction is plausible",
            "novelty_assessment": "known | underexplored | contradictory | artifact",
            "final_recommendation": "advance | hold | reject",
        },
    }


def _paper_title(filename: str) -> str:
    title = re.sub(r"^\d+\s*-\s*", "", filename)
    title = re.sub(r"\.pdf$", "", title, flags=re.I)
    return title.strip()


def _unique_nonempty(values: Sequence[str], *, limit: int) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _is_artifact(node: str) -> bool:
    return bool(GENERIC_OR_ARTIFACT.search(str(node or "").strip()))


def _looks_person_name(node: str) -> bool:
    text = str(node or "").strip()
    return bool(re.match(r"^[A-Z][a-z]{2,}$", text)) and not BIOMARKER_TERMS.search(text)


def _evidence(edge: Mapping[str, Any]) -> str:
    return str(edge.get("evidence_excerpt") or "")


def load_candidates(path: str | Path) -> List[Dict[str, Any]]:
    """Load raw candidates from the miner JSON output."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("candidate JSON must contain a list")
    return [dict(item) for item in payload]


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Review and triage raw post-KG hypothesis candidates.")
    parser.add_argument("--input", required=True, help="Raw candidates JSON from miner or pipeline")
    parser.add_argument("--output", help="Optional Markdown review path")
    parser.add_argument("--json-output", help="Optional JSON review path")
    args = parser.parse_args()

    candidates = load_candidates(args.input)
    reviewed = review_candidates(candidates)
    markdown = render_review_markdown(reviewed, title=f"Reviewed Hypotheses from {Path(args.input).name}")

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(reviewed_to_dicts(reviewed), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
