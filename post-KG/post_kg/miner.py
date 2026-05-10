"""Hypothesis discovery utilities for extracted biomedical KGs.

The miner looks for short mechanistic paths of the form:

    source --relation1--> bridge --relation2--> outcome

These paths are useful hypothesis seeds because they identify a possible
intermediate mechanism between two concepts while preserving provenance for the
supporting edges.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MECHANISTIC_RELATIONS = {
    "activates",
    "causes",
    "decreases",
    "increases",
    "inhibits",
    "mediates",
    "prevents",
    "promotes",
    "regulates",
    "suppresses",
    "treats",
}

INTERVENTION_RELATIONS = {
    "activates",
    "decreases",
    "inhibits",
    "increases",
    "prevents",
    "regulates",
    "suppresses",
    "treats",
}

GENERIC_NODE_PATTERNS = [
    re.compile(r"^\d+(?:[.\-]\d+)?\s*(?:mg|g|kg|ml|mm|cm|months?|weeks?|days?|years?)?$", re.I),
    re.compile(r"^(?:control|controls|case|cases|group|groups|patient|patients|children|mice|rats|study|trial)$", re.I),
    re.compile(r"^(?:high|low|normal|abnormal|baseline|follow-up|followup|significant)$", re.I),
    re.compile(r"^(?:sal|saline|vehicle|placebo|sham|td|nt)$", re.I),
]

TESTABLE_NODE_PATTERNS = [
    re.compile(r"\b(?:receptor|transporter|channel|enzyme|kinase|protein|gene|pathway|signaling)\b", re.I),
    re.compile(r"\b(?:level|levels|expression|activity|ratio|score|frequency|amplitude|concentration)\b", re.I),
    re.compile(r"\b(?:drug|therapy|agonist|antagonist|inhibitor|modulator|biomarker)\b", re.I),
    re.compile(r"^(?:[A-Z0-9]{2,}|IL-\d+|TNF|GABA|NMDA|NMDAR|KCC2|NKCC1)$"),
]


@dataclass(frozen=True)
class EdgeRecord:
    """Compact normalized representation of one KG triple."""

    id: int
    subject: str
    relation: str
    object: str
    weight: float
    source_count: int
    evidence_excerpt: str
    paper: str


@dataclass(frozen=True)
class HypothesisCandidate:
    """Ranked hypothesis seed backed by a two-edge KG path."""

    hypothesis: str
    study_idea: str
    score: float
    components: Dict[str, float]
    path: List[Dict[str, Any]]
    direct_connection: Optional[Dict[str, Any]]
    rationale: List[str]


def discover_hypotheses(
    graph: Mapping[str, Any],
    *,
    focus_terms: Optional[Sequence[str]] = None,
    focus_mode: str = "all",
    focus_scope: str = "path",
    top_k: int = 10,
    max_degree: int = 80,
    min_score: float = 0.0,
) -> List[HypothesisCandidate]:
    """Mine and rank candidate hypotheses from a graph dictionary.

    Parameters
    ----------
    graph:
        Graph JSON produced by ``main.py``.
    focus_terms:
        Optional terms used to keep candidates near a topic of interest.
    focus_mode:
        ``"all"`` requires every focus term to appear in the candidate context.
        ``"any"`` keeps candidates that match at least one focus term.
    focus_scope:
        ``"path"`` matches focus terms against source, bridge, and outcome
        nodes. ``"evidence"`` also allows evidence sentences to satisfy focus.
    top_k:
        Number of candidates to return.
    max_degree:
        Skip very high-degree bridge nodes, which are usually generic concepts.
    min_score:
        Drop candidates below this score.
    """

    edges = [_edge_from_triple(index, triple) for index, triple in enumerate(graph.get("triples", []))]
    edges = [edge for edge in edges if _specificity(edge.subject) > 0 and _specificity(edge.object) > 0]

    outgoing: Dict[str, List[EdgeRecord]] = defaultdict(list)
    direct_edges: Dict[Tuple[str, str], List[EdgeRecord]] = defaultdict(list)
    degree: Dict[str, int] = defaultdict(int)

    for edge in edges:
        outgoing[edge.subject].append(edge)
        direct_edges[(_norm(edge.subject), _norm(edge.object))].append(edge)
        degree[edge.subject] += 1
        degree[edge.object] += 1

    focus_norms = [_norm(term) for term in focus_terms or [] if _norm(term)]
    focus_mode = str(focus_mode or "all").casefold()
    if focus_mode not in {"all", "any"}:
        raise ValueError("focus_mode must be 'all' or 'any'")
    focus_scope = str(focus_scope or "path").casefold()
    if focus_scope not in {"path", "evidence"}:
        raise ValueError("focus_scope must be 'path' or 'evidence'")
    candidates: Dict[Tuple[str, str, str], HypothesisCandidate] = {}

    for first in edges:
        bridge = first.object
        if degree.get(bridge, 0) > max_degree:
            continue
        for second in outgoing.get(bridge, []):
            if first.subject == second.object:
                continue
            if first.subject == bridge or bridge == second.object:
                continue
            if _specificity(bridge) <= 0:
                continue

            path_focus_text = " ".join(
                [
                    first.subject,
                    _acronym(first.subject),
                    first.relation,
                    bridge,
                    _acronym(bridge),
                    second.relation,
                    second.object,
                    _acronym(second.object),
                ]
            )
            candidate_text = path_focus_text
            if focus_scope == "evidence":
                candidate_text = " ".join([candidate_text, first.evidence_excerpt, second.evidence_excerpt])
            if focus_norms and not _matches_focus(candidate_text, focus_norms, mode=focus_mode):
                continue

            direct = _summarize_direct_connection(direct_edges, first.subject, second.object)
            candidate = _score_path(first, second, direct)
            if candidate.score < min_score:
                continue
            key = (first.subject, bridge, second.object)
            previous = candidates.get(key)
            if previous is None or candidate.score > previous.score:
                candidates[key] = candidate

    ranked = sorted(candidates.values(), key=lambda item: (-item.score, item.hypothesis.casefold()))
    return ranked[: max(1, int(top_k))]


def render_markdown(candidates: Sequence[HypothesisCandidate], *, graph_name: str = "graph") -> str:
    """Render candidates as a compact, study-oriented Markdown report."""

    lines: List[str] = [
        f"# Hypothesis candidates from {graph_name}",
        "",
        "Method:",
        "1. Scan the KG for two-step paths: source -> bridge -> outcome.",
        "2. Keep paths near the requested focus terms, if any were provided.",
        "3. Penalize generic nodes and very common bridge nodes.",
        "4. Score each path for mechanism, evidence, novelty, testability, and specificity.",
        "5. Return hypotheses with the evidence sentences that support each edge.",
        "",
    ]

    if not candidates:
        lines.append("No candidate hypotheses passed the current filters.")
        return "\n".join(lines)

    for index, candidate in enumerate(candidates, start=1):
        lines.extend(
            [
                f"## {index}. {candidate.hypothesis}",
                "",
                f"Score: {candidate.score:.3f}",
                "Components: "
                + ", ".join(f"{name}={value:.2f}" for name, value in candidate.components.items()),
                "",
                f"Study idea: {candidate.study_idea}",
                "",
                "KG path:",
            ]
        )
        for edge in candidate.path:
            lines.append(
                f"- {edge['subject']} --[{edge['relation']}]--> {edge['object']} "
                f"(weight={edge['weight']:g}, sources={edge['source_count']})"
            )
            if edge.get("evidence_excerpt"):
                lines.append(f"  Evidence: {edge['evidence_excerpt']}")
            if edge.get("paper"):
                lines.append(f"  Paper: {edge['paper']}")

        if candidate.direct_connection:
            direct = candidate.direct_connection
            lines.append(
                "Direct connection: "
                f"{direct['subject']} --[{direct['relation']}]--> {direct['object']} "
                f"(weight={direct['weight']:g})"
            )
        else:
            lines.append("Direct connection: none found in this graph.")

        lines.append("Why this is interesting:")
        for reason in candidate.rationale:
            lines.append(f"- {reason}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def candidates_to_dicts(candidates: Sequence[HypothesisCandidate]) -> List[Dict[str, Any]]:
    """Convert candidates to JSON-serializable dictionaries."""

    return [asdict(candidate) for candidate in candidates]


def load_graph(path: str | Path) -> Dict[str, Any]:
    """Load a graph JSON file."""

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _edge_from_triple(index: int, triple: Mapping[str, Any]) -> EdgeRecord:
    sources = list(triple.get("sources") or [])
    first_source = sources[0] if sources else {}
    doc_meta = first_source.get("doc_meta") if isinstance(first_source, Mapping) else {}
    paper = ""
    if isinstance(doc_meta, Mapping):
        paper = str(doc_meta.get("filename") or doc_meta.get("title") or "")
    evidence = str(first_source.get("evidence") or "").strip() if isinstance(first_source, Mapping) else ""
    return EdgeRecord(
        id=index,
        subject=str(triple.get("subject") or "").strip(),
        relation=str(triple.get("relation") or "related to").strip(),
        object=str(triple.get("object") or "").strip(),
        weight=float(triple.get("weight") or 1.0),
        source_count=max(1, len(sources)),
        evidence_excerpt=_shorten(evidence),
        paper=paper,
    )


def _score_path(
    first: EdgeRecord,
    second: EdgeRecord,
    direct: Optional[Dict[str, Any]],
) -> HypothesisCandidate:
    mechanism = _mechanism_score(first.relation, second.relation)
    evidence = _evidence_score(first, second)
    novelty = _novelty_score(direct)
    testability = _testability_score(first, second)
    specificity = (_specificity(first.subject) + _specificity(first.object) + _specificity(second.object)) / 3.0

    score = (
        0.28 * mechanism
        + 0.22 * evidence
        + 0.20 * novelty
        + 0.18 * testability
        + 0.12 * specificity
    )

    hypothesis = (
        f"{first.subject} may influence {second.object} through {first.object}."
    )
    study_idea = (
        f"Test whether changing or measuring {first.object} alters the relationship between "
        f"{first.subject} and {second.object}."
    )
    rationale = _rationale(first, second, direct, mechanism, evidence, novelty, testability, specificity)

    return HypothesisCandidate(
        hypothesis=hypothesis,
        study_idea=study_idea,
        score=float(score),
        components={
            "mechanism": mechanism,
            "evidence": evidence,
            "novelty": novelty,
            "testability": testability,
            "specificity": specificity,
        },
        path=[_edge_payload(first), _edge_payload(second)],
        direct_connection=direct,
        rationale=rationale,
    )


def _edge_payload(edge: EdgeRecord) -> Dict[str, Any]:
    return {
        "id": edge.id,
        "subject": edge.subject,
        "relation": edge.relation,
        "object": edge.object,
        "weight": edge.weight,
        "source_count": edge.source_count,
        "evidence_excerpt": edge.evidence_excerpt,
        "paper": edge.paper,
    }


def _summarize_direct_connection(
    direct_edges: Mapping[Tuple[str, str], Sequence[EdgeRecord]],
    subject: str,
    obj: str,
) -> Optional[Dict[str, Any]]:
    direct = list(direct_edges.get((_norm(subject), _norm(obj)), []))
    direct.extend(direct_edges.get((_norm(obj), _norm(subject)), []))
    if not direct:
        return None
    best = max(direct, key=lambda edge: (edge.weight, edge.source_count))
    return {
        "subject": best.subject,
        "relation": best.relation,
        "object": best.object,
        "weight": best.weight,
        "source_count": best.source_count,
    }


def _mechanism_score(relation_one: str, relation_two: str) -> float:
    hits = sum(1 for relation in (relation_one, relation_two) if relation in MECHANISTIC_RELATIONS)
    if hits == 2:
        return 1.0
    if hits == 1:
        return 0.65
    return 0.25


def _evidence_score(first: EdgeRecord, second: EdgeRecord) -> float:
    weight_signal = math.log1p(min(first.weight, second.weight)) / math.log(6)
    source_signal = math.log1p(min(first.source_count, second.source_count)) / math.log(6)
    score = min(1.0, 0.65 * weight_signal + 0.35 * source_signal)
    if _has_negated_evidence(first.evidence_excerpt) or _has_negated_evidence(second.evidence_excerpt):
        score *= 0.6
    return score


def _novelty_score(direct: Optional[Mapping[str, Any]]) -> float:
    if not direct:
        return 1.0
    weight = float(direct.get("weight") or 1.0)
    if weight <= 1:
        return 0.55
    return max(0.1, 1.0 / (1.0 + weight))


def _testability_score(first: EdgeRecord, second: EdgeRecord) -> float:
    score = 0.0
    if first.relation in INTERVENTION_RELATIONS:
        score += 0.25
    if second.relation in INTERVENTION_RELATIONS:
        score += 0.25
    if _looks_testable(first.object):
        score += 0.35
    if _looks_testable(first.subject) or _looks_testable(second.object):
        score += 0.15
    return min(1.0, score)


def _specificity(node: str) -> float:
    text = str(node or "").strip()
    if not text:
        return 0.0
    for pattern in GENERIC_NODE_PATTERNS:
        if pattern.search(text):
            return 0.0
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return 0.0
    if len(tokens) == 1 and len(tokens[0]) <= 2 and not tokens[0].isupper():
        return 0.25
    if len(tokens) == 1 and tokens[0][:1].isupper() and not tokens[0].isupper():
        return 0.45
    if len(tokens) >= 5:
        return 0.65
    return 1.0


def _looks_testable(node: str) -> bool:
    text = str(node or "").strip()
    return any(pattern.search(text) for pattern in TESTABLE_NODE_PATTERNS)


def _has_negated_evidence(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:no|not|never|without|does not|do not|did not|"
            r"lack(?:s|ed|ing)?|failed to|does not support|no evidence)\b",
            str(text or ""),
            re.I,
        )
    )


def _rationale(
    first: EdgeRecord,
    second: EdgeRecord,
    direct: Optional[Mapping[str, Any]],
    mechanism: float,
    evidence: float,
    novelty: float,
    testability: float,
    specificity: float,
) -> List[str]:
    reasons = [
        f"Mechanistic path: {first.relation} followed by {second.relation}.",
        f"Evidence support is {evidence:.2f} based on edge weights and source counts.",
    ]
    if direct:
        reasons.append("A direct endpoint connection exists, so novelty is lower than an indirect-only path.")
    else:
        reasons.append("No direct endpoint connection was found, which makes the bridge worth checking.")
    if testability >= 0.5:
        reasons.append(f"{first.object} looks measurable or perturbable enough for follow-up work.")
    if mechanism < 0.5:
        reasons.append("The relation labels are broad, so this should be treated as exploratory.")
    if specificity < 0.7:
        reasons.append("One or more nodes are broad; inspect the evidence before prioritizing.")
    if novelty < 0.3:
        reasons.append("This may be more confirmatory than novel because the endpoint edge is already strong.")
    return reasons


def _matches_focus(text: str, focus_norms: Sequence[str], *, mode: str) -> bool:
    normalized = _norm(text)
    if mode == "any":
        return any(term in normalized for term in focus_norms)
    return all(term in normalized for term in focus_norms)


def _norm(text: str) -> str:
    text = re.sub(r"[_/]+", " ", str(text or ""))
    text = re.sub(r"[^0-9a-zA-Z\s-]+", " ", text)
    text = text.casefold().replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _acronym(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text or ""))
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens if token).casefold()


def _shorten(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _split_focus(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    terms: List[str] = []
    for value in values:
        terms.extend(part.strip() for part in str(value).split(",") if part.strip())
    return terms


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Mine study-ready hypothesis seeds from a KG JSON file.")
    parser.add_argument("--graph", required=True, help="Path to graph JSON produced by main.py")
    parser.add_argument("--focus", action="append", default=[], help="Focus term. Repeat or comma-separate terms.")
    parser.add_argument("--top", type=int, default=10, help="Number of hypotheses to return")
    parser.add_argument(
        "--focus-mode",
        choices=["all", "any"],
        default="all",
        help="Whether all focus terms must match or any one focus term is enough",
    )
    parser.add_argument(
        "--focus-scope",
        choices=["path", "evidence"],
        default="path",
        help="Match focus terms against path nodes only, or path nodes plus evidence text",
    )
    parser.add_argument("--max-degree", type=int, default=80, help="Skip bridge nodes above this graph degree")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum candidate score")
    parser.add_argument("--output", help="Optional report path")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")

    args = parser.parse_args()
    graph_path = Path(args.graph)
    graph = load_graph(graph_path)
    candidates = discover_hypotheses(
        graph,
        focus_terms=_split_focus(args.focus),
        focus_mode=args.focus_mode,
        focus_scope=args.focus_scope,
        top_k=args.top,
        max_degree=args.max_degree,
        min_score=args.min_score,
    )

    if args.json:
        payload = json.dumps(candidates_to_dicts(candidates), ensure_ascii=False, indent=2)
    else:
        payload = render_markdown(candidates, graph_name=graph_path.name)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
