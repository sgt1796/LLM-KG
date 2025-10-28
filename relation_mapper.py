# relation_mapper.py
from __future__ import annotations
import re, math, json, time, logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable, Any, Callable
import requests
import numpy as np
from llm_utils.Embedder import Embedder

log = logging.getLogger("relation_mapper")
OLSHOST = "https://www.ebi.ac.uk/ols4"

# ------- 1) Base canonical relations (portable, small) -------
BASE_CANON: Dict[str, Dict[str, Any]] = {
    # Causal
    "causes":              {"group": "causal", "aliases": ["leads to","results in","induces","triggers"]},
    "prevents":            {"group": "causal"},
    "predisposes":         {"group": "causal"},
    "treats":              {"group": "causal"},
    "complicates":         {"group": "causal"},
    "contraindicates":     {"group": "causal"},

    # Correlative
    "associated_with":     {"group": "correlative"},
    "co_occurs_with":      {"group": "correlative"},
    "correlated_with":     {"group": "correlative"},

    # Regulatory
    "positively_regulates":{"group": "regulatory"},
    "negatively_regulates":{"group": "regulatory"},
    "regulates":           {"group": "regulatory"},

    # Interaction / binding / activity
    "interacts_with":      {"group": "interaction"},
    "binds":               {"group": "interaction"},
    "activates":           {"group": "interaction"},
    "inhibits":            {"group": "interaction"},

    # Expression / location / partonomy / development
    "expressed_in":        {"group": "occurrence"},
    "located_in":          {"group": "occurrence"},
    "occurs_in":           {"group": "occurrence"},
    "part_of":             {"group": "structural"},
    "develops_from":       {"group": "structural"},

    # Measurement/effect
    "affects":             {"group": "effect"},
    "increases":           {"group": "effect"},
    "decreases":           {"group": "effect"},

    # Evidence/method (optional)
    "measured_by":         {"group": "evidence"},
    "indicated_by":        {"group": "evidence"},

    # Temporal/logical
    "precedes":            {"group": "temporal"},
    "follows":             {"group": "temporal"},
    "same_as":             {"group": "logical"},
    "subclass_of":         {"group": "logical"},
}

# Maps obvious surface patterns to canon (pattern-first override)
PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(bind|binds|bound)\b.*\b(to|with)\b", re.I), "binds"),
    (re.compile(r"\b(interact|interacts|interaction)\b", re.I), "interacts_with"),
    (re.compile(r"\b(inhibit|inhibits|suppresses|block(?:s|ed)?)\b", re.I), "inhibits"),
    (re.compile(r"\b(activat(?:e|es)|stimulat(?:e|es))\b", re.I), "activates"),
    (re.compile(r"\b(positively\s*regulat(?:e|es))\b", re.I), "positively_regulates"),
    (re.compile(r"\b(negatively\s*regulat(?:e|es)|down-?regulat(?:e|es))\b", re.I), "negatively_regulates"),
    (re.compile(r"\bregulat(?:e|es)\b", re.I), "regulates"),
    (re.compile(r"\b(cause[s]?|lead[s]? to|result[s]? in|induc(?:e|es)|trigger[s]?)\b", re.I), "causes"),
    (re.compile(r"\bprevent[s]?\b", re.I), "prevents"),
    (re.compile(r"\bpredispos(?:e|es)\b", re.I), "predisposes"),
    (re.compile(r"\b(treats?|treatment of)\b", re.I), "treats"),
    (re.compile(r"\b(complicat(?:e|es))\b", re.I), "complicates"),
    (re.compile(r"\b(associated|linked|correlat(?:e|es)|co-?occurs?)\b", re.I), "associated_with"),
    (re.compile(r"\b(part of)\b", re.I), "part_of"),
    (re.compile(r"\b(develops? from)\b", re.I), "develops_from"),
    (re.compile(r"\b(located in|occurs in|expressed in)\b", re.I), "occurs_in"),
    (re.compile(r"\b(precedes)\b", re.I), "precedes"),
    (re.compile(r"\b(follows)\b", re.I), "follows"),
]

NEGATION = re.compile(r"\b(no(?:t)?|never|without|lack of|fails? to)\b", re.I)
HEDGES  = re.compile(r"\b(may|might|could|suggests?|appears?|possibly|likely)\b", re.I)

@dataclass
class Predicate:
    id: str
    label: str
    description: str = ""
    source: str = "base"  # "base" | "ols:<ontoId>"
    group: Optional[str] = None
    mappings: Dict[str, str] = field(default_factory=dict)  # cross-refs (RO/SIO/UMLS IRIs)

class OLSClient:
    def __init__(self, host: str = OLSHOST, timeout: int = 20):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def list_ontologies(self) -> List[Dict[str, Any]]:
        url = f"{self.host}/api/ontologies"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # OLS4 returns a list
        return data if isinstance(data, list) else data.get("_embedded", {}).get("ontologies", [])

    def search_properties(self, q: str, ontology: Optional[str] = None, size: int = 200) -> List[Dict[str, Any]]:
        """
        Query object properties matching text q.
        OLS4 search supports 'type=property'. We also restrict queryFields to label,synonym,description where supported.
        """
        params = {"q": q, "type": "property", "size": str(size)}
        if ontology: params["ontology"] = ontology
        # queryFields is supported in OLS4 search backend
        params["queryFields"] = "label,synonym,description"
        url = f"{self.host}/api/search"
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        items = []
        # OLS4 may respond as a list or HAL-like; handle both
        if isinstance(js, dict):
            items = js.get("_embedded", {}).get("terms", []) or js.get("response", {}).get("docs", [])
        elif isinstance(js, list):
            items = js
        return items

def build_base_catalog() -> List[Predicate]:
    preds: List[Predicate] = []
    for lab, meta in BASE_CANON.items():
        preds.append(Predicate(
            id=f"canon:{lab}",
            label=lab.replace("_", " "),
            description=f"Canonical relation: {lab}",
            source="base",
            group=meta.get("group"),
        ))
    return preds

def pick_candidate_ontologies_from_texts(texts: List[str]) -> List[str]:
    """
    Heuristic: use TF-IDF to find top cue words and assign ontologies.
    We prefer RO and SIO by default, and add others if their names appear.
    """
    preferred = {"ro", "sio"}  # Relations Ontology & SIO
    # add more by cues
    corpus = " ".join(texts).lower()
    if any(k in corpus for k in ["umls", "semantic type", "sty"]):
        preferred.add("sty")  # UMLS semantic types (via BioPortal; still keep RO/SIO in OLS)
    return sorted(preferred)

def fetch_ols_predicates(ols: OLSClient, ontos: List[str], seeds: List[str]) -> List[Predicate]:
    out: List[Predicate] = []
    seen = set()
    for onto in ontos:
        for s in seeds:
            try:
                for term in ols.search_properties(q=s, ontology=onto, size=200):
                    label = term.get("label") or term.get("prefLabel") or ""
                    if not label: continue
                    iri = term.get("iri") or term.get("@id") or f"ols:{label}"
                    key = (onto, iri)
                    if key in seen: continue
                    seen.add(key)
                    desc = term.get("description") or term.get("definition") or ""
                    out.append(Predicate(
                        id=iri,
                        label=label,
                        description=desc if isinstance(desc, str) else " ".join(desc or []),
                        source=f"ols:{onto}",
                        group=None,
                        mappings={k:v for k,v in (("ontology", onto),)}
                    ))
            except Exception as e:
                log.warning("OLS search failed for onto=%s seed=%s: %s", onto, s, e)
                continue
    return out

class SemanticNNMapper:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self._labels = []
        self._ids = []
        self._meta = []
        self._mat = None

    def build(self, preds):
        texts = [f"{p.label}. {p.description or ''}" for p in preds]
        embs = np.array(self.embed_fn(texts), dtype=np.float32) # takes a list of string
        # normalize for cosine
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        self._mat = embs
        self._labels = [p.label for p in preds]
        self._ids = [p.id for p in preds]
        self._meta = preds

    def nearest(self, phrase: str, threshold: float = 0.55):
        v = np.array(self.embed_fn([phrase])[0], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        sims = np.dot(self._mat, v)
        i = int(np.argmax(sims))
        score = float(sims[i])
        pred = self._meta[i]
        if score < threshold:
            return "related_to", score, Predicate(id="canon:related_to", label="related to", description="", source="base")
        return pred.label.replace(" ", "_"), score, pred

def detect_neg_hedge(text: str) -> Dict[str, bool]:
    return {
        "negated": bool(NEGATION.search(text)),
        "uncertain": bool(HEDGES.search(text)),
    }

def pattern_override(text: str) -> Optional[str]:
    for pat, lab in PATTERNS:
        if pat.search(text):
            return lab
    return None

# ---------- Public API ----------
def build_dynamic_catalog_for_pdfs(pdf_texts: List[str],
                                   embed_fn: Callable[[List[str]], List[List[float]]],
                                   seeds: Iterable[str] = ("regulate","inhibit","activate","bind","interact",
                                                           "cause","prevent","treat","associate",
                                                           "part of","develops from","occurs in")) -> Tuple[List[Predicate], SemanticNNMapper]:
    """
    1) Start from BASE_CANON
    2) Pick ontologies (RO,SIO) based on texts
    3) Pull matching properties from OLS and add them
    4) Build NN index for mapping
    """
    preds = build_base_catalog()
    ols = OLSClient()
    ontos = pick_candidate_ontologies_from_texts(pdf_texts)  # e.g., ['ro','sio']
    ols_preds = fetch_ols_predicates(ols, ontos, list(seeds))
    # De-duplicate by label (casefold) while keeping canon first
    seen = set(p.label.casefold() for p in preds)
    for p in ols_preds:
        key = p.label.casefold()
        if key not in seen:
            preds.append(p)
            seen.add(key)

    mapper = SemanticNNMapper(embed_fn=embed_fn)
    mapper.build(preds)
    return preds, mapper

def map_relation_phrase(text_span: str,
                        mapper: SemanticNNMapper,
                        subj_type: Optional[str] = None,
                        obj_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Map a free-text relation phrase to a canonical label + flags.
    Add optional domain/range constraints (hook here) using your UMLS type system.
    """
    phrase = text_span.strip()
    # A) high-precision pattern
    pat = pattern_override(phrase)
    flags = detect_neg_hedge(phrase)
    if pat:
        return {"relation": pat, "score": 1.0, "source": "pattern", **flags}

    # B) semantic nearest neighbor to catalog
    rel, score, pred = mapper.nearest(phrase)
    # C) (optional) enforce type constraints using your STY pairs
    # if not is_allowed(rel, subj_type, obj_type): rel, score = "related_to", 0.0
    return {"relation": rel, "score": score, "source": pred.source, "iri": pred.id, **flags}
