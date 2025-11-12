# -*- coding: utf-8 -*-
"""
LLM-based NER using POP's structured output (OpenAI JSON schema).

API:
- LLMStructuredNER.extract(text) -> List[Tuple[str, Set[str]]]

Use when you want deterministic JSON from the model.
"""

from __future__ import annotations
from typing import List, Set, Tuple, Dict, Any, Optional
from pathlib import Path
import re
import sys
import json
from llm_utils.POP import PromptFunction
from kg_pipeline.label_store import LabelStore

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
LABELS_PATH = Path(".kg_cache/labels.json")
DEFAULT_LABELS = [
    # "DISEASE","DISORDER","SYMPTOM",
    # "DRUG","CHEMICAL",
    # "GENE","PROTEIN","BIO_PROCESS",
    # "MEASUREMENT","VALUE",
    # "PERSON","ORG","LOCATION",
    # "OTHER"
]

SYSTEM_PROMPT = """You are a biomedical NER extractor.
Return ONLY JSON matching the schema exactly.
Identify named entities that would plausibly form nodes in a knowledge graph.
Prefer biomedical concepts (diseases, drugs, genes/proteins), people, orgs, places, measurements.
De-duplicate within a sentence; preserve original surface forms (no stemming).
Do NOT include stop words or single letters.
"""

# Minimal instruction with explicit schema
USER_PROMPT = """Extract entities for EACH sentence provided.

Return JSON:
{
  "sentences": [
    { "text": "<original sentence>",
      "entities": ["Entity One", "Entity Two", "..."]
    },
    ...
  ]
}

TEXT:
<<<text>>>
"""

# OpenAI "json_schema" (fits the new response_format)
NER_SCHEMA: Dict[str, Any]
with open("llm_utils/schemas/biomedical_ner_extractor.json", "r") as f:
    NER_SCHEMA = json.load(f)
def _split_sentences(text: str, max_len: int = 1500) -> List[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    sents = _SENT_SPLIT.split(cleaned)
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_len:
            out.append(s)
            continue
        # crude chunking for very long sentences
        parts = re.split(r"([;:,-])", s)
        buf = ""
        for p in parts:
            if len(buf) + len(p) < max_len:
                buf += p
            else:
                if buf.strip():
                    out.append(buf.strip())
                buf = p
        if buf.strip():
            out.append(buf.strip())
    return out

# very light stopword filter for emergencies (keep minimal & safe)
_STOPish = set("""
a an the and or of with without into onto in on to for as by from at than then than
""".split())

def _filter_surface(e: str) -> bool:
    t = e.strip()
    if not t: return False
    # if len(t) == 1: return False
    low = t.lower()
    if low in _STOPish: return False
    return True


class LLMNER:
    def __init__(
        self,
        client: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_sent_len: int = 1500,
        label_store: Optional[LabelStore] = None
    ) -> None:
        self.fn = PromptFunction(sys_prompt=SYSTEM_PROMPT, prompt=USER_PROMPT, client=client)
        self.model = model  # if None, POP will use its default for that client
        self.temperature = temperature
        self.max_sent_len = max_sent_len
        self.label_store = label_store if label_store else LabelStore(LABELS_PATH, max_labels=384, min_promote=2, sim_threshold=0.86)

    def _call(self, text_block: str) -> Dict[str, Any]:
        # IMPORTANT: pass OpenAI-style structured output knobs through POP.
        # Your POP should forward these as response_format to OpenAI.
        raw = self.fn.execute(
            text=text_block,
            labels=self.label_store,  # LabelStore class has _str__ attributes
            temp=self.temperature,
            fmt=NER_SCHEMA
        )
        try:
            return json.loads(raw)
        except Exception:
            return raw

    def extract(self, text: str, mode="flat") -> List[Tuple[str, Set[str]]]:
        """Return pipeline-compatible: [(sentence, {entities...}), ...]
        
        Parameters
        ----------  
        text : str
            Full document text.
        mode : {"flat", "sentences", "structured"}
            - "flat": return a Set[str] of unique entity texts over the whole input.
            - "sentences": return List[Tuple[sentence, Set[str]]] like `process_text`.
            - "structured": return List[Dict] with offsets and labels (like `get_entities`).
        """
        sents = _split_sentences(text, self.max_sent_len)
        if not sents:
            return []

        payload = "\n".join(f"- {s}" for s in sents)
        data = self._call(payload)
        if mode == "flat":
            results: List[Tuple[str, Set[str]]] = []
        elif mode == "sentences":
            results: List[Tuple[str, Set[str]]] = []
        else:
            results: List[Dict] = []

        for item in data.get("sentences", []):
            st = (item.get("text") or "").strip()
            if not st:
                continue
            ents_raw = item.get("entities", []) or []
            ents = {e.strip() for e in ents_raw if _filter_surface(e)}
            if ents:
                if mode in ("flat", "sentences"):
                    results.append((st, ents))
                else:
                    results.append({
                        "text": st,
                        "entities": list(ents)
                    })
        print(results)
        return results

# ---------------------------
# CLI
# ---------------------------

def _demo() -> None:
    import json
    if len(sys.argv) < 2:
        print("Usage: python ner.py \"Your text here\" [--labels PERSON,ORG,GPE]")
        sys.exit(0)
    text = sys.argv[1]

    ner = LLMNER()
    results = ner.extract(text)
    print("\nPer-sentence entities:")
    for sent, ents in results:
        print(f"- {sent}\n  -> {sorted(ents)}")
    
    print("\nStructured entities:")
    print(json.dumps(ner.extract(text, mode="structured"), ensure_ascii=False, indent=2))

if __name__ == "__main__":  # pragma: no cover
    _demo()
