# -*- coding: utf-8 -*-
"""
Two-stage LLM-assisted biomedical NER.

Stage 1:
- start from cheap candidate proposals
- verify exact surface mentions per sentence

Stage 2:
- classify, filter, and normalize verified mentions
- retain both surface and canonical names

The builder-facing API still returns ``(sentence, entities)`` pairs that
contain only surface mentions, because downstream relation extraction
needs spans that occur verbatim in the sentence text.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json
import re
import sys

from llm_utils.POP import PromptFunction
from kg_pipeline.label_store import LabelStore
from kg_pipeline.ner_simple import NERExtractor

try:  # pragma: no cover - exercised indirectly when spaCy model exists
    from kg_pipeline.ner import SpacyNER
except Exception:  # pragma: no cover - safe fallback when spaCy/model is unavailable
    SpacyNER = None  # type: ignore[assignment]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_STOPISH = {
    "a", "an", "the", "and", "or", "of", "with", "without", "into",
    "onto", "in", "on", "to", "for", "as", "by", "from", "at", "than",
    "then", "we", "our", "they", "their", "this", "that", "these",
    "those",
}
_MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:%|mg|g|kg|ug|ng|pg|ml|mL|l|L|mm|cm|um|nm|"
    r"uM|nM|mM|mmHg|kDa|h|hr|hrs|day|days|week|weeks|month|months|"
    r"year|years)\b"
)
_BIOMED_TOKEN_PATTERN = re.compile(
    r"\b(?:[A-Za-z]+[-/]?\d+[A-Za-z0-9-]*|[A-Z]{2,}(?:-[A-Z0-9]+)*)\b"
)
_TITLECASE_TOKEN_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}(?:-[A-Za-z0-9]+)?\b")
ENTITY_LABELS = (
    "DISEASE",
    "DRUG",
    "GENE_PROTEIN",
    "PATHWAY",
    "CELL_TYPE",
    "MEASUREMENT",
    "OTHER",
)

SURFACE_STAGE_SYSTEM_PROMPT = """You are stage 1 of a biomedical NER pipeline.
Return ONLY JSON matching the schema exactly.
For each sentence, identify exact surface mentions that appear verbatim in the sentence.
Every mention must be copied as a contiguous substring from the sentence text.
Do not normalize, paraphrase, expand acronyms, merge mentions, or invent entities.
Prefer the provided candidate list and prune it aggressively; add a missing mention only when it appears verbatim in the sentence and is clearly a useful entity mention.
Do not include pronouns, stop words, section headers, malformed fragments, or generic filler terms.
Output sentence ids only. Never echo sentence text in the JSON output.
"""

SURFACE_STAGE_USER_PROMPT = """Verify exact entity surface mentions for each sentence.

Return JSON:
{
  "sentences": [
    { "id": 0, "mentions": ["Exact Mention", "Another Mention"] }
  ]
}

BATCH:
<<<payload_json>>>
"""

NORMALIZATION_STAGE_SYSTEM_PROMPT = """You are stage 2 of a biomedical NER pipeline.
Return ONLY JSON matching the schema exactly.
Given verified surface mentions, keep only useful knowledge-graph entities.
Assign one label from this schema:
DISEASE, DRUG, GENE_PROTEIN, PATHWAY, CELL_TYPE, MEASUREMENT, OTHER.
Keep `surface` exactly as provided. `canonical_name` should be a normalized name when obvious; otherwise repeat the surface form.
Remove junk, malformed spans, and generic mentions that are not useful entities.
Use OTHER only for meaningful entities that do not fit the biomedical labels above.
Output sentence ids only. Never echo sentence text in the JSON output.
"""

NORMALIZATION_STAGE_USER_PROMPT = """Classify, filter, and normalize verified mentions.

Return JSON:
{
  "sentences": [
    {
      "id": 0,
      "mentions": [
        {
          "surface": "IL-6",
          "label": "GENE_PROTEIN",
          "canonical_name": "interleukin-6"
        }
      ]
    }
  ]
}

BATCH:
<<<payload_json>>>
"""

SURFACE_STAGE_SCHEMA: Dict[str, Any] = {
    "name": "Biomedical_Surface_Mentions",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer"},
                        "mentions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["id", "mentions"],
                },
            }
        },
        "required": ["sentences"],
    },
}

NORMALIZATION_STAGE_SCHEMA: Dict[str, Any] = {
    "name": "Biomedical_Normalized_Mentions",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer"},
                        "mentions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "surface": {"type": "string"},
                                    "label": {
                                        "type": "string",
                                        "enum": list(ENTITY_LABELS),
                                    },
                                    "canonical_name": {"type": "string"},
                                },
                                "required": ["surface", "label", "canonical_name"],
                            },
                        },
                    },
                    "required": ["id", "mentions"],
                },
            }
        },
        "required": ["sentences"],
    },
}


def _split_sentences(text: str, max_len: int = 1500) -> List[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []

    sents = _SENT_SPLIT.split(cleaned)
    out: List[str] = []
    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_len:
            out.append(sent)
            continue

        parts = re.split(r"([;:,-])", sent)
        buf = ""
        for part in parts:
            if len(buf) + len(part) < max_len:
                buf += part
                continue
            if buf.strip():
                out.append(buf.strip())
            buf = part
        if buf.strip():
            out.append(buf.strip())
    return out


def _filter_surface(text: str) -> bool:
    mention = text.strip()
    if not mention:
        return False
    if mention.casefold() in _STOPISH:
        return False
    return True


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _sort_mentions_by_sentence(sentence: str, mentions: Iterable[str]) -> List[str]:
    return sorted(
        _dedupe_preserve_order(mentions),
        key=lambda item: (
            sentence.find(item) if sentence.find(item) >= 0 else sys.maxsize,
            -len(item),
            item.casefold(),
        ),
    )


class LLMNER:
    def __init__(
        self,
        client: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_sent_len: int = 1500,
        label_store: Optional[LabelStore] = None,
        proposer: Optional[Any] = None,
        surface_fn: Optional[Any] = None,
        normalizer_fn: Optional[Any] = None,
        sentence_batch_size: int = 8,
        batch_char_budget: int = 3200,
    ) -> None:
        self.surface_fn = surface_fn or PromptFunction(
            sys_prompt=SURFACE_STAGE_SYSTEM_PROMPT,
            prompt=SURFACE_STAGE_USER_PROMPT,
            client=client,
        )
        self.normalizer_fn = normalizer_fn or PromptFunction(
            sys_prompt=NORMALIZATION_STAGE_SYSTEM_PROMPT,
            prompt=NORMALIZATION_STAGE_USER_PROMPT,
            client=client,
        )
        self.model = model
        self.temperature = temperature
        self.max_sent_len = max_sent_len
        self.label_store = label_store
        self.sentence_batch_size = max(1, sentence_batch_size)
        self.batch_char_budget = max(self.max_sent_len, batch_char_budget)
        self.proposer = proposer or self._build_default_proposer()

    def _build_default_proposer(self) -> Any:
        if SpacyNER is not None:
            try:
                return SpacyNER()
            except Exception:
                pass
        return NERExtractor()

    def split_sentences(self, text: str) -> List[str]:
        return _split_sentences(text, self.max_sent_len)

    def iter_sentence_batches(self, sentences: Sequence[str]) -> Iterable[Tuple[int, List[str]]]:
        batch: List[str] = []
        batch_start = 0
        batch_chars = 0

        for idx, raw_sentence in enumerate(sentences):
            sentence = raw_sentence.strip()
            if not sentence:
                continue

            sentence_chars = len(sentence)
            if batch and (
                len(batch) >= self.sentence_batch_size
                or batch_chars + sentence_chars > self.batch_char_budget
            ):
                yield batch_start, batch
                batch = []
                batch_chars = 0

            if not batch:
                batch_start = idx

            batch.append(sentence)
            batch_chars += sentence_chars

        if batch:
            yield batch_start, batch

    def _propose_candidates(self, sentence: str) -> List[str]:
        candidates: List[str] = []

        if hasattr(self.proposer, "extract_entities_from_sentence"):
            proposed = self.proposer.extract_entities_from_sentence(sentence)
            if proposed:
                candidates.extend(str(item).strip() for item in proposed if str(item).strip())

        for pattern in (_MEASUREMENT_PATTERN, _BIOMED_TOKEN_PATTERN, _TITLECASE_TOKEN_PATTERN):
            for match in pattern.finditer(sentence):
                candidates.append(match.group(0).strip())

        filtered = [item for item in candidates if _filter_surface(item) and item in sentence]
        return _sort_mentions_by_sentence(sentence, filtered)

    def _execute_json(self, prompt_fn: Any, payload: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "temp": self.temperature,
            "fmt": schema,
        }
        if self.model is not None:
            kwargs["model"] = self.model

        raw = prompt_fn.execute(**kwargs)
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _run_surface_stage(self, indexed_sentences: Sequence[Dict[str, Any]]) -> Dict[int, List[str]]:
        if not indexed_sentences:
            return {}

        sentence_lookup = {item["id"]: item["text"] for item in indexed_sentences}
        payload = {
            "sentences": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "candidates": self._propose_candidates(item["text"]),
                }
                for item in indexed_sentences
            ]
        }
        data = self._execute_json(self.surface_fn, payload, SURFACE_STAGE_SCHEMA)

        verified: Dict[int, List[str]] = {item["id"]: [] for item in indexed_sentences}
        for item in data.get("sentences", []):
            sentence_id = item.get("id")
            sentence = sentence_lookup.get(sentence_id)
            if sentence is None:
                continue

            mentions = []
            for raw_mention in item.get("mentions", []) or []:
                mention = str(raw_mention).strip()
                if not _filter_surface(mention):
                    continue
                if mention not in sentence:
                    continue
                mentions.append(mention)

            verified[sentence_id] = _sort_mentions_by_sentence(sentence, mentions)
        return verified

    def _run_normalization_stage(
        self,
        indexed_sentences: Sequence[Dict[str, Any]],
        verified: Dict[int, List[str]],
    ) -> Dict[int, List[Dict[str, str]]]:
        stage_input = [
            {
                "id": item["id"],
                "text": item["text"],
                "mentions": verified.get(item["id"], []),
            }
            for item in indexed_sentences
            if verified.get(item["id"])
        ]
        if not stage_input:
            return {item["id"]: [] for item in indexed_sentences}

        sentence_lookup = {item["id"]: item["text"] for item in indexed_sentences}
        verified_lookup = {item["id"]: set(verified.get(item["id"], [])) for item in indexed_sentences}
        data = self._execute_json(
            self.normalizer_fn,
            {"sentences": stage_input},
            NORMALIZATION_STAGE_SCHEMA,
        )

        normalized: Dict[int, List[Dict[str, str]]] = {item["id"]: [] for item in indexed_sentences}
        for item in data.get("sentences", []):
            sentence_id = item.get("id")
            sentence = sentence_lookup.get(sentence_id)
            valid_surfaces = verified_lookup.get(sentence_id)
            if sentence is None or valid_surfaces is None:
                continue

            kept: List[Dict[str, str]] = []
            seen_surfaces: Set[str] = set()
            for raw_mention in item.get("mentions", []) or []:
                surface = str(raw_mention.get("surface", "")).strip()
                label = str(raw_mention.get("label", "")).strip()
                canonical_name = str(raw_mention.get("canonical_name", "")).strip() or surface

                if surface in seen_surfaces:
                    continue
                if surface not in valid_surfaces:
                    continue
                if surface not in sentence:
                    continue
                if label not in ENTITY_LABELS:
                    continue

                seen_surfaces.add(surface)
                kept.append(
                    {
                        "surface": surface,
                        "label": label,
                        "canonical_name": canonical_name,
                    }
                )

            kept.sort(
                key=lambda item_: (
                    sentence.find(item_["surface"]),
                    -len(item_["surface"]),
                    item_["surface"].casefold(),
                )
            )
            normalized[sentence_id] = kept

        return normalized

    def extract_sentences(
        self,
        sentences: Sequence[str],
        *,
        mode: str = "sentences",
        sentence_offset: int = 0,
    ):
        indexed_sentences = [
            {"id": sentence_offset + idx, "text": sentence.strip()}
            for idx, sentence in enumerate(sentences)
            if sentence and sentence.strip()
        ]

        if mode not in {"flat", "sentences", "structured"}:
            raise ValueError("mode must be one of: flat, sentences, structured")

        if not indexed_sentences:
            return set() if mode == "flat" else []

        verified = self._run_surface_stage(indexed_sentences)
        normalized = self._run_normalization_stage(indexed_sentences, verified)

        if mode == "flat":
            flat: Set[str] = set()
            for item in normalized.values():
                flat.update(entry["surface"] for entry in item)
            return flat

        if mode == "structured":
            return [
                {
                    "sentence_id": item["id"],
                    "sentence": item["text"],
                    "entities": normalized.get(item["id"], []),
                }
                for item in indexed_sentences
            ]

        return [
            (
                item["text"],
                {entry["surface"] for entry in normalized.get(item["id"], [])},
            )
            for item in indexed_sentences
        ]

    def extract(self, text: str, mode: str = "flat"):
        sentences = self.split_sentences(text)
        if not sentences:
            return set() if mode == "flat" else []

        if mode == "flat":
            flat: Set[str] = set()
            for batch_start, sentence_batch in self.iter_sentence_batches(sentences):
                flat.update(
                    self.extract_sentences(
                        sentence_batch,
                        mode="flat",
                        sentence_offset=batch_start,
                    )
                )
            return flat

        combined: List[Any] = []
        for batch_start, sentence_batch in self.iter_sentence_batches(sentences):
            combined.extend(
                self.extract_sentences(
                    sentence_batch,
                    mode=mode,
                    sentence_offset=batch_start,
                )
            )
        return combined


def _demo() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ner_llm.py \"Your text here\"")
        sys.exit(0)

    text = sys.argv[1]
    ner = LLMNER()

    print("\nPer-sentence entities:")
    for sent, ents in ner.extract(text, mode="sentences"):
        print(f"- {sent}\n  -> {sorted(ents)}")

    print("\nStructured entities:")
    print(json.dumps(ner.extract(text, mode="structured"), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _demo()
