# -*- coding: utf-8 -*-
"""
spaCy-based Named Entity Recognition (NER).

This module replaces the heuristic extractor with a robust spaCy pipeline.
It preserves the previous public methods so other parts of the project can
import and use it without changes:

- extract_entities_from_sentence(sentence) -> Set[str]
- process_text(text) -> List[Tuple[str, Set[str]]]

Extras:
- get_entities(text, return_spans=False): richer structured output
- Auto model selection with sensible fallbacks (SciSpaCy if available)
- Optional label filtering (e.g., keep only PERSON, ORG, GPE, etc.)
- Simple CLI: `python ner.py "Your text here"`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Dict, Any

import sys
import re

try:
    import spacy
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "spaCy is required for this module. Please install it with:\n"
        "  pip install spacy\n"
        "and download a model, e.g.:\n"
        "  python -m spacy download en_core_web_sm\n"
    ) from e


# ---------------------------
# Model auto-selection
# ---------------------------

_DEFAULT_MODEL_CANDIDATES: Sequence[str] = (
    # Prefer SciSpaCy small biomedical model if present
    "en_core_sci_sm",           # scispacy (if installed)
    "en_core_web_trf",          # spaCy transformer (if available)
    "en_core_web_md",           # medium English model
    "en_core_web_sm",           # small English model (fallback)
)


def _first_available_model(candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        try:
            spacy.util.get_package_path(name)  # type: ignore[attr-defined]
            return name
        except Exception:
            # Not installed
            continue
    return None


@dataclass
class SpacyNERConfig:
    model: Optional[str] = None
    # If set, keep only these entity labels (e.g., {"PERSON","ORG","GPE"}).
    keep_labels: Optional[Set[str]] = None
    # If True, collapse overlapping entities by preferring the longest span.
    prefer_longest: bool = True
    # Disable unused components for speed
    disable: Sequence[str] = ()


class SpacyNER:
    """
    Thin wrapper around spaCy's NER with a stable interface for this project.
    """

    SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    def __init__(self, config: Optional[SpacyNERConfig] = None) -> None:
        self.config = config or SpacyNERConfig()
        model_name = self.config.model or _first_available_model(_DEFAULT_MODEL_CANDIDATES)
        if not model_name:
            raise RuntimeError(
                "No spaCy model found. Please install one, e.g.:\n"
                "  python -m spacy download en_core_web_sm\n"
                "Or install a SciSpaCy model for biomedical text, e.g. en_core_sci_sm."
            )
        try:
            self.nlp = spacy.load(model_name, disable=list(self.config.disable))
            print(f"[SpacyNER] Loaded spaCy model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load spaCy model '{model_name}': {e}")
        self.model_name = model_name

    # ---------------------------
    # Public API (compatible with old module)
    # ---------------------------

    def extract_entities_from_sentence(self, sentence: str) -> Set[str]:
        """
        Return a set of unique entity strings found in `sentence`.
        """
        doc = self.nlp(sentence)
        ents = self._filter_and_dedup(doc.ents)
        return {ent.text.strip() for ent in ents}

    def process_text(self, text: str) -> List[Tuple[str, Set[str]]]:
        """
        Split `text` into sentences and extract a set of entities for each.
        Returns a list of (sentence, entities) pairs.
        """
        cleaned = " ".join(text.split())
        sentences = self.SENTENCE_SPLIT.split(cleaned) if cleaned else []
        results: List[Tuple[str, Set[str]]] = []
        for sent in sentences:
            ents = self.extract_entities_from_sentence(sent)
            if ents:
                results.append((sent, ents))
        return results

    def extract(self, text: str, mode: str = "flat", return_spans: bool = False, keep_labels: Optional[Set[str]] = None):
        """
        Unified extractor for compatibility with older pipelines expecting `.extract()`.

        Parameters
        ----------
        text : str
            Input sentence or multi-sentence text.
        mode : {"flat", "sentences", "structured"}
            - "flat": return a Set[str] of unique entity texts over the whole input.
            - "sentences": return List[Tuple[sentence, Set[str]]] like `process_text`.
            - "structured": return List[Dict] with offsets and labels (like `get_entities`).
        return_spans : bool
            Only used when mode == "structured". If True, includes spaCy Span objects under "_span".
        keep_labels : Optional[Set[str]]
            Optional override of labels to keep for this call only.

        Returns
        -------
        Set[str] | List[Tuple[str, Set[str]]] | List[Dict[str, Any]]
        """
        # Optionally override labels just for this call
        original_keep = self.config.keep_labels
        if keep_labels is not None:
            self.config.keep_labels = set(keep_labels)

        try:
            if mode == "sentences":
                return self.process_text(text)
            elif mode == "structured":
                return self.get_entities(text, return_spans=return_spans)
            else:  # "flat"
                # Union of sentence-level entities across the whole text
                per_sent = self.process_text(text)
                flat: Set[str] = set()
                for _, ents in per_sent:
                    flat.update(ents)
                return flat
        finally:
            # Restore original label filter
            self.config.keep_labels = original_keep

    # ---------------------------
    # Richer helpers
    # ---------------------------

    def get_entities(self, text: str, return_spans: bool = False) -> List[Dict[str, Any]]:
        """
        Return structured entities with offsets and labels for the whole text.

        Parameters
        ----------
        text : str
            Input text to analyze.
        return_spans : bool
            If True, also include the spaCy Span object under key "_span".

        Returns
        -------
        List[Dict[str, Any]]
            Each item has: {"text","label","start","end","sentence_id"} (+ "_span" if requested)
        """
        doc = self.nlp(text)
        entities = self._filter_and_dedup(doc.ents, dedup=False)
        out: List[Dict[str, Any]] = []
        sent_index_of_token = {}
        for i, sent in enumerate(doc.sents):
            for token in sent:
                sent_index_of_token[token.i] = i

        for ent in entities:
            item = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "sentence_id": sent_index_of_token.get(ent.start, 0),
            }
            if return_spans:
                item["_span"] = ent
            out.append(item)
        return out

    # ---------------------------
    # Internals
    # ---------------------------

    def _filter_and_dedup(self, ents: Iterable["spacy.tokens.Span"], dedup: bool = True) -> List["spacy.tokens.Span"]:
        # Label filtering
        if self.config.keep_labels is not None:
            ents = [e for e in ents if e.label_ in self.config.keep_labels]
        else:
            ents = list(ents)

        # Prefer longest spans when overlapping
        if self.config.prefer_longest and ents:
            ents = self._prefer_longest_non_overlapping(ents)

        # Deduplicate by (start,end,label)
        if dedup:
            seen = set()
            uniq = []
            for e in ents:
                key = (e.start_char, e.end_char, e.label_)
                if key not in seen:
                    seen.add(key)
                    uniq.append(e)
            return uniq
        return ents

    @staticmethod
    def _prefer_longest_non_overlapping(ents: List["spacy.tokens.Span"]) -> List["spacy.tokens.Span"]:
        # Sort by span length descending, then keep iff non-overlapping with previously kept
        ents_sorted = sorted(ents, key=lambda e: (e.end_char - e.start_char), reverse=True)
        kept: List["spacy.tokens.Span"] = []
        for e in ents_sorted:
            if all(e.end_char <= k.start_char or e.start_char >= k.end_char for k in kept):
                kept.append(e)
        # Return in document order
        return sorted(kept, key=lambda e: e.start_char)


# ---------------------------
# CLI
# ---------------------------

def _demo() -> None:
    import json
    if len(sys.argv) < 2:
        print("Usage: python ner.py \"Your text here\" [--labels PERSON,ORG,GPE]")
        sys.exit(0)

    text = sys.argv[1]
    keep_labels: Optional[Set[str]] = None
    if "--labels" in sys.argv:
        idx = sys.argv.index("--labels")
        if idx + 1 < len(sys.argv):
            keep_labels = set(sys.argv[idx + 1].split(","))

    ner = SpacyNER(SpacyNERConfig(keep_labels=keep_labels))
    print(f"[spaCy model: {ner.model_name}]")
    results = ner.process_text(text)
    print("\nPer-sentence entities:")
    for sent, ents in results:
        print(f"- {sent}\n  -> {sorted(ents)}")

    print("\nStructured entities:")
    print(json.dumps(ner.get_entities(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _demo()
