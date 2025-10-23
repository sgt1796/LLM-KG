"""Heuristic named entity extraction.

This module implements a very simple NER component without relying on
external NLP libraries.  Entities are identified using regular
expressions that look for capitalised phrases, uppercase acronyms and
mixed alphanumeric tokens (common for genes and chemical names).

For real biomedical applications you should replace this with a
domain‑specific NER model such as SciSpacy or BioBERT.  However,
because the runtime environment does not permit installing large
external packages, these heuristics provide a lightweight baseline.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set, Tuple
import sys
import json


class NERExtractor:
    """Extract simple entities from text using regex heuristics."""

    # Regex patterns for different entity types
    CAPITALISED_PHRASE = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    )  # e.g. "Large Language Model"
    UPPERCASE_ACRONYM = re.compile(r"\b([A-Z]{2,})\b")  # e.g. "LLM", "SES"
    ALPHANUMERIC = re.compile(
        r"\b([A-Za-z]+\d+[A-Za-z]*)\b"
    )  # e.g. "BRCA1", "H2O"

    SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, min_len: int = 2) -> None:
        """Initialise the extractor.

        Parameters
        ----------
        min_len : int
            Minimum length of an entity (in characters) to keep.  Very
            short tokens tend to be stop words or punctuation.
        """
        self.min_len = min_len

    def normalise(self, entity: str) -> str:
        """Normalise entity text by stripping punctuation and excess whitespace."""
        # Remove trailing punctuation
        cleaned = re.sub(r"[\-–,.;:]+$", "", entity.strip())
        return cleaned

    def extract_entities_from_sentence(self, sentence: str) -> Set[str]:
        """Extract entities from a single sentence.

        Returns a set of unique entity strings.
        """
        entities: Set[str] = set()
        # Search for capitalised multi‑word phrases
        for match in self.CAPITALISED_PHRASE.finditer(sentence):
            ent = self.normalise(match.group(1))
            if len(ent) >= self.min_len:
                entities.add(ent)
        # Search for uppercase acronyms
        for match in self.UPPERCASE_ACRONYM.finditer(sentence):
            ent = self.normalise(match.group(1))
            if len(ent) >= self.min_len:
                entities.add(ent)
        # Search for alphanumeric tokens
        for match in self.ALPHANUMERIC.finditer(sentence):
            ent = self.normalise(match.group(1))
            if len(ent) >= self.min_len:
                entities.add(ent)
        return entities

    def extract(self, text: str) -> List[Tuple[str, Set[str]]]:
        """Extract entities grouped by sentence.

        Parameters
        ----------
        text : str
            Full document text.

        Returns
        -------
        list of tuples
            Each tuple contains the original sentence string and the set
            of extracted entities in that sentence.
        """
        # Normalise whitespace and split into sentences
        cleaned = " ".join(text.split())
        sentences = self.SENTENCE_SPLIT.split(cleaned) if cleaned else []
        results: List[Tuple[str, Set[str]]] = []
        for sent in sentences:
            ents = self.extract_entities_from_sentence(sent)
            if ents:
                results.append((sent, ents))
        return results


def _demo() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ner_simple.py \"Your text here\"")
        sys.exit(0)

    text = sys.argv[1]
    ner = NERExtractor()
    results = ner.extract(text)

    print("\nPer-sentence entities:")
    for sent, ents in results:
        print(f"- {sent}\n  -> {sorted(ents)}")

    print("\nStructured entities:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _demo()