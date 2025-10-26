"""Placeholder for document chunking.

In a complete system long documents should be split into smaller
overlapping chunks before being embedded or processed by an LLM.
This module defines a simple interface and a trivial implementation
that returns the whole document as a single chunk.  Replace this
with your preferred chunking strategy (e.g. sentence/boundary
preserving split, token based fixed windows, etc.).
"""

from __future__ import annotations

import re
from typing import List

class DocumentChunker:
    """A trivial chunker that returns the entire document as one chunk."""

    def __init__(self, max_tokens: int = 512, overlap: int = 50) -> None:
        self.max_tokens = max_tokens
        self.overlap = overlap

    def no_chunk(self, text: str) -> List[str]:
        """Return the entire document as one chunk."""
        return [text]

    def chunk_by_sections(self, text: str) -> List[str]:
        """Chunking a journal article by sections."""
        section_headers = [
            "Abstract",
            "Introduction",
            "Methods",
            "Materials and Methods",
            "Results",
            "Discussion",
            "Conclusion",
            "References",
            "Acknowledgements",
        ]

        pattern = r'(?i)^\s*(' + '|'.join(section_headers) + r')\s*$'
        sections = re.split(pattern, text, flags=re.MULTILINE)

        chunked_sections = []
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            content = sections[i + 1].strip() if (i + 1) < len(sections) else ""
            if content and header not in ["References", "Acknowledgements"]:
                chunked_sections.append(f"{header}\n{content}")
        return chunked_sections

    def chunk_abstract_discussion(self, text: str) -> List[str]:
        """Extract only the Abstract and Discussion/Conclusion sections."""
        section_headers = [
            "Abstract",
            "Introduction",
            "Methods",
            "Materials and Methods",
            "Results",
            "Discussion",
            "Conclusion",
            "References",
            "Acknowledgements",
        ]
        pattern = r'(?i)^\s*(' + '|'.join(section_headers) + r')\s*$'
        sections = re.split(pattern, text, flags=re.MULTILINE)

        selected = []
        for i in range(1, len(sections), 2):
            header = sections[i].strip().lower()
            content = sections[i + 1].strip() if (i + 1) < len(sections) else ""
            if header in {"abstract", "discussion", "conclusion"} and content:
                selected.append(f"{header.title()}\n{content}")

        # fallback: if no section headers detected, take first and last 10%
        if not selected and len(text) > 1000:
            n = len(text)
            selected = [text[: n // 10], text[-n // 10 :]]
        return selected
