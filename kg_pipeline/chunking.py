"""Placeholder for document chunking.

In a complete system long documents should be split into smaller
overlapping chunks before being embedded or processed by an LLM.
This module defines a simple interface and a trivial implementation
that returns the whole document as a single chunk.  Replace this
with your preferred chunking strategy (e.g. sentence/boundary
preserving split, token based fixed windows, etc.).
"""

from __future__ import annotations

from typing import List


class DocumentChunker:
    """A trivial chunker that returns the entire document as one chunk."""

    def __init__(self, max_tokens: int = 512, overlap: int = 50) -> None:
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Return a list containing the original text.  Override this
        method to implement real chunking logic."""
        return [text]