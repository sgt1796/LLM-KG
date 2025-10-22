"""Document digestor.

This module provides a minimal document summariser.  Given a long
string, it extracts the first ``n`` sentences as a crude summary.  The
class is designed to be replaced by more sophisticated components
such as LLM-based summarisation via the ``PromptFunction`` from
the POP project.
"""

from __future__ import annotations

import re
from typing import List


class DocumentDigestor:
    """Produce a simple summary of the input text.

    Parameters
    ----------
    max_sentences : int
        The number of sentences to keep in the summary.
    """

    sentence_regex = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_sentences: int = 5) -> None:
        self.max_sentences = max_sentences

    def digest(self, text: str) -> str:
        """Return the first ``max_sentences`` sentences from ``text``.

        Parameters
        ----------
        text : str
            The full document text.

        Returns
        -------
        str
            A crude summary composed of the first ``max_sentences``
            sentences.  If fewer sentences are available, the entire
            text is returned.
        """
        # Normalise whitespace
        cleaned = " ".join(text.split())
        # Split into sentences using the regex; fallback to whole text
        sentences: List[str] = self.sentence_regex.split(cleaned) if cleaned else []
        if not sentences:
            return cleaned
        return " ".join(sentences[: self.max_sentences])