"""Module for acquiring and converting source documents into plain text.

At present this module supports PDF documents via the
`pdftotext` command from Poppler.  The extractor reads the
specified PDF and returns its textual content as a single string.

If additional document formats are required (e.g. HTML, Word
documents), extend the :meth:`DataAcquisition.read` method with
appropriate handlers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class DataAcquisition:
    """Convert source documents into plain text.

    Parameters
    ----------
    source : str or Path
        The path to the document to be processed.
    """

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source)

    def pdf_to_text(self, pdf_path: str | Path) -> str:
        """Convert a PDF file into plain text.

        This method invokes the external ``pdftotext`` utility which
        must be available on the host.  If the command fails an
        exception is raised.

        Parameters
        ----------
        pdf_path : str or Path
            Path to the PDF file.

        Returns
        -------
        str
            The extracted text.
        """
        pdf = Path(pdf_path)
        if not pdf.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf}")
        # Call pdftotext and capture stdout (- option writes to stdout)
        result = subprocess.run(
            ["pdftotext", str(pdf), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pdftotext failed for {pdf} with code {result.returncode}:\n{result.stderr}"
            )
        return result.stdout

    def read(self) -> str:
        """Read the source document into plain text.

        Currently only PDF documents are supported.  Additional
        formats can be handled by extending this method.
        """
        if self.source.suffix.lower() == ".pdf":
            return self.pdf_to_text(self.source)
        else:
            raise NotImplementedError(
                f"Unsupported file type: {self.source.suffix}. Only PDF is supported for now."
            )