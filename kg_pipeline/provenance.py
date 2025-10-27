# kg_pipeline/provenance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import hashlib
from pathlib import Path

def compute_doc_id(path: str | Path, *, extra: str = "") -> str:
    p = Path(path)
    # Stable content-agnostic hash: path, size, mtime (fast); switch to bytes hash if you prefer
    stat = p.stat()
    h = hashlib.sha256()
    h.update(str(p.resolve()).encode("utf-8"))
    h.update(str(stat.st_size).encode("utf-8"))
    h.update(str(int(stat.st_mtime)).encode("utf-8"))
    if extra:
        h.update(extra.encode("utf-8"))
    return h.hexdigest()[:16]

@dataclass
class DocMeta:
    title: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None

@dataclass
class DocContext:
    doc_id: str
    doc_meta: Dict[str, Any]
    chunk_id: int
    page_hint: Optional[int] = None

@dataclass
class Evidence:
    doc_id: str
    doc_meta: Dict[str, Any]
    chunk_id: int
    sentence_id: int
    page: Optional[int]
    char_span: Tuple[int, int]
    evidence: str
    confidence: float = 1.0

    def dedupe_key(self) -> Tuple:
        return (self.doc_id, self.chunk_id, self.sentence_id, self.char_span)
