"""Top level package for the knowledge‑graph extraction pipeline.

This module exposes the high‑level classes for data acquisition,
document digestion, named entity extraction, knowledge graph
construction, and merging.  Each component is deliberately kept
lightweight to minimise external dependencies.
"""

from .data_acquisition import DataAcquisition  # noqa: F401
from .digestor import DocumentDigestor         # noqa: F401
from .ner_simple import NERExtractor           # noqa: F401
from .ner import SpacyNER  # type: ignore  # noqa: F401
from .tuple_builder import WeightedTupleBuilder  # noqa: F401
from .tuple_merger import WeightedTupleMerger    # noqa: F401

# Triplet‑based builder and merger for subject–relation–object graphs
from .triple_builder import TripletKnowledgeGraphBuilder  # noqa: F401
from .triple_merger import TripletGraphMerger            # noqa: F401

__all__ = [
    "DataAcquisition",
    "DocumentDigestor",
    "NERExtractor",
    "SpacyNER",
    "WeightedTupleBuilder",
    "WeightedTupleMerger",
    "TripletKnowledgeGraphBuilder",
    "TripletGraphMerger",
]