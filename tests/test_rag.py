import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kg_pipeline.rag import build_index
from kg_rag_app import KGRagState, build_incident_triples, create_app


def _tokenize(text: str) -> list[str]:
    raw = [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token]
    tokens: list[str] = []
    for token in raw:
        tokens.append(token)
        stem = token
        if stem.endswith("ies") and len(stem) > 4:
            stem = stem[:-3] + "y"
        elif stem.endswith("ing") and len(stem) > 5:
            stem = stem[:-3]
        elif stem.endswith("ed") and len(stem) > 4:
            stem = stem[:-2]
        elif stem.endswith("s") and len(stem) > 3:
            stem = stem[:-1]
        if stem != token:
            tokens.append(stem)
    return tokens


class FakeEmbedder:
    def __init__(self, dims: int = 48) -> None:
        self.dims = dims
        self.calls = 0

    def get_embedding(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.vstack([self._encode(text) for text in texts]).astype("f")

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dims, dtype="f")
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dims
            vec[bucket] += 1.0
        return vec


def _sample_graph() -> dict:
    return {
        "nodes": [
            "Autism Spectrum Disorder",
            "Melatonin",
            "Stress",
            "IL-6",
            "TNF",
            "Cortisol",
        ],
        "triples": [
            {
                "subject": "Stress",
                "relation": "causes",
                "object": "Autism Spectrum Disorder",
                "weight": 3,
                "sources": [
                    {
                        "evidence": "Stress causes autism spectrum disorder in the synthetic example.",
                        "doc_meta": {"filename": "paper-a.pdf"},
                    }
                ],
            },
            {
                "subject": "Melatonin",
                "relation": "treats",
                "object": "Autism Spectrum Disorder",
                "weight": 2,
                "sources": [
                    {
                        "evidence": "Melatonin treats autism spectrum disorder symptoms.",
                        "doc_meta": {"filename": "paper-b.pdf"},
                    }
                ],
            },
            {
                "subject": "IL-6",
                "relation": "regulates",
                "object": "TNF",
                "weight": 4,
                "sources": [
                    {
                        "evidence": "IL-6 regulates TNF signaling.",
                        "doc_meta": {"filename": "paper-c.pdf"},
                    }
                ],
            },
            {
                "subject": "Cortisol",
                "relation": "associated with",
                "object": "Stress",
                "weight": 1,
                "sources": [
                    {
                        "evidence": "Cortisol is associated with stress.",
                        "doc_meta": {"filename": "paper-d.pdf"},
                    }
                ],
            },
        ],
    }


def _write_graph(tmpdir: str, name: str = "graph.json") -> Path:
    path = Path(tmpdir) / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_sample_graph(), handle, ensure_ascii=False, indent=2)
    return path


class HybridRetrieverTests(unittest.TestCase):
    def test_alias_resolution_maps_acronym_to_full_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder(), kge_enabled=False)

            result = retriever.query("What causes ASD?", top_k=5, hop_limit=2)

            self.assertEqual(result.debug_scores["anchors"][0]["node_id"], "Autism Spectrum Disorder")
            self.assertEqual(result.debug_scores["relation_intent"]["relation"], "causes")
            self.assertEqual(result.triples[0]["relation"], "causes")
            self.assertEqual(result.triples[0]["object"], "Autism Spectrum Disorder")

    def test_inverted_relation_phrase_returns_correct_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder(), kge_enabled=False)

            result = retriever.query("Autism Spectrum Disorder is treated with what?", top_k=5, hop_limit=2)

            self.assertEqual(result.debug_scores["relation_intent"]["relation"], "treats")
            self.assertEqual(result.debug_scores["relation_intent"]["direction"], -1)
            self.assertEqual(result.triples[0]["subject"], "Melatonin")
            self.assertEqual(result.triples[0]["relation"], "treats")
            self.assertEqual(result.triples[0]["object"], "Autism Spectrum Disorder")

    def test_cache_reload_reuses_existing_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            cache_root = Path(tmpdir) / "cache"

            embedder_one = FakeEmbedder()
            build_index(graph_path, cache_root, embedder_one, kge_enabled=False)
            self.assertEqual(embedder_one.calls, 3)

            embedder_two = FakeEmbedder()
            build_index(graph_path, cache_root, embedder_two, kge_enabled=False)
            self.assertEqual(embedder_two.calls, 0)

    def test_kge_unavailable_falls_back_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            cache_root = Path(tmpdir) / "cache"

            with patch("kg_pipeline.rag._train_rotate_embeddings", side_effect=ImportError("PyKEEN is not installed")):
                retriever = build_index(graph_path, cache_root, FakeEmbedder(), kge_enabled=True)

            self.assertFalse(retriever.kge.available)
            self.assertEqual(retriever.kge.metadata["status"], "unavailable")
            result = retriever.query("What causes ASD?", top_k=5, hop_limit=2)
            self.assertEqual(result.triples[0]["relation"], "causes")

    def test_related_question_returns_direct_triple_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder(), kge_enabled=False)

            result = retriever.query("How is IL-6 related to TNF?", top_k=5, hop_limit=2)

            self.assertEqual(result.triples[0]["subject"], "IL-6")
            self.assertEqual(result.triples[0]["object"], "TNF")
            self.assertTrue(result.paths)
            self.assertIn("IL-6", result.paths[0]["nodes"])
            self.assertIn("TNF", result.paths[0]["nodes"])

    def test_open_ended_question_uses_triple_text_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder(), kge_enabled=False)

            result = retriever.query("Which drugs treat autism?", top_k=5, hop_limit=2)

            self.assertEqual(result.triples[0]["subject"], "Melatonin")
            self.assertEqual(result.triples[0]["relation"], "treats")


class FlaskRetrieverRegressionTests(unittest.TestCase):
    def test_query_endpoint_returns_highlightable_focus_nodes_without_kge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder(), kge_enabled=False)
            node_ids = [record["id"] for record in retriever.node_records]
            state = KGRagState(
                graph=retriever.graph,
                retriever=retriever,
                graph_path=graph_path,
                cache_dir=Path(tmpdir) / "cache",
                incident_triples=build_incident_triples(node_ids, retriever.triples),
                visible_node_ids=node_ids,
            )
            app = create_app(state)
            client = app.test_client()

            with patch("kg_rag_app.call_llm_answer", return_value="synthetic answer"):
                response = client.post("/query", json={"query": "What causes ASD?", "top_n": 5})

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertIn("nodes", payload)
            self.assertIn("triples", payload)
            self.assertIn("paths", payload)
            self.assertTrue(payload["nodes"])
            self.assertIn(payload["nodes"][0]["id"], state.visible_node_ids)


if __name__ == "__main__":
    unittest.main()
