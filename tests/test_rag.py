import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kg_pipeline.rag import _load_graph, build_index
from kg_rag_app import KGGraphRegistry, KGRagState, build_incident_triples, create_app, create_app_from_env, load_graph


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


def _sample_graph_with_long_evidence() -> dict:
    graph = _sample_graph()
    graph["triples"][0]["sources"][0]["evidence"] = "Stress evidence " + ("long biomedical evidence. " * 30)
    return graph


def _write_graph(tmpdir: str, name: str = "graph.json") -> Path:
    path = Path(tmpdir) / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_sample_graph(), handle, ensure_ascii=False, indent=2)
    return path


class AleqRetrieverTests(unittest.TestCase):
    def test_alias_resolution_maps_acronym_to_full_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder())

            result = retriever.query("What causes ASD?", top_k=5, hop_limit=2)

            self.assertEqual(result.debug_scores["anchors"][0]["node_id"], "Autism Spectrum Disorder")
            self.assertEqual(result.debug_scores["relation_intent"]["relation"], "causes")
            self.assertEqual(result.triples[0]["relation"], "causes")
            self.assertEqual(result.triples[0]["object"], "Autism Spectrum Disorder")

    def test_inverted_relation_phrase_returns_correct_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder())

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
            build_index(graph_path, cache_root, embedder_one)
            self.assertEqual(embedder_one.calls, 3)

            embedder_two = FakeEmbedder()
            build_index(graph_path, cache_root, embedder_two)
            self.assertEqual(embedder_two.calls, 0)

    def test_aleq_reports_kge_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            cache_root = Path(tmpdir) / "cache"

            retriever = build_index(graph_path, cache_root, FakeEmbedder())

            self.assertFalse(retriever.kge.available)
            self.assertEqual(retriever.kge.metadata["status"], "disabled")
            result = retriever.query("What causes ASD?", top_k=5, hop_limit=2)
            self.assertEqual(result.triples[0]["relation"], "causes")

    def test_related_question_returns_direct_triple_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder())

            result = retriever.query("How is IL-6 related to TNF?", top_k=5, hop_limit=2)

            self.assertEqual(result.triples[0]["subject"], "IL-6")
            self.assertEqual(result.triples[0]["object"], "TNF")
            self.assertTrue(result.paths)
            self.assertIn("IL-6", result.paths[0]["nodes"])
            self.assertIn("TNF", result.paths[0]["nodes"])

    def test_open_ended_question_uses_triple_text_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder())

            result = retriever.query("Which drugs treat autism?", top_k=5, hop_limit=2)

            self.assertEqual(result.triples[0]["subject"], "Melatonin")
            self.assertEqual(result.triples[0]["relation"], "treats")

    def test_aleq_uses_project_schema_and_relation_lexicon(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(
                graph_path,
                Path(tmpdir) / "cache",
                FakeEmbedder(),
            )

            result = retriever.query("Autism Spectrum Disorder is treated with what?", top_k=5, hop_limit=2)

            self.assertEqual(retriever.method, "ALEQ")
            self.assertEqual(result.debug_scores["relation_intent"]["relation"], "treats")
            self.assertEqual(result.debug_scores["relation_intent"]["direction"], -1)
            self.assertEqual(result.triples[0]["subject"], "Melatonin")
            self.assertEqual(result.triples[0]["relation"], "treats")
            self.assertEqual(result.triples[0]["object"], "Autism Spectrum Disorder")
            self.assertTrue(result.focus_nodes)

    def test_reciprocal_triples_are_aggregated_without_losing_records(self) -> None:
        graph = {
            "nodes": ["A", "B"],
            "triples": [
                {"subject": "A", "relation": "activates", "object": "B", "weight": 2},
                {"subject": "B", "relation": "inhibits", "object": "A", "weight": 3},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "parallel.json"
            with open(graph_path, "w", encoding="utf-8") as handle:
                json.dump(graph, handle)

            structural_graph, node_ids, triples = _load_graph(graph_path)

            self.assertEqual(node_ids, ["A", "B"])
            self.assertEqual(len(triples), 2)
            self.assertEqual(structural_graph.number_of_edges(), 1)
            edge_data = structural_graph.get_edge_data("A", "B")
            self.assertEqual(edge_data["weight"], 5.0)
            self.assertEqual(edge_data["relations"], ["activates", "inhibits"])
            self.assertEqual(len(edge_data["raw_triples"]), 2)

            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder())
            self.assertEqual(len(retriever.triple_records), 2)
            best_id = retriever._best_edge_triple("A", "B")
            self.assertEqual(retriever.triple_records[best_id]["relation"], "inhibits")

            app_graph, _, _ = load_graph(graph_path)
            self.assertEqual(app_graph.get_edge_data("A", "B")["weight"], 5.0)


class FlaskRetrieverRegressionTests(unittest.TestCase):
    def test_no_graph_startup_serves_upload_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = KGGraphRegistry(Path(tmpdir) / "cache", embedder_factory=lambda _model: FakeEmbedder())
            app = create_app(registry)
            client = app.test_client()

            response = client.get("/")

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Upload graph JSON", response.data)
            self.assertIn(b'enctype="multipart/form-data"', response.data)

    def test_upload_valid_graph_redirects_to_graph_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = KGGraphRegistry(Path(tmpdir) / "cache", embedder_factory=lambda _model: FakeEmbedder())
            app = create_app(registry)
            client = app.test_client()

            payload = json.dumps(_sample_graph()).encode("utf-8")
            response = client.post(
                "/upload",
                data={"graph": (BytesIO(payload), "sample.json")},
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 302)
            self.assertIn("/graph/", response.headers["Location"])
            self.assertEqual(len(registry.records), 1)
            record = next(iter(registry.records.values()))
            self.assertTrue(record.graph_path.exists())

    def test_upload_invalid_graph_returns_error_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = KGGraphRegistry(Path(tmpdir) / "cache", embedder_factory=lambda _model: FakeEmbedder())
            app = create_app(registry)
            client = app.test_client()

            response = client.post(
                "/upload",
                data={"graph": (BytesIO(b"{not json"), "broken.json")},
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn(b"valid UTF-8 JSON", response.data)

    def test_graph_specific_query_uses_requested_graph_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            registry = KGGraphRegistry(Path(tmpdir) / "cache", embedder_factory=lambda _model: FakeEmbedder())
            record = registry.register_existing_graph(graph_path, graph_id="sample")
            app = create_app(registry)
            client = app.test_client()

            response = client.post(
                "/query",
                json={"graph_id": record.graph_id, "query": "What causes ASD?", "top_n": 5, "include_answer": False},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["graph_id"], "sample")
            self.assertEqual(payload["retriever_method"], "ALEQ")
            self.assertEqual(payload["triples"][0]["relation"], "causes")

    def test_query_defaults_to_aleq_without_method_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            registry = KGGraphRegistry(Path(tmpdir) / "cache", embedder_factory=lambda _model: FakeEmbedder())
            registry.register_existing_graph(graph_path, graph_id="sample")
            app = create_app(registry)
            client = app.test_client()

            default_response = client.post(
                "/api/search",
                json={"graph_id": "sample", "question": "What causes ASD?", "top_k": 5},
            )

            self.assertEqual(default_response.status_code, 200)
            self.assertEqual(default_response.get_json()["retriever_method"], "ALEQ")

    def test_provenance_endpoint_returns_limited_preview_and_full_node_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "graph.json"
            with open(graph_path, "w", encoding="utf-8") as handle:
                json.dump(_sample_graph_with_long_evidence(), handle)
            registry = KGGraphRegistry(Path(tmpdir) / "cache", embedder_factory=lambda _model: FakeEmbedder())
            registry.register_existing_graph(graph_path, graph_id="sample")
            app = create_app(registry)
            client = app.test_client()

            response = client.post(
                "/api/provenance",
                json={"graph_id": "sample", "type": "node", "node_id": "Stress"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertLessEqual(len(payload["preview"]), 240)
            self.assertTrue(payload["is_truncated"])
            self.assertIn("long biomedical evidence", payload["full_text"])

    def test_provenance_endpoint_returns_edge_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "graph.json"
            with open(graph_path, "w", encoding="utf-8") as handle:
                json.dump(_sample_graph_with_long_evidence(), handle)
            registry = KGGraphRegistry(Path(tmpdir) / "cache", embedder_factory=lambda _model: FakeEmbedder())
            registry.register_existing_graph(graph_path, graph_id="sample")
            app = create_app(registry)
            client = app.test_client()

            response = client.post(
                "/api/provenance",
                json={
                    "graph_id": "sample",
                    "type": "edge",
                    "from": "Stress",
                    "to": "Autism Spectrum Disorder",
                    "relation": "causes",
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertLessEqual(len(payload["preview"]), 240)
            self.assertEqual(payload["triples"][0]["relation"], "causes")
            self.assertIn("Stress --[causes]--> Autism Spectrum Disorder", payload["full_text"])

    def test_query_endpoint_returns_highlightable_focus_nodes_without_kge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder())
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
            self.assertIn("highlight", payload)
            self.assertTrue(payload["nodes"])
            self.assertIn(payload["nodes"][0]["id"], state.visible_node_ids)
            self.assertTrue(payload["highlight"]["node_ids"])
            self.assertTrue(payload["highlight"]["edges"])

    def test_agent_api_search_returns_structured_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            retriever = build_index(graph_path, Path(tmpdir) / "cache", FakeEmbedder())
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

            response = client.post("/api/search", json={"question": "What causes ASD?", "top_k": 5})

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertIn("triples", payload)
            self.assertIn("paths", payload)
            self.assertIn("context", payload)
            self.assertEqual(payload["triples"][0]["relation"], "causes")

    def test_create_app_from_env_uses_graph_and_cache_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = _write_graph(tmpdir)
            cache_dir = Path(tmpdir) / "cache"

            def fake_factory(model_name: str) -> FakeEmbedder:
                self.assertEqual(model_name, "fake-model")
                return FakeEmbedder()

            with patch.dict(
                "os.environ",
                {
                    "KG_GRAPH_PATH": str(graph_path),
                    "KG_CACHE_DIR": str(cache_dir),
                    "KG_OPENAI_EMBED_MODEL": "fake-model",
                },
                clear=False,
            ):
                app = create_app_from_env(embedder_factory=fake_factory)

            client = app.test_client()
            response = client.get("/healthz")
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["graph_path"], str(graph_path))
            self.assertEqual(payload["cache_dir"], str(cache_dir))

    def test_wsgi_module_exports_flask_app(self) -> None:
        sys.modules.pop("wsgi", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "KG_GRAPH_PATH": "",
                    "KG_CACHE_DIR": str(Path(tmpdir) / "cache"),
                    "KG_ENABLE_LLM_ANSWER": "false",
                },
                clear=False,
            ):
                module = importlib.import_module("wsgi")

        self.assertIsNotNone(module.app)
        self.assertTrue(hasattr(module.app, "test_client"))


if __name__ == "__main__":
    unittest.main()
