import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post_kg.miner import discover_hypotheses, render_markdown
from post_kg.reviewer import review_candidates


def _graph() -> dict:
    return {
        "nodes": [
            "Maternal immune activation",
            "GABA signaling",
            "Autism Spectrum Disorder",
            "Stress",
            "Cortisol",
            "0.5 mg",
        ],
        "triples": [
            {
                "subject": "Maternal immune activation",
                "relation": "decreases",
                "object": "GABA signaling",
                "weight": 3,
                "sources": [
                    {
                        "evidence": "Maternal immune activation decreases GABA signaling in this example.",
                        "doc_meta": {"filename": "paper-a.pdf"},
                    },
                    {
                        "evidence": "A second paper also links immune activation with reduced GABA signaling.",
                        "doc_meta": {"filename": "paper-b.pdf"},
                    },
                ],
            },
            {
                "subject": "GABA signaling",
                "relation": "regulates",
                "object": "Autism Spectrum Disorder",
                "weight": 2,
                "sources": [
                    {
                        "evidence": "GABA signaling regulates autism spectrum disorder phenotypes.",
                        "doc_meta": {"filename": "paper-c.pdf"},
                    }
                ],
            },
            {
                "subject": "Stress",
                "relation": "increases",
                "object": "Cortisol",
                "weight": 4,
                "sources": [
                    {
                        "evidence": "Stress increases cortisol.",
                        "doc_meta": {"filename": "paper-d.pdf"},
                    }
                ],
            },
            {
                "subject": "Cortisol",
                "relation": "causes",
                "object": "Autism Spectrum Disorder",
                "weight": 2,
                "sources": [
                    {
                        "evidence": "Cortisol causes autism spectrum disorder phenotypes in this synthetic graph.",
                        "doc_meta": {"filename": "paper-e.pdf"},
                    }
                ],
            },
            {
                "subject": "Stress",
                "relation": "associated with",
                "object": "Autism Spectrum Disorder",
                "weight": 5,
                "sources": [
                    {
                        "evidence": "Stress is associated with autism spectrum disorder.",
                        "doc_meta": {"filename": "paper-f.pdf"},
                    }
                ],
            },
            {
                "subject": "0.5 mg",
                "relation": "treats",
                "object": "GABA signaling",
                "weight": 1,
                "sources": [{"evidence": "A dose statement should not become a hypothesis seed."}],
            },
        ],
    }


class HypothesisDiscoveryTests(unittest.TestCase):
    def test_focus_mines_indirect_mechanistic_hypothesis(self) -> None:
        candidates = discover_hypotheses(_graph(), focus_terms=["GABA", "ASD"], top_k=5)

        self.assertTrue(candidates)
        top = candidates[0]
        self.assertIn("GABA signaling", top.hypothesis)
        self.assertIn("Autism Spectrum Disorder", top.hypothesis)
        self.assertIsNone(top.direct_connection)
        self.assertGreater(top.components["novelty"], 0.9)

    def test_existing_direct_connection_reduces_novelty(self) -> None:
        candidates = discover_hypotheses(_graph(), focus_terms=["Stress", "ASD"], top_k=5)
        stress_candidate = next(item for item in candidates if item.path[0]["subject"] == "Stress")

        self.assertIsNotNone(stress_candidate.direct_connection)
        self.assertLess(stress_candidate.components["novelty"], 0.3)

    def test_markdown_explains_method_and_evidence(self) -> None:
        candidates = discover_hypotheses(_graph(), focus_terms=["GABA"], top_k=1)
        report = render_markdown(candidates, graph_name="synthetic.json")

        self.assertIn("Method:", report)
        self.assertIn("KG path:", report)
        self.assertIn("Evidence:", report)

    def test_reviewer_rewrites_and_prioritizes_testable_candidate(self) -> None:
        candidates = discover_hypotheses(_graph(), focus_terms=["GABA", "ASD"], top_k=1)
        reviewed = review_candidates([candidate.__dict__ for candidate in candidates])

        self.assertEqual(reviewed[0].decision, "advance")
        self.assertIn(reviewed[0].category, {"biomarker", "mechanism"})
        self.assertTrue(reviewed[0].next_actions)
        self.assertTrue(reviewed[0].web_queries)
        self.assertIn("perplexity_search", reviewed[0].agent_task["recommended_tools"])


if __name__ == "__main__":
    unittest.main()
