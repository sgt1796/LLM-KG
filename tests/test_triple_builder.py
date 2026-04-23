import tempfile
import unittest
from pathlib import Path

from kg_pipeline.label_store import LabelStore
from kg_pipeline.triple_builder import TripletKnowledgeGraphBuilder


class TripletKnowledgeGraphBuilderTests(unittest.TestCase):
    def test_passive_voice_inverts_direction(self) -> None:
        builder = TripletKnowledgeGraphBuilder()

        triples = builder.extract_triplets(
            "IL-6 is inhibited by dexamethasone.",
            {"IL-6", "dexamethasone"},
        )

        self.assertIn(("dexamethasone", "inhibits", "IL-6"), triples)
        self.assertNotIn(("IL-6", "inhibits", "dexamethasone"), triples)

    def test_treated_with_inverts_direction(self) -> None:
        builder = TripletKnowledgeGraphBuilder()

        triples = builder.extract_triplets(
            "Psoriasis is treated with methotrexate.",
            {"Psoriasis", "methotrexate"},
        )

        self.assertIn(("methotrexate", "treats", "Psoriasis"), triples)

    def test_repeated_mentions_use_all_occurrences(self) -> None:
        builder = TripletKnowledgeGraphBuilder()

        triples = builder.extract_triplets(
            "IL-6 inhibits TNF and IL-6 activates STAT3.",
            {"IL-6", "TNF", "STAT3"},
        )

        self.assertIn(("IL-6", "inhibits", "TNF"), triples)
        self.assertIn(("IL-6", "activates", "STAT3"), triples)

    def test_negated_relation_is_skipped(self) -> None:
        builder = TripletKnowledgeGraphBuilder()

        triples = builder.extract_triplets(
            "IL-6 is not associated with dexamethasone.",
            {"IL-6", "dexamethasone"},
        )

        self.assertEqual(triples, [])

    def test_hedged_relation_is_skipped(self) -> None:
        builder = TripletKnowledgeGraphBuilder()

        triples = builder.extract_triplets(
            "IL-6 may increase TNF.",
            {"IL-6", "TNF"},
        )

        self.assertEqual(triples, [])

    def test_label_store_is_observed_once_per_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LabelStore(Path(tmpdir) / "labels.json", min_promote=2)
            builder = TripletKnowledgeGraphBuilder(label_store=store)

            builder.add_sentence("IL-6 inhibits TNF.", {"IL-6", "TNF"})
            self.assertEqual(store.labels["inhibits"]["count"], 1)
            self.assertEqual(builder.to_dict()["triples"], [])

            builder.add_sentence("IL-6 inhibits TNF.", {"IL-6", "TNF"})

            self.assertEqual(store.labels["inhibits"]["count"], 2)
            triples = builder.to_dict()["triples"]
            self.assertEqual(len(triples), 1)
            self.assertEqual(triples[0]["weight"], 1)


if __name__ == "__main__":
    unittest.main()
