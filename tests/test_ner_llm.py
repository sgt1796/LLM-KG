import json
import unittest

from kg_pipeline.ner_llm import LLMNER


class FakePromptFunction:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, str):
            return response
        return json.dumps(response)


class FakeProposer:
    def __init__(self, proposals):
        self.proposals = proposals

    def extract_entities_from_sentence(self, sentence: str):
        return set(self.proposals.get(sentence, []))


class LLMNERTests(unittest.TestCase):
    def test_extract_sentences_keeps_surface_forms_and_structured_canonical_names(self) -> None:
        surface_fn = FakePromptFunction(
            [
                {
                    "sentences": [
                        {"id": 5, "mentions": ["IL-6", "Interleukin-6", "TNF"]},
                        {"id": 6, "mentions": []},
                    ]
                },
                {
                    "sentences": [
                        {"id": 5, "mentions": ["IL-6", "Interleukin-6", "TNF"]},
                        {"id": 6, "mentions": []},
                    ]
                },
            ]
        )
        normalizer_fn = FakePromptFunction(
            [
                {
                    "sentences": [
                        {
                            "id": 5,
                            "mentions": [
                                {
                                    "surface": "IL-6",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "interleukin-6",
                                },
                                {
                                    "surface": "TNF",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "tumor necrosis factor",
                                },
                                {
                                    "surface": "Interleukin-6",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "interleukin-6",
                                },
                            ],
                        }
                    ]
                },
                {
                    "sentences": [
                        {
                            "id": 5,
                            "mentions": [
                                {
                                    "surface": "IL-6",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "interleukin-6",
                                },
                                {
                                    "surface": "TNF",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "tumor necrosis factor",
                                },
                                {
                                    "surface": "Interleukin-6",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "interleukin-6",
                                },
                            ],
                        }
                    ]
                },
            ]
        )
        proposer = FakeProposer({"IL-6 inhibits TNF.": ["IL-6", "TNF"]})
        ner = LLMNER(
            client="ollama",
            proposer=proposer,
            surface_fn=surface_fn,
            normalizer_fn=normalizer_fn,
        )

        sentence_results = ner.extract_sentences(
            ["IL-6 inhibits TNF.", "No entities here."],
            mode="sentences",
            sentence_offset=5,
        )
        structured_results = ner.extract_sentences(
            ["IL-6 inhibits TNF.", "No entities here."],
            mode="structured",
            sentence_offset=5,
        )

        self.assertEqual(
            sentence_results,
            [
                ("IL-6 inhibits TNF.", {"IL-6", "TNF"}),
                ("No entities here.", set()),
            ],
        )
        self.assertEqual(structured_results[0]["sentence_id"], 5)
        self.assertEqual(structured_results[1]["sentence_id"], 6)
        self.assertEqual(
            structured_results[0]["entities"],
            [
                {
                    "surface": "IL-6",
                    "label": "GENE_PROTEIN",
                    "canonical_name": "interleukin-6",
                },
                {
                    "surface": "TNF",
                    "label": "GENE_PROTEIN",
                    "canonical_name": "tumor necrosis factor",
                },
            ],
        )

    def test_model_and_temperature_are_forwarded_to_both_stages(self) -> None:
        surface_fn = FakePromptFunction([{"sentences": [{"id": 0, "mentions": ["IL-6"]}]}])
        normalizer_fn = FakePromptFunction(
            [
                {
                    "sentences": [
                        {
                            "id": 0,
                            "mentions": [
                                {
                                    "surface": "IL-6",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "interleukin-6",
                                }
                            ],
                        }
                    ]
                }
            ]
        )
        proposer = FakeProposer({"IL-6 inhibits TNF.": ["IL-6"]})
        ner = LLMNER(
            client="ollama",
            model="custom-biomed-model",
            temperature=0.2,
            proposer=proposer,
            surface_fn=surface_fn,
            normalizer_fn=normalizer_fn,
        )

        ner.extract_sentences(["IL-6 inhibits TNF."], mode="sentences")

        self.assertEqual(surface_fn.calls[0]["model"], "custom-biomed-model")
        self.assertEqual(normalizer_fn.calls[0]["model"], "custom-biomed-model")
        self.assertEqual(surface_fn.calls[0]["temp"], 0.2)
        self.assertEqual(normalizer_fn.calls[0]["temp"], 0.2)

    def test_extract_batches_small_sentence_groups(self) -> None:
        surface_fn = FakePromptFunction(
            [
                {"sentences": [{"id": 0, "mentions": ["IL-6"]}, {"id": 1, "mentions": ["TNF"]}]},
                {"sentences": [{"id": 2, "mentions": ["STAT3"]}]},
            ]
        )
        normalizer_fn = FakePromptFunction(
            [
                {
                    "sentences": [
                        {
                            "id": 0,
                            "mentions": [
                                {
                                    "surface": "IL-6",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "interleukin-6",
                                }
                            ],
                        },
                        {
                            "id": 1,
                            "mentions": [
                                {
                                    "surface": "TNF",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "tumor necrosis factor",
                                }
                            ],
                        },
                    ]
                },
                {
                    "sentences": [
                        {
                            "id": 2,
                            "mentions": [
                                {
                                    "surface": "STAT3",
                                    "label": "GENE_PROTEIN",
                                    "canonical_name": "signal transducer and activator of transcription 3",
                                }
                            ],
                        }
                    ]
                },
            ]
        )
        proposer = FakeProposer(
            {
                "IL-6 inhibits TNF.": ["IL-6"],
                "TNF activates STAT3.": ["TNF"],
                "STAT3 regulates genes.": ["STAT3"],
            }
        )
        ner = LLMNER(
            client="ollama",
            proposer=proposer,
            surface_fn=surface_fn,
            normalizer_fn=normalizer_fn,
            sentence_batch_size=2,
            batch_char_budget=200,
        )

        results = ner.extract(
            "IL-6 inhibits TNF. TNF activates STAT3. STAT3 regulates genes.",
            mode="sentences",
        )

        self.assertEqual(len(surface_fn.calls), 2)
        self.assertEqual(len(normalizer_fn.calls), 2)
        self.assertEqual(results[0], ("IL-6 inhibits TNF.", {"IL-6"}))
        self.assertEqual(results[1], ("TNF activates STAT3.", {"TNF"}))
        self.assertEqual(results[2], ("STAT3 regulates genes.", {"STAT3"}))


if __name__ == "__main__":
    unittest.main()
