import json
import unittest
from pathlib import Path

from fashion_rag.core.intent import (
    CERTAINTY_CLARIFICATION_REQUIRED,
    CERTAINTY_DETERMINISTIC,
    CERTAINTY_LLM_ASSISTED,
    coerce_intent_decision,
    route_from_keywords,
)
from fashion_rag.core.security import CommerceFactStreamFilter


CASES_FILE = Path(__file__).with_name("router_eval_cases.jsonl")


class RouterEvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [
            json.loads(line)
            for line in CASES_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_dataset_contains_a_few_dozen_cases(self):
        self.assertGreaterEqual(len(self.cases), 40)

    def test_high_precision_router_cases(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                decision = route_from_keywords(
                    case["query"],
                    has_image=case.get("has_image", False),
                )
                if case.get("fast_match") is False:
                    self.assertIsNone(decision)
                    continue
                self.assertIsNotNone(decision)
                self.assertEqual(decision.intent, case["intent"])
                self.assertEqual(decision.action, case["action"])
                self.assertEqual(decision.route, case["route"])
                if "missing_slot" in case:
                    self.assertIn(case["missing_slot"], decision.missing_slots)

    def test_llm_confidence_and_slots_do_not_control_policy(self):
        decision = coerce_intent_decision(
            {
                "intent": "product_discovery",
                "action": "search",
                "confidence": 0.999,
                "missing_slots": ["budget", "favorite_brand"],
            },
            "tìm một chiếc áo đẹp",
        )
        self.assertEqual(decision.certainty, CERTAINTY_LLM_ASSISTED)
        self.assertEqual(decision.confidence, 0.0)
        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.missing_slots, [])

    def test_unknown_llm_result_uses_python_slot_policy(self):
        decision = coerce_intent_decision(
            {"intent": "unknown", "action": "clarify", "missing_slots": ["anything"]},
            "mình muốn cái đẹp đẹp",
        )
        self.assertEqual(decision.certainty, CERTAINTY_CLARIFICATION_REQUIRED)
        self.assertEqual(decision.missing_slots, ["user_goal"])

    def test_deterministic_decision_has_auditable_certainty(self):
        decision = route_from_keywords("phối đồ đi làm")
        self.assertEqual(decision.certainty, CERTAINTY_DETERMINISTIC)

    def test_commerce_fact_filter_handles_split_stream_chunks(self):
        output_filter = CommerceFactStreamFilter()
        safe = "".join(
            output_filter.feed(chunk)
            for chunk in [
                "1. **Áo sơ mi trắng**\n- Mã ",
                "SP: FAKE-123\n- Giá: 1 đồng\n- Vì sao hợp: thanh lịch\n",
            ]
        ) + output_filter.finish()
        self.assertIn("Áo sơ mi trắng", safe)
        self.assertIn("Vì sao hợp", safe)
        self.assertNotIn("FAKE-123", safe)
        self.assertNotIn("1 đồng", safe)
        self.assertEqual(len(output_filter.removed_lines), 2)


if __name__ == "__main__":
    unittest.main()
