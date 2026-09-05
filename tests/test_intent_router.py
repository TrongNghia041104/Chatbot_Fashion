import unittest

from app.core.intent import (
    INTENT_OUTFIT_ADVICE,
    INTENT_PRODUCT_DISCOVERY,
    INTENT_PROFILE_ANALYSIS,
    INTENT_PROFILE_MANAGEMENT,
    INTENT_SOCIAL,
    MODALITY_IMAGE,
    ROUTE_IMAGE_OUTFIT_ADVICE,
    ROUTE_IMAGE_PRODUCT_SEARCH,
    ROUTE_PROFILE_STATE_HANDLER,
    ROUTE_PROFILE_VLM_ANALYSIS,
    ROUTE_SOCIAL_RESPONSE,
    ROUTE_TEXT_OUTFIT_ADVICE,
    ROUTE_TEXT_PRODUCT_SEARCH,
    resolve_route,
    route_user_request,
)


class IntentRouterTests(unittest.TestCase):
    def test_route_is_derived_from_semantics(self):
        self.assertEqual(
            resolve_route(INTENT_PRODUCT_DISCOVERY, "text", "search"),
            ROUTE_TEXT_PRODUCT_SEARCH,
        )
        self.assertEqual(
            resolve_route(INTENT_OUTFIT_ADVICE, "text_image", "style_image_item"),
            ROUTE_IMAGE_OUTFIT_ADVICE,
        )

    def test_clear_text_routes_do_not_need_llm(self):
        cases = {
            "tìm áo sơ mi trắng": (INTENT_PRODUCT_DISCOVERY, ROUTE_TEXT_PRODUCT_SEARCH),
            "phối đồ đi làm": (INTENT_OUTFIT_ADVICE, ROUTE_TEXT_OUTFIT_ADVICE),
            "xin chào": (INTENT_SOCIAL, ROUTE_SOCIAL_RESPONSE),
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                decision = route_user_request(query)
                self.assertEqual((decision.intent, decision.route), expected)
                self.assertNotEqual(decision.source, "llm")

    def test_image_only_requires_visual_understanding_first(self):
        decision = route_user_request("", has_image=True)
        self.assertEqual(decision.action, "inspect_image")
        self.assertIn("image_context", decision.missing_slots)

    def test_clear_fashion_item_defaults_to_similar_search_and_cta(self):
        decision = route_user_request(
            "",
            has_image=True,
            image_context={
                "subject": "product",
                "caption": "một áo blazer đen",
                "fashion_item": "áo blazer đen",
                "confidence": 0.91,
            },
        )
        self.assertEqual(decision.modality, MODALITY_IMAGE)
        self.assertEqual(decision.route, ROUTE_IMAGE_PRODUCT_SEARCH)
        self.assertTrue(decision.follow_up_question)
        self.assertEqual(decision.follow_up_options[0]["action"], "style_image_item")

    def test_unclear_image_asks_with_options(self):
        decision = route_user_request(
            "",
            has_image=True,
            image_context={"subject": "unclear", "caption": "một người", "confidence": 0.2},
        )
        self.assertTrue(decision.needs_clarification)
        self.assertIsNone(decision.route)
        self.assertEqual(len(decision.clarification_options), 3)

    def test_explicit_image_requests(self):
        outfit = route_user_request("phối đồ với áo này", has_image=True)
        profile = route_user_request("phân tích dáng người", has_image=True)
        compound = route_user_request("phân tích dáng người rồi phối đồ", has_image=True)
        self.assertEqual(outfit.route, ROUTE_IMAGE_OUTFIT_ADVICE)
        self.assertEqual(profile.route, ROUTE_PROFILE_VLM_ANALYSIS)
        self.assertEqual(compound.intent, INTENT_PROFILE_ANALYSIS)
        self.assertEqual(len(compound.workflow), 2)

    def test_identify_image_item_uses_vlm_then_existing_image_search_route(self):
        before_vision = route_user_request("Sản phẩm này của tôi là gì?", has_image=True)
        self.assertEqual(before_vision.intent, INTENT_PRODUCT_DISCOVERY)
        self.assertEqual(before_vision.action, "identify_image_item")
        self.assertEqual(before_vision.route, ROUTE_IMAGE_PRODUCT_SEARCH)
        self.assertFalse(before_vision.image_context)

        after_vision = route_user_request(
            "Sản phẩm này của tôi là gì?",
            has_image=True,
            image_context={
                "subject": "product",
                "caption": "một đôi giày thể thao nữ màu trắng",
                "fashion_item": "giày thể thao nữ",
                "confidence": 0.88,
            },
        )
        self.assertEqual(after_vision.action, "identify_image_item")
        self.assertEqual(after_vision.route, ROUTE_IMAGE_PRODUCT_SEARCH)
        self.assertEqual(after_vision.entities["identified_item"], "giày thể thao nữ")
        self.assertEqual(
            [option["action"] for option in after_vision.follow_up_options],
            ["view_similar_results", "style_image_item"],
        )

    def test_pending_profile_requires_explicit_confirmation(self):
        state = {"pending_profile_candidate": {"dang_nguoi": "Tam giác"}}
        decision = route_user_request("đồng ý lưu", state=state)
        self.assertEqual(decision.intent, INTENT_PROFILE_MANAGEMENT)
        self.assertEqual(decision.action, "confirm_candidate")
        self.assertEqual(decision.route, ROUTE_PROFILE_STATE_HANDLER)

    def test_profile_correction_deletes_wrong_field(self):
        decision = route_user_request("tôi không phải dáng chữ nhật")
        self.assertEqual(decision.action, "delete_field")
        self.assertEqual(decision.entities["profile_delete_fields"], ["dang_nguoi"])

    def test_more_without_history_never_blind_searches(self):
        decision = route_user_request("xem thêm")
        self.assertTrue(decision.needs_clarification)
        self.assertIsNone(decision.route)

    def test_stock_is_product_action_even_without_inventory_data(self):
        decision = route_user_request("áo này còn hàng không")
        self.assertEqual(decision.intent, INTENT_PRODUCT_DISCOVERY)
        self.assertEqual(decision.action, "stock_check")


if __name__ == "__main__":
    unittest.main()
