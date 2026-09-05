import unittest

from langchain_core.documents import Document

from app.api import (
    _browser_image_url,
    _decision_event,
    _docs_to_images,
    _image_identification_reply,
    _retrieval_progress_message,
)
from app.core.intent import route_user_request


class ApiUiPayloadTests(unittest.TestCase):
    def test_product_card_contains_commerce_fields(self):
        doc = Document(
            page_content="Áo sơ mi",
            metadata={
                "product_id": "SKU-001",
                "title": "Áo sơ mi nữ",
                "brand": "Demo Brand",
                "price": 299000,
                "category": "Áo",
                "images": ["images/SKU-001_MAIN.jpg"],
            },
        )
        card = _docs_to_images([doc])[0]
        self.assertEqual(card["title"], "Áo sơ mi nữ")
        self.assertEqual(card["brand"], "Demo Brand")
        self.assertEqual(card["price"], 299000)
        self.assertEqual(card["product_id"], "SKU-001")
        self.assertEqual(card["images"], ["/images/SKU-001_MAIN.jpg"])

    def test_main_image_is_used_when_images_list_is_empty(self):
        doc = Document(
            page_content="Giày",
            metadata={"product_id": "SHOE-1", "image_url": "SHOE-1_MAIN.jpg"},
        )
        self.assertEqual(_docs_to_images([doc])[0]["images"], ["/images/SHOE-1_MAIN.jpg"])

    def test_remote_image_url_is_preserved(self):
        self.assertEqual(_browser_image_url("https://example.com/a.jpg"), "https://example.com/a.jpg")

    def test_developer_mode_controls_router_trace(self):
        decision = route_user_request("tìm áo sơ mi")
        self.assertNotIn("trace", _decision_event(decision, "female", False))
        self.assertIn("trace", _decision_event(decision, "female", True))

    def test_progress_mentions_observable_filters(self):
        message = _retrieval_progress_message(
            {"categories": ["ao so mi"], "budget_text": "duoi 300k", "sizes": ["M"]}
        )
        self.assertIn("dưới 300k", message)
        self.assertIn("size M", message)

    def test_image_identification_reply_is_cautious_and_human_readable(self):
        reply = _image_identification_reply(
            {
                "fashion_item": "giày thể thao nữ",
                "caption": "một đôi giày thể thao nữ màu trắng dáng thấp",
            }
        )
        self.assertIn("có vẻ là", reply)
        self.assertIn("giày thể thao nữ", reply)
        self.assertIn("mẫu gần giống", reply)


if __name__ == "__main__":
    unittest.main()
