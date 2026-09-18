"""Value objects: the closed vocabulary a routing decision is built from.

Đây là các hằng số bất biến mô tả *hình dạng* của một quyết định routing
(modality/route/certainty). Thuật toán suy ra quyết định đó vẫn nằm ở
``fashion_rag.core.intent`` — module này chỉ giữ vocabulary dùng chung.
"""

from __future__ import annotations

# Input modality là chiều độc lập với intent.
MODALITY_TEXT = "text"          # Chỉ có văn bản
MODALITY_IMAGE = "image"        # Chỉ có ảnh (không có text kèm)
MODALITY_TEXT_IMAGE = "text_image"  # Vừa có văn bản vừa có ảnh

# Execution routes — Pipeline thực thi cuối cùng.
# CHỈ có hàm resolve_route() (core.intent) được phép gán các giá trị này.
#
#   ROUTE_TEXT_PRODUCT_SEARCH   → RAG vector search bằng văn bản (ViFashionCLIP)
#   ROUTE_IMAGE_PRODUCT_SEARCH  → RAG vector search bằng ảnh (FashionCLIP image)
#   ROUTE_TEXT_OUTFIT_ADVICE    → LLM sinh gợi ý outfit từ văn bản + Layer B rules
#   ROUTE_IMAGE_OUTFIT_ADVICE   → LLM sinh gợi ý outfit từ item trong ảnh
#   ROUTE_PROFILE_VLM_ANALYSIS  → VLM phân tích dáng người / tone da từ ảnh
#   ROUTE_PROFILE_STATE_HANDLER → CRUD profile trong session state
#   ROUTE_SOCIAL_RESPONSE       → Trả lời chào hỏi / cảm ơn bằng template cố định
#   ROUTE_OUT_OF_SCOPE_REDIRECT → Thông báo ngoài phạm vi + hướng dẫn dùng lại
ROUTE_TEXT_PRODUCT_SEARCH = "text_product_search"
ROUTE_IMAGE_PRODUCT_SEARCH = "image_product_search"
ROUTE_TEXT_OUTFIT_ADVICE = "text_outfit_advice"
ROUTE_IMAGE_OUTFIT_ADVICE = "image_outfit_advice"
ROUTE_PROFILE_VLM_ANALYSIS = "profile_vlm_analysis"
ROUTE_PROFILE_STATE_HANDLER = "profile_state_handler"
ROUTE_SOCIAL_RESPONSE = "social_response"
ROUTE_OUT_OF_SCOPE_REDIRECT = "out_of_scope_redirect"

EXECUTION_ROUTES = {
    ROUTE_TEXT_PRODUCT_SEARCH,
    ROUTE_IMAGE_PRODUCT_SEARCH,
    ROUTE_TEXT_OUTFIT_ADVICE,
    ROUTE_IMAGE_OUTFIT_ADVICE,
    ROUTE_PROFILE_VLM_ANALYSIS,
    ROUTE_PROFILE_STATE_HANDLER,
    ROUTE_SOCIAL_RESPONSE,
    ROUTE_OUT_OF_SCOPE_REDIRECT,
}

# Compatibility aliases — Tên cũ được giữ lại để code ngoài (notebooks, API cũ)
# không bị vỡ khi nội bộ đổi tên route. Code mới NÊN dùng tên explicit ở trên.
ROUTE_PRODUCT_SEARCH = ROUTE_TEXT_PRODUCT_SEARCH
ROUTE_OUTFIT_ADVICE = ROUTE_TEXT_OUTFIT_ADVICE
ROUTE_PROFILE_INQUIRY = ROUTE_PROFILE_STATE_HANDLER
ROUTE_OUT_OF_SCOPE = ROUTE_OUT_OF_SCOPE_REDIRECT
ROUTE_GREETING = ROUTE_SOCIAL_RESPONSE
ROUTE_CHITCHAT = ROUTE_SOCIAL_RESPONSE
ROUTE_CLARIFY = "clarify"  # Nhãn kiểm soát — KHÔNG bao giờ được trả về làm route thực

# `certainty` — Mức độ chắc chắn của quyết định routing.
# Khác với "confidence" (xác suất do LLM tự báo, không đáng tin),
# `certainty` là nhãn có thể kiểm chứng được dựa trên cơ chế ra quyết định:
#
#   DETERMINISTIC          — Quyết định bằng code Python thuần túy (cao nhất)
#   CONTEXTUAL             — Quyết định từ session state hoặc modality signal
#   SEMANTIC               — Quyết định bằng độ tương đồng embedding (Layer 3b)
#   LLM_ASSISTED           — LLM đã tham gia phân loại (thấp hơn)
#   CLARIFICATION_REQUIRED — Không đủ thông tin, cần hỏi lại người dùng
CERTAINTY_DETERMINISTIC = "deterministic"
CERTAINTY_CONTEXTUAL = "contextual"
CERTAINTY_SEMANTIC = "semantic"
CERTAINTY_LLM_ASSISTED = "llm_assisted"
CERTAINTY_CLARIFICATION_REQUIRED = "clarification_required"
