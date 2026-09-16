"""Domain entity: the single structured decision the router produces.

Thuật toán suy ra ``IntentDecision`` nằm ở ``fashion_rag.core.intent``
(``route_user_request()``). Module này chỉ định nghĩa *hình dạng* dữ liệu —
không phụ thuộc LangChain, Qdrant, hay bất kỳ hạ tầng nào.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from fashion_rag.domain.value_objects.enums import (
    CERTAINTY_DETERMINISTIC,
    MODALITY_TEXT,
    ROUTE_IMAGE_OUTFIT_ADVICE,
    ROUTE_IMAGE_PRODUCT_SEARCH,
    ROUTE_OUT_OF_SCOPE_REDIRECT,
    ROUTE_PROFILE_STATE_HANDLER,
    ROUTE_PROFILE_VLM_ANALYSIS,
    ROUTE_SOCIAL_RESPONSE,
    ROUTE_TEXT_OUTFIT_ADVICE,
    ROUTE_TEXT_PRODUCT_SEARCH,
)


@dataclass
class IntentDecision:
    """Structured semantic decision plus an execution route.

    Đây là **kết quả duy nhất** mà router trả về cho caller (api.py).
    Nó gói gọn toàn bộ quyết định routing vào một object duy nhất,
    bao gồm intent, modality, action, route thực thi, và trace debug.

    `trace` chứa các bước xử lý ngắn gọn để debug. Nó cố ý tránh
    chain-of-thought ẩn và chỉ lưu trữ kết quả policy có thể quan sát được.

    Attributes:
        intent (str): Mục đích nghiệp vụ — một trong các INTENT_* constants.
        modality (str): Kiểu đầu vào — text / image / text_image.
        action (str): Thao tác cụ thể trong intent (search, create_outfit, ...).
        route (str | None): Pipeline thực thi — một trong ROUTE_* constants.
            Nếu là None, caller phải xử lý clarification.
        confidence (float): **Deprecated** — không dùng để routing.
            Giữ lại tạm thời để notebooks cũ không bị vỡ.
        certainty (str): Mức độ tin cậy của quyết định (xem CERTAINTY_* constants).
            Đây là field đúng để routing, thay cho `confidence`.
        rewrite_query (str): Query đã được viết lại để tối ưu cho retrieval.
        entities (dict): Các thực thể trích xuất được: màu, category, dịp dùng, size.
        image_context (dict): Kết quả phân tích ảnh từ VLM (caption, subject, ...).
        missing_slots (list[str]): Các slot bị thiếu ngăn chặn thực thi.
        needs_clarification (bool): True nếu cần hỏi lại người dùng trước khi thực thi.
        clarification_question (str): Câu hỏi hiển thị cho người dùng khi cần làm rõ.
        clarification_options (list[dict]): Các lựa chọn quick-reply cho câu hỏi làm rõ.
        follow_up_question (str): Câu hỏi gợi ý sau khi đã thực thi (không chặn).
        follow_up_options (list[dict]): Các lựa chọn quick-reply cho follow-up.
        workflow (list[str]): Danh sách route thực thi theo thứ tự (dùng cho multi-step).
        reason (str): Lý do ngắn gọn tại sao chọn route/intent này (cho logging).
        source (str): Cơ chế ra quyết định ("keyword", "state", "llm", "fallback", ...).
        trace (list[dict]): Chuỗi các bước xử lý để debug/audit.
    """

    intent: str
    modality: str = MODALITY_TEXT
    action: str = "search"
    route: str | None = None
    # Deprecated — Routing phải dùng `certainty`, không dùng số này.
    # Giữ lại tạm thời để notebooks/UI cũ không bị vỡ.
    confidence: float = 0.0
    certainty: str = CERTAINTY_DETERMINISTIC
    rewrite_query: str = ""
    entities: dict = field(default_factory=dict)
    image_context: dict = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_options: list[dict] = field(default_factory=list)
    follow_up_question: str = ""
    follow_up_options: list[dict] = field(default_factory=list)
    workflow: list[str] = field(default_factory=list)
    reason: str = ""
    source: str = "router"
    trace: list[dict] = field(default_factory=list)

    @property
    def handler(self) -> str:
        """Compatibility execution kind used by the existing chat loop.

        Property này dịch route (chi tiết kỹ thuật) sang tên handler ngắn gọn
        mà vòng lặp chat cũ (api.py) đang dùng. Code mới nên dùng `route` trực tiếp.

        Returns:
            str: Tên handler ngắn — "search", "image_search", "outfit",
                 "profile_analysis", "profile_management", "social",
                 "out_of_scope", hoặc "clarify" nếu chưa có route.
        """
        if self.needs_clarification or not self.route:
            return "clarify"
        if self.route == ROUTE_TEXT_PRODUCT_SEARCH:
            return "search"
        if self.route == ROUTE_IMAGE_PRODUCT_SEARCH:
            return "image_search"
        if self.route in {ROUTE_TEXT_OUTFIT_ADVICE, ROUTE_IMAGE_OUTFIT_ADVICE}:
            return "outfit"
        if self.route == ROUTE_PROFILE_VLM_ANALYSIS:
            return "profile_analysis"
        if self.route == ROUTE_PROFILE_STATE_HANDLER:
            return "profile_management"
        if self.route == ROUTE_SOCIAL_RESPONSE:
            return "social"
        if self.route == ROUTE_OUT_OF_SCOPE_REDIRECT:
            return "out_of_scope"
        return "clarify"

    @property
    def legacy_intent(self) -> str:
        """Old short intent name retained only for external compatibility.

        Alias của `handler` — giữ lại để các caller cũ không phải đổi code.
        """
        return self.handler

    def to_debug_dict(self) -> dict:
        """Serialize toàn bộ decision thành dict, bao gồm cả computed property `handler`.

        Returns:
            dict: Toàn bộ fields của IntentDecision cộng thêm key ``handler``.
        """
        data = asdict(self)
        data["handler"] = self.handler
        return data


# Older imports continue to work while notebooks migrate to IntentDecision.
RouteDecision = IntentDecision
