"""Intent understanding and deterministic route resolution for the chatbot.

Module này là bộ não định tuyến (router) của chatbot thời trang. Nó nhận đầu vào
từ người dùng (văn bản, ảnh, hoặc cả hai) rồi quyết định **pipeline thực thi nào
sẽ xử lý yêu cầu đó**.

---

**Triết lý thiết kế cốt lõi (Core Design Philosophy)**::

    intent   = Người dùng muốn gì          (product_discovery, outfit_advice, ...)
    modality = Họ cung cấp đầu vào gì      (text, image, text+image)
    action   = Thao tác cụ thể bên trong   (search, find_similar, create_outfit, ...)
    route    = Pipeline thực thi được chọn  (do Python policy quyết định, KHÔNG phải LLM)

LLM chỉ được phép **phân loại ngôn ngữ mơ hồ** — nó không được phép trực tiếp
chọn route thực thi. Mọi quyết định route đều được thực hiện bởi code Python
deterministic (có thể kiểm chứng, không có hallucination).

---

**Thứ tự ưu tiên quyết định (Decision Priority Layers)**::

    Layer 1 — Modality gate    : Có ảnh → _route_image_request()
    Layer 2 — Session state    : Đang chờ xác nhận → _route_pending_state()
    Layer 3 — Keyword matching : Từ khóa độ chính xác cao → route_from_keywords()
    Layer 4 — LLM fallback     : Câu mơ hồ → classify_intent_llm()

Layer thấp hơn (1 < 2 < 3) luôn được ưu tiên hơn Layer 4 (LLM).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field

import ollama

from app.config import (
    DEFINITE_CHITCHAT,
    DEFINITE_GREETING,
    DEFINITE_OUTFIT,
    DEFINITE_PROFILE_INQUIRY,
    DEFINITE_SEARCH,
    FOLLOWUP_MORE_KEYWORDS,
    IMAGE_ROUTER_CONFIDENCE_THRESHOLD,
    LAYER_B_DANG_NGUOI,
    LAYER_B_TONE_DA,
    LLM_MODEL,
    MALE_KEYWORDS,
    OLLAMA_BASE_URL,
    STRICT_OUT_OF_SCOPE_PATTERNS,
)


# ---------------------------------------------------------------------------
# Stable business intents
# ---------------------------------------------------------------------------
# Các intent nghiệp vụ ổn định — đây là những "mục đích" mà người dùng có thể có.
# Chúng KHÔNG thay đổi theo thời gian (stable), khác với route có thể tái cấu trúc.
#
#   INTENT_PRODUCT_DISCOVERY  — Tìm/xem/so sánh sản phẩm, hỏi giá/size/tồn kho
#   INTENT_OUTFIT_ADVICE      — Hỏi phối đồ, tạo outfit, hỏi nên mặc gì
#   INTENT_PROFILE_ANALYSIS   — Gửi ảnh để phân tích dáng người / tone da
#   INTENT_PROFILE_MANAGEMENT — Đọc/cập nhật/xóa thông tin profile đã lưu
#   INTENT_SOCIAL             — Chào hỏi, cảm ơn, tạm biệt
#   INTENT_OUT_OF_SCOPE       — Câu hỏi không liên quan đến thời trang
#   INTENT_UNKNOWN            — Quá mơ hồ để xử lý an toàn (control result)
INTENT_PRODUCT_DISCOVERY = "product_discovery"
INTENT_OUTFIT_ADVICE = "outfit_advice"
INTENT_PROFILE_ANALYSIS = "profile_analysis"
INTENT_PROFILE_MANAGEMENT = "profile_management"
INTENT_SOCIAL = "social"
INTENT_OUT_OF_SCOPE = "out_of_scope"
INTENT_UNKNOWN = "unknown"  # Kết quả kiểm soát — không phải intent nghiệp vụ thực sự

BUSINESS_INTENTS = {
    INTENT_PRODUCT_DISCOVERY,
    INTENT_OUTFIT_ADVICE,
    INTENT_PROFILE_ANALYSIS,
    INTENT_PROFILE_MANAGEMENT,
    INTENT_SOCIAL,
    INTENT_OUT_OF_SCOPE,
}

# Input modality là chiều độc lập với intent.
# Một người dùng có thể muốn tìm sản phẩm (intent) bằng ảnh (modality),
# hoặc bằng văn bản, hoặc cả hai — đây là hai quyết định riêng biệt.
MODALITY_TEXT = "text"          # Chỉ có văn bản
MODALITY_IMAGE = "image"        # Chỉ có ảnh (không có text kèm)
MODALITY_TEXT_IMAGE = "text_image"  # Vừa có văn bản vừa có ảnh

# Execution routes — Pipeline thực thi cuối cùng.
# CHỈ có hàm resolve_route() được phép gán các giá trị này.
# Mỗi route tương ứng với một handler pipeline trong api.py:
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


PRODUCT_ACTIONS = {
    "search",
    "find_similar",
    "more",
    "price_check",
    "size_check",
    "stock_check",
    "compare",
}
OUTFIT_ACTIONS = {"create_outfit", "style_image_item", "refine_outfit"}
PROFILE_ANALYSIS_ACTIONS = {
    "analyze_body",
    "analyze_skin_tone",
    "analyze_full_profile",
    "analyze_then_style",
}
PROFILE_MANAGEMENT_ACTIONS = {
    "read",
    "update",
    "delete_field",
    "clear_all",
    "confirm_candidate",
    "reject_candidate",
}
SOCIAL_ACTIONS = {"greeting", "thanks", "goodbye", "social"}

# `certainty` — Mức độ chắc chắn của quyết định routing.
# Khác với "confidence" (xác suất do LLM tự báo, không đáng tin),
# `certainty` là nhãn có thể kiểm chứng được dựa trên cơ chế ra quyết định:
#
#   DETERMINISTIC          — Quyết định bằng code Python thuần túy (cao nhất)
#   CONTEXTUAL             — Quyết định từ session state hoặc modality signal
#   LLM_ASSISTED           — LLM đã tham gia phân loại (thấp hơn)
#   CLARIFICATION_REQUIRED — Không đủ thông tin, cần hỏi lại người dùng
CERTAINTY_DETERMINISTIC = "deterministic"
CERTAINTY_CONTEXTUAL = "contextual"
CERTAINTY_LLM_ASSISTED = "llm_assisted"
CERTAINTY_CLARIFICATION_REQUIRED = "clarification_required"

# Các source thuộc nhóm CONTEXTUAL (không phải LLM, không phải pure keyword)
CONTEXTUAL_SOURCES = {
    "state",               # Quyết định từ session state (pending confirmation, v.v.)
    "modality_gate",       # Quyết định từ việc phát hiện có ảnh
    "modality_keyword",    # Keyword đi kèm ảnh ("phân tích dáng", "phối đồ"...)
    "modality_override",   # Caller ép buộc route image search (force_image_search=True)
    "image_context_default",  # VLM nhận ra fashion item → tìm sản phẩm tương tự
    "image_context_clarify",  # VLM không chắc → hỏi lại người dùng
}

# BLOCKING_SLOTS — Chỉ những slot này mới được phép chặn thực thi và hỏi lại.
# Các thuộc tính mua sắm tùy chọn (màu sắc, ngân sách, size) KHÔNG được phép
# chặn — bot phải thực hiện tìm kiếm hữu ích trước, không thẩm vấn khách hàng.
#
#   user_goal       — Không biết người dùng muốn gì (quá mơ hồ)
#   previous_search — Yêu cầu "xem thêm" nhưng không có lượt tìm trước
#   image_context   — Có ảnh nhưng VLM chưa phân tích
#   image_goal      — VLM đã phân tích nhưng không chắc mục tiêu của người dùng
BLOCKING_SLOTS = {"user_goal", "previous_search", "image_context", "image_goal"}


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


def _trace(stage: str, result: str, detail: str = "") -> dict:
    """Tạo một bước trace nhỏ cho debug audit trail.

    Mỗi bước trace lưu lại tên giai đoạn, kết quả, và chi tiết tùy chọn.
    Toàn bộ trace được gắn vào `IntentDecision.trace` để caller có thể
    kiểm tra luồng ra quyết định mà không cần đọc log.

    Args:
        stage (str): Tên giai đoạn xử lý (vd: "keyword", "state", "route").
        result (str): Kết quả tại giai đoạn đó (vd: "greeting", "clarification").
        detail (str): Chi tiết bổ sung tùy chọn (vd: query gốc, confidence score).

    Returns:
        dict: ``{"stage": ..., "result": ..., "detail": ...}``
    """
    return {"stage": stage, "result": result, "detail": detail}


def strip_vietnamese_accents(text: str) -> str:
    """Remove Vietnamese diacritics for accent-insensitive matching.

    Chuẩn hóa văn bản tiếng Việt bằng cách bỏ dấu thanh và dấu phụ.
    Ví dụ: ``"Áo thun đỏ"`` → ``"Ao thun do"``.

    Sử dụng chuẩn hóa Unicode NFD (tách ký tự thành base + combining marks)
    rồi loại bỏ các combining mark (category "Mn"). Xử lý riêng chữ ``đ/Đ``
    vì chúng không có combining mark trong NFD.

    Args:
        text (str): Văn bản tiếng Việt có dấu (hoặc không có dấu).

    Returns:
        str: Văn bản đã bỏ dấu, chữ thường.
    """
    text = unicodedata.normalize("NFD", str(text or ""))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for consistent comparison.

    Chuẩn hóa văn bản: chuyển thành chữ thường và gộp mọi khoảng trắng
    liên tiếp thành một space duy nhất.

    Args:
        text (str): Văn bản đầu vào bất kỳ.

    Returns:
        str: Văn bản chữ thường, whitespace đã chuẩn hóa.
    """
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def plain_text(text: str) -> str:
    """Normalize and strip Vietnamese accents for accent-free matching.

    Kết hợp ``normalize_text`` và ``strip_vietnamese_accents``: kết quả
    là văn bản chữ thường, không dấu, whitespace chuẩn — dùng để khớp
    với các keyword trong danh sách accent-free (vd: COLOR_KEYWORDS).

    Args:
        text (str): Văn bản tiếng Việt bất kỳ.

    Returns:
        str: Văn bản bỏ dấu, chữ thường, whitespace chuẩn hóa.
    """
    return normalize_text(strip_vietnamese_accents(text))


def phrase_hit(text: str, phrase: str) -> bool:
    """Match one normalized word/phrase without substring false positives.

    Kiểm tra xem ``phrase`` có xuất hiện trong ``text`` hay không,
    **tại ranh giới từ** (word boundary) để tránh false positive từ substring.

    Ví dụ: ``phrase_hit("tim san pham", "tim")`` → True ("tìm" đúng)
    nhưng nếu không dùng word boundary, ``phrase_hit("vitamin", "tim")`` cũng True (sai).

    Cả ``text`` lẫn ``phrase`` đều được normalize trước khi so sánh.

    Args:
        text (str): Văn bản nguồn cần tìm trong đó.
        phrase (str): Từ/cụm từ cần tìm.

    Returns:
        bool: True nếu ``phrase`` xuất hiện tại ranh giới từ trong ``text``.
    """
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
            normalized_text,
        )
    )


def keyword_hit(query: str, keywords: list[str] | tuple[str, ...]) -> bool:
    """Match accented or accent-free keywords at word/phrase boundaries.

    Kiểm tra xem query có chứa bất kỳ keyword nào trong danh sách không.
    Hỗ trợ **cả hai dạng**: có dấu tiếng Việt và không dấu — người dùng
    có thể gõ ``"ao thun"`` hoặc ``"áo thun"`` đều được nhận ra.

    Cơ chế: mỗi keyword được kiểm tra 2 lần:
    - So sánh query gốc (normalized) với keyword gốc
    - So sánh query bỏ dấu (plain) với keyword bỏ dấu (plain)

    Args:
        query (str): Câu hỏi / văn bản của người dùng.
        keywords (list[str] | tuple[str, ...]): Danh sách từ/cụm từ cần tìm.

    Returns:
        bool: True nếu ít nhất một keyword xuất hiện trong query.
    """
    query_normalized = normalize_text(query)
    query_plain = plain_text(query)
    for keyword in keywords:
        keyword_normalized = normalize_text(keyword)
        keyword_plain = plain_text(keyword)
        if phrase_hit(query_normalized, keyword_normalized) or phrase_hit(query_plain, keyword_plain):
            return True
    return False


def word_hit(text: str, keyword: str) -> bool:
    """Match a single keyword at word boundaries in already-normalized text.

    Phiên bản đơn giản hơn ``phrase_hit`` — dành cho trường hợp text đã
    được normalize trước, và chỉ cần so sánh một keyword đơn.

    Args:
        text (str): Văn bản đã normalize.
        keyword (str): Từ khóa đơn cần tìm.

    Returns:
        bool: True nếu keyword xuất hiện tại ranh giới từ.
    """
    return bool(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text))


# ---------------------------------------------------------------------------
# Keyword Dictionaries và Entity Extraction Constants
# ---------------------------------------------------------------------------
# Các danh sách từ khóa được lưu trữ (được) dưới dạng accent-free (không dấu) để
# thiết lập match hai chiều với cả người dùng gõ có dấu lẫn không dấu.
#
# COLOR_KEYWORDS: danh sách màu sắc accent-free. Lưu ý: "tim" (accent-free của "tím")
# trùng với động từ "tìm" (tìm kiếm) — extract_color_entities() có logic
# đặc biệt để phân biệt hai từ này.
COLOR_KEYWORDS = [
    "den", "trang", "xanh", "do", "hong", "be", "kem", "nau", "xam",
    "ghi", "vang", "tim", "cam", "bac", "vang dong",
]
# CATEGORY_KEYWORDS: từ khóa danh mục được sắp xếp từ dài đến ngắn để khớp
# danh mục cụ thể nhất trước ("ao thun" trước "ao" để tránh false positive).
CATEGORY_KEYWORDS = [
    "ao thun", "ao so mi", "so mi", "ao khoac", "ao len", "ao",
    "quan jean", "quan dai", "quan short", "quan", "chan vay", "vay",
    "dam", "jumpsuit", "giay", "dep", "tui xach", "phu kien",
]
CATEGORY_ACCENTED_FORMS = {
    "ao thun": ["áo thun"],
    "ao so mi": ["áo sơ mi"],
    "so mi": ["sơ mi"],
    "ao khoac": ["áo khoác"],
    "ao len": ["áo len"],
    "ao": ["áo"],
    "quan jean": ["quần jean"],
    "quan dai": ["quần dài"],
    "quan short": ["quần short"],
    "quan": ["quần"],
    "chan vay": ["chân váy"],
    "vay": ["váy"],
    "dam": ["đầm"],
    "jumpsuit": ["jumpsuit"],
    "giay": ["giày"],
    "dep": ["dép"],
    "tui xach": ["túi xách"],
    "phu kien": ["phụ kiện"],
}
OCCASION_KEYWORDS = [
    "di lam", "di hoc", "di tiec", "di choi", "di bien", "hen ho",
    "du lich", "cong so", "hang ngay", "du dam cuoi", "troi lanh",
    "thoi tiet lanh",
]
IMAGE_OUTFIT_KEYWORDS = [
    "phoi", "phoi do", "phoi sao", "phoi nhu nao", "phoi the nao",
    "phoi voi gi", "mac voi gi", "mac cung gi", "di voi gi", "hop voi gi",
    "ket hop voi gi", "mix", "mix sao", "mix voi gi", "outfit",
    "goi y do", "goi y outfit",
]
IMAGE_PROFILE_KEYWORDS = [
    "dang nguoi", "body type", "tone da", "mau da", "phan tich nguoi",
    "phan tich dang", "phan tich voc dang",
]
PROFILE_CONFIRM_KEYWORDS = [
    "dong y", "xac nhan", "luu lai", "luu thong tin", "dung roi", "chinh xac",
]
PROFILE_REJECT_KEYWORDS = [
    "khong luu", "khong dong y", "khong dung", "bo qua", "huy",
]
PROFILE_CLEAR_ALL_KEYWORDS = [
    "xoa toan bo", "xoa het profile", "xoa het thong tin", "quen het thong tin",
]
PROFILE_DELETE_KEYWORDS = ["xoa", "bo", "quen", "khong luu"]
SOCIAL_GOODBYE_KEYWORDS = ["tam biet", "hen gap lai", "bye", "bai bai"]
SOCIAL_THANKS_KEYWORDS = ["cam on", "thank you", "thanks"]
CANCEL_PENDING_IMAGE_KEYWORDS = ["khong can", "bo qua anh", "thoi khong"]

# COLOR_ACCENTED_WORDS: map từ accent-free → các dạng có dấu tương ướng.
# Dùng trong extract_color_entities() để detect màu qua dấu ("tím" → không nhham với "tìm").
COLOR_ACCENTED_WORDS = {
    "den": ["đen"], "trang": ["trắng"], "do": ["đỏ"], "hong": ["hồng"],
    "nau": ["nâu"], "xam": ["xám"], "vang": ["vàng"], "tim": ["tím"],
    "bac": ["bạc"],
}
# COLOR_CONTEXT_WORDS: từ ngữ cảnh đi trước màu sắc (vd: "màu đỏ", "tône xanh").
# Dùng để contextual_hit: khớp nếu có từ ngữ cảnh đi liền trước tën màu.
COLOR_CONTEXT_WORDS = [
    "mau", "tone", "ao", "quan", "vay", "dam", "giay", "dep", "tui",
    "non", "kinh",
]


def extract_color_entities(query: str) -> list[str]:
    """Extract color names from query without confusing homophones.

    Trích xuất tên màu sắc từ câu query, xử lý đặc biệt vấn đề tiếng Việt:
    từ "tim" (accent-free của "tím") rất dễ bị nhầm với động từ "tìm" (tìm kiếm).

    **Logic 3 tầng để phân biệt màu và từ khác**:

    1. ``accented_hit``: Kiểm tra dạng có dấu ("tím" → chắc chắn là màu, không phải "tìm")
    2. ``contextual_hit``: Kiểm tra có từ ngữ cảnh ("màu tim", "tone tim") đi trước màu
    3. ``modifier_hit``: Kiểm tra có từ bổ nghĩa màu ("tim pastel", "tim đậm") đi sau màu

    Chỉ khi nhất một trong 3 tầng match thì màu mới được ghi nhận.

    Args:
        query (str): Câu query của người dùng (có dấu hoặc không dấu).

    Returns:
        list[str]: Danh sách màu accent-free nhận ra được (vd: ["tim", "xanh"]).
    """
    query_normalized = normalize_text(query)
    query_plain = plain_text(query)
    colors: list[str] = []
    for color in COLOR_KEYWORDS:
        accented_hit = any(
            word_hit(query_normalized, word)
            for word in COLOR_ACCENTED_WORDS.get(color, [])
        )
        contextual_hit = any(
            re.search(rf"(?<!\w){context}\s+{re.escape(color)}(?!\w)", query_plain)
            for context in COLOR_CONTEXT_WORDS
        )
        modifier_hit = re.search(
            rf"(?<!\w){re.escape(color)}\s+(pastel|dam|nhat|than|dong)(?!\w)",
            query_plain,
        )
        if accented_hit or contextual_hit or modifier_hit:
            colors.append(color)
    return colors


def category_hit(query: str, category: str) -> bool:
    """Match a product category while preserving Vietnamese accent disambiguation.

    Vấn đề: tiếng Việt có nhiều từ đồng âm khác dấu có nghĩa hoàn toàn khác nhau.
    Ví dụ: ``"đầm"`` (danh mục đầm — sản phẩm) khác với ``"đảm bảo"`` (cụm từ).

    Giải pháp: thay vì bỏ dấu trước khi so sánh (gây nhầm), hàm này kiểm tra:
    - Hoặc từ accent-free mà người dùng gõ (vd: "dam")
    - Hoặc một trong các dạng có dấu chính xác trong ``CATEGORY_ACCENTED_FORMS``

    Args:
        query (str): Câu query của người dùng.
        category (str): Từ khóa danh mục accent-free cần kiểm tra (vd: "dam", "vay").

    Returns:
        bool: True nếu query chứa danh mục này (có dấu hoặc không dấu).
    """
    query_normalized = normalize_text(query)
    forms = [category, *CATEGORY_ACCENTED_FORMS.get(category, [])]
    return any(phrase_hit(query_normalized, form) for form in forms)


def extract_basic_entities(query: str) -> dict:
    """Extract product filter entities from a Vietnamese shopping query.

    Trích xuất các bộ lọc tìm kiếm sản phẩm từ câu query:
    màu sắc, danh mục sản phẩm, dịp dùng, size, và ngân sách.

    Các entity này là **filter tùy chọn** — không có không sao, bot vẫn
    tìm kiếm bình thường. Chi tiết xem BLOCKING_SLOTS.

    Args:
        query (str): Câu query của người dùng.

    Returns:
        dict: Các key có thể có trong kết quả:

        - ``colors`` (list[str]): Màu sắc accent-free nhận ra được.
        - ``categories`` (list[str]): Danh mục, sắp xếp danh mục dài nhất lên trước
          để ưu tiên cụ thể ("ao thun" trước "ao").
        - ``occasions`` (list[str]): Dịp sử dụng ("di lam", "di tiec", ...).
        - ``sizes`` (list[str]): Size với chữ hoa ("M", "L", "XL", "34", ...).
        - ``budget_text`` (str): Đoạn văn bản ngân sách gốc (vd: "duoi 500k").
    """
    query_plain = plain_text(query)
    entities: dict = {}
    colors = extract_color_entities(query)
    if colors:
        entities["colors"] = colors
    categories = [category for category in CATEGORY_KEYWORDS if category_hit(query, category)]
    if categories:
        entities["categories"] = sorted(categories, key=len, reverse=True)
    occasions = [occasion for occasion in OCCASION_KEYWORDS if phrase_hit(query_plain, occasion)]
    if occasions:
        entities["occasions"] = occasions
    sizes = re.findall(
        r"\b(?:size\s*)?(xs|s|m|l|xl|xxl|xxxl|[2-5][0-9])\b",
        query_plain,
        flags=re.IGNORECASE,
    )
    if sizes:
        entities["sizes"] = sorted({size.upper() for size in sizes})
    budget = re.search(
        r"(?:duoi|nho hon|under|toi da|max)\s*(\d+(?:[\.,]\d+)?)\s*"
        r"(k|nghin|ngan|trieu|m|vnd|d)?",
        query_plain,
    )
    if budget:
        entities["budget_text"] = budget.group(0)
    return entities


def extract_profile_entities(query: str) -> dict:
    """Extract explicit profile updates or deletions from Vietnamese text.

    Phân tích câu query để tìm các thông tin profile mà người dùng
    đã chủ động cung cấp (cập nhật) hoặc muốn xóa.

    **Logic phân loại**:

    - Nếu có nhận diện được giá trị dương (``dang_nguoi``, ``tone_da``, ``gender``)
      và không có từ phủ định → trả về ``{"profile_updates": {...}}``
    - Nếu có từ phủ định ("không phải", "xoa", ...) kèm mention field profile
      → trả về ``{"profile_delete_fields": [...]}``
    - Nếu không nhận ra gì rõ ràng → trả về ``{}`` (rỗng)

    Args:
        query (str): Câu hỏi / tin nhắn của người dùng.

    Returns:
        dict: Một trong các dạng sau:

        - ``{"profile_updates": {field: value, ...}}`` — cập nhật profile
        - ``{"profile_delete_fields": [field, ...]}`` — xóa trường profile
        - ``{}`` — không phát hiện thông tin profile nào
    """
    query_plain = plain_text(query)
    updates: dict = {}
    delete_fields: list[str] = []

    if keyword_hit(query_plain, ["tu van do nam", "toi la nam", "gioi tinh nam"]):
        updates["gender"] = "male"
    if keyword_hit(query_plain, ["tu van do nu", "toi la nu", "gioi tinh nu"]):
        updates["gender"] = "female"

    negative_profile_value = keyword_hit(
        query_plain, ["khong phai", "khong co", "khong dung la"]
    )

    for label in LAYER_B_DANG_NGUOI:
        if phrase_hit(query_plain, plain_text(label)) and not negative_profile_value:
            updates["dang_nguoi"] = label
            break
    for label in LAYER_B_TONE_DA:
        if phrase_hit(query_plain, plain_text(label)) and not negative_profile_value:
            updates["tone_da"] = label
            break

    has_delete_verb = negative_profile_value or keyword_hit(query_plain, PROFILE_DELETE_KEYWORDS)
    if has_delete_verb:
        mentioned_body_label = any(
            phrase_hit(query_plain, plain_text(label)) for label in LAYER_B_DANG_NGUOI
        )
        mentioned_tone_label = any(
            phrase_hit(query_plain, plain_text(label)) for label in LAYER_B_TONE_DA
        )
        if mentioned_body_label or keyword_hit(query_plain, ["dang nguoi", "voc dang", "body type"]):
            delete_fields.append("dang_nguoi")
        if mentioned_tone_label or keyword_hit(query_plain, ["tone da", "mau da", "skin tone"]):
            delete_fields.append("tone_da")
        if phrase_hit(query_plain, "gioi tinh"):
            delete_fields.append("gender")

    if delete_fields:
        return {"profile_delete_fields": sorted(set(delete_fields))}
    if updates:
        return {"profile_updates": updates}
    return {}


def strict_out_of_scope_hit(query: str) -> bool:
    """Check if query matches strict out-of-scope patterns from config.

    Kiểm tra xem query có khớp với các mẫu regex ngoài phạm vi
    trong ``STRICT_OUT_OF_SCOPE_PATTERNS`` không.

    Các pattern này chỉ match những câu rõ ràng không liên quan đến
    thời trang (toán học, dự báo thời tiết, bóng đá...). Các câu mơ
    hồ hơn sẽ được để LLM phân loại.

    Args:
        query (str): Câu query của người dùng.

    Returns:
        bool: True nếu query khớp một pattern ngoài phạm vi.
    """
    query_plain = plain_text(query)
    return any(re.search(pattern, query_plain) for pattern in STRICT_OUT_OF_SCOPE_PATTERNS)


def is_more_request(query: str) -> bool:
    """Detect follow-up 'show more results' requests.

    Nhận biết câu yêu cầu "xem thêm" — người dùng muốn xem thêm kết quả
    từ lượt tìm kiếm trước (không phải một tìm kiếm mới).

    Args:
        query (str): Câu query của người dùng.

    Returns:
        bool: True nếu là yêu cầu xem thêm.
    """
    return keyword_hit(query, FOLLOWUP_MORE_KEYWORDS)


def infer_product_action(query: str) -> str:
    """Infer the specific product action from query content.

    Phân loại thao tác sản phẩm cụ thể dựa trên nội dung câu query.
    Tất cả các trường hợp dưới đây đều thuộc intent ``product_discovery``.

    Ủu tiên kiểm tra theo thứ tự (cái kiểm tra trước có ưu tiên cao hơn):

    1. **more**: Yêu cầu xem thêm kết quả ("xem thêm", "gợi ý thêm")
    2. **size_check**: Hỏi về size ("size M", "kích cỡ nào")
    3. **price_check**: Hỏi về giá ("giá bao nhiêu", "dưới 500k")
    4. **stock_check**: Hỏi tồn kho ("còn hàng không", "hết hàng")
    5. **compare**: So sánh sản phẩm ("so sánh", "nên chọn cái nào")
    6. **search**: Tìm kiếm chung (mặc định)

    Args:
        query (str): Câu query của người dùng.

    Returns:
        str: Tên action — một trong các PRODUCT_ACTIONS.
    """
    query_plain = plain_text(query)
    if is_more_request(query):
        return "more"
    if keyword_hit(query_plain, ["size", "kich co"]) or re.search(
        r"\b(xs|s|m|l|xl|xxl|xxxl)\b", query_plain
    ):
        return "size_check"
    if keyword_hit(query_plain, ["gia", "bao nhieu", "duoi", "tren", "re hon", "dat hon"]):
        return "price_check"
    if keyword_hit(query_plain, ["con hang", "het hang", "ton kho"]):
        return "stock_check"
    if keyword_hit(query_plain, ["so sanh", "khac nhau", "nen chon"]):
        return "compare"
    return "search"


def certainty_from_source(source: str, *, clarification: bool = False) -> str:
    """Return an auditable certainty level from the decision mechanism.

    Chuyển đổi tên ``source`` (cơ chế ra quyết định) sang mức ``certainty``
    có thể kiểm chứng được. Khác với confidence (xem CERTAINTY_* constants).

    Args:
        source (str): Cơ chế ra quyết định, vd: "keyword", "state", "llm", "fallback".
        clarification (bool): Nếu True, luôn trả về CLARIFICATION_REQUIRED bất kể source.

    Returns:
        str: Một trong các CERTAINTY_* constants.
    """
    if clarification:
        return CERTAINTY_CLARIFICATION_REQUIRED
    if source in {"llm", "fallback"}:
        return CERTAINTY_LLM_ASSISTED
    if source in CONTEXTUAL_SOURCES:
        return CERTAINTY_CONTEXTUAL
    return CERTAINTY_DETERMINISTIC


def derive_missing_slots(
    intent: str,
    action: str,
    modality: str,
    *,
    state: dict | None = None,
    image_context: dict | None = None,
) -> list[str]:
    """Derive blocking information gaps using Python policy, not LLM opinion.

    Xác định các "slot" bị thiếu có thể chặn thực thi và buộc phải hỏi lại.
    Chỉ các slot trong BLOCKING_SLOTS mới được phép chặn.

    **Tại sao không để LLM tự quyết định slot nào cần thiếu?**
    Vì LLM có xu hướng "over-ask" — thẩm vấn khách hàng cho mọi thực thể
    trước khi chịu tìm, gây trải nghiệm tồi. Chính sách Python ở đây
    đảm bảo bot luôn cá gắng thực hiện trước, chỉ hỏi khi thật sự không
    thể chọn được pipeline an toàn.

    Các trường hợp chặn:

    - ``INTENT_UNKNOWN``: Chưa biết người dùng muốn gì → slot ``user_goal``
    - Action ``more``/``refine_outfit`` mà không có lượt tước → slot ``previous_search``
    - Có ảnh nhưng VLM chưa phân tích → slot ``image_context``

    Args:
        intent (str): Intent đã phân loại được.
        action (str): Action cụ thể bên trong intent.
        modality (str): Kiểu đầu vào (text / image / text_image).
        state (dict | None): Session state hiện tại của conversation.
        image_context (dict | None): Kết quả phân tích ảnh từ VLM (nếu đã chạy).

    Returns:
        list[str]: Danh sách tên các slot bị chặn. Rỗng nếu không có gì chặn.
    """
    state = state or {}
    if intent == INTENT_UNKNOWN:
        return ["user_goal"]
    if action in {"more", "refine_outfit"}:
        previous = state.get("last_route_decision")
        if not previous or getattr(previous, "route", None) not in CONTINUABLE_ROUTES:
            return ["previous_search"]
    if modality in {MODALITY_IMAGE, MODALITY_TEXT_IMAGE} and action == "inspect_image":
        if not image_context:
            return ["image_context"]
    return []


def resolve_route(intent: str, modality: str, action: str) -> str | None:
    """Map semantic fields to one of the finite execution pipelines.

    Đây là hàm **duy nhất được phép chọn route**. Nó là một bảng lookup
    thuần túy — không có side effects, không gọi LLM.

    Bảng mapping Intent + Modality/Action → Route:

    +---------------------+------------------+-----------------+---------------------------+
    | Intent              | Modality/Action  | Route           | Pipeline mô tả            |
    +=====================+==================+=================+===========================+
    | product_discovery   | image/text_image | image_product…  | Vector search bằng ảnh    |
    | product_discovery   | action=find_sim… | image_product…  | Tìm sản phẩm tương tự     |
    | product_discovery   | text             | text_product…   | Vector search bằng text   |
    | outfit_advice       | image+style_img  | image_outfit…   | Phối đồ từ item trong ảnh|
    | outfit_advice       | text             | text_outfit…    | Gợi ý outfit bằng text    |
    | profile_analysis    | any              | profile_vlm…    | VLM phân tích ảnh         |
    | profile_management  | any              | profile_state…  | CRUD profile               |
    | social              | any              | social_response  | Template cố định          |
    | out_of_scope        | any              | out_of_scope…   | Redirect thư hướng dẫn  |
    +---------------------+------------------+-----------------+---------------------------+

    Args:
        intent (str): Intent nghiệp vụ — một trong INTENT_* constants.
        modality (str): Kiểu đầu vào (text / image / text_image).
        action (str): Thao tác cụ thể trong intent.

    Returns:
        str | None: Một trong EXECUTION_ROUTES, hoặc None nếu intent không khớp.
    """
    if intent == INTENT_PRODUCT_DISCOVERY:
        if modality in {MODALITY_IMAGE, MODALITY_TEXT_IMAGE} or action == "find_similar":
            return ROUTE_IMAGE_PRODUCT_SEARCH
        return ROUTE_TEXT_PRODUCT_SEARCH
    if intent == INTENT_OUTFIT_ADVICE:
        if modality in {MODALITY_IMAGE, MODALITY_TEXT_IMAGE} and action == "style_image_item":
            return ROUTE_IMAGE_OUTFIT_ADVICE
        return ROUTE_TEXT_OUTFIT_ADVICE
    if intent == INTENT_PROFILE_ANALYSIS:
        return ROUTE_PROFILE_VLM_ANALYSIS
    if intent == INTENT_PROFILE_MANAGEMENT:
        return ROUTE_PROFILE_STATE_HANDLER
    if intent == INTENT_SOCIAL:
        return ROUTE_SOCIAL_RESPONSE
    if intent == INTENT_OUT_OF_SCOPE:
        return ROUTE_OUT_OF_SCOPE_REDIRECT
    return None


def _decision(
    intent: str,
    modality: str,
    action: str,
    query: str,
    *,
    confidence: float,
    source: str,
    reason: str,
    entities: dict | None = None,
    image_context: dict | None = None,
    trace: list[dict] | None = None,
    workflow: list[str] | None = None,
    follow_up_question: str = "",
    follow_up_options: list[dict] | None = None,
) -> IntentDecision:
    """Internal factory: build a confirmed IntentDecision with a resolved route.

    Hàm nội bộ được dùng bởi tất cả các nánh routing khi đã xác định rõ
    intent, modality và action. Nó tự động gọi ``resolve_route()`` và dóng
    gói kết quả vào ``IntentDecision``.

    Hàm này luôn trả về một decision có thể thực thi người cần
    (``needs_clarification=False``). Nếu cần hỏi lại, dùng ``_clarification()``.

    Args:
        intent (str): Intent nghiệp vụ đã quyết định.
        modality (str): Kiểu đầu vào.
        action (str): Thao tác cụ thể.
        query (str): Query đã được viết lại cho retrieval.
        confidence (float): Giá trị debug tùy chọn (không dùng để routing).
        source (str): Cơ chế ra quyết định ("keyword", "state", "llm", ...).
        reason (str): Lý do ngắn gọn cho logging.
        entities (dict | None): Thực thể trích xuất được.
        image_context (dict | None): Kết quả phân tích ảnh từ VLM.
        trace (list[dict] | None): Trace đã có từ các bước trước.
        workflow (list[str] | None): Chuỗi route cho multi-step workflow.
        follow_up_question (str): Câu hỏi gợi ý sau thực thi (không chặn).
        follow_up_options (list[dict] | None): Lựa chọn quick-reply cho follow-up.

    Returns:
        IntentDecision: Decision hoàn chỉnh với route, certainty và trace.
    """
    route = resolve_route(intent, modality, action)
    stages = list(trace or [])
    stages.append(_trace("route", route or "none", f"{intent} + {modality} + {action}"))
    return IntentDecision(
        intent=intent,
        modality=modality,
        action=action,
        route=route,
        confidence=max(0.0, min(1.0, float(confidence))),
        certainty=certainty_from_source(source),
        rewrite_query=query.strip(),
        entities=entities or {},
        image_context=image_context or {},
        workflow=workflow or ([route] if route else []),
        follow_up_question=follow_up_question,
        follow_up_options=follow_up_options or [],
        reason=reason,
        source=source,
        trace=stages,
    )


def _clarification(
    query: str,
    *,
    modality: str,
    missing_slots: list[str],
    question: str,
    options: list[dict],
    reason: str,
    source: str,
    image_context: dict | None = None,
    trace: list[dict] | None = None,
) -> IntentDecision:
    """Internal factory: build an IntentDecision that requires user clarification.

    Tạo một decision yêu cầu làm rõ thông tin — thường được gọi khi:
    - Không nhận ra intent rõ ràng (INTENT_UNKNOWN)
    - Thiếu slot cần thiết để chọn pipeline (xem BLOCKING_SLOTS)

    Decision này có ``needs_clarification=True`` và ``route=None``,
    có nghĩa caller (api.py) sẽ hiển thị ``clarification_question``
    cho người dùng thay vì thực thi pipeline.

    Args:
        query (str): Query gốc của người dùng.
        modality (str): Kiểu đầu vào.
        missing_slots (list[str]): Các slot bị thiếu (chỉ giữ slot trong BLOCKING_SLOTS).
        question (str): Câu hỏi hiển thị cho người dùng.
        options (list[dict]): Các lựa chọn quick-reply (label, action, value).
        reason (str): Lý do ngắn gọn cho logging/debug.
        source (str): Cơ chế phát hiện ra việc thiếu slot.
        image_context (dict | None): Kết quả phân tích ảnh (nếu có).
        trace (list[dict] | None): Trace từ các bước trước.

    Returns:
        IntentDecision: Decision với ``needs_clarification=True``, ``route=None``.
    """
    missing_slots = [
        slot for slot in dict.fromkeys(missing_slots)
        if slot in BLOCKING_SLOTS
    ] or ["user_goal"]
    stages = list(trace or [])
    stages.append(_trace("route", "clarification", ", ".join(missing_slots)))
    return IntentDecision(
        intent=INTENT_UNKNOWN,
        modality=modality,
        action="clarify",
        route=None,
        confidence=0.0,
        certainty=CERTAINTY_CLARIFICATION_REQUIRED,
        rewrite_query=query.strip(),
        image_context=image_context or {},
        missing_slots=missing_slots,
        needs_clarification=True,
        clarification_question=question,
        clarification_options=options,
        reason=reason,
        source=source,
        trace=stages,
    )


def _image_description(image_context: dict) -> str:
    """Extract a human-readable description from VLM image analysis output.

    Lấy mô tả ngắn gọn về ảnh từ kết quả VLM, dùng để hiển thị
    trong câu hỏi làm rõ hoặc follow-up cho người dùng.

    Ưu tiên: ``caption`` > ``fashion_item`` > chuỗi dự phòng.

    Args:
        image_context (dict): Kết quả phân tích từ VLM vision module.

    Returns:
        str: Mô tả ảnh để hiển thị cho người dùng.
    """
    return str(
        image_context.get("caption")
        or image_context.get("fashion_item")
        or "ảnh thời trang bạn vừa gửi"
    ).strip()


def _route_image_request(query: str, image_context: dict | None = None) -> IntentDecision:
    """Route a request that includes an image through a multi-stage decision tree.

    Xử lý yêu cầu có kèm ảnh. Đây là nánh phức tạp nhất của router vì
    ảnh có thể đi kèm nhiều mục đích khác nhau.

    **Cây quyết định (Decision Tree) theo thứ tự ưu tiên**::

        1. profile + outfit keyword  → analyze_then_style (multi-step workflow)
        2. profile keyword            → profile_analysis (analyze_body / skin_tone)
        3. outfit keyword             → outfit_advice (style_image_item)
        4. product keyword / category → product_discovery (find_similar)
        5. image_context is None      → action=inspect_image (API must run VLM first)
        6. VLM confidence >= threshold→ product_discovery (find_similar) + follow-up
        7. VLM confidence < threshold → clarification (hỏi người dùng muốn gì)

    Trong trường hợp 5 (``image_context is None``), hàm trả về một decision
    đặc biệt với ``action="inspect_image"`` và ``missing_slots=["image_context"]``.
    API cần gọi VLM rồi route lại thêm một lần với ``image_context`` đã có.

    Args:
        query (str): Văn bản đi kèm ảnh (có thể rỗng nếu chỉ gửi ảnh).
        image_context (dict | None): Kết quả VLM nếu đã chạy, None nếu chưa.

    Returns:
        IntentDecision: Decision với route hợp lệ hoặc yêu cầu clarification.
    """
    modality = MODALITY_TEXT_IMAGE if query.strip() else MODALITY_IMAGE
    entities = extract_basic_entities(query)
    trace = [_trace("modality", modality, "has_image=True")]

    profile_hit = keyword_hit(query, IMAGE_PROFILE_KEYWORDS)
    outfit_hit = keyword_hit(query, IMAGE_OUTFIT_KEYWORDS) or keyword_hit(query, DEFINITE_OUTFIT)
    product_hit = keyword_hit(query, DEFINITE_SEARCH) or bool(extract_basic_entities(query).get("categories"))

    if profile_hit and outfit_hit:
        return _decision(
            INTENT_PROFILE_ANALYSIS,
            modality,
            "analyze_then_style",
            query,
            confidence=0.99,
            source="modality_keyword",
            reason="Người dùng yêu cầu phân tích profile và phối đồ trong cùng lượt.",
            entities=entities,
            trace=trace + [_trace("keyword", "profile+outfit", query)],
            workflow=[ROUTE_PROFILE_VLM_ANALYSIS, ROUTE_TEXT_OUTFIT_ADVICE],
        )
    if profile_hit:
        action = "analyze_skin_tone" if keyword_hit(query, ["tone da", "mau da"]) else "analyze_full_profile"
        return _decision(
            INTENT_PROFILE_ANALYSIS,
            modality,
            action,
            query,
            confidence=0.98,
            source="modality_keyword",
            reason="Ảnh đi kèm yêu cầu phân tích người rõ ràng.",
            entities=entities,
            trace=trace + [_trace("keyword", "profile_analysis", query)],
        )
    if outfit_hit:
        return _decision(
            INTENT_OUTFIT_ADVICE,
            modality,
            "style_image_item",
            query,
            confidence=0.99,
            source="modality_keyword",
            reason="Ảnh đi kèm yêu cầu phối món trong ảnh.",
            entities=entities,
            trace=trace + [_trace("keyword", "image_outfit", query)],
        )
    if product_hit:
        return _decision(
            INTENT_PRODUCT_DISCOVERY,
            modality,
            "find_similar",
            query,
            confidence=0.98,
            source="modality_keyword",
            reason="Ảnh đi kèm yêu cầu tìm/xem sản phẩm rõ ràng.",
            entities=entities,
            trace=trace + [_trace("keyword", "image_product_search", query)],
        )

    if image_context is None:
        # API should call the VLM image-understanding helper, then route again with
        # the resulting structured context. This is not a clarification yet.
        decision = IntentDecision(
            intent=INTENT_UNKNOWN,
            modality=modality,
            action="inspect_image",
            route=None,
            confidence=0.0,
            certainty=CERTAINTY_CONTEXTUAL,
            rewrite_query=query.strip(),
            missing_slots=["image_context"],
            reason="Cần hiểu nội dung ảnh trước khi chọn hoặc hỏi lại pipeline.",
            source="modality_gate",
            trace=trace + [_trace("image_understanding", "required", "VLM has not run")],
        )
        return decision

    subject = str(image_context.get("subject", "unclear")).lower()
    confidence = float(image_context.get("confidence", 0.0) or 0.0)
    fashion_item = str(image_context.get("fashion_item", "")).strip()
    caption = _image_description(image_context)
    trace.append(
        _trace(
            "image_understanding",
            subject,
            f"confidence={confidence:.2f}; fashion_item={fashion_item or 'none'}",
        )
    )

    if fashion_item and confidence >= IMAGE_ROUTER_CONFIDENCE_THRESHOLD:
        search_query = query.strip() or f"Tìm sản phẩm giống {fashion_item} trong ảnh"
        return _decision(
            INTENT_PRODUCT_DISCOVERY,
            modality,
            "find_similar",
            search_query,
            confidence=confidence,
            source="image_context_default",
            reason="VLM nhận diện được món thời trang; mặc định tìm sản phẩm tương tự.",
            entities=entities,
            image_context=image_context,
            trace=trace,
            follow_up_question=(
                f"Mình nhận ra {caption}. Bạn có muốn mình gợi ý cách phối đồ với món này không?"
            ),
            follow_up_options=[
                {
                    "label": "Phối đồ với món này",
                    "value": "Phối đồ với món này",
                    "action": "style_image_item",
                },
                {
                    "label": "Không cần phối đồ",
                    "value": "Không cần",
                    "action": "keep_search_results",
                },
            ],
        )

    return _clarification(
        query,
        modality=modality,
        missing_slots=["image_goal"],
        question=(
            f"Mình thấy {caption}, nhưng chưa xác định chắc món bạn quan tâm. "
            "Bạn muốn tìm sản phẩm tương tự, phối đồ, hay phân tích người trong ảnh?"
        ),
        options=[
            {"label": "Tìm sản phẩm tương tự", "action": "find_similar"},
            {"label": "Gợi ý phối đồ", "action": "style_image_item"},
            {"label": "Phân tích người trong ảnh", "action": "analyze_full_profile"},
        ],
        reason="Ảnh hoặc món thời trang có độ tin cậy thấp, cần người dùng xác nhận mục tiêu.",
        source="image_context_clarify",
        image_context=image_context,
        trace=trace,
    )


def _route_pending_state(query: str, state: dict) -> IntentDecision | None:
    """Handle queries when the session has pending state awaiting user response.

    Xử lý yêu cầu khi session đang có trạng thái chờ xác nhận từ người dùng.
    Nếu không có pending state nào, hàm trả về ``None`` để router tiếp tục.

    **Các pending state được xử lý**:

    1. **pending_profile_candidate**: VLM vừa phân tích ảnh và đã có kết quả profile
       (dáng người, tone da). Bot hỏi người dùng có muốn lưu không.

       - Người dùng đồng ý ("ok", "đúng rồi"...) → ``confirm_candidate``
       - Người dùng từ chối ("không", "bỏ qua"...) → ``reject_candidate``

    2. **pending_image_docs**: Có ảnh candidates từ lượt trước chưa được
       xử lý hết.

       - Người dùng hỏi phối đồ → ``style_image_item`` kế thừa candidates
       - Người dùng từ chối ảnh → xóa pending image, trả lời social

    Args:
        query (str): Câu hỏi hiện tại của người dùng.
        state (dict): Session state hệ thống đang lưu giữ.

    Returns:
        IntentDecision | None: Decision nếu có pending state, None nếu không.
    """
    query_plain = plain_text(query)

    pending_profile = state.get("pending_profile_candidate")
    if pending_profile:
        trace = [_trace("state", "pending_profile_candidate", "awaiting confirmation")]
        if keyword_hit(query_plain, PROFILE_CONFIRM_KEYWORDS):
            return _decision(
                INTENT_PROFILE_MANAGEMENT,
                MODALITY_TEXT,
                "confirm_candidate",
                query,
                confidence=0.99,
                source="state",
                reason="Người dùng xác nhận lưu profile VLM đang chờ.",
                entities={"profile_candidate": dict(pending_profile)},
                trace=trace,
            )
        if keyword_hit(query_plain, PROFILE_REJECT_KEYWORDS):
            return _decision(
                INTENT_PROFILE_MANAGEMENT,
                MODALITY_TEXT,
                "reject_candidate",
                query,
                confidence=0.99,
                source="state",
                reason="Người dùng từ chối profile VLM đang chờ.",
                trace=trace,
            )

    pending_image_docs = state.get("pending_image_docs") or []
    if pending_image_docs:
        trace = [_trace("state", "pending_image_context", f"candidates={len(pending_image_docs)}")]
        if keyword_hit(query, IMAGE_OUTFIT_KEYWORDS) or keyword_hit(query, DEFINITE_OUTFIT):
            return _decision(
                INTENT_OUTFIT_ADVICE,
                MODALITY_TEXT_IMAGE,
                "style_image_item",
                query,
                confidence=0.98,
                source="state",
                reason="Yêu cầu phối đồ kế thừa ảnh và candidates ở lượt trước.",
                entities=extract_basic_entities(query),
                image_context=dict(state.get("pending_image_context") or {}),
                trace=trace,
            )
        if keyword_hit(query_plain, CANCEL_PENDING_IMAGE_KEYWORDS):
            return _decision(
                INTENT_SOCIAL,
                MODALITY_TEXT,
                "social",
                query,
                confidence=0.98,
                source="state",
                reason="Người dùng từ chối gợi ý tiếp theo từ ảnh.",
                entities={"clear_pending_image": True},
                trace=trace,
            )
    return None


def _route_profile_management(query: str) -> IntentDecision | None:
    """Detect and route explicit profile management operations from text.

    Phân loại các thao tác quản lý profile người dùng dựa trên từ khóa.
    Nếu không nhận ra thao tác profile nào, trả về ``None``.

    **Thứ tự ưu tiên kiểm tra**:

    1. **clear_all**: "xóa toàn bộ", "xóa hết profile" → xóa toàn bộ thông tin
    2. **update**: Phát hiện giá trị profile mới (dáng người, tone da, giới tính)
    3. **delete_field**: Phát hiện yêu cầu xóa trường cụ thể
    4. **read**: Hỏi xem profile đã lưu ("dang nguoi toi", "profile cua toi")

    Args:
        query (str): Câu query của người dùng.

    Returns:
        IntentDecision | None: Decision quản lý profile nếu nhận ra, None nếu không.
    """
    query_plain = plain_text(query)
    trace = [_trace("keyword", "profile_management", query)]
    if keyword_hit(query_plain, PROFILE_CLEAR_ALL_KEYWORDS):
        return _decision(
            INTENT_PROFILE_MANAGEMENT,
            MODALITY_TEXT,
            "clear_all",
            query,
            confidence=0.99,
            source="keyword",
            reason="Người dùng yêu cầu xóa toàn bộ profile.",
            trace=trace,
        )
    profile_entities = extract_profile_entities(query)
    if profile_entities.get("profile_updates"):
        return _decision(
            INTENT_PROFILE_MANAGEMENT,
            MODALITY_TEXT,
            "update",
            query,
            confidence=0.97,
            source="keyword",
            reason="Phát hiện giá trị profile mới rõ ràng.",
            entities=profile_entities,
            trace=trace,
        )
    if profile_entities.get("profile_delete_fields"):
        return _decision(
            INTENT_PROFILE_MANAGEMENT,
            MODALITY_TEXT,
            "delete_field",
            query,
            confidence=0.97,
            source="keyword",
            reason="Người dùng yêu cầu xóa một số trường profile.",
            entities=profile_entities,
            trace=trace,
        )
    if keyword_hit(query, DEFINITE_PROFILE_INQUIRY):
        return _decision(
            INTENT_PROFILE_MANAGEMENT,
            MODALITY_TEXT,
            "read",
            query,
            confidence=0.99,
            source="keyword",
            reason="Người dùng hỏi thông tin profile đã lưu.",
            trace=trace,
        )
    return None


CONTINUABLE_ROUTES = {
    ROUTE_TEXT_PRODUCT_SEARCH,
    ROUTE_IMAGE_PRODUCT_SEARCH,
    ROUTE_TEXT_OUTFIT_ADVICE,
    ROUTE_IMAGE_OUTFIT_ADVICE,
}


def route_from_keywords(
    query: str,
    state: dict | None = None,
    has_image: bool = False,
    image_context: dict | None = None,
) -> IntentDecision | None:
    """Resolve modality, state and high-precision language without calling LLM.

    Đây là **orchestrator Layer 1–3** của router: chạy toàn bộ logic
    deterministic/contextual trước khi gọi LLM. Nếu một trong các kiểm
    tra này khớp, LLM sẽ không được gọi.

    **Thứ tự kiểm tra** (sớm hơn = ưu tiên cao hơn):

    1. **Modality gate** (có ảnh): Đi thẳng vào ``_route_image_request()``
    2. **Pending state**: Hỏi xác nhận / hủy ảnh / profile candidate
    3. **Profile management**: Xóa/sử a/đọc profile
    4. **Follow-up "xem thêm"**: Kế thừa lượt truước nếu có, hỏi nếu không
    5. **Greeting/chitchat**: Từ khóa lời chào, cảm ơn, tạm biệt
    6. **Out-of-scope**: Regex pattern ngoài phạm vi nghiêm ngặt
    7. **Outfit keyword**: Các từ khóa phối đồ độ chính xác cao
    8. **Search keyword**: Các từ khóa tìm sản phẩm độ chính xác cao
    9. **Category heuristic**: Có danh mục thời trang rõ ràng (confidence thấp hơn)
   10. **None** → gọi LLM (``classify_intent_llm()``)

    Args:
        query (str): Câu query của người dùng.
        state (dict | None): Session state hiện tại.
        has_image (bool): True nếu request đi kèm file ảnh.
        image_context (dict | None): Kết quả VLM nếu đã chạy.

    Returns:
        IntentDecision | None: Decision nếu nhận ra, None nếu phải gọi LLM.
    """
    state = state or {}
    if has_image:
        return _route_image_request(query, image_context=image_context)

    pending_decision = _route_pending_state(query, state)
    if pending_decision:
        return pending_decision

    profile_decision = _route_profile_management(query)
    if profile_decision:
        return profile_decision

    if is_more_request(query):
        previous = state.get("last_route_decision")
        if previous and getattr(previous, "route", None) in CONTINUABLE_ROUTES:
            intent = getattr(previous, "intent", INTENT_PRODUCT_DISCOVERY)
            modality = getattr(previous, "modality", MODALITY_TEXT)
            action = "more" if intent == INTENT_PRODUCT_DISCOVERY else "refine_outfit"
            return _decision(
                intent,
                modality,
                action,
                getattr(previous, "rewrite_query", "") or state.get("last_query", query),
                confidence=0.98,
                source="state",
                reason="Follow-up kế thừa intent, modality và query trước.",
                entities=dict(getattr(previous, "entities", {}) or {}),
                image_context=dict(state.get("pending_image_context") or {}),
                trace=[_trace("state", "continue_previous", previous.route)],
            )
        return _clarification(
            query,
            modality=MODALITY_TEXT,
            missing_slots=["previous_search"],
            question="Bạn muốn xem thêm nhóm sản phẩm nào?",
            options=[
                {"label": "Áo", "action": "search_category", "value": "Áo"},
                {"label": "Quần", "action": "search_category", "value": "Quần"},
                {"label": "Váy/Đầm", "action": "search_category", "value": "Đầm"},
            ],
            reason="Không có lượt tìm kiếm trước để kế thừa.",
            source="state",
            trace=[_trace("state", "missing_previous_search")],
        )

    if keyword_hit(query, DEFINITE_GREETING):
        return _decision(
            INTENT_SOCIAL, MODALITY_TEXT, "greeting", query,
            confidence=0.99, source="keyword", reason="Khớp lời chào.",
            trace=[_trace("keyword", "greeting", query)],
        )
    if keyword_hit(query, DEFINITE_CHITCHAT):
        action = "goodbye" if keyword_hit(query, SOCIAL_GOODBYE_KEYWORDS) else "thanks"
        return _decision(
            INTENT_SOCIAL, MODALITY_TEXT, action, query,
            confidence=0.99, source="keyword", reason="Khớp câu xã giao.",
            trace=[_trace("keyword", action, query)],
        )
    if strict_out_of_scope_hit(query):
        return _decision(
            INTENT_OUT_OF_SCOPE, MODALITY_TEXT, "redirect", query,
            confidence=0.95, source="keyword", reason="Khớp mẫu ngoài phạm vi nghiêm ngặt.",
            trace=[_trace("keyword", "out_of_scope", query)],
        )
    if keyword_hit(query, DEFINITE_OUTFIT):
        return _decision(
            INTENT_OUTFIT_ADVICE,
            MODALITY_TEXT,
            "create_outfit",
            query,
            confidence=0.99,
            source="keyword",
            reason="Khớp từ khóa phối đồ độ chính xác cao.",
            entities=extract_basic_entities(query),
            trace=[_trace("keyword", "outfit_advice", query)],
        )
    if keyword_hit(query, DEFINITE_SEARCH):
        action = infer_product_action(query)
        return _decision(
            INTENT_PRODUCT_DISCOVERY,
            MODALITY_TEXT,
            action,
            query,
            confidence=0.99,
            source="keyword",
            reason="Khớp từ khóa tìm/hỏi sản phẩm độ chính xác cao.",
            entities=extract_basic_entities(query),
            trace=[_trace("keyword", action, query)],
        )
    if extract_basic_entities(query).get("categories"):
        action = infer_product_action(query)
        return _decision(
            INTENT_PRODUCT_DISCOVERY,
            MODALITY_TEXT,
            action,
            query,
            confidence=0.85,
            source="keyword_heuristic",
            reason="Có category thời trang rõ ràng; không cần gọi LLM router.",
            entities=extract_basic_entities(query),
            trace=[_trace("category_heuristic", action, query)],
        )
    return None


# ---------------------------------------------------------------------------
# LLM Classification Fallback
# ---------------------------------------------------------------------------
# ROUTE_CLASSIFY_PROMPT là prompt được đưa cho LLM để phân loại intent khi
# keyword routing không đủ để quyết định.
#
# **Thiết kế cố ý của prompt**:
# - LLM được phép chọn INTENT và ACTION
# - LLM KHÔNG được phép chọn ROUTE (dòng "Do not choose an execution route")
# - LLM KHÔNG được phép tự báo confidence score (tà thời gựt cậu) 
# - Schema JSON được inline vào prompt để buộc LLM trả về JSON có cấu trúc
# - context_block: đoạn tin nhắn bot trước (tối đa 300 ký tự) cho context
ROUTE_CLASSIFY_PROMPT = (
    "You classify the business intent of a Vietnamese fashion-shopping request.\n"
    "Return exactly one compact JSON object. Do not choose an execution route.\n\n"
    "Allowed intents:\n"
    "- product_discovery: find/view/compare products; ask price, size or stock.\n"
    "- outfit_advice: create/refine an outfit or ask what to wear.\n"
    "- profile_management: read/update/delete saved profile information.\n"
    "- social: greeting, thanks or goodbye.\n"
    "- out_of_scope: clearly unrelated to fashion and cannot be redirected naturally.\n"
    "- unknown: too vague to act safely.\n\n"
    "Stock data is unavailable; still classify stock questions as product_discovery.\n"
    "Weather plus 'what to wear' is outfit_advice, not out_of_scope.\n\n"
    "Previous context: {context_block}\n"
    "User query: {query}\n\n"
    "Do not report a confidence score and do not decide which fields are required.\n"
    "If the main goal is ambiguous, return intent=unknown and action=clarify.\n\n"
    "Schema:\n"
    "{{\"intent\":\"product_discovery|outfit_advice|profile_management|social|out_of_scope|unknown\","
    "\"action\":\"search|more|price_check|size_check|stock_check|compare|create_outfit|refine_outfit|read|update|delete_field|clear_all|greeting|thanks|goodbye|redirect|clarify\","
    "\"rewrite_query\":\"retrieval-ready Vietnamese query\","
    "\"entities\":{{}},\"reason\":\"short operational reason\"}}"
)

ollama_client = ollama.Client(host=OLLAMA_BASE_URL)


def extract_json_object(text: str) -> dict | None:
    """Robustly extract a JSON object from LLM output text.

    Phân tích an toàn JSON từ output của LLM, vốn thường không hoàn toàn
    sạch (có thể có suy nghĩ khởi đầu hoặc văn bản dư sau JSON).

    **Chiến lược 2 bước**:

    1. Cố parse toàn bộ text như JSON thuần túy (trường hợp lý tưởng)
    2. Nếu thất bại, dùng regex tìm đoạn ``{...}`` trong text rồi parse

    Args:
        text (str): Output thô của LLM.

    Returns:
        dict | None: Dict nếu parse được, None nếu không tìm thấy JSON hợp lệ.
    """
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def coerce_intent_decision(
    data: dict,
    query: str,
    *,
    modality: str = MODALITY_TEXT,
    source: str = "llm",
    state: dict | None = None,
    image_context: dict | None = None,
) -> IntentDecision:
    """Sanitize and validate LLM classification output into a safe IntentDecision.

    Biến output JSON từ LLM (có thể chứa giá trị sai hoặc thiếu) thành
    một ``IntentDecision`` đã được kiểm tra và an toàn.

    **Các bước sanitization**:

    1. Validate intent: phải nằm trong BUSINESS_INTENTS, nếu không → INTENT_UNKNOWN
    2. Bỏ qua confidence LLM tự báo (không đáng tin, ghi log để debug)
    3. Kiểm tra missing slots — nếu thiếu slot blocking, hỏi lại
    4. Coerce action về giá trị hợp lệ nếu LLM trả về action không nằm trong
       tập actions cho phép của intent đó
    5. Gọi ``resolve_route()`` qua ``_decision()`` để chọn route

    Args:
        data (dict): JSON đã parse từ LLM output.
        query (str): Query gốc của người dùng (dự phòng nếu LLM không trả rewrite_query).
        modality (str): Kiểu đầu vào hiện tại.
        source (str): Nguồn gọi hàm này (thường là "llm").
        state (dict | None): Session state để kiểm tra missing slots.
        image_context (dict | None): Kết quả VLM nếu đã chạy.

    Returns:
        IntentDecision: Decision đã sanitize, sẵn sàng cho caller sử dụng.
    """
    intent = str(data.get("intent", INTENT_UNKNOWN)).strip().lower()
    if intent not in BUSINESS_INTENTS | {INTENT_UNKNOWN}:
        intent = INTENT_UNKNOWN
    action = str(data.get("action") or "search").strip().lower()
    rewrite_query = str(data.get("rewrite_query") or query).strip() or query
    # Never use a confidence value self-reported by the model for routing.
    # Keep it only as an observable debug hint when an older model returns it.
    llm_reported_confidence = data.get("confidence")
    entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    if not entities:
        entities = extract_basic_entities(rewrite_query)
    reason = str(data.get("reason", "LLM intent classification."))
    trace_detail = f"action={action}"
    if llm_reported_confidence is not None:
        trace_detail += "; ignored_model_confidence=true"
    trace = [_trace("llm_intent", intent, trace_detail)]

    missing_slots = derive_missing_slots(
        intent,
        action,
        modality,
        state=state,
        image_context=image_context,
    )
    if missing_slots:
        return _clarification(
            query,
            modality=modality,
            missing_slots=missing_slots,
            question="Bạn muốn tìm một sản phẩm cụ thể hay muốn mình gợi ý cách phối đồ?",
            options=[
                {"label": "Tìm sản phẩm", "action": "search"},
                {"label": "Gợi ý phối đồ", "action": "create_outfit"},
            ],
            reason=reason,
            source=source,
            trace=trace,
        )

    allowed_actions = {
        INTENT_PRODUCT_DISCOVERY: PRODUCT_ACTIONS,
        INTENT_OUTFIT_ADVICE: OUTFIT_ACTIONS,
        INTENT_PROFILE_MANAGEMENT: PROFILE_MANAGEMENT_ACTIONS,
        INTENT_SOCIAL: SOCIAL_ACTIONS,
        INTENT_OUT_OF_SCOPE: {"redirect"},
    }.get(intent, {action})
    if action not in allowed_actions:
        action = {
            INTENT_PRODUCT_DISCOVERY: "search",
            INTENT_OUTFIT_ADVICE: "create_outfit",
            INTENT_PROFILE_MANAGEMENT: "read",
            INTENT_SOCIAL: "social",
            INTENT_OUT_OF_SCOPE: "redirect",
        }.get(intent, "clarify")

    return _decision(
        intent,
        modality,
        action,
        rewrite_query,
        # Compatibility only; certainty="llm_assisted" is the routing signal.
        confidence=0.0,
        source=source,
        reason=reason,
        entities=entities,
        trace=trace,
    )


def coerce_route_decision(
    data: dict,
    query: str,
    source: str = "llm",
) -> IntentDecision:
    """Backward-compatible wrapper for old positional notebook calls."""
    return coerce_intent_decision(data, query, source=source)


def classify_intent_llm(
    query: str,
    last_bot_msg: str = "",
    state: dict | None = None,
) -> IntentDecision:
    """Layer 4 fallback: call LLM to classify ambiguous Vietnamese queries.

    Đây là **Layer 4 (cuối cùng)** trong cỗi ra quyết định. Chỉ được gọi
    khi ``route_from_keywords()`` trả về ``None``.

    **Quy trình**:
    1. Format prompt với query + 300 ký tự context từ tin nhắn trước
    2. Gọi Ollama với LLM_MODEL, temperature=0 (deterministic nhất có thể)
    3. Parse JSON từ output bằng ``extract_json_object()``
    4. Sanitize qua ``coerce_intent_decision()``
    5. Nếu LLM lỗi hoặc parse thất bại: hỏi lại người dùng (fallback an toàn)

    **Phân loại chỉ, không route**: Xem comment ROUTE_CLASSIFY_PROMPT.

    Args:
        query (str): Câu query của người dùng.
        last_bot_msg (str): Tin nhắn bot trước (context cho LLM).
        state (dict | None): Session state để kiểm tra slots.

    Returns:
        IntentDecision: Decision từ LLM, hoặc clarification nếu LLM lỗi.
    """
    context_block = last_bot_msg[:300] if last_bot_msg else "none"
    try:
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": ROUTE_CLASSIFY_PROMPT.format(
                    query=query,
                    context_block=context_block,
                ),
            }],
            options={"temperature": 0, "num_predict": 100},
            format="json",
        )
        raw = response["message"]["content"].strip()
        parsed = extract_json_object(raw)
        if parsed:
            return coerce_intent_decision(parsed, query, source="llm", state=state)
    except Exception as exc:
        print(f"[WARN] LLM intent classify error: {exc}")

    return _clarification(
        query,
        modality=MODALITY_TEXT,
        missing_slots=["user_goal"],
        question="Bạn muốn tìm sản phẩm cụ thể hay muốn mình tư vấn phối đồ?",
        options=[
            {"label": "Tìm sản phẩm", "action": "search"},
            {"label": "Tư vấn phối đồ", "action": "create_outfit"},
        ],
        reason="Không phân loại chắc chắn được; hỏi lại an toàn hơn tự tìm sai.",
        source="fallback",
        trace=[_trace("llm_intent", "failed")],
    )


# Legacy name retained for callers that explicitly request the LLM layer.
classify_route_llm = classify_intent_llm


def route_user_request(
    query: str,
    last_bot_msg: str = "",
    state: dict | None = None,
    force_image_search: bool = False,
    has_image: bool = False,
    image_context: dict | None = None,
) -> IntentDecision:
    """Main public entry point: return one routing decision for a user request.

    Đây là **hàm duy nhất mà api.py nên gọi** khi cần định tuyến một yêu
    cầu. Nó chạy toàn bộ chuỗi layer đưới đây rồi trả về đúng một
    ``IntentDecision``.

    **Thứ tự thực hiện**::

        1. Nếu force_image_search=True  → lock route image_product_search ngay lập tức
        2. Gọi route_from_keywords()     → Layer 1–3 (deterministic/contextual)
        3. Nếu Layer 1–3 không match    → gọi classify_intent_llm() (Layer 4)

    **force_image_search** là cờ dành cho trường hợp API đã chạy image retrieval
    thành công và muốn đảm bảo route không bị thay đổi dù query có gì.

    Args:
        query (str): Câu query / tin nhắn của người dùng.
        last_bot_msg (str): Tin nhắn bot trước để cấp context cho LLM.
        state (dict | None): Session state hiện tại của conversation.
        force_image_search (bool): Nếu True, luôn trả về ROUTE_IMAGE_PRODUCT_SEARCH.
        has_image (bool): True nếu request có kèm ảnh.
        image_context (dict | None): Kết quả VLM nếu đã chạy.

    Returns:
        IntentDecision: Một quyết định routing duy nhất đã hoàn chỉnh.
    """
    state = state or {}
    if force_image_search:
        modality = MODALITY_TEXT_IMAGE if query.strip() else MODALITY_IMAGE
        return _decision(
            INTENT_PRODUCT_DISCOVERY,
            modality,
            "find_similar",
            query or "Tìm sản phẩm giống ảnh đã tải lên",
            confidence=1.0,
            source="modality_override",
            reason="Image retrieval đã có candidates, route được khóa vào image search.",
            entities=extract_basic_entities(query),
            image_context=image_context,
            trace=[_trace("modality_override", "image_product_search")],
        )
    fast_decision = route_from_keywords(
        query,
        state=state,
        has_image=has_image,
        image_context=image_context,
    )
    if fast_decision:
        return fast_decision
    return classify_intent_llm(query, last_bot_msg=last_bot_msg, state=state)


def detect_intent_llm(query: str, last_bot_msg: str = "") -> str:
    """Convenience wrapper: return handler string using LLM path only.

    Wrapper tiện lợi — gọi ``classify_intent_llm()`` và trả về
    handler string ngắn gọn thay vì full IntentDecision object.

    Args:
        query (str): Câu query của người dùng.
        last_bot_msg (str): Tin nhắn bot trước (context).

    Returns:
        str: Handler string ("search", "outfit", "clarify", ...).
    """
    return classify_intent_llm(query, last_bot_msg=last_bot_msg).handler


def detect_intent(query: str, last_bot_msg: str = "") -> str:
    """Convenience wrapper: return handler string using full routing pipeline.

    Wrapper tiện lợi — gọi ``route_user_request()`` (toàn bộ cỗi router)
    và trả về handler string ngắn gọn. Dùng khi chỉ cần biết handler
    mà không cần chi tiết routing đầy đủ.

    Args:
        query (str): Câu query của người dùng.
        last_bot_msg (str): Tin nhắn bot trước (context cho LLM nếu cần).

    Returns:
        str: Handler string ("search", "outfit", "clarify", ...).
    """
    return route_user_request(query, last_bot_msg=last_bot_msg).handler


def detect_gender(query: str) -> str:
    """Detect user gender context from Vietnamese query text.

    Nhận biết ngữ cảnh giới tính từ câu query (không phải giới tính thực
    của người dùng). Dùng để tùy biến gợi ý sản phẩm theo đối tượng
    đang được hỏi đến (vd: "tìm đồ cho bạn trai" → male).

    Args:
        query (str): Câu query của người dùng.

    Returns:
        str: "male", "female", hoặc "unknown" nếu không nhận ra.
    """
    query_plain = plain_text(query)
    female_keywords = ["nu", "con gai", "ban gai", "phu nu", "vo"]
    if keyword_hit(query, MALE_KEYWORDS):
        return "male"
    if keyword_hit(query_plain, female_keywords):
        return "female"
    return "unknown"


# ---------------------------------------------------------------------------
# Static Response Template Helpers
# ---------------------------------------------------------------------------
# Các hàm này trả về chuỗi phản hồi cố định không cần gọi LLM.
# Chúng được dùng bởi api.py để trả lời nhanh cho các route
# ROUTE_SOCIAL_RESPONSE, ROUTE_OUT_OF_SCOPE_REDIRECT và clarification.

def get_social_response(action: str) -> str:
    """Return a canned social response string for greeting, thanks or goodbye.

    Trả về câu trả lời xã giao cố định, không cần gọi LLM.
    Được sử dụng bởi ROUTE_SOCIAL_RESPONSE handler.

    Args:
        action (str): Loại tương tác xã hội: "greeting", "goodbye", hoặc bất kỳ (thanks).

    Returns:
        str: Câu trả lời xã giao phù hợp.
    """
    if action == "greeting":
        return (
            "Xin chào! Mình là trợ lý tư vấn thời trang của shop. "
            "Bạn muốn tìm sản phẩm, phối đồ, hay gửi ảnh để mình gợi ý hôm nay?"
        )
    if action == "goodbye":
        return "Cảm ơn bạn đã ghé shop. Hẹn gặp lại bạn nhé!"
    return "Rất vui được hỗ trợ bạn. Khi cần tìm đồ hoặc phối outfit, cứ nhắn mình nhé."


def get_greeting_response() -> str:
    """Return the standard greeting response string.

    Wrapper tiện lợi gọi ``get_social_response("greeting")``.

    Returns:
        str: Lời chào mừng của chatbot.
    """
    return get_social_response("greeting")


def get_chitchat_response(query: str) -> str:
    """Return an appropriate social response based on chitchat content.

    Phân loại "tạm biệt" hay "cảm ơn" rồi trả về câu phản hồi phù hợp.

    Args:
        query (str): Câu xã giao của người dùng.

    Returns:
        str: Câu phản hồi xã giao phù hợp.
    """
    action = "goodbye" if keyword_hit(query, SOCIAL_GOODBYE_KEYWORDS) else "thanks"
    return get_social_response(action)


def get_profile_inquiry_response(profile: dict) -> str:
    """Return a human-readable summary of the user's saved profile.

    Tạo câu trả lời mô tả profile hiện tại của người dùng khi họ hỏi xem
    hệ thống đang lưu gì về mình.

    Args:
        profile (dict): Dữ liệu profile người dùng đã lưu (gender, dang_nguoi, tone_da).

    Returns:
        str: Tóm tắt profile dưới dạng người dùng dễ đọc, hoặc thông báo chưa có thông tin.
    """
    if not profile:
        return (
            "Mình chưa biết nhiều về vóc dáng và sở thích của bạn. Bạn có thể chia sẻ trực tiếp "
            "hoặc gửi một bức ảnh; mình chỉ ghi nhớ khi bạn đồng ý."
        )
    return (
        "Những điều mình đang ghi nhớ về bạn: "
        f"giới tính/ngữ cảnh là {profile.get('gender', 'chưa rõ')}, "
        f"dáng người là {profile.get('dang_nguoi', 'chưa rõ')}, "
        f"tone da là {profile.get('tone_da', 'chưa rõ')}."
    )


def get_out_of_scope_response(query: str) -> str:
    """Return a polite redirect response for out-of-scope queries.

    Trả về câu trả lời lịch sự khi người dùng hỏi ngoài phạm vi thời trang,
    kèm hướng dẫn về những gì bot có thể hỗ trợ.

    Args:
        query (str): Câu query ngoài phạm vi (không dùng trong template này).

    Returns:
        str: Thông báo ngoài phạm vi + hướng dẫn dùng lại chatbot.
    """
    return (
        "Câu này hơi nằm ngoài phạm vi tư vấn thời trang của mình. "
        "Mình có thể giúp bạn chọn đồ theo thời tiết, dịp sử dụng, ngân sách hoặc phong cách."
    )


def get_clarify_response(decision: IntentDecision | None = None) -> str:
    """Return clarification question from decision, or a default fallback.

    Lấy câu hỏi làm rõ từ decision nếu có, hoặc trả về câu hỏi mặc định.
    Dùng bởi handler "clarify" trong api.py.

    Args:
        decision (IntentDecision | None): Decision chứa câu hỏi (nếu có).

    Returns:
        str: Câu hỏi làm rõ từ decision hoặc câu hỏi mặc định.
    """
    if decision and decision.clarification_question:
        return decision.clarification_question
    return "Bạn muốn tìm một sản phẩm cụ thể hay muốn mình tư vấn phối đồ?"
