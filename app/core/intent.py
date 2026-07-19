"""Intent understanding and deterministic route resolution for the chatbot.

The central design rule is intentionally simple:

    intent = what the user wants
    modality = what input they supplied
    action = the operation inside that intent
    route = the execution pipeline derived by Python policy

The LLM may classify ambiguous language, but it never invents or directly
selects an execution route.
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
INTENT_PRODUCT_DISCOVERY = "product_discovery"
INTENT_OUTFIT_ADVICE = "outfit_advice"
INTENT_PROFILE_ANALYSIS = "profile_analysis"
INTENT_PROFILE_MANAGEMENT = "profile_management"
INTENT_SOCIAL = "social"
INTENT_OUT_OF_SCOPE = "out_of_scope"
INTENT_UNKNOWN = "unknown"  # control result, not a business intent

BUSINESS_INTENTS = {
    INTENT_PRODUCT_DISCOVERY,
    INTENT_OUTFIT_ADVICE,
    INTENT_PROFILE_ANALYSIS,
    INTENT_PROFILE_MANAGEMENT,
    INTENT_SOCIAL,
    INTENT_OUT_OF_SCOPE,
}

# Input modality is orthogonal to intent.
MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
MODALITY_TEXT_IMAGE = "text_image"

# Execution routes. Only resolve_route() chooses these values.
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

# Compatibility aliases for older callers. New code should use the explicit
# route names above. They point to the same pipeline rather than duplicating it.
ROUTE_PRODUCT_SEARCH = ROUTE_TEXT_PRODUCT_SEARCH
ROUTE_OUTFIT_ADVICE = ROUTE_TEXT_OUTFIT_ADVICE
ROUTE_PROFILE_INQUIRY = ROUTE_PROFILE_STATE_HANDLER
ROUTE_OUT_OF_SCOPE = ROUTE_OUT_OF_SCOPE_REDIRECT
ROUTE_GREETING = ROUTE_SOCIAL_RESPONSE
ROUTE_CHITCHAT = ROUTE_SOCIAL_RESPONSE
ROUTE_CLARIFY = "clarify"  # control label only; never returned as a route


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

# `certainty` describes how the decision was produced. Unlike a probability
# emitted by an LLM, these values have an observable operational meaning.
CERTAINTY_DETERMINISTIC = "deterministic"
CERTAINTY_CONTEXTUAL = "contextual"
CERTAINTY_LLM_ASSISTED = "llm_assisted"
CERTAINTY_CLARIFICATION_REQUIRED = "clarification_required"

CONTEXTUAL_SOURCES = {
    "state",
    "modality_gate",
    "modality_keyword",
    "modality_override",
    "image_context_default",
    "image_context_clarify",
}

# Only these slots are allowed to block execution. Product attributes such as
# color, budget and size are optional filters, so the bot must not interrogate
# the customer for every field before performing a useful search.
BLOCKING_SLOTS = {"user_goal", "previous_search", "image_context", "image_goal"}


@dataclass
class IntentDecision:
    """Structured semantic decision plus an execution route.

    `trace` contains short operational stages for debugging. It deliberately
    avoids hidden chain-of-thought and stores only observable policy outcomes.
    """

    intent: str
    modality: str = MODALITY_TEXT
    action: str = "search"
    route: str | None = None
    # Deprecated compatibility field. Routing must use `certainty`, never this
    # numeric value. It remains temporarily so older notebooks/UI do not break.
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
        """Compatibility execution kind used by the existing chat loop."""
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
        """Old short intent name retained only for external compatibility."""
        return self.handler

    def to_debug_dict(self) -> dict:
        data = asdict(self)
        data["handler"] = self.handler
        return data


# Older imports continue to work while notebooks migrate to IntentDecision.
RouteDecision = IntentDecision


def _trace(stage: str, result: str, detail: str = "") -> dict:
    return {"stage": stage, "result": result, "detail": detail}


def strip_vietnamese_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text or ""))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def plain_text(text: str) -> str:
    return normalize_text(strip_vietnamese_accents(text))


def phrase_hit(text: str, phrase: str) -> bool:
    """Match one normalized word/phrase without substring false positives."""
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
    """Match accented or accent-free keywords at word/phrase boundaries."""
    query_normalized = normalize_text(query)
    query_plain = plain_text(query)
    for keyword in keywords:
        keyword_normalized = normalize_text(keyword)
        keyword_plain = plain_text(keyword)
        if phrase_hit(query_normalized, keyword_normalized) or phrase_hit(query_plain, keyword_plain):
            return True
    return False


def word_hit(text: str, keyword: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text))


COLOR_KEYWORDS = [
    "den", "trang", "xanh", "do", "hong", "be", "kem", "nau", "xam",
    "ghi", "vang", "tim", "cam", "bac", "vang dong",
]
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

COLOR_ACCENTED_WORDS = {
    "den": ["đen"], "trang": ["trắng"], "do": ["đỏ"], "hong": ["hồng"],
    "nau": ["nâu"], "xam": ["xám"], "vang": ["vàng"], "tim": ["tím"],
    "bac": ["bạc"],
}
COLOR_CONTEXT_WORDS = [
    "mau", "tone", "ao", "quan", "vay", "dam", "giay", "dep", "tui",
    "non", "kinh",
]


def extract_color_entities(query: str) -> list[str]:
    """Extract colors without confusing the verb `tìm` with the color `tím`."""
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
    """Match a category while preserving Vietnamese accent disambiguation.

    For example, `đầm` is a product category while `đảm bảo` is not. We match
    either the literal accent-free phrase typed by the user or an approved
    accented spelling, instead of stripping every accent before comparison.
    """
    query_normalized = normalize_text(query)
    forms = [category, *CATEGORY_ACCENTED_FORMS.get(category, [])]
    return any(phrase_hit(query_normalized, form) for form in forms)


def extract_basic_entities(query: str) -> dict:
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
    """Extract explicit profile updates/deletions from Vietnamese text."""
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
    query_plain = plain_text(query)
    return any(re.search(pattern, query_plain) for pattern in STRICT_OUT_OF_SCOPE_PATTERNS)


def is_more_request(query: str) -> bool:
    return keyword_hit(query, FOLLOWUP_MORE_KEYWORDS)


def infer_product_action(query: str) -> str:
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
    """Return an auditable certainty level from the decision mechanism."""
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

    Optional shopping filters (category, budget, color and size) never block a
    request. The system asks only when it cannot choose a safe workflow.
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
    """Map semantic fields to one of the finite execution pipelines."""
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
    return str(
        image_context.get("caption")
        or image_context.get("fashion_item")
        or "ảnh thời trang bạn vừa gửi"
    ).strip()


def _route_image_request(query: str, image_context: dict | None = None) -> IntentDecision:
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
    """Resolve modality, state and high-precision language without calling LLM."""
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
    """Return one semantic decision using deterministic stages before LLM."""
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
    return classify_intent_llm(query, last_bot_msg=last_bot_msg).handler


def detect_intent(query: str, last_bot_msg: str = "") -> str:
    return route_user_request(query, last_bot_msg=last_bot_msg).handler


def detect_gender(query: str) -> str:
    query_plain = plain_text(query)
    female_keywords = ["nu", "con gai", "ban gai", "phu nu", "vo"]
    if keyword_hit(query, MALE_KEYWORDS):
        return "male"
    if keyword_hit(query_plain, female_keywords):
        return "female"
    return "unknown"


def get_social_response(action: str) -> str:
    if action == "greeting":
        return (
            "Xin chào! Mình là trợ lý tư vấn thời trang của shop. "
            "Bạn muốn tìm sản phẩm, phối đồ, hay gửi ảnh để mình gợi ý hôm nay?"
        )
    if action == "goodbye":
        return "Cảm ơn bạn đã ghé shop. Hẹn gặp lại bạn nhé!"
    return "Rất vui được hỗ trợ bạn. Khi cần tìm đồ hoặc phối outfit, cứ nhắn mình nhé."


def get_greeting_response() -> str:
    return get_social_response("greeting")


def get_chitchat_response(query: str) -> str:
    action = "goodbye" if keyword_hit(query, SOCIAL_GOODBYE_KEYWORDS) else "thanks"
    return get_social_response(action)


def get_profile_inquiry_response(profile: dict) -> str:
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
    return (
        "Câu này hơi nằm ngoài phạm vi tư vấn thời trang của mình. "
        "Mình có thể giúp bạn chọn đồ theo thời tiết, dịp sử dụng, ngân sách hoặc phong cách."
    )


def get_clarify_response(decision: IntentDecision | None = None) -> str:
    if decision and decision.clarification_question:
        return decision.clarification_question
    return "Bạn muốn tìm một sản phẩm cụ thể hay muốn mình tư vấn phối đồ?"
