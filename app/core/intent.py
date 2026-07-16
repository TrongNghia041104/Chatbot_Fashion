"""Structured request routing for the Fashion RAG chatbot."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

import ollama

from app.config import (
    DEFINITE_CHITCHAT,
    DEFINITE_GREETING,
    DEFINITE_OUTFIT,
    DEFINITE_PROFILE_INQUIRY,
    DEFINITE_SEARCH,
    FOLLOWUP_MORE_KEYWORDS,
    LLM_MODEL,
    MALE_KEYWORDS,
    STRICT_OUT_OF_SCOPE_PATTERNS,
)


ROUTE_PRODUCT_SEARCH = "product_search"
ROUTE_IMAGE_PRODUCT_SEARCH = "image_product_search"
ROUTE_OUTFIT_ADVICE = "outfit_advice"
ROUTE_PROFILE_INQUIRY = "profile_inquiry"
ROUTE_OUT_OF_SCOPE = "out_of_scope"
ROUTE_GREETING = "greeting"
ROUTE_CHITCHAT = "chitchat"
ROUTE_CLARIFY = "clarify"

ROUTE_TO_INTENT = {
    ROUTE_PRODUCT_SEARCH: "search",
    ROUTE_IMAGE_PRODUCT_SEARCH: "image_search",
    ROUTE_OUTFIT_ADVICE: "outfit",
    ROUTE_PROFILE_INQUIRY: "profile_inquiry",
    ROUTE_OUT_OF_SCOPE: "out_of_scope",
    ROUTE_GREETING: "greeting",
    ROUTE_CHITCHAT: "chitchat",
    ROUTE_CLARIFY: "clarify",
}

ROUTE_CLASSIFY_PROMPT = (
    "You are the router for a Vietnamese fashion shopping chatbot.\n"
    "Pick exactly one route:\n"
    "- product_search: find/view/compare products, ask price, size, stock, images.\n"
    "- outfit_advice: styling/mix-match/what to wear for occasion, body shape, weather context.\n"
    "- profile_inquiry: user asks what body shape, skin tone, gender, budget, size, or preferences the bot remembers.\n"
    "- out_of_scope: clearly unrelated to fashion/shopping and cannot be gracefully redirected.\n"
    "- greeting: short hello/start conversation.\n"
    "- chitchat: thanks, goodbye, social message.\n"
    "- clarify: too vague and previous context is not enough.\n\n"
    "Important rules:\n"
    "- If a query mentions weather but asks what to wear, choose outfit_advice, not out_of_scope.\n"
    "- If a query asks 'con size L khong' choose product_search with action=size_check, but note actual stock may be unknown.\n"
    "- If user says 'xem them' and previous context exists, state resolver handles it before this prompt.\n"
    "- Be strict with out_of_scope, but friendly: only choose it when no fashion redirection is natural.\n\n"
    "Last bot message/context: {context_block}\n"
    "User query: {query}\n\n"
    "Return exactly one compact JSON object, no markdown, with this schema:\n"
    "{{\"route\":\"product_search|outfit_advice|profile_inquiry|out_of_scope|greeting|chitchat|clarify\","
    "\"action\":\"search|more|mix_match|size_check|price_check|answer|redirect|clarify\","
    "\"confidence\":0.0,"
    "\"rewrite_query\":\"short retrieval-ready query\","
    "\"entities\":{{}},"
    "\"missing_slots\":[],"
    "\"reason\":\"short reason\"}}"
)


@dataclass
class RouteDecision:
    """Structured routing result: stable route + expandable action/entities."""

    route: str
    action: str = "answer"
    confidence: float = 0.0
    rewrite_query: str = ""
    entities: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    reason: str = ""
    source: str = "router"

    @property
    def intent(self) -> str:
        return ROUTE_TO_INTENT.get(self.route, "search")


CONTINUABLE_ROUTES = {ROUTE_PRODUCT_SEARCH, ROUTE_IMAGE_PRODUCT_SEARCH, ROUTE_OUTFIT_ADVICE}

COLOR_KEYWORDS = [
    "den",
    "trang",
    "xanh",
    "do",
    "hong",
    "be",
    "kem",
    "nau",
    "xam",
    "ghi",
    "vang",
    "tim",
    "cam",
    "bac",
    "vang dong",
]
CATEGORY_KEYWORDS = [
    "ao thun",
    "ao so mi",
    "so mi",
    "ao khoac",
    "ao len",
    "ao",
    "quan jean",
    "quan dai",
    "quan short",
    "quan",
    "chan vay",
    "vay",
    "dam",
    "jumpsuit",
    "giay",
    "dep",
    "tui xach",
    "phu kien",
]
OCCASION_KEYWORDS = [
    "di lam",
    "di hoc",
    "di tiec",
    "di choi",
    "di bien",
    "hen ho",
    "du lich",
    "cong so",
    "hang ngay",
    "du dam cuoi",
    "troi lanh",
    "thoi tiet lanh",
]

COLOR_ACCENTED_WORDS = {
    "den": ["đen"],
    "trang": ["trắng"],
    "do": ["đỏ"],
    "hong": ["hồng"],
    "nau": ["nâu"],
    "xam": ["xám"],
    "vang": ["vàng"],
    "tim": ["tím"],
    "bac": ["bạc"],
}
COLOR_CONTEXT_WORDS = [
    "mau",
    "tone",
    "ao",
    "quan",
    "vay",
    "dam",
    "giay",
    "dep",
    "tui",
    "non",
    "kinh",
]


def strip_vietnamese_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def plain_text(text: str) -> str:
    return normalize_text(strip_vietnamese_accents(text))


def keyword_hit(query: str, keywords: list[str]) -> bool:
    q_norm = normalize_text(query)
    q_plain = plain_text(query)
    for keyword in keywords:
        kw_norm = normalize_text(keyword)
        kw_plain = plain_text(keyword)
        if kw_norm and (kw_norm in q_norm or kw_plain in q_plain):
            return True
    return False


def word_hit(text: str, keyword: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text))


def extract_color_entities(query: str) -> list[str]:
    """Extract color words without confusing search verbs like 'tìm' with 'tím'."""
    q_norm = normalize_text(query)
    q_plain = plain_text(query)
    colors = []

    for color in COLOR_KEYWORDS:
        color_plain = plain_text(color)
        accented_words = COLOR_ACCENTED_WORDS.get(color_plain, [])
        has_accented_hit = any(word_hit(q_norm, word) for word in accented_words)
        has_contextual_plain_hit = any(
            re.search(rf"(?<!\w){context}\s+{re.escape(color_plain)}(?!\w)", q_plain)
            for context in COLOR_CONTEXT_WORDS
        )
        has_modifier_hit = re.search(
            rf"(?<!\w){re.escape(color_plain)}\s+(pastel|dam|nhat|than|dong)(?!\w)",
            q_plain,
        )
        if has_accented_hit or has_contextual_plain_hit or has_modifier_hit:
            colors.append(color)

    return colors


def strict_out_of_scope_hit(query: str) -> bool:
    q_plain = plain_text(query)
    return any(re.search(pattern, q_plain) for pattern in STRICT_OUT_OF_SCOPE_PATTERNS)


def is_more_request(query: str) -> bool:
    q_plain = plain_text(query)
    return any(plain_text(keyword) in q_plain for keyword in FOLLOWUP_MORE_KEYWORDS)


def infer_product_action(query: str) -> str:
    q = plain_text(query)
    if is_more_request(query):
        return "more"
    if "size" in q or "kich co" in q or re.search(r"\b(xs|s|m|l|xl|xxl|xxxl)\b", q):
        return "size_check"
    if any(keyword in q for keyword in ["gia", "bao nhieu", "duoi", "tren", "re hon", "dat hon"]):
        return "price_check"
    if any(keyword in q for keyword in ["con hang", "het hang", "ton kho"]):
        return "stock_check"
    if any(keyword in q for keyword in ["so sanh", "khac nhau", "nen chon"]):
        return "compare"
    return "search"


def extract_basic_entities(query: str) -> dict:
    q = plain_text(query)
    entities = {}
    colors = extract_color_entities(query)
    if colors:
        entities["colors"] = colors
    categories = [cat for cat in CATEGORY_KEYWORDS if cat in q]
    if categories:
        entities["categories"] = sorted(categories, key=len, reverse=True)
    occasions = [occasion for occasion in OCCASION_KEYWORDS if occasion in q]
    if occasions:
        entities["occasions"] = occasions
    sizes = re.findall(r"\b(?:size\s*)?(xs|s|m|l|xl|xxl|xxxl|[2-5][0-9])\b", q, flags=re.IGNORECASE)
    if sizes:
        entities["sizes"] = sorted(set(size.upper() for size in sizes))
    budget = re.search(r"(?:duoi|nho hon|under|toi da|max)\s*(\d+(?:[\.,]\d+)?)\s*(k|nghin|ngan|trieu|m|vnd|d)?", q)
    if budget:
        entities["budget_text"] = budget.group(0)
    return entities


def normalize_route(raw_route: str) -> str:
    route = plain_text(raw_route).replace(" ", "_").replace("-", "_")
    mapping = {
        "search": ROUTE_PRODUCT_SEARCH,
        "product": ROUTE_PRODUCT_SEARCH,
        "product_search": ROUTE_PRODUCT_SEARCH,
        "image_search": ROUTE_IMAGE_PRODUCT_SEARCH,
        "image_product_search": ROUTE_IMAGE_PRODUCT_SEARCH,
        "outfit": ROUTE_OUTFIT_ADVICE,
        "outfit_advice": ROUTE_OUTFIT_ADVICE,
        "mix_match": ROUTE_OUTFIT_ADVICE,
        "profile": ROUTE_PROFILE_INQUIRY,
        "profile_inquiry": ROUTE_PROFILE_INQUIRY,
        "user_profile": ROUTE_PROFILE_INQUIRY,
        "out_of_scope": ROUTE_OUT_OF_SCOPE,
        "oos": ROUTE_OUT_OF_SCOPE,
        "greeting": ROUTE_GREETING,
        "hello": ROUTE_GREETING,
        "chitchat": ROUTE_CHITCHAT,
        "smalltalk": ROUTE_CHITCHAT,
        "clarify": ROUTE_CLARIFY,
        "unclear": ROUTE_CLARIFY,
    }
    return mapping.get(route, ROUTE_PRODUCT_SEARCH)


def route_from_keywords(query: str, state: dict | None = None) -> RouteDecision | None:
    state = state or {}
    if is_more_request(query):
        previous = state.get("last_route_decision")
        if previous and getattr(previous, "route", None) in CONTINUABLE_ROUTES:
            return RouteDecision(
                route=previous.route,
                action="more",
                confidence=0.98,
                rewrite_query=previous.rewrite_query or state.get("last_query", query),
                entities=dict(previous.entities),
                reason="Follow-up more: reuse previous route/query.",
                source="state",
            )
        return RouteDecision(
            route=ROUTE_CLARIFY,
            action="need_previous_search",
            confidence=0.90,
            rewrite_query=query,
            missing_slots=["previous_search"],
            reason="User asked for more, but there is no previous searchable turn.",
            source="keyword",
        )

    if keyword_hit(query, DEFINITE_GREETING):
        return RouteDecision(ROUTE_GREETING, "answer", 0.99, query, reason="Matched greeting keyword.", source="keyword")
    if keyword_hit(query, DEFINITE_CHITCHAT):
        return RouteDecision(ROUTE_CHITCHAT, "answer", 0.99, query, reason="Matched chitchat keyword.", source="keyword")
    if keyword_hit(query, DEFINITE_PROFILE_INQUIRY):
        return RouteDecision(ROUTE_PROFILE_INQUIRY, "answer", 0.99, query, reason="User asks about saved profile.", source="keyword")
    if strict_out_of_scope_hit(query):
        return RouteDecision(ROUTE_OUT_OF_SCOPE, "redirect", 0.95, query, reason="Strict out-of-scope pattern.", source="keyword")
    if keyword_hit(query, DEFINITE_OUTFIT):
        return RouteDecision(
            route=ROUTE_OUTFIT_ADVICE,
            action="mix_match",
            confidence=0.99,
            rewrite_query=query,
            entities=extract_basic_entities(query),
            reason="Matched high-precision outfit keyword.",
            source="keyword",
        )
    if keyword_hit(query, DEFINITE_SEARCH):
        return RouteDecision(
            route=ROUTE_PRODUCT_SEARCH,
            action=infer_product_action(query),
            confidence=0.99,
            rewrite_query=query,
            entities=extract_basic_entities(query),
            reason="Matched high-precision product-search keyword.",
            source="keyword",
        )
    # Heuristic: query mentions a clothing category → almost always product search.
    # This avoids calling the LLM router for common queries like "áo thún màu đen" or "váy đi tiệc".
    if keyword_hit(query, CATEGORY_KEYWORDS):
        return RouteDecision(
            route=ROUTE_PRODUCT_SEARCH,
            action=infer_product_action(query),
            confidence=0.85,
            rewrite_query=query,
            entities=extract_basic_entities(query),
            reason="Clothing category keyword heuristic → product search (no LLM needed).",
            source="keyword_heuristic",
        )
    return None


def extract_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def coerce_route_decision(data: dict, query: str, source: str) -> RouteDecision:
    route = normalize_route(str(data.get("route", ROUTE_PRODUCT_SEARCH)))
    action = str(data.get("action") or ("mix_match" if route == ROUTE_OUTFIT_ADVICE else "search"))
    rewrite_query = str(data.get("rewrite_query") or query).strip() or query
    try:
        confidence = float(data.get("confidence", 0.70))
    except Exception:
        confidence = 0.70
    entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    missing_slots = data.get("missing_slots") if isinstance(data.get("missing_slots"), list) else []
    if not entities:
        entities = extract_basic_entities(rewrite_query)
    return RouteDecision(
        route=route,
        action=action,
        confidence=max(0.0, min(1.0, confidence)),
        rewrite_query=rewrite_query,
        entities=entities,
        missing_slots=missing_slots,
        reason=str(data.get("reason", "LLM route classification.")),
        source=source,
    )


def classify_route_llm(query: str, last_bot_msg: str = "") -> RouteDecision:
    context_block = last_bot_msg[:300] if last_bot_msg else "none"
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": ROUTE_CLASSIFY_PROMPT.format(query=query, context_block=context_block),
                }
            ],
            options={"temperature": 0, "num_predict": 80},  # Reduced from 220 — JSON route is short
            format="json",
        )
        raw = response["message"]["content"].strip()
        parsed = extract_json_object(raw)
        if parsed:
            return coerce_route_decision(parsed, query, source="llm")
        return RouteDecision(
            normalize_route(raw),
            "search",
            0.55,
            query,
            extract_basic_entities(query),
            reason=f"LLM non-JSON: {raw[:80]}",
            source="llm_fallback",
        )
    except Exception as exc:
        print(f"[WARN] LLM route classify error: {exc} -> fallback product_search")
    return RouteDecision(
        ROUTE_PRODUCT_SEARCH,
        infer_product_action(query),
        0.40,
        query,
        extract_basic_entities(query),
        reason="Router failed; safest fallback is product search.",
        source="fallback",
    )


def route_user_request(
    query: str,
    last_bot_msg: str = "",
    state: dict | None = None,
    force_image_search: bool = False,
) -> RouteDecision:
    if force_image_search:
        return RouteDecision(
            ROUTE_IMAGE_PRODUCT_SEARCH,
            "search_similar_image",
            1.0,
            query,
            extract_basic_entities(query),
            reason="Product image retrieval already returned candidates.",
            source="modality",
        )
    fast = route_from_keywords(query, state=state)
    if fast:
        return fast
    return classify_route_llm(query, last_bot_msg=last_bot_msg)


def detect_intent_llm(query: str, last_bot_msg: str = "") -> str:
    return classify_route_llm(query, last_bot_msg=last_bot_msg).intent


def detect_intent(query: str, last_bot_msg: str = "") -> str:
    return route_user_request(query, last_bot_msg=last_bot_msg).intent


def detect_gender(query: str) -> str:
    return "male" if keyword_hit(query, MALE_KEYWORDS) else "unknown"


def get_greeting_response() -> str:
    return (
        "Xin chào! Mình là trợ lý tư vấn thời trang của shop. "
        "Bạn muốn tìm sản phẩm, phối đồ, hay gửi ảnh để mình gợi ý hôm nay?"
    )


def get_chitchat_response(query: str) -> str:
    return "Rất vui được hỗ trợ bạn. Khi cần tìm đồ hoặc phối outfit, cứ nhắn mình nhé."


def get_profile_inquiry_response(profile: dict) -> str:
    if not profile:
        return "Mình chưa có thông tin dáng người hoặc tone da của bạn. Bạn có thể gửi ảnh người hoặc mô tả vóc dáng để mình tư vấn sát hơn nhé."

    gender = profile.get("gender", "chưa rõ")
    body = profile.get("dang_nguoi", "chưa rõ")
    tone = profile.get("tone_da", "chưa rõ")
    return (
        f"Mình đang lưu thông tin của bạn như sau: giới tính/ngữ cảnh: {gender}, "
        f"dáng người: {body}, tone da: {tone}. "
        "Mình sẽ dùng các thông tin này để gợi ý outfit và sản phẩm phù hợp hơn."
    )


def get_out_of_scope_response(query: str) -> str:
    return (
        "Câu này hơi nằm ngoài phạm vi tư vấn thời trang của mình. "
        "Nhưng nếu bạn muốn, mình có thể kéo nó về chuyện mặc đẹp: ví dụ chọn đồ theo thời tiết, dịp đi chơi, ngân sách hoặc phong cách bạn thích."
    )


def get_clarify_response(decision: RouteDecision | None = None) -> str:
    if decision and "previous_search" in decision.missing_slots:
        return "Bạn muốn xem thêm nhóm sản phẩm nào? Ví dụ: 'xem thêm áo thun trắng' hoặc 'xem thêm váy đi tiệc'."
    return "Mình cần bạn nói rõ hơn một chút: bạn muốn tìm sản phẩm cụ thể hay muốn mình tư vấn phối đồ?"
