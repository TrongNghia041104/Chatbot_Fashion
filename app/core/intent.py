"""
app/core/intent.py — Intent Detection & Response Helpers
=========================================================
Phân loại intent của câu hỏi người dùng bằng hybrid approach:
1. Keyword matching nhanh (ưu tiên)
2. LLM classification (fallback)

Intent types: outfit | search | greeting | chitchat
"""

import ollama

from app.config import (
    LLM_MODEL,
    DEFINITE_GREETING,
    DEFINITE_CHITCHAT,
    DEFINITE_OUTFIT,
    DEFINITE_SEARCH,
    MALE_KEYWORDS,
)
from app.core.llm import INTENT_CLASSIFY_PROMPT


def detect_intent_llm(query: str, last_bot_msg: str = "") -> str:
    """
    Phân loại intent bằng LLM (dùng khi keyword matching không đủ rõ).

    Returns:
        "outfit" | "search" | "chitchat" | "greeting"
    """
    context_block = ""
    if last_bot_msg:
        context_block = f'\nContext — Bot vừa nói: "{last_bot_msg[:120]}..."\n'

    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": INTENT_CLASSIFY_PROMPT.format(
                    query=query,
                    context_block=context_block,
                ),
            }],
            options={"temperature": 0, "num_predict": 10},
        )
        result = resp["message"]["content"].strip().upper()
        for intent in ["OUTFIT", "SEARCH", "CHITCHAT", "GREETING"]:
            if intent in result:
                return intent.lower()
    except Exception as e:
        print(f"[WARN] LLM intent lỗi: {e} → fallback search")

    return "search"


def detect_intent(query: str, last_bot_msg: str = "") -> str:
    """
    Phân loại intent bằng hybrid: keyword → LLM fallback.

    Args:
        query: Câu hỏi của người dùng.
        last_bot_msg: Tin nhắn cuối cùng của bot (để context).

    Returns:
        "outfit" | "search" | "greeting" | "chitchat"
    """
    q = query.lower().strip()
    if any(kw in q for kw in DEFINITE_OUTFIT):   return "outfit"
    if any(kw in q for kw in DEFINITE_SEARCH):   return "search"
    if any(kw in q for kw in DEFINITE_GREETING): return "greeting"
    if any(kw in q for kw in DEFINITE_CHITCHAT): return "chitchat"
    return detect_intent_llm(query, last_bot_msg)


def detect_gender(query: str) -> str:
    """
    Phát hiện giới tính từ câu hỏi. Mặc định là "female".

    Returns:
        "male" hoặc "female"
    """
    return "male" if any(kw in query.lower() for kw in MALE_KEYWORDS) else "female"


def get_greeting_response() -> str:
    """Trả về câu chào hỏi mặc định của bot."""
    return (
        "Xin chào! Mình là trợ lý tư vấn thời trang của shop. "
        "Bạn cần tìm sản phẩm hay muốn được gợi ý phối đồ hôm nay? 😊"
    )


def get_chitchat_response(query: str) -> str:
    """Trả về câu phản hồi chitchat mặc định."""
    return "Rất vui được hỗ trợ bạn! Bạn còn muốn hỏi thêm gì về thời trang không?"
