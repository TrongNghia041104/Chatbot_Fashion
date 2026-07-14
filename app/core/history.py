"""
app/core/history.py — Redis Chat History với Summarization
============================================================
Quản lý lịch sử hội thoại qua Redis.
Tự động tóm tắt khi lịch sử vượt ngưỡng để giữ context window nhỏ gọn.
"""

import ollama
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import SystemMessage

from app.config import (
    REDIS_URL,
    LLM_MODEL,
    HISTORY_MAX_MESSAGES,
    HISTORY_RECENT_KEEP,
    SUMMARIZE_MAX_TOKENS,
)
from app.core.llm import SUMMARIZE_PROMPT


def summarize_history(messages: list) -> str:
    """
    Dùng LLM tóm tắt lịch sử hội thoại cũ thành đoạn văn ngắn.

    Args:
        messages: Danh sách LangChain message objects.

    Returns:
        Chuỗi tóm tắt.
    """
    history_text = "\n".join([
        f"{'Khách' if m.type == 'human' else 'Bot'}: {m.content[:300]}"
        for m in messages
    ])
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": SUMMARIZE_PROMPT.format(history_text=history_text),
        }],
        options={"temperature": 0, "num_predict": SUMMARIZE_MAX_TOKENS},
    )
    return resp["message"]["content"].strip()


def get_message_history(session_id: str) -> RedisChatMessageHistory:
    """
    Lấy lịch sử hội thoại từ Redis, có auto-summarization.

    Chiến lược:
    - Dưới HISTORY_MAX_MESSAGES: giữ nguyên, không tóm tắt.
    - Trên HISTORY_MAX_MESSAGES: tóm tắt phần cũ, giữ HISTORY_RECENT_KEEP messages gần nhất.

    Args:
        session_id: ID phiên hội thoại.

    Returns:
        RedisChatMessageHistory đã được xử lý.
    """
    history  = RedisChatMessageHistory(session_id, url=REDIS_URL)
    messages = history.messages

    # Chưa vượt ngưỡng → không cần tóm tắt
    if len(messages) <= HISTORY_MAX_MESSAGES:
        return history

    # Vượt ngưỡng → tóm tắt phần cũ, giữ messages gần nhất
    old_messages    = messages[:-HISTORY_RECENT_KEEP]
    recent_messages = messages[-HISTORY_RECENT_KEEP:]

    summary_text = summarize_history(old_messages)

    history.clear()
    history.add_message(SystemMessage(
        content=f"[TÓM TẮT HỘI THOẠI TRƯỚC]: {summary_text}",
    ))
    history.add_messages(recent_messages)

    return history


print("[OK] Redis history ready.")
