"""
app/core/history.py — Redis Chat History với Summarization
============================================================
Quản lý lịch sử hội thoại qua Redis.
Mặc định giữ một cửa sổ gần; có thể bật tóm tắt bằng biến môi trường.
"""

import ollama
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import SystemMessage

from app.config import (
    HISTORY_ENABLE_SUMMARIZATION,
    HISTORY_MAX_MESSAGES,
    HISTORY_RECENT_KEEP,
    SUMMARIZE_MAX_TOKENS,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    REDIS_URL,
)
from app.core.llm import SUMMARIZE_PROMPT

ollama_client = ollama.Client(host=OLLAMA_BASE_URL)


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
    resp = ollama_client.chat(
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
    - Dưới HISTORY_MAX_MESSAGES: giữ nguyên.
    - Trên ngưỡng: giữ HISTORY_RECENT_KEEP message gần nhất.
    - Chỉ gọi LLM tóm tắt nếu HISTORY_ENABLE_SUMMARIZATION=True.

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

    # Vượt ngưỡng → mặc định chỉ giữ cửa sổ gần để không phát sinh thêm LLM call.
    old_messages    = messages[:-HISTORY_RECENT_KEEP]
    recent_messages = messages[-HISTORY_RECENT_KEEP:]

    history.clear()
    if HISTORY_ENABLE_SUMMARIZATION:
        summary_text = summarize_history(old_messages)
        history.add_message(SystemMessage(
            content=f"[TÓM TẮT HỘI THOẠI TRƯỚC]: {summary_text}",
        ))
    history.add_messages(recent_messages)

    return history


print("[OK] Redis history ready.")
