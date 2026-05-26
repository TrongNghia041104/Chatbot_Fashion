"""
app/core/__init__.py — Core package cho Fashion RAG Chatbot
============================================================
Re-export tất cả public API để import gọn hơn:

    from app.core import full_chat_chain, outfit_chain_with_history, vector_db
    from app.core import detect_intent, detect_gender, build_outfit_context
    from app.core import detect_image_type, analyze_person_image, caption_product_image
    from app.core import get_greeting_response, get_chitchat_response
"""

from app.core.embeddings import BGEM3Embeddings
from app.core.vector_store import vector_db, retriever, client as qdrant_client
from app.core.vision import detect_image_type, analyze_person_image, caption_product_image
from app.core.intent import (
    detect_intent, detect_gender,
    get_greeting_response, get_chitchat_response,
)
from app.core.outfit import build_outfit_context
from app.core.chains import full_chat_chain, outfit_chain_with_history

__all__ = [
    "BGEM3Embeddings",
    "vector_db", "retriever", "qdrant_client",
    "detect_image_type", "analyze_person_image", "caption_product_image",
    "detect_intent", "detect_gender",
    "get_greeting_response", "get_chitchat_response",
    "build_outfit_context",
    "full_chat_chain", "outfit_chain_with_history",
]
