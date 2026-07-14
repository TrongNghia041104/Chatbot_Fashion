"""Lazy public core API for the Fashion RAG chatbot.

This package intentionally avoids eager imports. Importing app.core.intent should
not load LangChain chains, torch, transformers, or model checkpoints.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "BGEM3Embeddings": "app.core.embeddings",
    "ViFashionCLIPTextEmbeddings": "app.core.embeddings",
    "get_product_embeddings": "app.core.embeddings",
    "get_rule_embeddings": "app.core.embeddings",
    "get_qdrant_client": "app.core.vector_store",
    "get_product_vector_db": "app.core.vector_store",
    "get_product_retriever": "app.core.vector_store",
    "detect_image_type": "app.core.vision",
    "analyze_person_image": "app.core.vision",
    "caption_product_image": "app.core.vision",
    "search_products_by_image": "app.core.image_search",
    "RouteDecision": "app.core.intent",
    "route_user_request": "app.core.intent",
    "detect_intent": "app.core.intent",
    "detect_gender": "app.core.intent",
    "get_greeting_response": "app.core.intent",
    "get_chitchat_response": "app.core.intent",
    "get_profile_inquiry_response": "app.core.intent",
    "get_out_of_scope_response": "app.core.intent",
    "get_clarify_response": "app.core.intent",
    "build_outfit_context": "app.core.outfit",
    "get_full_chat_chain": "app.core.chains",
    "get_product_answer_chain": "app.core.chains",
    "get_image_search_chain": "app.core.chains",
    "get_outfit_chain": "app.core.chains",
    "format_documents_for_llm": "app.core.llm",
    "validate_user_query": "app.core.security",
    "extract_product_ids_from_docs": "app.core.security",
    "extract_product_ids_from_text": "app.core.security",
    "check_answer_grounding": "app.core.security",
    "append_chat_turn_log": "app.core.security",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
