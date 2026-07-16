"""Lazy LangChain pipeline assembly."""

from __future__ import annotations

from threading import Lock

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.history import get_message_history
from app.core.llm import QA_PROMPT, contextualize_q_prompt, doc_prompt, llm, outfit_prompt
from app.core.vector_store import get_product_retriever


_full_chat_chain = None
_fast_search_chain = None
_product_answer_chain = None
_outfit_chain = None
_chain_lock = Lock()


def get_full_chat_chain():
    """Full text-search RAG chain: rewrite -> retrieve -> answer."""
    global _full_chat_chain
    if _full_chat_chain is None:
        with _chain_lock:
            if _full_chat_chain is None:
                history_aware_retriever = create_history_aware_retriever(
                    llm,
                    get_product_retriever(),
                    contextualize_q_prompt,
                )
                document_chain = create_stuff_documents_chain(
                    llm=llm,
                    prompt=QA_PROMPT,
                    document_prompt=doc_prompt,
                )
                rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)
                _full_chat_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_message_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
    return _full_chat_chain


def get_fast_search_chain():
    """Fast text-search RAG chain: retrieve with pre-rewritten query -> answer.

    Skips the LLM-based history-aware query rewrite step entirely.
    The caller is expected to pass ``input=decision.rewrite_query`` which has
    already been cleaned by the keyword router or a single lightweight LLM call.
    This eliminates one full LLM round-trip (~3-8 s) per search request.
    """
    global _fast_search_chain
    if _fast_search_chain is None:
        with _chain_lock:
            if _fast_search_chain is None:
                document_chain = create_stuff_documents_chain(
                    llm=llm,
                    prompt=QA_PROMPT,
                    document_prompt=doc_prompt,
                )
                # Plain retriever — no LLM rewrite; uses `input` directly as search query.
                rag_chain = create_retrieval_chain(get_product_retriever(), document_chain)
                _fast_search_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_message_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
    return _fast_search_chain


def get_product_answer_chain():
    """LLM-only product answer chain for documents pre-fetched by image search."""
    global _product_answer_chain
    if _product_answer_chain is None:
        with _chain_lock:
            if _product_answer_chain is None:
                product_answer_llm_chain = QA_PROMPT | llm
                _product_answer_chain = RunnableWithMessageHistory(
                    product_answer_llm_chain,
                    get_message_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                )
    return _product_answer_chain


def get_image_search_chain():
    """Backward-compatible alias for product-answer chain."""
    return get_product_answer_chain()


def get_outfit_chain():
    """LLM-only outfit chain fed by build_outfit_context()."""
    global _outfit_chain
    if _outfit_chain is None:
        with _chain_lock:
            if _outfit_chain is None:
                outfit_llm_chain = outfit_prompt | llm
                _outfit_chain = RunnableWithMessageHistory(
                    outfit_llm_chain,
                    get_message_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                )
    return _outfit_chain
