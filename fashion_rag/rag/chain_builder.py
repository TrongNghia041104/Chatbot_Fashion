"""
Xây dựng toàn bộ RAG pipeline: LLM + Retriever + Chains + Redis History.
"""
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from fashion_rag.config.settings import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT,
    LLM_NUM_PREDICT, LLM_NUM_CTX,
    REDIS_URL, MAX_HISTORY_MESSAGES,
    RETRIEVER_SEARCH_TYPE, RETRIEVER_TOP_K, RETRIEVER_SCORE_THRESHOLD,
)
from fashion_rag.rag.prompts import (
    QA_PROMPT, contextualize_q_prompt, doc_prompt, outfit_prompt,
)


def _create_llm():
    """Khởi tạo LLM Qwen qua Ollama."""
    print("[THÔNG BÁO] Đang khởi tạo mô hình Qwen local...")
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
        num_predict=LLM_NUM_PREDICT,
        num_ctx=LLM_NUM_CTX,
    )
    print("[THÔNG BÁO] Đã khởi tạo LLM thành công!")
    return llm


def get_message_history(session_id: str):
    """Lấy Redis chat history với giới hạn số message."""
    history = RedisChatMessageHistory(session_id, url=REDIS_URL)
    current_messages = history.messages
    if len(current_messages) > MAX_HISTORY_MESSAGES:
        kept_messages = current_messages[-MAX_HISTORY_MESSAGES:]
        history.clear()
        history.add_messages(kept_messages)
    return history


def build_rag_pipeline(vector_db):
    """
    Xây dựng toàn bộ RAG pipeline.

    Returns:
        tuple: (full_chat_chain, outfit_chain_with_history, llm)
    """
    llm = _create_llm()

    # Thiết lập retriever
    retriever = vector_db.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={
            "k": RETRIEVER_TOP_K,
            "score_threshold": RETRIEVER_SCORE_THRESHOLD,
        },
    )
    print("[THÔNG BÁO] Đã khởi tạo Retriever thành công!")

    # Chuỗi 1: Viết lại câu hỏi dựa trên lịch sử
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Chuỗi 2: Nhồi documents vào prompt chính
    document_chain = create_stuff_documents_chain(
        llm=llm, prompt=QA_PROMPT, document_prompt=doc_prompt
    )

    # Chuỗi 3: Ghép retriever + document chain
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Bọc Redis history cho RAG chain
    full_chat_chain = RunnableWithMessageHistory(
        rag_chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Outfit chain (không qua retriever)
    outfit_llm_chain = outfit_prompt | llm
    outfit_chain_with_history = RunnableWithMessageHistory(
        outfit_llm_chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("[THÔNG BÁO] Hệ thống RAG Pipeline đã sẵn sàng!")
    return full_chat_chain, outfit_chain_with_history, llm
