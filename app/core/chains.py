"""
app/core/chains.py — RAG Pipeline Assembly
===========================================
Lắp ráp các LangChain chains từ các components đã khởi tạo:
- full_chat_chain: Chain tìm kiếm sản phẩm (RAG + history)
- outfit_chain_with_history: Chain tư vấn phối đồ (LLM + history)
"""

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.llm import llm, QA_PROMPT, contextualize_q_prompt, doc_prompt, outfit_prompt
from app.core.vector_store import retriever
from app.core.history import get_message_history

print("[INFO] Đang lắp ráp RAG Pipeline...")

# ── RAG Chain (Search) ────────────────────────────────────────────────────────
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt,
)
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=QA_PROMPT,
    document_prompt=doc_prompt,
)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

full_chat_chain = RunnableWithMessageHistory(
    rag_chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ── Outfit Chain ──────────────────────────────────────────────────────────────
outfit_llm_chain = outfit_prompt | llm

outfit_chain_with_history = RunnableWithMessageHistory(
    outfit_llm_chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

print("[OK] RAG Pipeline sẵn sàng!")
