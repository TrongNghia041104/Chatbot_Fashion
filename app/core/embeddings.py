"""
app/core/embeddings.py — BGE-M3 Embedding Wrapper
==================================================
Wrapper cho model BGE-M3 chạy qua Ollama (SSH tunnel → Vast.ai GPU).
Vector 1024 chiều — khớp với Qdrant collection đã tạo.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from app.config import OLLAMA_BASE_URL, EMBEDDING_MODEL


class BGEM3Embeddings(Embeddings):
    """
    Custom embedding wrapper cho BGE-M3 qua Ollama.

    Sử dụng:
        embeddings = BGEM3Embeddings()
        vector = embeddings.embed_query("áo thun trắng")
        vectors = embeddings.embed_documents(["áo thun trắng", "quần jean xanh"])
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.ollama_embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.ollama_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.ollama_embeddings.embed_query(text)


# ── Singleton instance dùng chung toàn app ───────────────────────────────────
print("[INFO] Đang load Embedding model BGE-M3...")
custom_embeddings = BGEM3Embeddings()
print("[OK] Embedding model sẵn sàng!")
