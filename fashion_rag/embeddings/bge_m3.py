"""
Custom Embedding wrapper cho mô hình BAAI/bge-m3.
BGE-M3 hỗ trợ đa ngôn ngữ (tiếng Việt) và không cần tách từ thủ công.
Vector size: 1024 (lớn hơn PhoBERT 768).
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from fashion_rag.config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


class BGEM3Embeddings(Embeddings):
    """
    Wrapper embedding sử dụng BGE-M3 qua HuggingFace.
    Không cần word_tokenize (underthesea) — đẩy trực tiếp text thô.
    """

    def __init__(self, model_name: str = None):
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name or EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vector hóa danh sách documents."""
        return self.hf_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Vector hóa câu truy vấn của người dùng."""
        return self.hf_embeddings.embed_query(text)
