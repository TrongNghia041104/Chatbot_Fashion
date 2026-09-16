"""Ports (interfaces) for the infrastructure adapters this system can swap out.

Không định nghĩa Protocol mới — LangChain đã cung cấp đúng ranh giới
port/adapter cần thiết, và toàn bộ adapter hiện có đã implement chúng:

- ``EmbedderPort``  = ``langchain_core.embeddings.Embeddings``
  (``BGEM3Embeddings``, ``ViFashionCLIPTextEmbeddings``,
  ``RemoteViFashionCLIPTextEmbeddings`` trong ``infrastructure/embeddings/``.)
- ``RetrieverPort`` = ``langchain_core.retrievers.BaseRetriever``
  (``get_product_retriever()`` trong ``infrastructure/vectorstores/``.)
- ``LLMPort``       = ``langchain_core.runnables.Runnable``
  (``ChatOllama`` trong ``infrastructure/llms/llm.py``, và mọi chain trong
  ``application/chat/chains.py``.)

Viết lại các interface này bằng ``Protocol`` riêng sẽ chỉ nhân bản mà không
thêm giá trị. Muốn thay Qdrant/Ollama/BGE-M3 bằng công nghệ khác, chỉ cần
viết adapter mới implement đúng 1 trong 3 interface dưới đây.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings as EmbedderPort
from langchain_core.retrievers import BaseRetriever as RetrieverPort
from langchain_core.runnables import Runnable as LLMPort

__all__ = ["EmbedderPort", "RetrieverPort", "LLMPort"]
