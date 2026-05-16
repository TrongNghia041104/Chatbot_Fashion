"""
Fashion RAG - Hệ thống tư vấn thời trang Multimodal RAG
========================================================
Kiến trúc 2 tầng:
  - Layer A: RAG pipeline (Qdrant + BGE-M3 + LLM) cho tìm kiếm sản phẩm
  - Layer B: Rule-based matching engine cho tư vấn phối đồ
  - Vision : Qwen2.5-VL cho xử lý ảnh đầu vào
"""
