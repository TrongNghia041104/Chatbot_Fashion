"""Shared technical utilities: configuration, security, telemetry, and the intent router.

Import trực tiếp từ submodule cần dùng (vd: ``from fashion_rag.core.intent import
route_user_request``). Các adapter hạ tầng (Qdrant, embeddings, LLM, vision, Redis)
và use case (chains, outfit, profile, retrieval, ingestion) nằm ở
``infrastructure/``, ``application/`` và ``modules/`` — không còn re-export qua đây.
"""
