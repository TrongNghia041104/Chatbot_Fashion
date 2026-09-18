"""Layer 3b: semantic intent routing via embedding similarity.

Nằm giữa keyword routing (Layer 3) và LLM fallback (Layer 4) trong
``fashion_rag.core.intent``. Khi keyword không khớp, module này nhúng câu
người dùng bằng BGE-M3 (embedder đã dùng cho Layer B — không thêm model mới)
rồi so cosine với các câu mẫu của từng intent. Nếu một intent vượt ngưỡng và
cách biệt rõ so với intent nhì thì trả về intent đó; ngược lại trả None để
LLM (Layer 4) xử lý câu thật sự mơ hồ.

Logic quyết định (``_best_intent``) là hàm thuần, tách khỏi lời gọi embedding
để test được mà không cần Ollama — xem self-check ``demo()`` ở cuối file.
"""

from __future__ import annotations

from threading import Lock

import numpy as np

from fashion_rag.config import (
    SEMANTIC_INTENT_EXAMPLES,
    SEMANTIC_ROUTER_MARGIN,
    SEMANTIC_ROUTER_THRESHOLD,
)


_index_lock = Lock()
# intent -> ma trận (n_examples x dim) câu mẫu đã chuẩn hóa L2. Dựng một lần.
_example_index: dict[str, np.ndarray] | None = None


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


def _score_intents(query_vec: np.ndarray, index: dict[str, np.ndarray]) -> dict[str, float]:
    """Điểm mỗi intent = cosine cao nhất giữa query và các câu mẫu của intent đó."""
    query_vec = query_vec / max(float(np.linalg.norm(query_vec)), 1e-9)
    scores: dict[str, float] = {}
    for intent, matrix in index.items():
        if matrix.size:
            scores[intent] = float(np.max(matrix @ query_vec))
    return scores


def _best_intent(
    scores: dict[str, float],
    threshold: float = SEMANTIC_ROUTER_THRESHOLD,
    margin: float = SEMANTIC_ROUTER_MARGIN,
) -> tuple[str, float] | None:
    """Chọn intent chỉ khi đủ chắc.

    Điều kiện: điểm cao nhất ``>= threshold`` VÀ cách biệt với intent nhì
    ``>= margin``. Cách biệt nhỏ nghĩa là câu nằm giữa hai intent → nhường LLM.
    """
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score >= threshold and (top_score - second_score) >= margin:
        return top_intent, top_score
    return None


def _get_index() -> dict[str, np.ndarray]:
    """Lazy singleton: nhúng câu mẫu một lần rồi cache. Dùng BGE-M3 qua Ollama."""
    global _example_index
    if _example_index is None:
        with _index_lock:
            if _example_index is None:
                from fashion_rag.infrastructure.embeddings.embeddings import get_rule_embeddings

                embedder = get_rule_embeddings()
                index: dict[str, np.ndarray] = {}
                for intent, examples in SEMANTIC_INTENT_EXAMPLES.items():
                    if examples:
                        vectors = np.asarray(embedder.embed_documents(list(examples)), dtype=np.float32)
                        index[intent] = _l2_normalize(vectors)
                _example_index = index
    return _example_index


def classify_intent_semantic(query: str) -> tuple[str, float] | None:
    """Trả ``(intent, score)`` nếu semantic đủ chắc, ``None`` nếu để LLM xử lý."""
    text = str(query or "").strip()
    if not text:
        return None
    from fashion_rag.infrastructure.embeddings.embeddings import get_rule_embeddings

    query_vec = np.asarray(get_rule_embeddings().embed_query(text), dtype=np.float32)
    return _best_intent(_score_intents(query_vec, _get_index()))


def demo() -> None:
    """Self-check hàm quyết định bằng vector giả — không cần Ollama/torch."""
    index = {
        "product_discovery": _l2_normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32)),
        "outfit_advice": _l2_normalize(np.array([[0.0, 1.0, 0.0]], dtype=np.float32)),
    }
    # Rõ ràng nghiêng về product_discovery.
    clear = _score_intents(np.array([0.9, 0.1, 0.0], dtype=np.float32), index)
    assert _best_intent(clear, 0.55, 0.05)[0] == "product_discovery"
    # Nhập nhằng giữa 2 intent (cách biệt < margin) -> None, nhường LLM.
    tie = _score_intents(np.array([0.71, 0.71, 0.0], dtype=np.float32), index)
    assert _best_intent(tie, 0.55, 0.05) is None
    # Yếu, dưới threshold -> None.
    weak = _score_intents(np.array([0.3, 0.2, 0.9], dtype=np.float32), index)
    assert _best_intent(weak, 0.55, 0.05) is None
    print("[OK] semantic_router self-check passed.")


if __name__ == "__main__":
    demo()
