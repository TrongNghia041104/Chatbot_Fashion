"""
app/core/vector_store.py — Qdrant Vector Store & Retriever
===========================================================
Khởi tạo kết nối Qdrant, vector store và retriever.
Module này cũng chịu trách nhiệm index Layer B knowledge vào Qdrant.
"""

import json

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from app.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_FASHION,
    QDRANT_COLLECTION_LAYER_B_F,
    QDRANT_COLLECTION_LAYER_B_M,
    QDRANT_VECTOR_SIZE,
    RETRIEVER_K,
    RETRIEVER_SCORE_THRESHOLD,
    LAYER_B_FEMALE_PATH,
    LAYER_B_MALE_PATH,
)
from app.core.embeddings import custom_embeddings


# ── Qdrant Client ─────────────────────────────────────────────────────────────
print("[INFO] Đang kết nối Qdrant Docker (localhost:6333)...")
client = QdrantClient(url=QDRANT_URL)

# ── Vector Store (Layer A — fashion products) ─────────────────────────────────
vector_db = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION_FASHION,
    embedding=custom_embeddings,
)

retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": RETRIEVER_K, "score_threshold": RETRIEVER_SCORE_THRESHOLD},
)
print("[OK] Qdrant + Retriever sẵn sàng!")


# ── Layer B Indexing ──────────────────────────────────────────────────────────
def _load_layer_b(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_layer_b(data: list, collection_name: str) -> None:
    """Index Layer B knowledge vào Qdrant (bỏ qua nếu collection đã tồn tại)."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"[SKIP] {collection_name} đã tồn tại — bỏ qua index.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
    )
    points = []
    for i, rule in enumerate(data):
        text = f"{rule['rule_key']} {rule['phong_cach']} {rule['boi_canh']} {rule['ly_do_tu_van']}"
        vector = custom_embeddings.embed_query(text)
        points.append(PointStruct(id=i, vector=vector, payload=rule))

    client.upsert(collection_name=collection_name, points=points)
    print(f"[OK] Indexed {len(points)} rules → {collection_name}")


# ── Load & Index Layer B khi module được import ───────────────────────────────
layer_b_female = _load_layer_b(LAYER_B_FEMALE_PATH)
layer_b_male   = _load_layer_b(LAYER_B_MALE_PATH)
print(f"[OK] Layer B: {len(layer_b_female)} rules Nữ | {len(layer_b_male)} rules Nam")

index_layer_b(layer_b_female, QDRANT_COLLECTION_LAYER_B_F)
index_layer_b(layer_b_male,   QDRANT_COLLECTION_LAYER_B_M)
