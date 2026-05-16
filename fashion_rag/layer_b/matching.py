"""
Semantic matching engine cho Layer B.
Sử dụng BGE-M3 embedding + Qdrant để tìm quy tắc phối đồ phù hợp nhất.
"""
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

from fashion_rag.config.settings import (
    QDRANT_LAYER_B_FEMALE_COLLECTION,
    QDRANT_LAYER_B_MALE_COLLECTION,
    EMBEDDING_VECTOR_SIZE,
    LAYER_B_SCORE_THRESHOLD,
)
from fashion_rag.layer_b.knowledge import get_knowledge_by_gender


def index_layer_b(data, collection_name, vector_db):
    """Embed toàn bộ Layer B bằng BGE-M3 và lưu vào Qdrant (chạy 1 lần)."""
    qdrant_client = vector_db.client
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in existing:
        count = qdrant_client.count(collection_name).count
        print(f"[SKIP] {collection_name} đã tồn tại ({count} rules)")
        return

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = []
    for i, rule in enumerate(data):
        text = f"{rule['rule_key']} {rule['phong_cach']} {rule['boi_canh']} {rule['ly_do_tu_van']}"
        vector = vector_db.embeddings.embed_query(text)
        points.append(PointStruct(id=i, vector=vector, payload=rule))

    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"[OK] Indexed {len(points)} rules → {collection_name}")


def find_matching_rule(user_query, gender="female", profile=None, vector_db=None):
    """Tìm rule Layer B phù hợp nhất bằng semantic search + profile filter."""
    collection = QDRANT_LAYER_B_FEMALE_COLLECTION if gender == "female" else QDRANT_LAYER_B_MALE_COLLECTION
    qdrant_client = vector_db.client
    query_vector = vector_db.embeddings.embed_query(user_query)

    conditions = []
    if profile:
        if profile.get("dang_nguoi"):
            conditions.append(FieldCondition(key="dang_nguoi", match=MatchAny(any=[profile["dang_nguoi"], "Mọi dáng người"])))
        if profile.get("tone_da"):
            conditions.append(FieldCondition(key="tone_da", match=MatchAny(any=[profile["tone_da"], "Mọi tone da"])))

    search_filter = Filter(must=conditions) if conditions else None

    results = qdrant_client.search(
        collection_name=collection, query_vector=query_vector,
        query_filter=search_filter, limit=1, score_threshold=LAYER_B_SCORE_THRESHOLD,
    )

    if not results and search_filter:
        results = qdrant_client.search(
            collection_name=collection, query_vector=query_vector,
            limit=1, score_threshold=LAYER_B_SCORE_THRESHOLD,
        )

    return results[0].payload if results else None


def find_outfit_details(base_rule, gender="female"):
    """Tìm rule chi tiết cho mỗi category trong goi_y_phoi_cung (exact match)."""
    knowledge = get_knowledge_by_gender(gender)
    outfit_rules = {}
    for category in base_rule.get("goi_y_phoi_cung", []):
        matched = [
            r for r in knowledge
            if r["rule_key"].startswith(category)
            and r["phong_cach"] == base_rule["phong_cach"]
            and r["boi_canh"] == base_rule["boi_canh"]
        ]
        if matched:
            outfit_rules[category] = matched[0]
    return outfit_rules
