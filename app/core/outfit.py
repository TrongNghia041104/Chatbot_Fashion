"""
app/core/outfit.py — Outfit Matching & Context Builder
========================================================
Logic phối đồ (Layer B):
- Tìm rule phù hợp từ Layer B knowledge base
- Tìm sản phẩm thực tế từ Layer A (Qdrant)
- Xây dựng context string để gửi vào LLM
"""

from qdrant_client.http.models import Filter, FieldCondition, MatchAny

from app.config import (
    CATEGORY_MAPPING,
    PHU_KIEN_KEYWORD_ROUTER,
    LAYER_B_SCORE_THRESHOLD,
    LAYER_B_LIMIT,
)
from app.core.embeddings import custom_embeddings
from app.core.vector_store import client, layer_b_female, layer_b_male, vector_db


# ── Layer B: Tìm rule phù hợp ─────────────────────────────────────────────────
def find_matching_rule(
    user_query: str,
    gender: str = "female",
    profile: dict = None,
) -> dict | None:
    """
    Tìm fashion rule phù hợp nhất từ Layer B knowledge base.

    Áp dụng fallback 3 bước:
    1. Lọc theo dang_nguoi + tone_da
    2. Chỉ lọc dang_nguoi (bỏ tone_da)
    3. Không lọc gì (pure semantic search)

    Returns:
        dict rule payload hoặc None nếu không tìm thấy.
    """
    collection   = f"layer_b_{gender}"
    query_vector = custom_embeddings.embed_query(user_query)
    conditions   = []

    if profile:
        if profile.get("dang_nguoi"):
            conditions.append(FieldCondition(
                key="dang_nguoi",
                match=MatchAny(any=[profile["dang_nguoi"], "Mọi vóc dáng"]),
            ))
        if profile.get("tone_da"):
            conditions.append(FieldCondition(
                key="tone_da",
                match=MatchAny(any=[profile["tone_da"], "Mọi tone da"]),
            ))

    search_filter = Filter(must=conditions) if conditions else None
    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=search_filter,
        limit=LAYER_B_LIMIT,
        score_threshold=LAYER_B_SCORE_THRESHOLD,
    )
    results = response.points

    # Fallback 1: bỏ tone_da, chỉ giữ dang_nguoi
    if not results and profile and profile.get("tone_da") and profile.get("dang_nguoi"):
        conds2 = [FieldCondition(
            key="dang_nguoi",
            match=MatchAny(any=[profile["dang_nguoi"], "Mọi vóc dáng"]),
        )]
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=Filter(must=conds2),
            limit=LAYER_B_LIMIT,
            score_threshold=LAYER_B_SCORE_THRESHOLD,
        )
        results = response.points

    # Fallback 2: không lọc gì
    if not results:
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=LAYER_B_LIMIT,
            score_threshold=LAYER_B_SCORE_THRESHOLD,
        )
        results = response.points

    return results[0].payload if results else None


def find_outfit_details(base_rule: dict, gender: str = "female") -> dict:
    """
    Tìm chi tiết rules cho từng món đồ cần phối (goi_y_phoi_cung).

    Returns:
        dict {category: rule} chứa rules cho từng loại sản phẩm trong outfit.
    """
    knowledge    = layer_b_female if gender == "female" else layer_b_male
    outfit_rules = {}

    for category in base_rule.get("goi_y_phoi_cung", []):
        matched = [
            r for r in knowledge
            if r["rule_key"].startswith(category)
            and r["phong_cach"] == base_rule["phong_cach"]
            and r["boi_canh"]   == base_rule["boi_canh"]
        ]
        if matched:
            outfit_rules[category] = matched[0]

    return outfit_rules


# ── Category Routing: Layer B → Layer A ───────────────────────────────────────
def get_layer_a_categories(layer_b_category: str, product_type: str) -> list[str]:
    """
    Map category từ Layer B sang category tương ứng trong Layer A (Qdrant).

    Với "Phụ kiện", dùng keyword routing để xác định loại phụ kiện cụ thể.
    """
    if layer_b_category != "Phụ kiện":
        return CATEGORY_MAPPING.get(layer_b_category, [])

    ptype_lower = product_type.lower()
    for cat, keywords in PHU_KIEN_KEYWORD_ROUTER.items():
        if any(kw in ptype_lower for kw in keywords):
            return [cat]

    return ["Phụ kiện hỗ trợ"]


def get_products_for_outfit(
    product_type: str,
    layer_b_category: str,
    phong_cach: str,
    vdb=None,
) -> list:
    """
    Tìm sản phẩm thực tế từ Layer A (Qdrant) cho một món trong outfit.

    Args:
        product_type: Loại sản phẩm cụ thể (vd: "Áo thun cổ tròn").
        layer_b_category: Category theo Layer B (vd: "Áo mặc trong (áo thun/sơ mi)").
        phong_cach: Phong cách thời trang (vd: "Casual").
        vdb: Vector store instance (default: dùng global vector_db).

    Returns:
        List Document từ Qdrant.
    """
    _vdb = vdb or vector_db
    target_categories = get_layer_a_categories(layer_b_category, product_type)
    search_filter = None

    if target_categories:
        search_filter = Filter(must=[FieldCondition(
            key="metadata.category",
            match=MatchAny(any=target_categories),
        )])

    return _vdb.similarity_search(
        query=f"{product_type} {phong_cach}",
        k=3,
        filter=search_filter,
    )


# ── Build Outfit Context ───────────────────────────────────────────────────────
def build_outfit_context(
    user_query: str,
    gender: str = "female",
    profile: dict = None,
) -> tuple[str, list]:
    """
    Xây dựng toàn bộ context phối đồ để gửi vào LLM.

    Returns:
        Tuple (context_str, images_data):
        - context_str: Chuỗi CÔNG THỨC PHỐI ĐỒ + SẢN PHẨM GỢI Ý
        - images_data: list[dict] {product_id, category, images: [url...]}
    """
    base_rule = find_matching_rule(user_query, gender, profile)
    if not base_rule:
        return "", []

    outfit_rules = find_outfit_details(base_rule, gender)
    if not outfit_rules:
        return "", []

    # ── Thu thập sản phẩm cho từng món trong outfit ───────────────────────
    outfit_products = {}
    for layer_b_category, rule in outfit_rules.items():
        product_type = rule["rule_key"].split("|")[1].strip()
        products     = get_products_for_outfit(
            product_type, layer_b_category, base_rule["phong_cach"],
        )
        outfit_products[layer_b_category] = {
            "product_type": product_type,
            "ly_do":        rule["ly_do_tu_van"],
            "products":     products,
        }

    # ── Xây dựng context string ───────────────────────────────────────────
    lines = [
        "CÔNG THỨC PHỐI ĐỒ:",
        f"  Phong cách : {base_rule['phong_cach']}",
        f"  Bối cảnh   : {base_rule['boi_canh']}",
        f"  Lý do chính: {base_rule['ly_do_tu_van']}",
    ]
    if profile and profile.get("dang_nguoi"):
        lines.append(f"  Dáng người : {profile['dang_nguoi']}")
    if profile and profile.get("tone_da"):
        lines.append(f"  Tone da    : {profile['tone_da']}")

    lines += ["", "SẢN PHẨM GỢI Ý:"]
    images_data: list[dict] = []

    for cat, data in outfit_products.items():
        lines.append(f"\n[{cat} – {data['product_type']}]")
        lines.append(f"  Lý do: {data['ly_do']}")

        if data["products"]:
            for doc in data["products"]:
                pid = doc.metadata.get("product_id", "N/A")
                price_raw = doc.metadata.get("price", "N/A")
                try:
                    price_fmt = f"{int(price_raw):,}".replace(",", ".")
                except Exception:
                    price_fmt = price_raw

                lines.append(f"  • (Mã SP: {pid} | Giá: {price_fmt} VNĐ)")
                lines.append(f"    {doc.page_content[:600]}")

                # Thu thập ảnh
                doc_images = [url for url in doc.metadata.get("images", []) if url]
                if doc_images:
                    images_data.append({
                        "product_id": pid,
                        "category":   cat,
                        "images":     doc_images[:2],   # Tối đa 2 ảnh/SP
                    })
        else:
            lines.append("  • (Chưa có sản phẩm phù hợp trong kho)")

    return "\n".join(lines), images_data
