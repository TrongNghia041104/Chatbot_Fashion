"""Layer B outfit matching and Layer A product context building."""

from __future__ import annotations

from qdrant_client.http.models import FieldCondition, Filter, MatchAny

from app.config import (
    CATEGORY_MAPPING,
    LAYER_B_FALLBACK_SCORE_THRESHOLD,
    LAYER_B_SCORE_THRESHOLD,
    LAYER_B_WILDCARD_DANG,
    LAYER_B_WILDCARD_TONE,
    PHU_KIEN_KEYWORD_ROUTER,
)
from app.core.embeddings import get_rule_embeddings
from app.core.vector_store import (
    client,
    diversity_filter_documents,
    ensure_layer_b_indexed,
    get_product_vector_db,
    layer_b_female,
    layer_b_male,
    normalize_product_metadata,
)


def find_matching_rule(user_query: str, gender: str = "female", profile: dict | None = None) -> dict | None:
    """Find the best Layer B styling rule, with body/tone filter fallbacks."""
    ensure_layer_b_indexed()
    collection = f"layer_b_{gender}"
    query_vector = get_rule_embeddings().embed_query(user_query)

    conditions = []
    if profile:
        if profile.get("dang_nguoi"):
            conditions.append(
                FieldCondition(
                    key="dang_nguoi",
                    match=MatchAny(any=[profile["dang_nguoi"], LAYER_B_WILDCARD_DANG]),
                )
            )
        if profile.get("tone_da"):
            conditions.append(
                FieldCondition(
                    key="tone_da",
                    match=MatchAny(any=[profile["tone_da"], LAYER_B_WILDCARD_TONE]),
                )
            )

    search_filter = Filter(must=conditions) if conditions else None
    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=search_filter,
        limit=1,
        score_threshold=LAYER_B_SCORE_THRESHOLD,
    )
    results = response.points

    if not results and profile and profile.get("tone_da") and profile.get("dang_nguoi"):
        fallback_cond = [
            FieldCondition(
                key="dang_nguoi",
                match=MatchAny(any=[profile["dang_nguoi"], LAYER_B_WILDCARD_DANG]),
            )
        ]
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=Filter(must=fallback_cond),
            limit=1,
            score_threshold=LAYER_B_SCORE_THRESHOLD,
        )
        results = response.points

    if not results and search_filter:
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=1,
            score_threshold=LAYER_B_SCORE_THRESHOLD,
        )
        results = response.points

    return results[0].payload if results else None


def find_outfit_details(base_rule: dict, gender: str = "female") -> dict:
    """Find detail rules for each item category in the selected outfit formula."""
    ensure_layer_b_indexed()
    collection = f"layer_b_{gender}"
    outfit_rules = {}
    style_query = f"{base_rule['phong_cach']} {base_rule['boi_canh']}"
    data_source = layer_b_female if gender == "female" else layer_b_male

    for category in base_rule.get("goi_y_phoi_cung", []):
        exact = [
            rule
            for rule in data_source
            if rule["rule_key"].startswith(category)
            and rule["phong_cach"] == base_rule["phong_cach"]
            and rule["boi_canh"] == base_rule["boi_canh"]
        ]
        if exact:
            outfit_rules[category] = exact[0]
            continue

        query_vector = get_rule_embeddings().embed_query(f"{category} {style_query}")
        raw_response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=10,
            score_threshold=LAYER_B_FALLBACK_SCORE_THRESHOLD,
        )
        category_results = [
            point
            for point in raw_response.points
            if point.payload["rule_key"].startswith(category)
        ]
        if category_results:
            outfit_rules[category] = category_results[0].payload

    return outfit_rules


def get_layer_a_categories(layer_b_category: str, product_type: str) -> list[str]:
    """Map Layer B category labels to Layer A product categories."""
    if layer_b_category != "Phụ kiện":
        return CATEGORY_MAPPING.get(layer_b_category, [])

    product_type_lower = product_type.lower()
    for category, keywords in PHU_KIEN_KEYWORD_ROUTER.items():
        if any(keyword in product_type_lower for keyword in keywords):
            return [category]
    return ["Phụ kiện hỗ trợ"]


def get_products_for_outfit(
    product_type: str,
    layer_b_category: str,
    phong_cach: str,
    vdb=None,
) -> list:
    """Search Layer A products for one outfit item."""
    vector_db = vdb or get_product_vector_db()
    target_categories = get_layer_a_categories(layer_b_category, product_type)
    search_filter = None
    if target_categories:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.category",
                    match=MatchAny(any=target_categories),
                )
            ]
        )

    query = f"{product_type} {phong_cach}"
    raw_results = vector_db.similarity_search_with_score(query=query, k=8, filter=search_filter)
    valid_products = [
        normalize_product_metadata(doc)
        for doc, score in raw_results
        if score >= 0.30
    ]
    return diversity_filter_documents(valid_products, max_docs=3)


def build_outfit_context(
    user_query: str,
    gender: str = "female",
    profile: dict | None = None,
) -> tuple[str, list]:
    """Build the full outfit context and product image payload for the UI."""
    base_rule = find_matching_rule(user_query, gender, profile)
    if not base_rule:
        return "", []

    outfit_rules = find_outfit_details(base_rule, gender)
    if not outfit_rules:
        return "", []

    outfit_products = {}
    for layer_b_category, rule in outfit_rules.items():
        product_type = rule["rule_key"].split("|")[1].strip()
        products = get_products_for_outfit(product_type, layer_b_category, base_rule["phong_cach"])
        outfit_products[layer_b_category] = {
            "product_type": product_type,
            "ly_do": rule["ly_do_tu_van"],
            "products": products,
        }

    lines = [
        "CÔNG THỨC PHỐI ĐỒ:",
        f"  Phong cách: {base_rule['phong_cach']}",
        f"  Bối cảnh : {base_rule['boi_canh']}",
        f"  Lý do    : {base_rule['ly_do_tu_van']}",
    ]
    if profile and profile.get("dang_nguoi"):
        lines.append(f"  Dáng người: {profile['dang_nguoi']}")
    if profile and profile.get("tone_da"):
        lines.append(f"  Tone da   : {profile['tone_da']}")
    lines += ["", "SẢN PHẨM GỢI Ý:"]

    images_data: list[dict] = []
    total_products = 0
    for category, data in outfit_products.items():
        lines.append(f"\n[{category} — {data['product_type']}]")
        lines.append(f"  Lý do: {data['ly_do']}")
        if data["products"]:
            for doc in data["products"]:
                if total_products >= 3:
                    break
                doc = normalize_product_metadata(doc)
                product_id = doc.metadata.get("product_id", "N/A")
                price_raw = doc.metadata.get("price", "N/A")
                image_url = doc.metadata.get("image_url", "")
                try:
                    price_fmt = f"{int(price_raw):,}".replace(",", ".")
                except Exception:
                    price_fmt = price_raw
                lines.append(f"  - (Mã SP: {product_id} | Giá: {price_fmt} VND | IMAGE_URL: {image_url})")
                lines.append(f"    {doc.page_content[:600]}")
                doc_images = [url for url in doc.metadata.get("images", []) if url]
                if doc_images:
                    images_data.append(
                        {
                            "product_id": product_id,
                            "category": category,
                            "images": doc_images[:2],
                        }
                    )
                total_products += 1
        else:
            lines.append("  - (Chưa có sản phẩm phù hợp trong kho)")
        if total_products >= 3:
            break

    return "\n".join(lines), images_data
