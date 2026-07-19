"""Layer B outfit matching and Layer A product context building."""

from __future__ import annotations

import time
from collections import Counter

from qdrant_client.http.models import FieldCondition, Filter, MatchAny

from app.config import (
    CATEGORY_MAPPING,
    LAYER_B_FALLBACK_SCORE_THRESHOLD,
    LAYER_B_SCORE_THRESHOLD,
    LAYER_B_WILDCARD_DANG,
    LAYER_B_WILDCARD_TONE,
    OUTFIT_MAX_PRODUCT_SLOTS,
    PHU_KIEN_KEYWORD_ROUTER,
)

LAYER_A_TO_LAYER_B_CATEGORY = {
    "Áo": "Áo mặc trong (áo thun/sơ mi)",
    "Áo khoác": "Áo khoác ngoài",
    "Quần": "Quần/Chân váy",
    "Chân váy": "Quần/Chân váy",
    "Đầm": "Đầm/Jumpsuit",
    "Jumpsuit": "Đầm/Jumpsuit",
    "Giày": "Giày dép",
    "Túi xách": "Túi xách",
    "Mũ": "Phụ kiện",
    "Găng tay": "Phụ kiện",
    "Kính mắt": "Phụ kiện",
    "Đồng hồ": "Phụ kiện",
    "Dây chuyền": "Phụ kiện",
    "Bông tai": "Phụ kiện",
    "Vòng tay": "Phụ kiện",
    "Nhẫn": "Phụ kiện",
    "Ghim cài áo": "Phụ kiện",
    "Phụ kiện hỗ trợ": "Phụ kiện",
}


def _first_text(value, default: str = "") -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else default
    if value is None:
        return default
    return str(value).strip()


def _metadata_detail(metadata: dict, key: str) -> str:
    details = metadata.get("details") or {}
    return _first_text(details.get(key))


def analyze_image_item_context(image_docs: list) -> dict:
    """
    Summarize fast image retrieval results into the text slots Layer B needs.

    This deliberately uses product metadata first. A VLM should only enrich this
    when retrieval confidence is weak or the user asks for deeper visual analysis.
    """
    docs = [normalize_product_metadata(doc) for doc in image_docs if doc is not None]
    if not docs:
        return {
            "confidence": "low",
            "confidence_score": 0.0,
            "needs_vlm": True,
            "reason": "No image retrieval candidates.",
        }

    category_counts = Counter(_first_text(doc.metadata.get("category"), "Khác") for doc in docs)
    department_counts = Counter(_first_text(doc.metadata.get("department"), "Unisex") for doc in docs)
    top_category, top_category_count = category_counts.most_common(1)[0]
    top_department = department_counts.most_common(1)[0][0]
    top_doc = docs[0]
    top_meta = top_doc.metadata
    scores = [float(doc.metadata.get("image_search_score") or 0.0) for doc in docs]
    top_score = scores[0] if scores else 0.0
    score_gap = top_score - scores[1] if len(scores) > 1 else top_score
    category_agreement = top_category_count / max(len(docs), 1)

    color = _metadata_detail(top_meta, "main_color")
    material = _metadata_detail(top_meta, "material")
    pattern = _metadata_detail(top_meta, "pattern")
    season = _first_text(top_meta.get("season"))
    occasion = ", ".join(str(x) for x in (top_meta.get("occasion") or [])[:2])
    title = _first_text(top_meta.get("title"))

    confidence_score = min(
        1.0,
        0.45 * category_agreement
        + 0.35 * min(top_score / 0.35, 1.0)
        + 0.20 * min(max(score_gap, 0.0) / 0.08, 1.0),
    )
    confidence = "high" if confidence_score >= 0.65 else "medium" if confidence_score >= 0.45 else "low"

    return {
        "layer_a_category": top_category,
        "layer_b_category": LAYER_A_TO_LAYER_B_CATEGORY.get(top_category, "Phụ kiện"),
        "department": top_department,
        "product_type": title or top_category,
        "color": color,
        "material": material,
        "pattern": pattern,
        "season": season,
        "occasion": occasion,
        "top_score": round(top_score, 4),
        "score_gap": round(score_gap, 4),
        "category_agreement": round(category_agreement, 4),
        "confidence": confidence,
        "confidence_score": round(confidence_score, 4),
        "needs_vlm": confidence == "low",
        "source": "image_retrieval_metadata",
    }


def build_layer_b_query_from_image_context(item_context: dict, user_query: str, profile: dict | None = None) -> str:
    """Turn a retrieved image item into a Layer B semantic rule query."""
    parts = [
        f"Phối đồ với {item_context.get('layer_b_category', '')}",
        item_context.get("product_type", ""),
    ]
    if item_context.get("color"):
        parts.append(f"màu {item_context['color']}")
    if item_context.get("material"):
        parts.append(f"chất liệu {item_context['material']}")
    if item_context.get("pattern"):
        parts.append(f"họa tiết {item_context['pattern']}")
    if item_context.get("occasion"):
        parts.append(f"dịp {item_context['occasion']}")
    if item_context.get("season"):
        parts.append(f"mùa {item_context['season']}")
    if item_context.get("department") and item_context["department"] != "Unisex":
        parts.append(f"cho {item_context['department']}")
    # Profile slots are categorical filters, so they are applied in
    # find_matching_rule() instead of being appended to the semantic query.
    if user_query:
        parts.append(user_query)
    return " ".join(part for part in parts if part).strip()


from app.core.embeddings import get_product_embeddings, get_rule_embeddings
from app.core.vector_store import (
    client,
    diversity_filter_documents,
    ensure_layer_b_indexed,
    get_product_vector_db,
    layer_b_female,
    layer_b_male,
    normalize_product_metadata,
)


def _count_metric(metrics: dict | None, name: str, count: int = 1) -> None:
    """Increment one internal retrieval counter when diagnostics are enabled."""
    if metrics is None or count <= 0:
        return
    calls = metrics.setdefault("model_calls", {})
    calls[name] = int(calls.get(name, 0)) + int(count)


def _count_vectors(metrics: dict | None, name: str, count: int) -> None:
    """Record how many vectors one batched embedding request produced."""
    if metrics is None or count <= 0:
        return
    vectors = metrics.setdefault("model_vectors", {})
    vectors[name] = int(vectors.get(name, 0)) + int(count)


def _time_metric(metrics: dict | None, name: str, started_at: float) -> None:
    """Record one internal outfit-stage duration."""
    if metrics is None:
        return
    timings = metrics.setdefault("timings", {})
    timings[name] = round(time.perf_counter() - started_at, 4)


def _iter_rule_values(rule: dict, key: str) -> list[str]:
    value = rule.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _wildcard_values(data_source: list[dict], key: str, base_value: str) -> list[str]:
    values = {base_value}
    for rule in data_source:
        for value in _iter_rule_values(rule, key):
            if value == base_value or value.startswith(f"{base_value} "):
                values.add(value)
    return sorted(values)


def _profile_match_values(data_source: list[dict], key: str, user_value: str, wildcard_base: str) -> list[str]:
    return [user_value, *_wildcard_values(data_source, key, wildcard_base)]


def find_matching_rule(
    user_query: str,
    gender: str = "female",
    profile: dict | None = None,
    metrics: dict | None = None,
) -> dict | None:
    """Find the best Layer B styling rule, with body/tone filter fallbacks."""
    ensure_layer_b_indexed()
    collection = f"layer_b_{gender}"
    data_source = layer_b_female if gender == "female" else layer_b_male
    query_vector = get_rule_embeddings().embed_query(user_query)
    _count_metric(metrics, "bge_m3")
    _count_vectors(metrics, "bge_m3", 1)

    conditions = []
    if profile:
        if profile.get("dang_nguoi"):
            conditions.append(
                FieldCondition(
                    key="dang_nguoi",
                    match=MatchAny(
                        any=_profile_match_values(
                            data_source,
                            "dang_nguoi",
                            profile["dang_nguoi"],
                            LAYER_B_WILDCARD_DANG,
                        )
                    ),
                )
            )
        if profile.get("tone_da"):
            conditions.append(
                FieldCondition(
                    key="tone_da",
                    match=MatchAny(
                        any=_profile_match_values(
                            data_source,
                            "tone_da",
                            profile["tone_da"],
                            LAYER_B_WILDCARD_TONE,
                        )
                    ),
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
                match=MatchAny(
                    any=_profile_match_values(
                        data_source,
                        "dang_nguoi",
                        profile["dang_nguoi"],
                        LAYER_B_WILDCARD_DANG,
                    )
                ),
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


def find_outfit_details(
    base_rule: dict,
    gender: str = "female",
    metrics: dict | None = None,
) -> dict:
    """Find detail rules, batching semantic fallbacks into one BGE-M3 call."""
    ensure_layer_b_indexed()
    collection = f"layer_b_{gender}"
    outfit_rules = {}
    style_query = f"{base_rule['phong_cach']} {base_rule['boi_canh']}"
    data_source = layer_b_female if gender == "female" else layer_b_male
    semantic_fallbacks: list[tuple[str, str]] = []

    # Prompt/UI chỉ hiển thị tối đa ba món, nên không retrieve các slot chắc chắn bị bỏ.
    selected_categories = base_rule.get("goi_y_phoi_cung", [])[:OUTFIT_MAX_PRODUCT_SLOTS]
    for category in selected_categories:
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

        semantic_fallbacks.append((category, f"{category} {style_query}"))

    if not semantic_fallbacks:
        return outfit_rules

    # Embed toàn bộ slot còn thiếu trong một request thay vì gọi VastAI tuần tự.
    query_vectors = get_rule_embeddings().embed_documents(
        [query for _, query in semantic_fallbacks]
    )
    _count_metric(metrics, "bge_m3")
    _count_vectors(metrics, "bge_m3", len(query_vectors))

    for (category, _), query_vector in zip(semantic_fallbacks, query_vectors):
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
    result = get_products_for_outfit_batch(
        [(layer_b_category, product_type)],
        phong_cach,
        vdb=vdb,
    )
    return result.get(layer_b_category, [])


def _layer_a_filter(layer_b_category: str, product_type: str) -> Filter | None:
    """Build the Qdrant category filter for one outfit slot."""
    target_categories = get_layer_a_categories(layer_b_category, product_type)
    if not target_categories:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="metadata.category",
                match=MatchAny(any=target_categories),
            )
        ]
    )


def get_products_for_outfit_batch(
    items: list[tuple[str, str]],
    phong_cach: str,
    vdb=None,
    metrics: dict | None = None,
) -> dict[str, list]:
    """Embed all outfit product queries once, then query Qdrant per slot.

    Args:
        items: Pairs of ``(layer_b_category, product_type)``.
        phong_cach: Style label shared by the selected outfit formula.
        vdb: Optional initialized Layer A vector store.
        metrics: Optional mutable diagnostic dictionary.

    Returns:
        Mapping from each Layer B category to its diverse product candidates.
    """
    if not items:
        return {}

    vector_db = vdb or get_product_vector_db()
    queries = [f"{product_type} {phong_cach}" for _, product_type in items]

    # ViFashionCLIP local xử lý cả outfit trong một batch để giảm overhead CPU.
    vectors = get_product_embeddings().embed_documents(queries)
    _count_metric(metrics, "vifashionclip_text")
    _count_vectors(metrics, "vifashionclip_text", len(vectors))

    results: dict[str, list] = {}
    for (layer_b_category, product_type), vector in zip(items, vectors):
        search_filter = _layer_a_filter(layer_b_category, product_type)
        raw_results = vector_db.similarity_search_with_score_by_vector(
            embedding=vector,
            k=8,
            filter=search_filter,
        )
        valid_products = [
            normalize_product_metadata(doc)
            for doc, score in raw_results
            if score >= 0.30
        ]
        results[layer_b_category] = diversity_filter_documents(valid_products, max_docs=3)
    return results


def _product_card_payload(doc, slot: str, fallback_title: str) -> dict:
    """Serialize one product with every available image for the commerce UI."""
    doc = normalize_product_metadata(doc)
    images = [url for url in doc.metadata.get("images", []) if url]
    main_image = doc.metadata.get("image_url", "")
    if main_image and main_image not in images:
        images.insert(0, main_image)
    return {
        "product_id": doc.metadata.get("product_id", "N/A"),
        "title": doc.metadata.get("title", "") or fallback_title,
        "category": doc.metadata.get("category", "Thời trang"),
        "slot": slot,
        "brand": doc.metadata.get("brand", "") or "Thương hiệu khác",
        "price": doc.metadata.get("price", 0),
        "images": images,
    }


def _format_outfit_context(
    base_rule: dict,
    outfit_rules: dict,
    profile: dict | None = None,
    base_item_context: dict | None = None,
    metrics: dict | None = None,
) -> tuple[str, list]:
    """Format selected Layer B rules and Layer A products for the answer chain."""
    if not outfit_rules:
        return "", []

    product_types = {
        layer_b_category: rule["rule_key"].split("|")[1].strip()
        for layer_b_category, rule in outfit_rules.items()
    }
    products_by_category = get_products_for_outfit_batch(
        list(product_types.items()),
        base_rule["phong_cach"],
        metrics=metrics,
    )
    outfit_products = {}
    for layer_b_category, rule in outfit_rules.items():
        product_type = product_types[layer_b_category]
        outfit_products[layer_b_category] = {
            "product_type": product_type,
            "ly_do": rule["ly_do_tu_van"],
            "products": products_by_category.get(layer_b_category, []),
        }

    lines = [
        "CÔNG THỨC PHỐI ĐỒ:",
        f"  Phong cách: {base_rule['phong_cach']}",
        f"  Bối cảnh : {base_rule['boi_canh']}",
        f"  Lý do    : {base_rule['ly_do_tu_van']}",
    ]
    if base_item_context:
        lines = [
            "MÓN GỐC TỪ ẢNH:",
            f"  Category Layer A: {base_item_context.get('layer_a_category', 'Không rõ')}",
            f"  Category Layer B: {base_item_context.get('layer_b_category', 'Không rõ')}",
            f"  Sản phẩm gần nhất: {base_item_context.get('product_type', 'Không rõ')}",
            f"  Độ tin cậy image retrieval: {base_item_context.get('confidence', 'unknown')} "
            f"({base_item_context.get('confidence_score', 0)})",
            "",
            *lines,
        ]
    if profile and profile.get("dang_nguoi"):
        lines.append(f"  Dáng người: {profile['dang_nguoi']}")
    if profile and profile.get("tone_da"):
        lines.append(f"  Tone da   : {profile['tone_da']}")
    lines += ["", "SẢN PHẨM GỢI Ý:"]

    images_data: list[dict] = []
    for category, data in outfit_products.items():
        lines.append(f"\n[{category} — {data['product_type']}]")
        lines.append(f"  Lý do: {data['ly_do']}")
        if data["products"]:
            # Chọn đúng một món tốt nhất cho mỗi slot; các candidate còn lại
            # được gửi kèm để nút "Xem lựa chọn khác" không cần gọi lại RAG.
            selected_doc = normalize_product_metadata(data["products"][0])
            product_id = selected_doc.metadata.get("product_id", "N/A")
            title = selected_doc.metadata.get("title", "") or data["product_type"]
            price_raw = selected_doc.metadata.get("price", "N/A")
            try:
                price_fmt = f"{int(price_raw):,}".replace(",", ".")
            except Exception:
                price_fmt = price_raw
            lines.append(
                f"  - TÊN_CHÍNH_XÁC: {title} | MÃ_SP: {product_id} | "
                f"GIÁ_CHÍNH_XÁC: {price_fmt} VND"
            )
            lines.append(f"    {selected_doc.page_content[:600]}")

            selected_card = _product_card_payload(selected_doc, category, data["product_type"])
            selected_card["alternatives"] = [
                _product_card_payload(doc, category, data["product_type"])
                for doc in data["products"][1:3]
            ]
            images_data.append(selected_card)
        else:
            lines.append("  - (Chưa có sản phẩm phù hợp trong kho)")

    return "\n".join(lines), images_data


def build_outfit_context(
    user_query: str,
    gender: str = "female",
    profile: dict | None = None,
    metrics: dict | None = None,
) -> tuple[str, list]:
    """Build the full outfit context and product image payload for the UI."""
    started_at = time.perf_counter()
    base_rule = find_matching_rule(user_query, gender, profile, metrics=metrics)
    _time_metric(metrics, "layer_b_base_rule", started_at)
    if not base_rule:
        return "", []

    started_at = time.perf_counter()
    outfit_rules = find_outfit_details(base_rule, gender, metrics=metrics)
    _time_metric(metrics, "layer_b_detail_rules", started_at)
    if not outfit_rules:
        return "", []

    started_at = time.perf_counter()
    result = _format_outfit_context(
        base_rule,
        outfit_rules,
        profile=profile,
        metrics=metrics,
    )
    _time_metric(metrics, "layer_a_products", started_at)
    return result


def build_outfit_context_from_image_docs(
    image_docs: list,
    user_query: str = "",
    gender: str = "female",
    profile: dict | None = None,
    metrics: dict | None = None,
) -> tuple[str, list, dict]:
    """
    Build outfit context for "style this uploaded product image".

    Fast path: use FashionCLIP image retrieval metadata to infer the base item.
    The returned diagnostics tell the caller whether a VLM fallback would be useful.
    """
    item_context = analyze_image_item_context(image_docs)
    layer_b_query = build_layer_b_query_from_image_context(item_context, user_query, profile)
    item_context["layer_b_query"] = layer_b_query

    started_at = time.perf_counter()
    base_rule = find_matching_rule(layer_b_query, gender, profile, metrics=metrics)
    _time_metric(metrics, "layer_b_base_rule", started_at)
    if not base_rule:
        item_context["retrieval_metrics"] = metrics or {}
        return "", [], item_context

    started_at = time.perf_counter()
    outfit_rules = find_outfit_details(base_rule, gender, metrics=metrics)
    _time_metric(metrics, "layer_b_detail_rules", started_at)
    if not outfit_rules:
        item_context["retrieval_metrics"] = metrics or {}
        return "", [], item_context

    started_at = time.perf_counter()
    context, images = _format_outfit_context(
        base_rule,
        outfit_rules,
        profile=profile,
        base_item_context=item_context,
        metrics=metrics,
    )
    _time_metric(metrics, "layer_a_products", started_at)
    item_context["retrieval_metrics"] = metrics or {}
    return context, images, item_context
