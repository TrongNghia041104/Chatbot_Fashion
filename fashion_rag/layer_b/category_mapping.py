"""
Ánh xạ category từ Layer B sang Layer A và tìm sản phẩm thực.
"""
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

# Ánh xạ 7/8 category Layer B → danh mục metadata.category của Layer A
CATEGORY_MAPPING = {
    "Áo mặc trong (áo thun/sơ mi)": ["Áo"],
    "Áo khoác ngoài":                ["Áo khoác"],
    "Áo khoác nhẹ/Áo len":           ["Áo khoác"],
    "Quần/Chân váy":                  ["Quần", "Chân váy"],
    "Đầm/Jumpsuit":                   ["Đầm", "Jumpsuit"],
    "Giày dép":                       ["Giày"],
    "Túi xách":                       ["Túi xách"],
    "Phụ kiện":                       None,
}

# Keyword router cho "Phụ kiện" → đúng danh mục Layer A
PHU_KIEN_KEYWORD_ROUTER = {
    "Mũ":             ["beret", "hat", "cap", "beanie", "fedora", "bucket", "brim", "flat cap", "earflap", "snood", "balaclava", "trapper"],
    "Găng tay":        ["gloves", "glove", "arm warmer"],
    "Kính mắt":        ["glasses", "sunglasses", "sunglass"],
    "Đồng hồ":         ["watch"],
    "Dây chuyền":      ["necklace", "chain pendant", "chain"],
    "Bông tai":        ["earring", "earrings"],
    "Vòng tay":        ["bracelet"],
    "Nhẫn":            ["ring"],
    "Ghim cài áo":     ["brooch", "pin", "badge"],
    "Phụ kiện hỗ trợ": ["socks", "sock", "scarf", "tie", "belt", "bandana", "headband", "suspender"],
}


def get_layer_a_categories(layer_b_category, product_type):
    """Trả về danh sách tên category Layer A cần filter khi tìm Qdrant."""
    if layer_b_category != "Phụ kiện":
        return CATEGORY_MAPPING.get(layer_b_category, [])

    ptype_lower = product_type.lower()
    for layer_a_cat, keywords in PHU_KIEN_KEYWORD_ROUTER.items():
        if any(kw in ptype_lower for kw in keywords):
            return [layer_a_cat]

    return ["Phụ kiện hỗ trợ"]


def get_products_for_outfit(product_type, layer_b_category, phong_cach, vector_db):
    """Tìm sản phẩm thật trong Layer A (Qdrant) cho 1 món đồ trong outfit."""
    target_categories = get_layer_a_categories(layer_b_category, product_type)

    search_filter = None
    if target_categories:
        search_filter = Filter(
            must=[FieldCondition(key="metadata.category", match=MatchAny(any=target_categories))]
        )

    query = f"{product_type} {phong_cach}"
    results = vector_db.similarity_search(query=query, k=3, filter=search_filter)
    return results
