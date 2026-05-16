"""
Xây dựng outfit context từ Layer B → Layer A → context string cho LLM.
"""
from fashion_rag.layer_b.matching import find_matching_rule, find_outfit_details
from fashion_rag.layer_b.category_mapping import get_products_for_outfit


def build_outfit_context(user_query, gender="female", profile=None, vector_db=None):
    """
    Luồng Layer B hoàn chỉnh:
      1. find_matching_rule()  — semantic + profile filter
      2. find_outfit_details() — exact match cùng phong_cach + boi_canh
      3. get_products_for_outfit() — Qdrant Layer A + category filter
      4. Trả về context string cho LLM
    """
    base_rule = find_matching_rule(user_query, gender, profile, vector_db)
    if not base_rule:
        return ""

    outfit_rules = find_outfit_details(base_rule, gender)
    if not outfit_rules:
        return ""

    outfit_products = {}
    for layer_b_category, rule in outfit_rules.items():
        product_type = rule["rule_key"].split("|")[1].strip()
        products = get_products_for_outfit(
            product_type=product_type,
            layer_b_category=layer_b_category,
            phong_cach=base_rule["phong_cach"],
            vector_db=vector_db,
        )
        outfit_products[layer_b_category] = {
            "product_type": product_type,
            "ly_do": rule["ly_do_tu_van"],
            "products": products,
        }

    lines = [
        "CÔNG THỨC PHỐI ĐỒ:",
        f"  Phong cách : {base_rule['phong_cach']}",
        f"  Bối cảnh   : {base_rule['boi_canh']}",
        f"  Lý do chính: {base_rule['ly_do_tu_van']}",
        "",
    ]
    if profile and profile.get("dang_nguoi"):
        lines.append(f"  Dáng người : {profile['dang_nguoi']}")
    if profile and profile.get("tone_da"):
        lines.append(f"  Tone da    : {profile['tone_da']}")
    lines.append("")
    lines.append("SẢN PHẨM GỢI Ý:")

    for cat, data in outfit_products.items():
        lines.append(f"\n[{cat} – {data['product_type']}]")
        lines.append(f"  Lý do: {data['ly_do']}")
        if data["products"]:
            for doc in data["products"]:
                pid = doc.metadata.get("product_id", "N/A")
                price = doc.metadata.get("price", "N/A")
                lines.append(f"  • (Mã SP: {pid} | Giá: {price} VNĐ)")
                lines.append(f"    {doc.page_content[:200]}")
        else:
            lines.append("  • (Chưa có sản phẩm phù hợp trong kho)")

    return "\n".join(lines)
