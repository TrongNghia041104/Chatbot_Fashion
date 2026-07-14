"""Product metadata helpers shared by retrieval and indexing scripts."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def extract_image_urls(item: dict) -> list[str]:
    """Return all high-quality image paths/URLs from one product item."""
    return [img.get("large") for img in item.get("images", []) if img.get("large")]


def normalize_to_text(value, default: str = "Không rõ") -> str:
    """Normalize metadata values into display/indexing text."""
    if value is None or value == "":
        return default
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        return ", ".join(cleaned) if cleaned else default
    return str(value).strip() or default


def build_product_metadata(item: dict) -> dict:
    """Build stable LangChain metadata for product documents."""
    image_urls = extract_image_urls(item)
    return {
        "product_id": item.get("product_id", ""),
        "title": item.get("title", ""),
        "category": item.get("category", ""),
        "department": item.get("department", ""),
        "brand": item.get("brand", ""),
        "price": item.get("price", 0),
        "images": image_urls,
        "image_url": image_urls[0] if image_urls else "",
    }


def build_product_page_content(item: dict) -> str:
    """Build the labeled Vietnamese text used by ViFashionCLIP retrieval."""
    details = item.get("details", {}) or {}
    fields = [
        ("Tên sản phẩm", item.get("title")),
        ("Mã sản phẩm", item.get("product_id")),
        ("Danh mục", item.get("category")),
        ("Đối tượng", item.get("department")),
        ("Thương hiệu", item.get("brand")),
        ("Giá", f"{item.get('price', 0)} VND"),
        ("Màu sắc", details.get("main_color")),
        ("Chất liệu", details.get("material")),
        ("Kích cỡ", details.get("size")),
        ("Họa tiết", details.get("pattern")),
        ("Mùa phù hợp", item.get("season")),
        ("Dịp sử dụng", item.get("occasion")),
        ("Mô tả", item.get("description")),
    ]
    return "\n".join(f"{label}: {normalize_to_text(value)}" for label, value in fields)


def process_fashion_metadata(file_path: str | Path) -> list[Document]:
    """Read product JSONL into LangChain Documents and print basic quality stats."""
    file_path = Path(file_path)
    documents: list[Document] = []
    stats = {
        "total_lines": 0,
        "json_errors": 0,
        "missing_product_id": 0,
        "missing_category": 0,
        "missing_image": 0,
    }

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["total_lines"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue

            metadata = build_product_metadata(item)
            if not metadata["product_id"]:
                stats["missing_product_id"] += 1
            if not metadata["category"]:
                stats["missing_category"] += 1
            if not metadata["image_url"]:
                stats["missing_image"] += 1

            documents.append(
                Document(
                    page_content=build_product_page_content(item),
                    metadata=metadata,
                )
            )

    print(
        "[STATS] "
        f"lines={stats['total_lines']} | docs={len(documents)} | "
        f"json_errors={stats['json_errors']} | "
        f"missing_id={stats['missing_product_id']} | "
        f"missing_category={stats['missing_category']} | "
        f"missing_image={stats['missing_image']}"
    )
    return documents
