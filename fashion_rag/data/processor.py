"""
Xử lý dữ liệu metadata sản phẩm thời trang từ file JSONL.
Chuyển đổi dữ liệu thô có cấu trúc thành LangChain Documents
để phục vụ cho quá trình embedding và lưu vào vector database.
"""
import json
import os

from langchain_core.documents import Document


def process_fashion_metadata(file_path: str) -> list[Document]:
    """
    Đọc file JSONL và chuyển đổi thành danh sách các LangChain Documents.

    Mỗi Document bao gồm:
    - page_content (string): Văn bản hóa từ metadata sản phẩm → dùng để embedding
    - metadata (dict): Thông tin bổ trợ giữ nguyên cấu trúc
    """
    documents = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # Trích xuất danh sách link ảnh từ mảng images
            image_urls = [
                img.get("large") for img in item.get("images", []) if "large" in img
            ]

            # Metadata của sản phẩm
            metadata = {
                "product_id": item.get("product_id", ""),
                "category": item.get("category", ""),
                "department": item.get("department", ""),
                "brand": item.get("brand", ""),
                "price": item.get("price", 0),
                "images": image_urls,
            }

            # Trích xuất thông tin chi tiết
            details = item.get("details", {})
            main_color = details.get("main_color", "Không có thông tin về màu sắc")
            material = details.get("material", "Không có thông tin về chất liệu")
            size = details.get("size", "Không có thông tin về kích thước")
            pattern = details.get("pattern", "Không có thông tin về họa tiết")

            # Văn bản hóa dữ liệu có cấu trúc
            page_content = (
                f"Sản phẩm {item.get('title', 'Không có tên sản phẩm')} thuộc danh mục {item.get('category', '')} "
                f"dành cho {item.get('department', '')}. "
                f"Thương hiệu: {item.get('brand', 'Không có thông tin về thương hiệu')}. "
                f"Mức giá: {item.get('price', 0)} VNĐ. "
                f"Đặc điểm chi tiết: có các loại màu sắc: {main_color}, chất liệu: {material}, kích cỡ: {size}, họa tiết: {pattern}. "
                f"Phù hợp sử dụng cho {item.get('season', '')} và các dịp {item.get('occasion', '')}. "
                f"Mô tả chi tiết: {item.get('description', '')}"
            )

            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

    return documents


def process_all_directory(directory_path: str) -> list[Document]:
    """
    Lặp qua tất cả các file .jsonl trong thư mục và gộp thành danh sách Documents.
    """
    all_documents = []

    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory_path, filename)
            print(f"[THÔNG BÁO] Đang đọc file: {filename}...")
            docs = process_fashion_metadata(file_path)
            all_documents.extend(docs)

    return all_documents
