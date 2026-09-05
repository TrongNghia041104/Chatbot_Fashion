"""Refine notebook 04 and standardize notebook headings to the BƯỚC format."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "notebooks" / "research_demo_v3_split"
NB01 = SPLIT_DIR / "01_environment_config_models.ipynb"
NB02 = SPLIT_DIR / "02_product_data_pipeline.ipynb"
NB04 = SPLIT_DIR / "04_image_retrieval_debug.ipynb"


def source(text: str) -> list[str]:
    text = text.strip("\n")
    return [line + "\n" for line in text.splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def replace_markdown(notebook_path: Path, replacements: dict[str, str]) -> None:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        for old, new in replacements.items():
            text = text.replace(old, new)
        cell["source"] = source(text)
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def standardize_headings() -> None:
    replace_markdown(
        NB01,
        {
            "## PHẦN 0: Kiểm Tra Môi Trường": "## BƯỚC 1: Kiểm Tra Môi Trường",
            "## PHẦN 1: Import Thư Viện Cần Cho Notebook Này": "## BƯỚC 2: Import Thư Viện Cần Cho Notebook Này",
            "## PHẦN 2A: Constants Và Cấu Hình Mô Hình": "## BƯỚC 3: Constants Và Cấu Hình Mô Hình",
            "## PHẦN 2B: Tìm Thư Mục Dự Án Và Checkpoint": "## BƯỚC 4: Tìm Thư Mục Dự Án Và Checkpoint",
            "## PHẦN 2B.1: Kiểm Tra Nhanh Các Dịch Vụ Nền": "## BƯỚC 5: Kiểm Tra Nhanh Các Dịch Vụ Nền",
            "## PHẦN 2C: Các Khối Neural Network Của ViFashionCLIP": "## BƯỚC 6: Các Khối Neural Network Của ViFashionCLIP",
            "## PHẦN 2D: `BGEM3Embeddings` Cho Layer B": "## BƯỚC 7: `BGEM3Embeddings` Cho Layer B",
            "## PHẦN 2E: `ViFashionCLIPTextEmbeddings` Cho Layer A Text": "## BƯỚC 8: `ViFashionCLIPTextEmbeddings` Cho Layer A Text",
            "## PHẦN 2F: `FashionCLIPImageEmbeddings` Cho Layer A Image": "## BƯỚC 9: `FashionCLIPImageEmbeddings` Cho Layer A Image",
        },
    )
    replace_markdown(
        NB02,
        {
            "| Phần | Dùng để làm gì | Ví dụ |": "| Mảnh dữ liệu | Dùng để làm gì | Ví dụ |",
            "## PHẦN 3A: Tư Duy Data Pipeline - Biến Product Thô Thành `Document`": "## BƯỚC 1: Tư Duy Data Pipeline - Biến Product Thô Thành `Document`",
            "## PHẦN 3A.1: Preview Một Sản Phẩm Trước Khi Index": "## BƯỚC 2: Preview Một Sản Phẩm Trước Khi Index",
            "## PHẦN 3B: Đọc JSONL Và Index Vào Qdrant": "## BƯỚC 3: Đọc JSONL Và Index Vào Qdrant",
        },
    )


def split_notebook_04_step_9(cells: list[dict]) -> list[dict]:
    """Tách cell code quá dài của bước 9 thành nhiều cell nhỏ, dễ đọc hơn trong notebook."""
    output: list[dict] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            output.append(cell)
            continue

        text = "".join(cell.get("source", []))
        if "def image_doc_row" not in text or "def debug_image_and_text" not in text:
            output.append(cell)
            continue

        markers = [
            "def image_doc_row",
            "def extract_json_object",
            "def build_clarification_diagnostic",
            "def debug_image_and_text",
            "def calibrate_image_threshold",
        ]
        positions = [(marker, text.index(marker)) for marker in markers]
        chunks: dict[str, str] = {}
        for idx, (marker, start) in enumerate(positions):
            end = positions[idx + 1][1] if idx + 1 < len(positions) else len(text)
            chunks[marker] = text[start:end].strip()

        output.extend(
            [
                markdown_cell(
                    """
## BƯỚC 9A: Chuẩn Hóa Output Và Hiển Thị Ảnh

Nhóm hàm này không làm retrieval mới. Nó chỉ biến `Document` thành thứ dễ đọc/dễ nhìn:

- `image_doc_row`: rút gọn metadata quan trọng của một sản phẩm.
- `resolve_doc_image_path`: tìm file ảnh local từ `main_image_path`, `image_url`, hoặc `images`.
- `show_image_file`: hiện ảnh query.
- `show_retrieval_gallery`: hiện gallery top results.
- `print_docs_table`: in bảng text gồm score, product_id, category, brand, title, image_url.
- `debug_image_query`: chạy riêng image-only retrieval để kiểm tra ảnh đầu vào có đang tìm đúng loại sản phẩm không.
- `compare_retrieval_sets`: so sánh overlap giữa hai danh sách kết quả.

Nếu cell này lỗi, vấn đề thường là path ảnh hoặc metadata ảnh, chưa phải lỗi retrieval.
"""
                ),
                code_cell(chunks["def image_doc_row"]),
                markdown_cell(
                    """
## BƯỚC 9B: VLM Rerank Candidate

Nhóm hàm này chỉ chạy khi bạn bật `use_vlm_rerank=True`.

Luồng xử lý:

```text
query image + candidate image + text query
  -> gửi sang Qwen-VL qua Ollama/Vast.ai
  -> VLM trả JSON {"score": ..., "reason": ...}
  -> sắp xếp lại top candidates theo `vlm_score`
```

Ý nghĩa từng hàm:

- `extract_json_object`: cố parse JSON ngay cả khi VLM lỡ trả thêm chữ bên ngoài.
- `clamp_score`: ép score về khoảng 0..1.
- `get_vlm_ollama_client`: tạo client trỏ tới `VLM_OLLAMA_BASE_URL`, có thể là local hoặc Vast.ai.
- `vlm_score_candidate`: chấm một candidate bằng VLM.
- `vlm_rerank_documents`: chấm top-k candidate rồi sắp xếp lại.

VLM rerank rất chậm vì mỗi candidate là một lần đọc 2 ảnh. Khi test tốc độ, dùng `vlm_top_k=1` hoặc `vlm_top_k=3`.
"""
                ),
                code_cell(chunks["def extract_json_object"]),
                markdown_cell(
                    """
## BƯỚC 9C: Quyết Định Có Nên Hỏi Lại Người Dùng Không

Notebook này làm theo hướng: **search trước, hỏi lại sau**.

`build_clarification_diagnostic` không tự sửa kết quả retrieval. Nó chỉ nhìn các tín hiệu debug:

- có final docs không;
- category suy ra từ image-only có đủ chắc không;
- score gap giữa top 1 và top 2 có quá sát không;
- text query có tín hiệu department/gender nhưng hệ thống đang hard-lock category không.

Nếu các tín hiệu này yếu, notebook gợi ý câu hỏi cần hỏi lại người dùng.
"""
                ),
                code_cell(chunks["def build_clarification_diagnostic"]),
                markdown_cell(
                    """
## BƯỚC 9D: Pipeline Chính Cho Ảnh + Text

Đây là hàm bạn sẽ dùng nhiều nhất: `debug_multimodal_retrieval`.

Nó chạy theo thứ tự:

```text
1. image-only search
2. suy ra intent từ text query
3. suy ra category từ image-only results
4. chọn hard_lock hoặc soft_boost
5. text-only baseline
6. composed image+text retrieval
7. metadata-aware rerank
8. optional VLM rerank
9. in summary + clarification diagnostic
```

Điểm cần nhớ:

- `intent="auto"`: notebook tự đoán search/outfit bằng keyword trong text.
- `category_mode="auto"`: nếu intent là search thì hard-lock, nếu outfit thì soft-boost.
- `use_vlm_rerank=False`: nhanh hơn, phù hợp debug retrieval.
- `use_vlm_rerank=True`: chậm hơn, phù hợp kiểm tra chất lượng top candidates bằng VLM.
"""
                ),
                code_cell(chunks["def debug_image_and_text"]),
                markdown_cell(
                    """
## BƯỚC 9E: Calibrate Threshold Và Ví Dụ Chạy

`calibrate_image_threshold` dùng self-search: lấy ảnh trong dataset làm query rồi xem sản phẩm gốc có được tìm lại không.

Mục tiêu của nó là hiểu ngưỡng `IMAGE_SEARCH_SCORE_THRESHOLD=0.15`:

- nếu positive match thường cao hơn 0.15 nhiều, threshold này đang khá thoáng;
- nếu nhiều positive match thấp hơn 0.15, threshold quá cao sẽ làm mất kết quả đúng;
- nếu negative cũng cao, tăng threshold không đủ, cần rerank/filter tốt hơn.

Ví dụ 3 là ví dụ quan trọng nhất cho multimodal retrieval.
"""
                ),
                code_cell(chunks["def calibrate_image_threshold"]),
            ]
        )
    return output


def refine_notebook_04() -> None:
    nb = json.loads(NB04.read_text(encoding="utf-8"))
    metadata = nb.get("metadata", {})
    cells = [
        markdown_cell(
            """
# 04 - Image Retrieval Debug

Notebook này dùng để soi riêng nhánh **tìm sản phẩm bằng ảnh**.

Góc nhìn cần giữ trong đầu:

```text
ảnh query hoặc ảnh MAIN sản phẩm
  -> FashionCLIPImageEmbeddings
  -> vector ảnh 512 chiều
  -> Qdrant image collection
  -> raw image points
  -> Document có page_content + metadata
  -> final docs gửi cho LLM
```

Notebook này lấy image retrieval làm trọng tâm, nhưng có thêm text-only và composed image+text retrieval để đối chiếu. Nó vẫn không debug prompt/chat loop. Nếu ảnh search sai, hãy kiểm tra theo thứ tự: ảnh có tồn tại không, vector ảnh có tạo được không, Qdrant image collection có đúng không, score có quá thấp không.

Notebook có thể chạy độc lập vì tự import config từ `app/`. Model ảnh chỉ load ở bước cần encode ảnh.
"""
        ),
        markdown_cell(
            """
## BƯỚC 1: Setup Tối Thiểu

Cell này tìm thư mục gốc `Chatbot_Fashion/`, thêm vào `sys.path`, rồi import các thư viện nhẹ.

Chưa load FashionCLIP ở bước này. Load model ảnh quá sớm sẽ làm notebook chậm và khiến bạn khó biết cell nào đang tốn thời gian.
"""
        ),
        code_cell(
            """
import json
import math
import random
import re
import sys
import unicodedata
import uuid
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, VectorParams
from qdrant_client.models import PointStruct
from tqdm.auto import tqdm


def find_chatbot_fashion_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "app" / "config.py").exists():
            return candidate
    raise RuntimeError("Không tìm thấy thư mục gốc Chatbot_Fashion chứa app/config.py")


CHATBOT_FASHION_DIR = find_chatbot_fashion_root()
if str(CHATBOT_FASHION_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_FASHION_DIR))

from app.config import (
    IMAGE_EMBEDDING_BATCH_SIZE,
    IMAGE_SEARCH_MAX_PRODUCTS as APP_IMAGE_SEARCH_MAX_PRODUCTS,
    IMAGE_SEARCH_SCORE_THRESHOLD as APP_IMAGE_SEARCH_SCORE_THRESHOLD,
    IMAGE_SEARCH_TOP_K as APP_IMAGE_SEARCH_TOP_K,
    IMAGE_VECTOR_SIZE,
    METADATA_FILE as APP_METADATA_FILE,
    OLLAMA_BASE_URL,
    PRODUCT_IMAGE_ROOT as APP_PRODUCT_IMAGE_ROOT,
    PRODUCT_SEARCH_BRAND_LIMIT as APP_PRODUCT_SEARCH_BRAND_LIMIT,
    PRODUCT_SEARCH_CANDIDATE_K as APP_PRODUCT_SEARCH_CANDIDATE_K,
    PRODUCT_SEARCH_PAGE_SIZE as APP_PRODUCT_SEARCH_PAGE_SIZE,
    QWEN_VL_MODEL,
    QDRANT_COLLECTION_FASHION,
    QDRANT_COLLECTION_PRODUCT_IMAGE,
    QDRANT_URL,
)

print("[OK] Setup notebook 04 hoàn tất")
print(f"Project root: {CHATBOT_FASHION_DIR}")
"""
        ),
        markdown_cell(
            """
## BƯỚC 2: Cấu Hình Image Retrieval

Các biến cần nhìn khi debug:

- `PRODUCT_IMAGE_ROOT`: thư mục ảnh local.
- `METADATA_FILE`: file JSONL chứa metadata sản phẩm.
- `PRODUCT_IMAGE_COLLECTION`: collection Qdrant lưu vector ảnh.
- `IMAGE_SEARCH_SCORE_THRESHOLD`: ngưỡng score khi search bằng ảnh.

Nếu notebook báo không thấy ảnh, kiểm tra `PRODUCT_IMAGE_ROOT` trước. Nếu có ảnh nhưng search rỗng, kiểm tra collection Qdrant.
"""
        ),
        code_cell(
            """
PRODUCT_IMAGE_ROOT = Path(APP_PRODUCT_IMAGE_ROOT)
METADATA_FILE = Path(APP_METADATA_FILE)
PRODUCT_COLLECTION = QDRANT_COLLECTION_FASHION
PRODUCT_IMAGE_COLLECTION = QDRANT_COLLECTION_PRODUCT_IMAGE

IMAGE_SEARCH_TOP_K = APP_IMAGE_SEARCH_TOP_K
IMAGE_SEARCH_MAX_PRODUCTS = APP_IMAGE_SEARCH_MAX_PRODUCTS
IMAGE_SEARCH_SCORE_THRESHOLD = APP_IMAGE_SEARCH_SCORE_THRESHOLD
TEXT_SEARCH_CANDIDATE_K = APP_PRODUCT_SEARCH_CANDIDATE_K
TEXT_SEARCH_MAX_PRODUCTS = APP_PRODUCT_SEARCH_PAGE_SIZE
TEXT_SEARCH_BRAND_LIMIT = APP_PRODUCT_SEARCH_BRAND_LIMIT
MULTIMODAL_CANDIDATE_K = max(IMAGE_SEARCH_TOP_K, 30)
MULTIMODAL_FINAL_K = IMAGE_SEARCH_MAX_PRODUCTS
CATEGORY_LOCK_MIN_SHARE = 0.50
CATEGORY_LOCK_MIN_COUNT = 2
CONFIDENCE_SCORE_GAP_MIN = 0.02
VLM_RERANK_TOP_K = 5
VLM_OLLAMA_BASE_URL = OLLAMA_BASE_URL

qdrant = QdrantClient(url=QDRANT_URL, timeout=20, check_compatibility=False)
_image_embeddings_instance = None
_text_embeddings_instance = None
_text_vector_db = None


def get_image_embeddings():
    \"\"\"Load FashionCLIP image encoder đúng lúc cần encode ảnh, không load ở đầu notebook.\"\"\"
    global _image_embeddings_instance
    if _image_embeddings_instance is None:
        from app.core.image_search import FashionCLIPImageEmbeddings

        print("[INFO] Load FashionCLIP image encoder lần đầu...")
        _image_embeddings_instance = FashionCLIPImageEmbeddings(batch_size=IMAGE_EMBEDDING_BATCH_SIZE)
    return _image_embeddings_instance


def get_text_vector_db():
    \"\"\"Load text embedding/vector store khi cần so sánh image retrieval với text retrieval.\"\"\"
    global _text_vector_db
    if _text_vector_db is None:
        print("[INFO] Chuẩn bị text vector store để so sánh với image retrieval...")
        _text_vector_db = QdrantVectorStore(
            client=qdrant,
            collection_name=PRODUCT_COLLECTION,
            embedding=get_text_embeddings(),
        )
    return _text_vector_db


def get_text_embeddings():
    \"\"\"Load ViFashionCLIP text embedder 512-dim, cùng không gian với FashionCLIP image vector.\"\"\"
    global _text_embeddings_instance
    if _text_embeddings_instance is None:
        from app.core.embeddings import get_product_embeddings

        _text_embeddings_instance = get_product_embeddings()
    return _text_embeddings_instance


print("[OK] Image retrieval config")
print(f"Image root : {PRODUCT_IMAGE_ROOT}")
print(f"Metadata   : {METADATA_FILE}")
print(f"Collection : {PRODUCT_IMAGE_COLLECTION}")
print(f"Search     : top-{IMAGE_SEARCH_TOP_K} | threshold={IMAGE_SEARCH_SCORE_THRESHOLD} | final={IMAGE_SEARCH_MAX_PRODUCTS}")
print(f"Text compare: top-{TEXT_SEARCH_CANDIDATE_K} -> final {TEXT_SEARCH_MAX_PRODUCTS}")
print(f"Multimodal : candidates={MULTIMODAL_CANDIDATE_K} -> final {MULTIMODAL_FINAL_K} | VLM top-{VLM_RERANK_TOP_K}")
print(f"VLM model  : {QWEN_VL_MODEL}")
print(f"VLM Ollama : {VLM_OLLAMA_BASE_URL}")
"""
        ),
        markdown_cell(
            """
## BƯỚC 3: Tìm Ảnh MAIN Của Sản Phẩm

Ảnh trong metadata thường là relative path hoặc URL. Ba hàm dưới đây biến metadata thành file ảnh thật trên máy:

- `get_main_image_relative_path`: chọn ảnh đại diện tốt nhất.
- `resolve_main_image_path`: đổi relative path thành local path.
- `iter_products_with_main_image`: chỉ yield sản phẩm có ảnh tồn tại.

Nếu bước này sai, image retrieval phía sau chắc chắn sai vì Qdrant sẽ index nhầm hoặc thiếu ảnh.
"""
        ),
        code_cell(
            """
def get_main_image_relative_path(item: dict) -> str:
    \"\"\"Chọn ảnh MAIN theo thứ tự ưu tiên: variant MAIN, tên có _MAIN, rồi ảnh đầu tiên.\"\"\"
    images = item.get("images", []) or []

    for image in images:
        if str(image.get("variant", "")).upper() == "MAIN" and image.get("large"):
            return image["large"]

    for image in images:
        large = str(image.get("large", ""))
        if "_MAIN" in large.upper():
            return large

    if images and images[0].get("large"):
        return images[0]["large"]

    product_id = item.get("product_id")
    return f"images/{product_id}_MAIN.jpg" if product_id else ""


def resolve_main_image_path(relative_path: str, image_root: str | Path = PRODUCT_IMAGE_ROOT) -> Path:
    \"\"\"Bỏ prefix `images/` nếu có, rồi ghép với thư mục ảnh local.\"\"\"
    rel = Path(str(relative_path).replace("\\\\", "/"))
    if rel.parts and rel.parts[0].lower() == "images":
        rel = Path(*rel.parts[1:])
    return Path(image_root) / rel


def iter_products_with_main_image(
    metadata_file: str | Path = METADATA_FILE,
    image_root: str | Path = PRODUCT_IMAGE_ROOT,
):
    \"\"\"Đọc JSONL và yield `(item, rel_path, img_path)` cho sản phẩm có ảnh MAIN tồn tại.\"\"\"
    metadata_file = Path(metadata_file)
    image_root = Path(image_root)
    with metadata_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            rel_path = get_main_image_relative_path(item)
            img_path = resolve_main_image_path(rel_path, image_root)
            if img_path.exists():
                yield item, rel_path, img_path


print("[OK] Image path helpers ready")
"""
        ),
        markdown_cell(
            """
## BƯỚC 4: Preview Một Ảnh MAIN Bất Kỳ

Cell này không index và không load model. Nó chỉ random một sản phẩm có ảnh thật để bạn kiểm tra:

- metadata có `product_id`, `title`, `category` không;
- `rel_path` từ JSONL là gì;
- `img_path` local có tồn tại không.
"""
        ),
        code_cell(
            """
def preview_random_main_image(
    metadata_file: str | Path = METADATA_FILE,
    image_root: str | Path = PRODUCT_IMAGE_ROOT,
):
    \"\"\"Random một sản phẩm có ảnh MAIN tồn tại để kiểm tra path trước khi index/search.\"\"\"
    chosen = None
    count = 0
    for item, rel_path, img_path in iter_products_with_main_image(metadata_file, image_root):
        count += 1
        if random.randrange(count) == 0:
            chosen = (item, rel_path, img_path)

    if chosen is None:
        print("[WARN] Không tìm thấy sản phẩm nào có ảnh MAIN tồn tại.")
        return None

    item, rel_path, img_path = chosen
    print(f"Scanned products with MAIN image: {count}")
    print(f"product_id : {item.get('product_id', '')}")
    print(f"title      : {item.get('title', '')}")
    print(f"category   : {item.get('category', '')}")
    print(f"rel_path   : {rel_path}")
    print(f"img_path   : {img_path}")
    print(f"exists     : {img_path.exists()}")
    return {"item": item, "rel_path": rel_path, "img_path": img_path}


# Ví dụ:
# sample = preview_random_main_image()
"""
        ),
        markdown_cell(
            """
## BƯỚC 5: Build Payload Ảnh Cho Qdrant

Payload là phần đi kèm vector ảnh trong Qdrant. Vector giúp tìm ảnh giống nhau, còn payload giúp ta biết ảnh đó thuộc sản phẩm nào.

`page_content` vẫn quan trọng: sau khi image search tìm được sản phẩm, LLM cần text sản phẩm để trả lời có căn cứ.
"""
        ),
        code_cell(
            """
def extract_image_urls(item: dict) -> list[str]:
    \"\"\"Lấy các ảnh chất lượng cao từ metadata sản phẩm.\"\"\"
    return [image.get("large") for image in item.get("images", []) if image.get("large")]


def normalize_to_text(value, default: str = "Không rõ") -> str:
    \"\"\"Đổi giá trị metadata thành text ổn định để đưa vào page_content.\"\"\"
    if value is None or value == "":
        return default
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        return ", ".join(cleaned) if cleaned else default
    return str(value).strip() or default


def build_product_metadata(item: dict) -> dict:
    \"\"\"Tạo metadata gọn, ổn định cho một sản phẩm.\"\"\"
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
    \"\"\"Tạo text mô tả sản phẩm để LLM có context sau khi image search trả về kết quả.\"\"\"
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
        ("Dịp sử dụng", item.get("occasion")),
        ("Mô tả", item.get("description")),
    ]
    return "\\n".join(f"{label}: {normalize_to_text(value)}" for label, value in fields)


def build_image_payload(item: dict, rel_path: str, img_path: Path) -> dict:
    \"\"\"Tạo payload lưu vào Qdrant cùng vector ảnh MAIN.\"\"\"
    metadata = build_product_metadata(item)
    metadata["main_image_path"] = str(img_path)
    metadata["main_image_relpath"] = rel_path
    metadata["image_url"] = metadata.get("image_url") or rel_path
    return {
        "product_id": metadata.get("product_id", ""),
        "title": metadata.get("title", ""),
        "category": metadata.get("category", ""),
        "department": metadata.get("department", ""),
        "brand": metadata.get("brand", ""),
        "price": metadata.get("price", 0),
        "image_url": metadata.get("image_url", ""),
        "main_image_path": str(img_path),
        "main_image_relpath": rel_path,
        "page_content": build_product_page_content(item),
        "metadata": metadata,
    }


def preview_image_payload(sample: dict | None) -> dict | None:
    \"\"\"In payload rút gọn để kiểm tra trước khi index vào Qdrant.\"\"\"
    if not sample:
        return None
    payload = build_image_payload(sample["item"], sample["rel_path"], sample["img_path"])
    print("product_id:", payload["product_id"])
    print("title     :", payload["title"])
    print("category  :", payload["category"])
    print("image_url :", payload["image_url"])
    print("content   :", payload["page_content"][:300].replace("\\n", " | "))
    return payload


# Ví dụ:
# payload = preview_image_payload(sample)
"""
        ),
        markdown_cell(
            """
## BƯỚC 6: Kiểm Tra Image Collection Trong Qdrant

Trước khi search bằng ảnh, cần biết collection ảnh đã tồn tại chưa.

Nếu collection chưa tồn tại, đừng debug LLM vội. Khi đó hệ thống chưa có kho vector ảnh để tìm kiếm.
"""
        ),
        code_cell(
            """
def check_image_collection(collection_name: str = PRODUCT_IMAGE_COLLECTION) -> dict | None:
    \"\"\"Kiểm tra collection ảnh có tồn tại và đang có bao nhiêu point.\"\"\"
    if not qdrant.collection_exists(collection_name):
        print(f"[WARN] Chưa có image collection: {collection_name}")
        return None

    count = qdrant.count(collection_name).count
    info = qdrant.get_collection(collection_name)
    print(f"[OK] Collection: {collection_name}")
    print(f"Points     : {count}")
    print(f"Status     : {getattr(info, 'status', '')}")
    return {"collection": collection_name, "points": count, "info": info}


# Ví dụ:
# image_collection_info = check_image_collection()
"""
        ),
        markdown_cell(
            """
## BƯỚC 7: Index Ảnh MAIN Vào Qdrant

Chỉ chạy bước này khi collection ảnh chưa có hoặc chưa đủ.

Pipeline index:

```text
JSONL -> ảnh MAIN tồn tại -> FashionCLIP image vector -> Qdrant point
```

Với 65k sản phẩm, bước này có thể lâu. Debug bình thường nên kiểm tra collection trước rồi mới quyết định index lại.
"""
        ),
        code_cell(
            """
def run_main_image_index_pipeline(
    metadata_file: str | Path = METADATA_FILE,
    image_root: str | Path = PRODUCT_IMAGE_ROOT,
    collection_name: str = PRODUCT_IMAGE_COLLECTION,
    batch_size: int = 64,
) -> None:
    \"\"\"Index một ảnh MAIN cho mỗi sản phẩm vào Qdrant, có resume theo số point hiện có.\"\"\"
    metadata_file = Path(metadata_file)
    image_root = Path(image_root)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata: {metadata_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_root}")

    items = list(iter_products_with_main_image(metadata_file, image_root))
    print(f"[OK] Tìm thấy {len(items)} sản phẩm có ảnh MAIN")

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=IMAGE_VECTOR_SIZE, distance=Distance.COSINE),
        )
        current_count = 0
    else:
        current_count = qdrant.count(collection_name).count
        print(f"[INFO] Collection đã có {current_count} vectors")

    remaining = items[current_count:]
    if not remaining:
        print("[OK] Image collection đã được index đầy đủ.")
        return

    embeddings = get_image_embeddings()
    with tqdm(total=len(items), initial=current_count, desc="Image index", unit="img") as progress:
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start:start + batch_size]
            image_paths = [img_path for _, _, img_path in batch]
            vectors = embeddings.encode_image_paths(image_paths)
            points = []

            for (item, rel_path, img_path), vector in zip(batch, vectors):
                if vector is None:
                    continue
                product_id = str(item.get("product_id", ""))
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"fashion-main-image:{product_id}"))
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=build_image_payload(item, rel_path, img_path),
                ))

            if points:
                qdrant.upsert(collection_name=collection_name, points=points)
            progress.update(len(batch))

    print(f"[OK] Index ảnh hoàn tất -> {collection_name}")


# Chỉ bỏ comment khi thật sự muốn index lại:
# run_main_image_index_pipeline()
"""
        ),
        markdown_cell(
            """
## BƯỚC 8: Search Bằng Ảnh, Text Và Vector Ghép

Đây là nơi định nghĩa ba kiểu retrieval:

```text
image-only  : ảnh -> image vector -> image collection
text-only   : text -> text vector -> product text collection
composed    : ảnh + text -> một vector ghép -> image collection
```

Composed retrieval ở đây là zero-shot: không train thêm model, chỉ dùng vector ảnh và vector text đang có, cộng có trọng số rồi normalize lại.
"""
        ),
        code_cell(
            """
def normalize_product_metadata(doc: Document) -> Document:
    \"\"\"Đảm bảo Document có `images` dạng list và `image_url` ổn định.\"\"\"
    images = doc.metadata.get("images", [])
    if isinstance(images, str):
        images = [images] if images else []
    doc.metadata["images"] = images
    doc.metadata["image_url"] = doc.metadata.get("image_url") or (images[0] if images else "")
    return doc


def image_point_to_document(point, score_key: str = "image_search_score") -> Document:
    \"\"\"Đổi Qdrant image point thành Document dùng chung với prompt LLM.\"\"\"
    payload = point.payload or {}
    metadata = payload.get("metadata", {}) or {}
    metadata[score_key] = getattr(point, "score", None)
    metadata["image_url"] = payload.get("image_url") or metadata.get("image_url", "")
    return normalize_product_metadata(Document(
        page_content=payload.get("page_content", ""),
        metadata=metadata,
    ))


def query_image_collection_by_vector(
    query_vector: list[float],
    collection_name: str = PRODUCT_IMAGE_COLLECTION,
    top_k: int = IMAGE_SEARCH_TOP_K,
    max_products: int = IMAGE_SEARCH_MAX_PRODUCTS,
    score_threshold: float | None = IMAGE_SEARCH_SCORE_THRESHOLD,
    score_key: str = "image_search_score",
    query_filter: Filter | None = None,
) -> list[Document]:
    \"\"\"Query image collection bằng một vector 512-dim đã normalize.\"\"\"
    if not qdrant.collection_exists(collection_name):
        print(f"[WARN] Collection ảnh chưa tồn tại: {collection_name}")
        return []

    kwargs = {
        "collection_name": collection_name,
        "query": query_vector,
        "limit": top_k,
        "with_payload": True,
    }
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    if query_filter is not None:
        kwargs["query_filter"] = query_filter

    response = qdrant.query_points(**kwargs)
    docs: list[Document] = []
    seen_ids: set[str] = set()
    for point in response.points:
        doc = image_point_to_document(point, score_key=score_key)
        product_id = str(doc.metadata.get("product_id", ""))
        if product_id and product_id in seen_ids:
            continue
        docs.append(doc)
        if product_id:
            seen_ids.add(product_id)
        if len(docs) >= max_products:
            break
    return docs


def search_products_by_image(
    image_path: str | Path,
    collection_name: str = PRODUCT_IMAGE_COLLECTION,
    top_k: int = IMAGE_SEARCH_TOP_K,
    max_products: int = IMAGE_SEARCH_MAX_PRODUCTS,
    score_threshold: float | None = IMAGE_SEARCH_SCORE_THRESHOLD,
) -> list[Document]:
    \"\"\"Tìm sản phẩm tương tự ảnh query, lọc trùng product_id và giới hạn số kết quả cuối.\"\"\"
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[LỖI] Không tìm thấy ảnh: {image_path}")
        return []
    if not qdrant.collection_exists(collection_name):
        print(f"[WARN] Collection ảnh chưa tồn tại: {collection_name}")
        return []

    query_vector = get_image_embeddings().embed_image(image_path)
    if query_vector is None:
        print("[WARN] Không encode được ảnh query.")
        return []

    return query_image_collection_by_vector(
        query_vector=query_vector,
        collection_name=collection_name,
        top_k=top_k,
        max_products=max_products,
        score_threshold=score_threshold,
        score_key="image_search_score",
    )


def diversity_filter_documents(
    docs: list[Document],
    max_docs: int = TEXT_SEARCH_MAX_PRODUCTS,
    max_per_brand: int = TEXT_SEARCH_BRAND_LIMIT,
) -> list[Document]:
    \"\"\"Lọc trùng product_id và hạn chế quá nhiều sản phẩm cùng brand.\"\"\"
    selected: list[Document] = []
    seen_product_ids: set[str] = set()
    brand_counts: dict[str, int] = {}

    for doc in docs:
        doc = normalize_product_metadata(doc)
        product_id = str(doc.metadata.get("product_id", "")).strip().lower()
        brand = str(doc.metadata.get("brand", "")).strip().lower()

        if product_id and product_id in seen_product_ids:
            continue
        if brand and brand_counts.get(brand, 0) >= max_per_brand:
            continue

        selected.append(doc)
        if product_id:
            seen_product_ids.add(product_id)
        if brand:
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        if len(selected) >= max_docs:
            return selected

    return selected


def search_products_by_text(
    text_query: str,
    k: int = TEXT_SEARCH_CANDIDATE_K,
    max_products: int = TEXT_SEARCH_MAX_PRODUCTS,
) -> list[Document]:
    \"\"\"Chạy text retrieval gọn để đối chiếu với image retrieval trong cùng notebook.\"\"\"
    if not text_query or not text_query.strip():
        return []

    vector_db = get_text_vector_db()
    pairs = vector_db.similarity_search_with_score(query=text_query, k=k)
    docs: list[Document] = []
    for rank, (doc, score) in enumerate(pairs, start=1):
        doc = normalize_product_metadata(doc)
        doc.metadata["text_rank"] = rank
        doc.metadata["text_score"] = float(score)
        docs.append(doc)
    return diversity_filter_documents(docs, max_docs=max_products)


def strip_vietnamese_accents(text: str) -> str:
    \"\"\"Đưa text về dạng không dấu để bắt keyword ổn hơn khi user gõ thiếu dấu.\"\"\"
    normalized = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def normalize_debug_text(text: str) -> str:
    \"\"\"Lowercase + bỏ dấu + gom khoảng trắng, chỉ dùng cho rule debug đơn giản.\"\"\"
    text = strip_vietnamese_accents(text).lower()
    return re.sub(r"\\s+", " ", text).strip()


def infer_retrieval_intent(text_query: str) -> str:
    \"\"\"Suy ra intent ở mức debug: search/similar giữ loại sản phẩm, outfit thì không khóa cứng category.\"\"\"
    q = normalize_debug_text(text_query)
    outfit_keywords = [
        "phoi", "outfit", "mix", "mac voi", "di voi", "ket hop", "goi y set", "hop voi",
        "nen mac", "mac cung", "style voi",
    ]
    if any(keyword in q for keyword in outfit_keywords):
        return "outfit"
    return "search"


def infer_target_departments(text_query: str) -> list[str]:
    \"\"\"Bắt tín hiệu giới tính/departement từ câu hỏi để boost mềm, không dùng làm filter cứng.\"\"\"
    q = normalize_debug_text(text_query)
    if any(keyword in q for keyword in ["nu", "female", "woman", "women", "girl", "feminine", "nang", "ban gai"]):
        return ["Nữ", "Unisex"]
    if any(keyword in q for keyword in ["nam", "male", "man", "men", "boy", "ban trai"]):
        return ["Nam", "Unisex"]
    return []


def infer_category_from_docs(
    docs: list[Document],
    min_share: float = CATEGORY_LOCK_MIN_SHARE,
    min_count: int = CATEGORY_LOCK_MIN_COUNT,
) -> dict:
    \"\"\"Lấy category chiếm đa số trong image-only results để quyết định có nên khóa category không.\"\"\"
    counts: dict[str, int] = {}
    for doc in docs:
        category = str(doc.metadata.get("category", "")).strip()
        if category:
            counts[category] = counts.get(category, 0) + 1

    total = sum(counts.values())
    if not counts or total == 0:
        return {"category": None, "share": 0.0, "count": 0, "counts": counts, "confident": False}

    category, count = max(counts.items(), key=lambda item: item[1])
    share = count / total
    confident = count >= min_count and share >= min_share
    return {"category": category, "share": share, "count": count, "counts": counts, "confident": confident}


def build_category_filter(category: str | None) -> Filter | None:
    \"\"\"Tạo Qdrant filter cho payload top-level `category` trong image collection.\"\"\"
    if not category:
        return None
    return Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])


def score_gap(docs: list[Document], score_key: str) -> float | None:
    \"\"\"Khoảng cách score giữa rank 1 và rank 2; gap nhỏ nghĩa là top result chưa thật sự nổi bật.\"\"\"
    scores = [doc.metadata.get(score_key) for doc in docs if doc.metadata.get(score_key) is not None]
    if len(scores) < 2:
        return None
    return float(scores[0]) - float(scores[1])


def metadata_rerank_documents(
    docs: list[Document],
    text_query: str,
    locked_category: str | None = None,
    category_mode: str = "none",
    intent: str = "search",
    final_k: int = MULTIMODAL_FINAL_K,
) -> list[Document]:
    \"\"\"Rerank nhẹ bằng metadata: category từ ảnh, department từ text, và score retrieval gốc.

    Đây không phải model học máy. Nó là lớp debug/rule để bạn nhìn rõ: hệ thống đang ưu tiên gì
    trước khi đưa candidate sang VLM rerank hoặc LLM.
    \"\"\"
    target_departments = infer_target_departments(text_query)
    reranked: list[Document] = []

    for doc in docs:
        doc = normalize_product_metadata(doc)
        meta = doc.metadata
        base_score = (
            meta.get("composed_score")
            if meta.get("composed_score") is not None
            else meta.get("image_search_score")
            if meta.get("image_search_score") is not None
            else meta.get("text_score")
            if meta.get("text_score") is not None
            else 0.0
        )
        try:
            score = float(base_score)
        except (TypeError, ValueError):
            score = 0.0

        reasons = [f"base={score:.3f}"]
        category = str(meta.get("category", "")).strip()
        department = str(meta.get("department", "") or meta.get("gender", "")).strip()

        if locked_category:
            if category == locked_category:
                bonus = 0.20 if category_mode == "hard_lock" else 0.12
                score += bonus
                reasons.append(f"category={locked_category} +{bonus:.2f}")
            elif category_mode == "hard_lock":
                score -= 0.50
                reasons.append(f"khác category {locked_category} -0.50")

        if target_departments:
            if department in target_departments:
                score += 0.10
                reasons.append(f"department={department} +0.10")
            elif not department and any(word in normalize_debug_text(doc.page_content) for word in [" nu ", " nu,", " nu.", "female", "women"]):
                score += 0.05
                reasons.append("text có tín hiệu nữ +0.05")

        if intent == "outfit" and locked_category and category != locked_category:
            score += 0.05
            reasons.append("outfit cần mở category +0.05")

        meta["metadata_rerank_score"] = score
        meta["metadata_rerank_reason"] = "; ".join(reasons)
        reranked.append(doc)

    reranked.sort(key=lambda doc: float(doc.metadata.get("metadata_rerank_score", 0.0)), reverse=True)
    return reranked[:final_k]


def l2_normalize_vector(vector: list[float] | np.ndarray) -> list[float]:
    \"\"\"Normalize vector về độ dài 1 để cosine search ổn định.\"\"\"
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Vector rỗng hoặc norm bằng 0, không thể normalize.")
    return (arr / norm).astype(float).tolist()


def embed_image_query(image_path: str | Path) -> list[float]:
    \"\"\"Encode ảnh query thành vector FashionCLIP 512-dim đã normalize.\"\"\"
    vector = get_image_embeddings().embed_image(image_path)
    if vector is None:
        raise RuntimeError(f"Không encode được ảnh: {image_path}")
    return l2_normalize_vector(vector)


def embed_text_query_for_image_space(text_query: str) -> list[float]:
    \"\"\"Encode text query tiếng Việt thành vector 512-dim cùng không gian với image collection.\"\"\"
    if not text_query or not text_query.strip():
        raise ValueError("text_query đang rỗng.")
    vector = get_text_embeddings().embed_query(text_query)
    return l2_normalize_vector(vector)


def compose_image_text_vector(
    image_vector: list[float],
    text_vector: list[float],
    image_weight: float = 0.6,
    text_weight: float = 0.4,
) -> list[float]:
    \"\"\"Ghép ảnh + text ở vector level: normalize(w_img * image + w_text * text).\"\"\"
    if image_weight < 0 or text_weight < 0:
        raise ValueError("image_weight và text_weight phải >= 0.")
    if image_weight == 0 and text_weight == 0:
        raise ValueError("Ít nhất một weight phải > 0.")

    image_arr = np.asarray(image_vector, dtype=np.float32)
    text_arr = np.asarray(text_vector, dtype=np.float32)
    if image_arr.shape != text_arr.shape:
        raise ValueError(f"Vector ảnh và text khác chiều: {image_arr.shape} vs {text_arr.shape}")

    composed = image_weight * image_arr + text_weight * text_arr
    return l2_normalize_vector(composed)


def search_products_by_composed_image_text(
    image_path: str | Path,
    text_query: str,
    image_weight: float = 0.6,
    text_weight: float = 0.4,
    top_k: int = IMAGE_SEARCH_TOP_K,
    max_products: int = IMAGE_SEARCH_MAX_PRODUCTS,
    score_threshold: float | None = None,
    query_filter: Filter | None = None,
) -> list[Document]:
    \"\"\"True zero-shot composed retrieval: ảnh + text -> một vector chung -> image collection.\"\"\"
    image_vector = embed_image_query(image_path)
    text_vector = embed_text_query_for_image_space(text_query)
    composed_vector = compose_image_text_vector(
        image_vector=image_vector,
        text_vector=text_vector,
        image_weight=image_weight,
        text_weight=text_weight,
    )
    docs = query_image_collection_by_vector(
        query_vector=composed_vector,
        collection_name=PRODUCT_IMAGE_COLLECTION,
        top_k=top_k,
        max_products=max_products,
        score_threshold=score_threshold,
        score_key="composed_score",
        query_filter=query_filter,
    )
    for doc in docs:
        doc.metadata["image_weight"] = image_weight
        doc.metadata["text_weight"] = text_weight
    return docs
"""
        ),
        markdown_cell(
            """
## BƯỚC 9: Debug Composed Retrieval Và Calibrate Threshold

Hàm chính ở bước này:

- `debug_image_query`: image-only, có hiển thị ảnh.
- `debug_image_and_text`: chạy image-only, text-only, composed image+text rồi so sánh.
- `calibrate_image_threshold`: self-search ảnh trong dataset để xem ngưỡng `0.15` có hợp lý không.
"""
        ),
        code_cell(
            """
def image_doc_row(doc: Document, rank: int) -> dict:
    \"\"\"Rút gọn Document thành dict dễ đọc khi debug retrieval.\"\"\"
    doc = normalize_product_metadata(doc)
    return {
        "rank": rank,
        "product_id": doc.metadata.get("product_id", ""),
        "title": doc.metadata.get("title", ""),
        "category": doc.metadata.get("category", ""),
        "brand": doc.metadata.get("brand", ""),
        "image_score": doc.metadata.get("image_search_score"),
        "text_score": doc.metadata.get("text_score"),
        "composed_score": doc.metadata.get("composed_score"),
        "metadata_rerank_score": doc.metadata.get("metadata_rerank_score"),
        "metadata_rerank_reason": doc.metadata.get("metadata_rerank_reason", ""),
        "vlm_score": doc.metadata.get("vlm_score"),
        "vlm_reason": doc.metadata.get("vlm_reason", ""),
        "image_url": doc.metadata.get("image_url", ""),
        "preview": doc.page_content[:220].replace("\\n", " | "),
    }


def resolve_doc_image_path(doc: Document) -> Path | None:
    \"\"\"Tìm file ảnh local tốt nhất từ metadata của Document.\"\"\"
    doc = normalize_product_metadata(doc)

    main_image_path = doc.metadata.get("main_image_path")
    if main_image_path and Path(main_image_path).exists():
        return Path(main_image_path)

    image_url = doc.metadata.get("image_url", "")
    if image_url and not str(image_url).startswith(("http://", "https://")):
        candidate = resolve_main_image_path(str(image_url), PRODUCT_IMAGE_ROOT)
        if candidate.exists():
            return candidate

    images = doc.metadata.get("images", []) or []
    for image in images:
        if image and not str(image).startswith(("http://", "https://")):
            candidate = resolve_main_image_path(str(image), PRODUCT_IMAGE_ROOT)
            if candidate.exists():
                return candidate
    return None


def show_image_file(image_path: str | Path, title: str = "Ảnh query") -> None:
    \"\"\"Hiển thị một ảnh local ngay trong notebook.\"\"\"
    from IPython.display import display
    from PIL import Image as PILImage

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[WARN] Không thấy ảnh để hiển thị: {image_path}")
        return
    print(title)
    display(PILImage.open(image_path).convert("RGB"))


def show_retrieval_gallery(title: str, docs: list[Document], score_key: str = "image_search_score") -> None:
    \"\"\"Hiển thị ảnh top results thành gallery nhỏ để đánh giá retrieval bằng mắt.\"\"\"
    if not docs:
        print(f"\\n=== {title}: không có kết quả ===")
        return

    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    items = []
    for rank, doc in enumerate(docs, start=1):
        img_path = resolve_doc_image_path(doc)
        if img_path is None:
            continue
        items.append((rank, doc, img_path))

    if not items:
        print(f"[WARN] Có {len(docs)} kết quả nhưng không resolve được file ảnh local để hiển thị.")
        return

    cols = min(3, len(items))
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = list(axes.ravel())

    for ax in axes:
        ax.axis("off")

    for ax, (rank, doc, img_path) in zip(axes, items):
        meta = doc.metadata
        score = meta.get(score_key)
        score_text = "" if score is None else f" | {float(score):.3f}"
        title_text = f"#{rank}{score_text}\\n{meta.get('category', '')} | {meta.get('brand', '')}\\n{meta.get('product_id', '')}"
        ax.imshow(PILImage.open(img_path).convert("RGB"))
        ax.set_title(title_text, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def print_docs_table(title: str, docs: list[Document], score_key: str) -> None:
    print(f"\\n=== {title} ({len(docs)}) ===")
    for rank, doc in enumerate(docs, start=1):
        row = image_doc_row(doc, rank)
        score = doc.metadata.get(score_key)
        score_text = "" if score is None else f" score={float(score):.4f}"
        print(f"#{rank}{score_text} | {row['product_id']} | {row['category']} | {row['brand']}")
        print(f"  {row['title']}")
        if row["metadata_rerank_reason"]:
            print(f"  metadata_rerank: {row['metadata_rerank_reason']}")
        if row["vlm_reason"]:
            print(f"  vlm: score={row['vlm_score']} | {row['vlm_reason']}")
        print(f"  image_url: {row['image_url']}")
        print(f"  {row['preview']}")


def debug_image_query(
    image_path: str | Path,
    top_k: int = IMAGE_SEARCH_TOP_K,
    max_products: int = IMAGE_SEARCH_MAX_PRODUCTS,
    score_threshold: float | None = IMAGE_SEARCH_SCORE_THRESHOLD,
    show_images: bool = True,
) -> list[Document]:
    print(f"IMAGE: {Path(image_path)}")
    if show_images:
        show_image_file(image_path, title="Ảnh query")

    docs = search_products_by_image(
        image_path=image_path,
        top_k=top_k,
        max_products=max_products,
        score_threshold=score_threshold,
    )
    print_docs_table("Image search results", docs, score_key="image_search_score")
    if show_images:
        show_retrieval_gallery("Top image retrieval results", docs, score_key="image_search_score")
    return docs


def compare_retrieval_sets(image_docs: list[Document], text_docs: list[Document]) -> dict:
    \"\"\"So sánh nhanh overlap giữa kết quả image retrieval và text retrieval.\"\"\"
    image_ids = {str(doc.metadata.get("product_id", "")) for doc in image_docs if doc.metadata.get("product_id")}
    text_ids = {str(doc.metadata.get("product_id", "")) for doc in text_docs if doc.metadata.get("product_id")}
    image_categories = {str(doc.metadata.get("category", "")) for doc in image_docs if doc.metadata.get("category")}
    text_categories = {str(doc.metadata.get("category", "")) for doc in text_docs if doc.metadata.get("category")}

    overlap_ids = image_ids & text_ids
    overlap_categories = image_categories & text_categories
    summary = {
        "image_count": len(image_docs),
        "text_count": len(text_docs),
        "overlap_product_ids": sorted(overlap_ids),
        "overlap_categories": sorted(overlap_categories),
    }
    print("\\n=== Image vs Text Retrieval Summary ===")
    print(f"Image docs        : {summary['image_count']}")
    print(f"Text docs         : {summary['text_count']}")
    print(f"Trùng product_id  : {summary['overlap_product_ids'] or 'Không có'}")
    print(f"Trùng category    : {summary['overlap_categories'] or 'Không có'}")
    return summary


def extract_json_object(text: str) -> dict:
    \"\"\"Parse JSON từ VLM output, chịu được trường hợp model lỡ thêm vài chữ bên ngoài.\"\"\"
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError(f"Không parse được JSON từ VLM output: {text[:200]}")


def clamp_score(value, default: float = 0.0) -> float:
    \"\"\"Đưa score về khoảng 0..1 để VLM rerank không làm vỡ pipeline.\"\"\"
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def get_vlm_ollama_client():
    \"\"\"Tạo Ollama client cho VLM. Đổi `VLM_OLLAMA_BASE_URL` để trỏ sang Vast.ai nếu cần.\"\"\"
    import ollama

    return ollama.Client(host=VLM_OLLAMA_BASE_URL)


def vlm_score_candidate(
    query_image_path: str | Path,
    text_query: str,
    candidate_doc: Document,
    intent: str = "search",
    locked_category: str | None = None,
) -> dict:
    \"\"\"Dùng VLM chấm một candidate bằng ảnh query + ảnh candidate + câu user.

    Output cố ý ngắn: `score` và `reason`. Đây là lý do debug, không phải chain-of-thought dài.
    \"\"\"
    candidate_image_path = resolve_doc_image_path(candidate_doc)
    if candidate_image_path is None:
        return {"score": 0.0, "reason": "Không resolve được ảnh candidate local."}

    from app.core.vision import _preprocess_image

    query_b64 = _preprocess_image(str(query_image_path))
    candidate_b64 = _preprocess_image(str(candidate_image_path))
    meta = candidate_doc.metadata
    prompt = f\"\"\"
Bạn là reranker sản phẩm thời trang.
Ảnh 1 là ảnh query của người dùng. Ảnh 2 là sản phẩm candidate.

Yêu cầu người dùng: {text_query}
Intent đã suy ra: {intent}
Category khóa/boost từ ảnh query: {locked_category or "không có"}

Candidate metadata:
- title: {meta.get("title", "")}
- category: {meta.get("category", "")}
- brand: {meta.get("brand", "")}
- department: {meta.get("department", "") or meta.get("gender", "")}

Hãy chấm mức phù hợp của candidate với ảnh query và yêu cầu người dùng.
Nếu intent là search/similar item, ưu tiên giữ đúng loại sản phẩm từ ảnh query.
Nếu intent là outfit/phối đồ, candidate có thể khác loại sản phẩm nếu hợp để phối.
Trả về JSON duy nhất theo format:
{{"score": 0.0, "reason": "lý do ngắn dưới 25 từ"}}
\"\"\".strip()

    client = get_vlm_ollama_client()
    response = client.chat(
        model=QWEN_VL_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [query_b64, candidate_b64]}],
        options={"temperature": 0},
    )
    content = response.get("message", {}).get("content", "")
    parsed = extract_json_object(content)
    return {
        "score": clamp_score(parsed.get("score")),
        "reason": str(parsed.get("reason", "")).strip()[:240],
    }


def vlm_rerank_documents(
    query_image_path: str | Path,
    text_query: str,
    docs: list[Document],
    intent: str = "search",
    locked_category: str | None = None,
    top_k: int = VLM_RERANK_TOP_K,
) -> list[Document]:
    \"\"\"VLM rerank top candidates. Nếu VLM lỗi, giữ nguyên thứ tự metadata rerank để notebook vẫn chạy.\"\"\"
    reranked: list[Document] = []
    for rank, doc in enumerate(docs[:top_k], start=1):
        try:
            result = vlm_score_candidate(
                query_image_path=query_image_path,
                text_query=text_query,
                candidate_doc=doc,
                intent=intent,
                locked_category=locked_category,
            )
        except Exception as exc:
            result = {"score": 0.0, "reason": f"VLM lỗi ở rank {rank}: {type(exc).__name__}: {exc}"}
        doc.metadata["vlm_score"] = result["score"]
        doc.metadata["vlm_reason"] = result["reason"]
        reranked.append(doc)

    reranked.sort(key=lambda doc: float(doc.metadata.get("vlm_score", 0.0)), reverse=True)
    return reranked + docs[top_k:]


def build_clarification_diagnostic(
    text_query: str,
    intent: str,
    category_mode: str,
    category_info: dict,
    composed_docs: list[Document],
    final_docs: list[Document],
) -> dict:
    \"\"\"Search trước, rồi quyết định có nên hỏi lại user không.\"\"\"
    questions: list[str] = []
    composed_gap = score_gap(composed_docs, "composed_score")
    final_gap = score_gap(final_docs, "metadata_rerank_score")

    if not final_docs:
        questions.append("Bạn có thể mô tả rõ hơn sản phẩm muốn tìm không?")
    if intent == "search" and not category_info.get("confident"):
        questions.append("Ảnh này bạn muốn tìm đúng loại sản phẩm nào?")
    if composed_gap is not None and composed_gap < CONFIDENCE_SCORE_GAP_MIN:
        questions.append("Các kết quả đang khá sát nhau. Bạn ưu tiên giống kiểu dáng, màu sắc hay thương hiệu?")
    if infer_target_departments(text_query) and category_mode == "hard_lock":
        questions.append("Bạn muốn giữ đúng loại sản phẩm trong ảnh nhưng đổi sang phong cách/department nào?")

    diagnostic = {
        "needs_clarification": bool(questions),
        "questions": questions,
        "composed_score_gap": composed_gap,
        "final_score_gap": final_gap,
    }

    print("\\n=== Clarification diagnostic ===")
    print(f"needs_clarification: {diagnostic['needs_clarification']}")
    print(f"composed_score_gap : {composed_gap}")
    print(f"final_score_gap    : {final_gap}")
    for question in questions[:3]:
        print(f"- {question}")
    return diagnostic


def debug_image_and_text(
    image_path: str | Path,
    text_query: str,
    image_weight: float = 0.6,
    text_weight: float = 0.4,
    composed_threshold: float | None = None,
    intent: str = "auto",
    category_mode: str = "auto",
    use_vlm_rerank: bool = False,
    vlm_top_k: int = VLM_RERANK_TOP_K,
    show_images: bool = True,
) -> dict:
    \"\"\"Debug 3 nhánh: image-only, text-only và true composed image+text retrieval.\"\"\"
    image_docs = debug_image_query(image_path, show_images=show_images)
    resolved_intent = infer_retrieval_intent(text_query) if intent == "auto" else intent
    category_info = infer_category_from_docs(image_docs)
    locked_category = category_info.get("category") if category_info.get("confident") else None

    if category_mode == "auto":
        resolved_category_mode = "hard_lock" if resolved_intent == "search" else "soft_boost"
    else:
        resolved_category_mode = category_mode

    if resolved_category_mode == "hard_lock" and locked_category is None:
        query_filter = None
        print("\\n[WARN] Muốn hard_lock category nhưng image-only chưa đủ chắc, tạm không filter cứng.")
    elif resolved_category_mode == "hard_lock":
        query_filter = build_category_filter(locked_category)
    else:
        query_filter = None

    filter_text = f"category={locked_category}" if query_filter is not None else "không dùng filter cứng"
    print("\\n=== Multimodal decision ===")
    print(f"intent        : {resolved_intent}")
    print(f"category_mode : {resolved_category_mode}")
    print(f"category_info : {category_info}")
    print(f"qdrant_filter : {filter_text}")

    print(f"\\nTEXT QUERY: {text_query}")
    text_docs = search_products_by_text(text_query)
    print_docs_table("Text search results", text_docs, score_key="text_score")
    if show_images:
        show_retrieval_gallery("Top text retrieval results", text_docs, score_key="text_score")

    print(
        f"\\nCOMPOSED QUERY: image_weight={image_weight} | "
        f"text_weight={text_weight} | threshold={composed_threshold}"
    )
    composed_docs = search_products_by_composed_image_text(
        image_path=image_path,
        text_query=text_query,
        image_weight=image_weight,
        text_weight=text_weight,
        top_k=MULTIMODAL_CANDIDATE_K,
        max_products=MULTIMODAL_CANDIDATE_K,
        score_threshold=composed_threshold,
        query_filter=query_filter,
    )
    print_docs_table("Composed image+text results", composed_docs, score_key="composed_score")
    if show_images:
        show_retrieval_gallery("Top composed image+text results", composed_docs, score_key="composed_score")

    metadata_docs = metadata_rerank_documents(
        docs=composed_docs,
        text_query=text_query,
        locked_category=locked_category,
        category_mode=resolved_category_mode,
        intent=resolved_intent,
        final_k=MULTIMODAL_FINAL_K,
    )
    print_docs_table("Metadata-aware rerank results", metadata_docs, score_key="metadata_rerank_score")
    if show_images:
        show_retrieval_gallery("Top metadata-aware rerank results", metadata_docs, score_key="metadata_rerank_score")

    final_docs = metadata_docs
    if use_vlm_rerank:
        print(f"\\nVLM RERANK: model={QWEN_VL_MODEL} | top_k={vlm_top_k} | base_url={VLM_OLLAMA_BASE_URL}")
        final_docs = vlm_rerank_documents(
            query_image_path=image_path,
            text_query=text_query,
            docs=metadata_docs,
            intent=resolved_intent,
            locked_category=locked_category,
            top_k=vlm_top_k,
        )
        print_docs_table("VLM rerank results", final_docs[:MULTIMODAL_FINAL_K], score_key="vlm_score")
        if show_images:
            show_retrieval_gallery("Top VLM rerank results", final_docs[:MULTIMODAL_FINAL_K], score_key="vlm_score")

    image_text_summary = compare_retrieval_sets(image_docs, text_docs)
    image_composed_summary = compare_retrieval_sets(image_docs, composed_docs)
    text_composed_summary = compare_retrieval_sets(text_docs, composed_docs)
    clarification = build_clarification_diagnostic(
        text_query=text_query,
        intent=resolved_intent,
        category_mode=resolved_category_mode,
        category_info=category_info,
        composed_docs=composed_docs,
        final_docs=final_docs,
    )
    summary = {
        "intent": resolved_intent,
        "category_mode": resolved_category_mode,
        "category_info": category_info,
        "clarification": clarification,
        "image_vs_text": image_text_summary,
        "image_vs_composed": image_composed_summary,
        "text_vs_composed": text_composed_summary,
    }
    return {
        "image_docs": image_docs,
        "text_docs": text_docs,
        "composed_docs": composed_docs,
        "metadata_docs": metadata_docs,
        "final_docs": final_docs,
        "summary": summary,
    }


def debug_multimodal_retrieval(*args, **kwargs) -> dict:
    \"\"\"Alias dễ nhớ hơn cho `debug_image_and_text`.\"\"\"
    return debug_image_and_text(*args, **kwargs)


def calibrate_image_threshold(
    sample_size: int = 20,
    top_k: int = 12,
    seed: int = 42,
) -> list[dict]:
    \"\"\"Self-search ảnh trong dataset để xem positive match thường có score bao nhiêu.\"\"\"
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    for item, rel_path, img_path in iter_products_with_main_image():
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append((item, rel_path, img_path))
        else:
            j = rng.randrange(seen)
            if j < sample_size:
                reservoir[j] = (item, rel_path, img_path)

    rows = []
    for idx, (item, rel_path, img_path) in enumerate(reservoir, start=1):
        product_id = str(item.get("product_id", ""))
        docs = search_products_by_image(
            image_path=img_path,
            top_k=top_k,
            max_products=top_k,
            score_threshold=None,
        )
        hit_rank = None
        hit_score = None
        top_score = None
        if docs:
            top_score = docs[0].metadata.get("image_search_score")
        for rank, doc in enumerate(docs, start=1):
            if str(doc.metadata.get("product_id", "")) == product_id:
                hit_rank = rank
                hit_score = doc.metadata.get("image_search_score")
                break
        row = {
            "query_product_id": product_id,
            "category": item.get("category", ""),
            "hit_rank": hit_rank,
            "hit_score": hit_score,
            "top_score": top_score,
            "image_path": str(img_path),
        }
        rows.append(row)
        print(
            f"[{idx}/{len(reservoir)}] {product_id} | "
            f"hit_rank={hit_rank} | hit_score={hit_score} | top_score={top_score}"
        )

    hit_scores = [float(row["hit_score"]) for row in rows if row["hit_score"] is not None]
    recall_at_k = sum(row["hit_rank"] is not None for row in rows) / max(1, len(rows))
    print("\\n=== Threshold calibration summary ===")
    print(f"sample_size : {len(rows)}")
    print(f"recall@{top_k}   : {recall_at_k:.2%}")
    if hit_scores:
        print(f"min positive score : {min(hit_scores):.4f}")
        print(f"median positive    : {float(np.median(hit_scores)):.4f}")
        print(f"p10 positive       : {float(np.percentile(hit_scores, 10)):.4f}")
        print(f"p25 positive       : {float(np.percentile(hit_scores, 25)):.4f}")
        print(f"current threshold  : {IMAGE_SEARCH_SCORE_THRESHOLD}")
    return rows


# Ví dụ 1: random một ảnh trong dataset rồi image search
# sample = preview_random_main_image()
# image_docs = debug_image_query(sample["img_path"])

# Ví dụ 2: nhập ảnh bất kỳ trên máy
# image_docs = debug_image_query(r"D:\\path\\to\\your_image.jpg")

# Ví dụ 3: true composed image+text retrieval
# result = debug_multimodal_retrieval(
#     image_path=r"D:\\path\\to\\your_image.jpg",
#     text_query="giống ảnh này nhưng màu trắng, mặc đi làm",
#     image_weight=0.6,
#     text_weight=0.4,
#     intent="search",
#     category_mode="auto",
#     use_vlm_rerank=True,
#     vlm_top_k=3,
# )

# Ví dụ 4: calibrate threshold 0.15 bằng self-search, chạy ít mẫu trước vì sẽ load model ảnh
# threshold_rows = calibrate_image_threshold(sample_size=20, top_k=12)
"""
        ),
    ]

    cells = split_notebook_04_step_9(cells)

    nb["cells"] = cells
    nb["metadata"] = metadata
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5
    NB04.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    standardize_headings()
    refine_notebook_04()
    print(f"[OK] Standardized: {NB01}")
    print(f"[OK] Standardized: {NB02}")
    print(f"[OK] Refined: {NB04}")


if __name__ == "__main__":
    main()
