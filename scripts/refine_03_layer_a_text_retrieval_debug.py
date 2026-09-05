"""Refine notebook 03 into a focused Layer A retrieval debugger."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "research_demo_v3_split" / "03_layer_a_text_retrieval_debug.ipynb"


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


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = nb.get("metadata", {})

    cells = [
        markdown_cell(
            """
# 03 - Layer A Text Retrieval Debug

Notebook này dùng để soi riêng nhánh **tìm sản phẩm bằng text**.

Góc nhìn cần giữ trong đầu:

```text
query tiếng Việt
  -> ViFashionCLIPTextEmbeddings
  -> Qdrant lấy raw candidates
  -> optional reranker chấm lại
  -> diversity filter giảm trùng sản phẩm/brand
  -> final docs gửi cho LLM
```

Notebook này **không debug prompt, không debug chat loop, không debug Layer B phối đồ**. Nếu final docs đã sai, đừng sửa prompt vội; hãy sửa retrieval trước.

Bạn có thể mở notebook này độc lập. Nó import class/config trực tiếp từ thư mục `app/`, không cần chạy notebook 01 trước.

Notebook 01 vẫn nên đọc trước nếu bạn muốn hiểu mô hình embedding được khai báo như thế nào.
"""
        ),
        markdown_cell(
            """
## BƯỚC 1: Setup Tối Thiểu

Cell này tìm thư mục gốc `Chatbot_Fashion/`, thêm nó vào `sys.path`, rồi import thẳng từ code app.

Cách này tốt hơn việc phụ thuộc vào “đã chạy notebook 01 chưa”, vì mỗi notebook có thể chạy độc lập và ít lỗi thiếu biến hơn.
Import model embedding được để sang bước 3 để cell setup không bị chậm.
"""
        ),
        code_cell(
            """
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


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
    PRODUCT_IMAGE_ROOT as APP_PRODUCT_IMAGE_ROOT,
    PRODUCT_SEARCH_BRAND_LIMIT as APP_PRODUCT_SEARCH_BRAND_LIMIT,
    PRODUCT_SEARCH_CANDIDATE_K as APP_PRODUCT_SEARCH_CANDIDATE_K,
    PRODUCT_SEARCH_PAGE_SIZE as APP_PRODUCT_SEARCH_PAGE_SIZE,
    QDRANT_COLLECTION_FASHION,
    QDRANT_URL as APP_QDRANT_URL,
    RERANKER_BATCH_SIZE as APP_RERANKER_BATCH_SIZE,
    RERANKER_MODEL_NAME as APP_RERANKER_MODEL_NAME,
    RERANKER_TOP_N as APP_RERANKER_TOP_N,
)

print("[OK] Setup notebook 03 hoàn tất")
print(f"Project root: {CHATBOT_FASHION_DIR}")
"""
        ),
        markdown_cell(
            """
## BƯỚC 2: Cấu Hình Retrieval

Các số dưới đây ảnh hưởng trực tiếp đến chất lượng và tốc độ:

- `APP_DENSE_CANDIDATE_K`: số ứng viên dense mà app production đang cấu hình.
- `PRODUCT_SEARCH_CANDIDATE_K`: số ứng viên dense trong notebook debug. Ở đây cố ý để 30 cho cùng thang với BM25/Hybrid, giúp nhìn rộng hơn khi phân tích lỗi.
- `RERANKER_TOP_N`: rerank bao nhiêu ứng viên đầu.
- `PRODUCT_SEARCH_PAGE_SIZE`: cuối cùng giữ bao nhiêu sản phẩm.
- `ENABLE_PRODUCT_RERANKER`: nên để `False` khi mới debug raw retrieval; bật `True` sau khi raw candidates đã có sản phẩm đúng.

Vì vậy nếu app đang để dense raw 15 mà notebook debug để 30 thì không mâu thuẫn. Debug thường cần lấy rộng hơn production để xem “sản phẩm đúng có nằm đâu đó trong candidate pool không?”.
"""
        ),
        code_cell(
            """
QDRANT_URL = APP_QDRANT_URL
PRODUCT_COLLECTION = QDRANT_COLLECTION_FASHION
PRODUCT_IMAGE_ROOT = Path(APP_PRODUCT_IMAGE_ROOT)

APP_DENSE_CANDIDATE_K = APP_PRODUCT_SEARCH_CANDIDATE_K
PRODUCT_SEARCH_CANDIDATE_K = 30
PRODUCT_SEARCH_PAGE_SIZE = APP_PRODUCT_SEARCH_PAGE_SIZE
PRODUCT_SEARCH_BRAND_LIMIT = APP_PRODUCT_SEARCH_BRAND_LIMIT
BM25_CANDIDATE_K = 30
HYBRID_CANDIDATE_K = 30

# Để debug nhanh, mặc định tắt reranker.
# Khi muốn so sánh chất lượng sau dense retrieval, đổi thành True.
ENABLE_PRODUCT_RERANKER = False
RERANKER_MODEL_NAME = APP_RERANKER_MODEL_NAME
RERANKER_TOP_N = APP_RERANKER_TOP_N
RERANKER_BATCH_SIZE = APP_RERANKER_BATCH_SIZE

print("[OK] Retrieval config")
print(f"Collection : {PRODUCT_COLLECTION}")
print(f"Image root : {PRODUCT_IMAGE_ROOT}")
print(f"App dense k: {APP_DENSE_CANDIDATE_K}")
flow = f"top-{PRODUCT_SEARCH_CANDIDATE_K} raw"
if ENABLE_PRODUCT_RERANKER:
    flow += f" -> rerank top-{RERANKER_TOP_N}"
flow += f" -> diversity -> final {PRODUCT_SEARCH_PAGE_SIZE}"
print(f"Flow       : {flow}")
print(f"Reranker   : {ENABLE_PRODUCT_RERANKER}")
print(f"BM25 debug : top-{BM25_CANDIDATE_K} lexical candidates")
print(f"Hybrid RRF : top-{HYBRID_CANDIDATE_K} fused candidates")
"""
        ),
        markdown_cell(
            """
## BƯỚC 3: Kết Nối Qdrant Và Embedding

Cell này là cell nặng nhất của notebook 03 vì nó khởi tạo `ViFashionCLIPTextEmbeddings`.

Nếu chạy chậm hoặc lỗi:

- Qdrant có thể chưa bật ở `localhost:6333`.
- ViFashionCLIP checkpoint/model có thể load chậm.
- Nếu `PRODUCT_EMBEDDING_BACKEND=remote`, notebook sẽ gọi embedding service giống app.
"""
        ),
        code_cell(
            """
print("[INFO] Khởi tạo ViFashionCLIP text embedding...")
from app.core.embeddings import get_product_embeddings

product_embeddings = get_product_embeddings()

print("[INFO] Kết nối Qdrant...")
client = QdrantClient(url=QDRANT_URL, timeout=20, check_compatibility=False)

if not client.collection_exists(PRODUCT_COLLECTION):
    raise RuntimeError(f"Không thấy collection Qdrant: {PRODUCT_COLLECTION}")

count = client.count(PRODUCT_COLLECTION).count
print(f"[OK] Collection `{PRODUCT_COLLECTION}` có {count} points")

vector_db = QdrantVectorStore(
    client=client,
    collection_name=PRODUCT_COLLECTION,
    embedding=product_embeddings,
)
print("[OK] vector_db sẵn sàng")
"""
        ),
        markdown_cell(
            """
## BƯỚC 4: Chuẩn Hóa Metadata Và Lọc Đa Dạng

Hai hàm này xử lý phần sau retrieval:

- `normalize_product_metadata`: đảm bảo document có `image_url`.
- `diversity_filter_documents`: giảm lặp cùng `product_id` hoặc cùng brand quá nhiều.

Nếu bạn thấy chatbot gợi ý nhiều món giống nhau, hãy nhìn hàm diversity này.
"""
        ),
        code_cell(
            """
def normalize_product_metadata(doc: Document) -> Document:
    \"\"\"Đảm bảo mỗi Document có `images` dạng list và `image_url` ổn định.\"\"\"
    images = doc.metadata.get("images", [])
    if isinstance(images, str):
        images = [images] if images else []
    doc.metadata["images"] = images
    doc.metadata["image_url"] = doc.metadata.get("image_url") or (images[0] if images else "")
    return doc


def diversity_filter_documents(
    docs: list[Document],
    max_docs: int = PRODUCT_SEARCH_PAGE_SIZE,
    max_per_brand: int = PRODUCT_SEARCH_BRAND_LIMIT,
) -> list[Document]:
    \"\"\"Lọc trùng product_id và giới hạn số sản phẩm cùng brand.\"\"\"
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
"""
        ),
        markdown_cell(
            """
## BƯỚC 5: Dense Retrieval - Xem Raw Candidates Từ Qdrant

Đây là bước quan trọng nhất.

Nếu top-30 raw candidates đã không có sản phẩm đúng, vấn đề thường nằm ở:

- query đưa vào chưa đúng ý;
- `page_content` sản phẩm thiếu tín hiệu;
- embedding chưa tốt;
- collection Qdrant không đúng dữ liệu.
"""
        ),
        code_cell(
            """
def doc_row(doc: Document, rank: int) -> dict:
    \"\"\"Rút gọn một Document thành dict dễ đọc khi debug.\"\"\"
    doc = normalize_product_metadata(doc)
    return {
        "rank": rank,
        "product_id": doc.metadata.get("product_id", ""),
        "title": doc.metadata.get("title", ""),
        "category": doc.metadata.get("category", ""),
        "brand": doc.metadata.get("brand", ""),
        "price": doc.metadata.get("price", ""),
        "dense_score": doc.metadata.get("dense_score"),
        "bm25_score": doc.metadata.get("bm25_score"),
        "hybrid_rrf_score": doc.metadata.get("hybrid_rrf_score"),
        "rerank_score": doc.metadata.get("rerank_score"),
        "image_url": doc.metadata.get("image_url", ""),
        "preview": doc.page_content[:220].replace("\\n", " | "),
    }


def print_doc_rows(title: str, docs: list[Document], limit: int | None = None) -> None:
    print(f"\\n=== {title} ({len(docs)}) ===")
    rows = docs if limit is None else docs[:limit]
    for rank, doc in enumerate(rows, start=1):
        row = doc_row(doc, rank)
        score = row["dense_score"]
        bm25 = row["bm25_score"]
        hybrid = row["hybrid_rrf_score"]
        rerank = row["rerank_score"]
        score_text = "" if score is None else f" dense={float(score):.4f}"
        bm25_text = "" if bm25 is None else f" bm25={float(bm25):.4f}"
        hybrid_text = "" if hybrid is None else f" rrf={float(hybrid):.4f}"
        rerank_text = "" if rerank is None else f" rerank={float(rerank):.4f}"
        print(f"#{rank}{score_text}{bm25_text}{hybrid_text}{rerank_text} | {row['product_id']} | {row['category']} | {row['brand']}")
        print(f"  {row['title']}")
        if row["image_url"]:
            print(f"  image_url: {row['image_url']}")
        print(f"  {row['preview']}")


def image_value_to_path_text(value) -> str:
    \"\"\"Lấy path ảnh từ metadata dạng string hoặc dict.\"\"\"
    if isinstance(value, dict):
        for key in ["large", "image_url", "url", "main_image", "path"]:
            if value.get(key):
                return str(value[key])
        return ""
    return str(value or "")


def resolve_product_image_path(doc: Document) -> Path | None:
    \"\"\"Resolve ảnh local từ thông tin sản phẩm, không cần query image collection.\"\"\"
    doc = normalize_product_metadata(doc)

    main_image_path = doc.metadata.get("main_image_path")
    if main_image_path and Path(main_image_path).exists():
        return Path(main_image_path)

    candidates = []
    if doc.metadata.get("image_url"):
        candidates.append(doc.metadata.get("image_url"))
    candidates.extend(doc.metadata.get("images", []) or [])

    for value in candidates:
        path_text = image_value_to_path_text(value)
        if not path_text or path_text.startswith(("http://", "https://")):
            continue
        rel = Path(path_text.replace("\\\\", "/"))
        if rel.parts and rel.parts[0].lower() == "images":
            rel = Path(*rel.parts[1:])
        candidate = PRODUCT_IMAGE_ROOT / rel
        if candidate.exists():
            return candidate

    product_id = str(doc.metadata.get("product_id", "")).strip()
    if product_id:
        for suffix in ["_MAIN.jpg", "_MAIN.jpeg", "_MAIN.png", ".jpg", ".jpeg", ".png"]:
            candidate = PRODUCT_IMAGE_ROOT / f"{product_id}{suffix}"
            if candidate.exists():
                return candidate
    return None


def show_final_doc_images(docs: list[Document], title: str = "Final docs images") -> None:
    \"\"\"Hiển thị ảnh của final docs để kiểm tra bằng mắt trước khi gửi vào LLM.\"\"\"
    if not docs:
        return

    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    items = []
    for rank, doc in enumerate(docs, start=1):
        img_path = resolve_product_image_path(doc)
        if img_path is not None:
            items.append((rank, doc, img_path))

    if not items:
        print("[WARN] Final docs có sản phẩm nhưng không resolve được ảnh local từ metadata.")
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
        score = (
            meta.get("rerank_score")
            if meta.get("rerank_score") is not None
            else meta.get("hybrid_rrf_score")
            if meta.get("hybrid_rrf_score") is not None
            else meta.get("dense_score")
        )
        score_text = "" if score is None else f" | {float(score):.3f}"
        ax.imshow(PILImage.open(img_path).convert("RGB"))
        ax.set_title(
            f"#{rank}{score_text}\\n{meta.get('category', '')} | {meta.get('brand', '')}\\n{meta.get('product_id', '')}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def retrieve_raw_candidates(query: str, k: int = PRODUCT_SEARCH_CANDIDATE_K) -> list[Document]:
    \"\"\"Gọi Qdrant similarity search và gắn dense_score/dense_rank vào metadata.\"\"\"
    pairs = vector_db.similarity_search_with_score(query=query, k=k)
    docs: list[Document] = []
    for rank, (doc, score) in enumerate(pairs, start=1):
        doc = normalize_product_metadata(doc)
        doc.metadata["dense_rank"] = rank
        doc.metadata["dense_score"] = float(score)
        docs.append(doc)
    return docs
"""
        ),
        markdown_cell(
            """
## BƯỚC 6: Lexical/BM25 Baseline Và Hybrid Debug

Dense retrieval giỏi bắt ý nghĩa gần đúng, nhưng yếu với keyword chính xác như brand, mã sản phẩm, size, giá.

BM25 thì ngược lại: nó không “hiểu nghĩa” sâu, nhưng bám chữ rất tốt. Vì vậy notebook này dùng BM25 để trả lời câu hỏi:

```text
Nếu dense sai, keyword search có tìm đúng hơn không?
Nếu BM25 đúng còn dense sai, vấn đề nằm ở embedding/query rewrite.
Nếu cả hai đều sai, vấn đề có thể nằm ở dữ liệu hoặc query quá mơ hồ.
```

BM25 trong notebook này hoạt động như sau:

1. Tách query và `page_content` của từng sản phẩm thành token.
2. Với mỗi token trong query, xem token đó xuất hiện bao nhiêu lần trong từng sản phẩm (`tf`).
3. Token càng hiếm trong toàn bộ collection thì càng có trọng lượng cao (`idf`).
4. Document quá dài sẽ bị chuẩn hóa bằng `doc_len / avgdl`, để sản phẩm dài không thắng chỉ vì chứa nhiều chữ.
5. Tổng điểm của các token query tạo thành `bm25_score`.

Nói ngắn gọn: BM25 không hiểu hình ảnh hay ngữ nghĩa sâu. Nó chỉ trả lời câu: “sản phẩm nào chứa nhiều từ quan trọng giống query nhất?”.

Hybrid ở đây dùng RRF - Reciprocal Rank Fusion: lấy rank từ dense và BM25 rồi gộp lại, không cần scale score tuyệt đối.
"""
        ),
        code_cell(
            """
bm25_index = None


VIETNAMESE_STOPWORDS_WITH_ACCENTS = {
    "cái", "mẫu", "này", "đó", "đẹp", "quá", "có", "nào", "tương", "tự",
    "cho", "mình", "tôi", "bạn", "và", "là", "nhưng", "hơn", "với", "của",
    "một", "các", "những", "được", "kiểu", "dạng", "sản", "phẩm",
}

# Metadata sản phẩm có thể chứa title tiếng Anh. Nhóm này không phải stopword tiếng Việt,
# chỉ là các từ nhiễu rất phổ biến nếu bạn muốn BM25 tập trung vào brand/category/thuộc tính.
EXTRA_METADATA_NOISE_WORDS = {"the", "and", "for", "with", "of", "to", "a", "an"}


def strip_vietnamese_accents(text: str) -> str:
    \"\"\"Bỏ dấu để BM25 bắt được cả query có dấu và không dấu.\"\"\"
    return "".join(
        char for char in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d").replace("Đ", "D")


VIETNAMESE_STOPWORDS = VIETNAMESE_STOPWORDS_WITH_ACCENTS | {
    strip_vietnamese_accents(word) for word in VIETNAMESE_STOPWORDS_WITH_ACCENTS
}
BM25_STOPWORDS = VIETNAMESE_STOPWORDS | EXTRA_METADATA_NOISE_WORDS


def tokenize_for_bm25(text: str) -> list[str]:
    \"\"\"Tokenize đơn giản cho debug BM25, ưu tiên dễ hiểu hơn là tối ưu ngôn ngữ học.\"\"\"
    raw_text = str(text).lower()
    ascii_text = strip_vietnamese_accents(raw_text)
    tokens = re.findall(r"[^\W_]+", raw_text, flags=re.UNICODE)
    tokens.extend(re.findall(r"[a-z0-9]+", ascii_text))
    return [token for token in tokens if len(token) >= 2 and token not in BM25_STOPWORDS]


def point_to_document(point) -> Document:
    \"\"\"Đổi một Qdrant point thành Document, robust với payload của LangChain/Qdrant.\"\"\"
    payload = point.payload or {}
    metadata = payload.get("metadata", {}) or {}
    page_content = (
        payload.get("page_content")
        or payload.get("content")
        or metadata.get("page_content")
        or ""
    )
    if not metadata:
        metadata = {key: value for key, value in payload.items() if key not in {"page_content", "content"}}
    return normalize_product_metadata(Document(page_content=str(page_content), metadata=metadata))


def build_bm25_index_from_qdrant(collection_name: str = PRODUCT_COLLECTION, batch_size: int = 512) -> dict:
    \"\"\"Load payload từ Qdrant và build BM25 index trong RAM để debug keyword retrieval.\"\"\"
    docs: list[Document] = []
    token_counts: list[Counter] = []
    doc_lens: list[int] = []
    df: Counter = Counter()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            doc = point_to_document(point)
            tokens = tokenize_for_bm25(doc.page_content)
            counts = Counter(tokens)
            docs.append(doc)
            token_counts.append(counts)
            doc_lens.append(sum(counts.values()))
            df.update(counts.keys())

        if offset is None:
            break

    avgdl = sum(doc_lens) / max(1, len(doc_lens))
    print(f"[OK] BM25 index: docs={len(docs)} | avgdl={avgdl:.1f}")
    return {
        "docs": docs,
        "token_counts": token_counts,
        "doc_lens": doc_lens,
        "df": df,
        "avgdl": avgdl,
        "total_docs": len(docs),
    }


def get_bm25_index() -> dict:
    global bm25_index
    if bm25_index is None:
        print("[INFO] Build BM25 index từ Qdrant payload, chạy lần đầu sẽ hơi lâu...")
        bm25_index = build_bm25_index_from_qdrant()
    return bm25_index


def bm25_search(query: str, k: int = BM25_CANDIDATE_K, k1: float = 1.5, b: float = 0.75) -> list[Document]:
    \"\"\"BM25 lexical search trên page_content để so sánh với dense retrieval.\"\"\"
    index = get_bm25_index()
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []

    total_docs = index["total_docs"]
    avgdl = index["avgdl"]
    scored: list[tuple[float, int]] = []

    for doc_idx, counts in enumerate(index["token_counts"]):
        doc_len = index["doc_lens"][doc_idx]
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            n_q = index["df"].get(token, 0)
            idf = math.log(1 + (total_docs - n_q + 0.5) / (n_q + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            scored.append((score, doc_idx))

    scored.sort(reverse=True, key=lambda item: item[0])
    docs: list[Document] = []
    for rank, (score, doc_idx) in enumerate(scored[:k], start=1):
        doc = index["docs"][doc_idx]
        doc.metadata["bm25_rank"] = rank
        doc.metadata["bm25_score"] = float(score)
        docs.append(doc)
    return docs


def rrf_fuse_documents(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    k: int = HYBRID_CANDIDATE_K,
    rrf_k: int = 60,
) -> list[Document]:
    \"\"\"Gộp dense rank và BM25 rank bằng Reciprocal Rank Fusion.\"\"\"
    by_id: dict[str, Document] = {}
    scores: dict[str, float] = {}

    def doc_key(doc: Document) -> str:
        return str(doc.metadata.get("product_id") or id(doc))

    for rank, doc in enumerate(dense_docs, start=1):
        key = doc_key(doc)
        by_id[key] = doc
        doc.metadata["dense_rank"] = doc.metadata.get("dense_rank", rank)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc_key(doc)
        if key not in by_id:
            by_id[key] = doc
        else:
            by_id[key].metadata["bm25_score"] = doc.metadata.get("bm25_score")
            by_id[key].metadata["bm25_rank"] = rank
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)[:k]
    output: list[Document] = []
    for rank, key in enumerate(ranked_keys, start=1):
        doc = by_id[key]
        doc.metadata["hybrid_rank"] = rank
        doc.metadata["hybrid_rrf_score"] = scores[key]
        output.append(doc)
    return output


def print_bm25_rows(title: str, docs: list[Document], limit: int | None = None) -> None:
    print(f"\\n=== {title} ({len(docs)}) ===")
    rows = docs if limit is None else docs[:limit]
    for rank, doc in enumerate(rows, start=1):
        doc = normalize_product_metadata(doc)
        bm25 = doc.metadata.get("bm25_score")
        dense = doc.metadata.get("dense_score")
        hybrid = doc.metadata.get("hybrid_rrf_score")
        parts = []
        if dense is not None:
            parts.append(f"dense={float(dense):.4f}")
        if bm25 is not None:
            parts.append(f"bm25={float(bm25):.4f}")
        if hybrid is not None:
            parts.append(f"rrf={float(hybrid):.4f}")
        score_text = " " + " ".join(parts) if parts else ""
        print(f"#{rank}{score_text} | {doc.metadata.get('product_id', '')} | {doc.metadata.get('category', '')} | {doc.metadata.get('brand', '')}")
        print(f"  {doc.metadata.get('title', '')}")
        if doc.metadata.get("image_url"):
            print(f"  image_url: {doc.metadata.get('image_url')}")
        print(f"  {doc.page_content[:220].replace(chr(10), ' | ')}")
"""
        ),
        markdown_cell(
            """
## BƯỚC 7: Optional Reranker

Reranker đọc cặp `(query, product text)` kỹ hơn dense retrieval, nhưng chậm hơn.

Khi mới debug, hãy xem raw candidates trước. Chỉ bật reranker khi bạn muốn so sánh:

```python
ENABLE_PRODUCT_RERANKER = True
```
"""
        ),
        code_cell(
            """
product_reranker = None


class ProductCrossEncoderReranker:
    \"\"\"Cross-encoder reranker, chỉ load khi `ENABLE_PRODUCT_RERANKER=True`.\"\"\"

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        print(f"[OK] Reranker loaded: {model_name} on {self.device}")

    def score_pairs(self, query: str, docs: list[Document]) -> list[float]:
        pairs = [(query, doc.page_content[:1400]) for doc in docs]
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), RERANKER_BATCH_SIZE):
                batch = pairs[start:start + RERANKER_BATCH_SIZE]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits
                batch_scores = logits[:, -1] if logits.ndim == 2 and logits.shape[-1] > 1 else logits.reshape(-1)
                scores.extend(batch_scores.detach().float().cpu().tolist())
        return scores


def get_product_reranker():
    global product_reranker, ENABLE_PRODUCT_RERANKER
    if not ENABLE_PRODUCT_RERANKER:
        return None
    if product_reranker is None:
        try:
            product_reranker = ProductCrossEncoderReranker()
        except Exception as exc:
            ENABLE_PRODUCT_RERANKER = False
            print(f"[WARN] Không load được reranker, dùng dense order: {exc}")
            return None
    return product_reranker


def rerank_documents(query: str, docs: list[Document], top_n: int = RERANKER_TOP_N) -> list[Document]:
    reranker = get_product_reranker()
    if reranker is None or not docs:
        return docs

    scores = reranker.score_pairs(query, docs[:top_n])
    ranked = sorted(zip(docs[:top_n], scores), key=lambda pair: pair[1], reverse=True)
    output: list[Document] = []
    for rank, (doc, score) in enumerate(ranked, start=1):
        doc.metadata["rerank_rank"] = rank
        doc.metadata["rerank_score"] = float(score)
        output.append(doc)
    return output
"""
        ),
        markdown_cell(
            """
## BƯỚC 8: Debug Một Query Từ Đầu Đến Cuối

Hàm này in ba lát cắt:

1. `Raw Qdrant candidates`: Qdrant lấy gì.
2. `BM25 lexical candidates`: keyword search lấy gì.
3. `Hybrid RRF candidates`: dense + BM25 hợp nhất ra sao.
4. `After rerank`: reranker có đổi thứ tự không.
5. `Final docs`: các sản phẩm thật sự sẽ gửi vào LLM, kèm ảnh resolve từ metadata nếu tìm được file local.

Các candidate list mặc định sẽ in hết, không cắt top-10 nữa. Nếu output quá dài, bạn có thể gọi riêng `print_doc_rows(..., limit=10)`.
"""
        ),
        code_cell(
            """
def print_score_gap_diagnostics(title: str, docs: list[Document], score_key: str) -> None:
    \"\"\"In chênh lệch score/rank để biết retriever có tự tin không.\"\"\"
    scores = [doc.metadata.get(score_key) for doc in docs if doc.metadata.get(score_key) is not None]
    if len(scores) < 2:
        return
    scores = [float(score) for score in scores]
    print(f"\\n=== {title} score diagnostics ===")
    print(f"top1={scores[0]:.4f} | top2={scores[1]:.4f} | gap={scores[0] - scores[1]:.4f}")
    print(f"top1-top{min(10, len(scores))}={scores[0] - scores[min(9, len(scores)-1)]:.4f}")


def debug_layer_a_query(
    query: str,
    k: int = PRODUCT_SEARCH_CANDIDATE_K,
    use_bm25: bool = True,
    show_final_images: bool = True,
) -> dict[str, list[Document]]:
    print(f"QUERY: {query}")

    raw_docs = retrieve_raw_candidates(query, k=k)
    print_doc_rows("Raw Qdrant candidates", raw_docs)
    print_score_gap_diagnostics("Dense", raw_docs, score_key="dense_score")

    bm25_docs: list[Document] = []
    hybrid_docs: list[Document] = raw_docs
    if use_bm25:
        bm25_docs = bm25_search(query, k=BM25_CANDIDATE_K)
        print_bm25_rows("BM25 lexical candidates", bm25_docs)

        hybrid_docs = rrf_fuse_documents(raw_docs, bm25_docs, k=HYBRID_CANDIDATE_K)
        print_bm25_rows("Hybrid RRF candidates", hybrid_docs)

    ranked_docs = rerank_documents(query, hybrid_docs, top_n=RERANKER_TOP_N)
    print_doc_rows("After optional rerank", ranked_docs)

    final_docs = diversity_filter_documents(ranked_docs, max_docs=PRODUCT_SEARCH_PAGE_SIZE)
    print_doc_rows("Final docs sent to LLM", final_docs)
    if show_final_images:
        show_final_doc_images(final_docs)

    return {
        "raw_docs": raw_docs,
        "bm25_docs": bm25_docs,
        "hybrid_docs": hybrid_docs,
        "ranked_docs": ranked_docs,
        "final_docs": final_docs,
    }


# Ví dụ:
# result = debug_layer_a_query("tìm áo sơ mi trắng đi làm cho nữ")
# result = debug_layer_a_query("nike nữ giày thể thao", use_bm25=True)
"""
        ),
    ]

    nb["cells"] = cells
    nb["metadata"] = metadata
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Refined notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()

