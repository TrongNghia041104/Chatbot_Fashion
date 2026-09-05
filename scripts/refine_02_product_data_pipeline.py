"""Refine notebook 02 for product-data-pipeline debugging."""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "research_demo_v3_split" / "02_product_data_pipeline.ipynb"


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


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return no_marks.replace("đ", "d").replace("Đ", "D").lower()


def first_line(cell: dict) -> str:
    text = "".join(cell.get("source", [])).strip()
    return text.splitlines()[0] if text else ""


def find_cell(cells: list[dict], startswith: str | None = None, contains: str | None = None) -> int:
    start_key = fold(startswith) if startswith else None
    contains_key = fold(contains) if contains else None
    for idx, cell in enumerate(cells):
        text = "".join(cell.get("source", []))
        if start_key and fold(first_line(cell)).startswith(start_key):
            return idx
        if contains_key and contains_key in fold(text):
            return idx
    raise KeyError(startswith or contains)


def set_cell(cells: list[dict], idx: int, text: str) -> None:
    cells[idx]["source"] = source(text)
    if cells[idx].get("cell_type") == "code":
        cells[idx]["execution_count"] = None
        cells[idx]["outputs"] = []


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    set_cell(
        cells,
        0,
        """
# 02 - Product Data Pipeline: Từ Dữ Liệu Thô Đến Bằng Chứng Retrieval

Notebook này trả lời câu hỏi nền tảng nhất của RAG:

> Trước khi Qdrant tìm được sản phẩm đúng, mỗi sản phẩm đã được biến thành “bằng chứng” như thế nào?

Một sản phẩm thô trong JSONL không tự nhiên trở thành context tốt cho LLM. Ta phải tách nó thành hai phần:

| Phần | Dùng để làm gì | Ví dụ |
|---|---|---|
| `metadata` | Filter, hiển thị, render ảnh, kiểm tra hallucination | `product_id`, `category`, `brand`, `price`, `image_url` |
| `page_content` | Text đem đi embedding và gửi cho LLM | tên, mã, danh mục, màu, chất liệu, size, dịp, mô tả |

Góc nhìn mới: **retrieval không bắt đầu ở Qdrant, mà bắt đầu từ cách bạn viết `page_content` và `metadata`.** Nếu hai thứ này nghèo hoặc sai, embedding tốt mấy cũng khó cứu.

**Output nên quan sát khi chạy notebook này**

- Một raw product item có field nào.
- `metadata` có đủ khóa để filter/render/grounding không.
- `page_content` có đủ tín hiệu thời trang để retrieval không.
- Thống kê dữ liệu: thiếu id, thiếu category, thiếu ảnh, lỗi JSON.
- Chỉ chạy index khi đã kiểm tra preview ổn.
""",
    )

    set_cell(
        cells,
        1,
        """
from pathlib import Path
import json
import random

from langchain_core.documents import Document


def resolve_chatbot_fashion_dir() -> Path:
    \"\"\"Tìm thư mục gốc `Chatbot_Fashion/` nếu notebook 01 chưa được chạy trong kernel này.\"\"\"
    cwd = Path.cwd()
    if cwd.name == "notebooks" and cwd.parent.name == "Chatbot_Fashion":
        return cwd.parent
    if cwd.name == "Chatbot_Fashion":
        return cwd
    candidate = cwd / "Chatbot_Fashion"
    if candidate.exists():
        return candidate
    for parent in cwd.parents:
        candidate = parent / "Chatbot_Fashion"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy thư mục Chatbot_Fashion/")


# Nếu notebook 01 đã chạy, biến CHATBOT_FASHION_DIR có thể đã tồn tại.
# Nếu chưa, cell này tự resolve để notebook 02 vẫn đọc được dữ liệu.
CHATBOT_FASHION_DIR = globals().get("CHATBOT_FASHION_DIR", resolve_chatbot_fashion_dir())
METADATA_FILE = CHATBOT_FASHION_DIR / "data" / "metadata" / "meta_Amazon_Lazada_Fashion_65k.jsonl"

print(f"[OK] Project root  : {CHATBOT_FASHION_DIR}")
print(f"[OK] Metadata file : {METADATA_FILE}")
print(f"     Exists        : {METADATA_FILE.exists()}")
""",
    )

    set_cell(
        cells,
        2,
        """
## PHẦN 3A: Tư Duy Data Pipeline - Biến Product Thô Thành `Document`

LangChain `Document` là hợp đồng giữa dữ liệu và retrieval:

```text
Document(
    page_content = \"đoạn text đem đi embedding và gửi cho LLM\",
    metadata     = {thông tin có cấu trúc để filter, render, kiểm tra}
)
```

Trong hệ này, mỗi sản phẩm là một chunk. 65k sản phẩm tương ứng khoảng 65k `Document`.

```text
JSONL file
  -> đọc từng dòng bằng json.loads
  -> raw product dict
  -> extract_image_urls()
  -> build_product_metadata()
  -> build_product_page_content()
  -> Document(page_content, metadata)
```

Điều cần chú ý: `page_content` không phải văn bản trang trí. Nó là thứ quyết định vector của sản phẩm. Nếu thiếu màu/chất liệu/dịp/size, query tương ứng sẽ khó tìm đúng sản phẩm.
""",
    )

    set_cell(
        cells,
        3,
        """
### Bốn Hàm Nhỏ Nhưng Quyết Định Chất Lượng Retrieval

- `extract_image_urls`: lấy danh sách ảnh để UI hiển thị và LLM có `IMAGE_URL`.
- `normalize_to_text`: biến dữ liệu lộn xộn thành text ổn định, tránh `None` hoặc list thô làm bẩn context.
- `build_product_metadata`: tạo phần dữ liệu có cấu trúc cho filter, display, grounding.
- `build_product_page_content`: tạo đoạn text có nhãn rõ ràng để embed bằng ViFashionCLIP.

Nếu hệ thống lấy sai sản phẩm, hãy kiểm tra `build_product_page_content` trước khi đổ lỗi cho Qdrant.
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="**`extract_image_urls`"),
        """
**`extract_image_urls` - chọn ảnh có ích cho UI và context**

Dữ liệu raw thường có nhiều ảnh cho một sản phẩm. Hàm này chỉ lấy field `large` vì đó là ảnh chất lượng cao nhất trong metadata.
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="def extract_image_urls"),
        """
def extract_image_urls(item: dict) -> list[str]:
    \"\"\"
    Lấy tất cả URL/path ảnh chất lượng cao từ một product item.

    Input:
        item: dict của một dòng JSONL sản phẩm.

    Output:
        list[str] chứa các ảnh có field `large`.

    Vì sao cần hàm riêng?
        - UI cần ảnh để render sản phẩm.
        - Prompt LLM cần `IMAGE_URL` nếu muốn trả markdown image.
        - Grounding/debug dễ hơn khi mỗi Document có image_url ổn định.
    \"\"\"
    images = item.get("images", []) or []
    return [img.get("large") for img in images if isinstance(img, dict) and img.get("large")]
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="**`normalize_to_text`"),
        """
**`normalize_to_text` - dọn dữ liệu trước khi biến thành context**

Raw metadata có thể là `None`, chuỗi rỗng, list, số, hoặc object lạ. Retrieval cần text ổn định, nên mọi field phải được chuẩn hóa về chuỗi.
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="def normalize_to_text"),
        """
def normalize_to_text(value, default: str = "Không rõ") -> str:
    \"\"\"
    Chuẩn hóa một field bất kỳ thành chuỗi dùng được trong `page_content`.

    Quy tắc:
        - None hoặc chuỗi rỗng -> default.
        - list -> bỏ item rỗng rồi nối bằng dấu phẩy.
        - kiểu khác -> ép sang str và strip.

    Vì sao không để trống?
        Khi xem debug context, `Không rõ` giúp ta biết field thật sự thiếu,
        thay vì nhầm là hàm build text bị lỗi.
    \"\"\"
    if value is None or value == "":
        return default

    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned) if cleaned else default

    return str(value).strip() or default
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="**`build_product_metadata`"),
        """
**`build_product_metadata` - phần có cấu trúc của một sản phẩm**

`metadata` không phải thứ để embedding chính. Nó là “thẻ căn cước” của sản phẩm: dùng để filter Qdrant, render UI, log, và kiểm tra LLM có bịa mã sản phẩm không.
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="def build_product_metadata"),
        """
def build_product_metadata(item: dict) -> dict:
    \"\"\"
    Tạo metadata dict cho LangChain Document.

    Metadata nên ngắn, ổn định, dễ filter:
        product_id : khóa định danh và grounding.
        title      : tên hiển thị.
        category   : filter theo kệ hàng.
        department : nam/nữ/trẻ em nếu có.
        brand      : diversity filter có thể giới hạn brand lặp.
        price      : hiển thị và filter giá sau này.
        images     : danh sách ảnh.
        image_url  : ảnh đại diện đầu tiên.

    Lưu ý:
        Đừng nhồi mô tả dài vào metadata. Text dài thuộc `page_content`.
    \"\"\"
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
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="**`build_product_page_content`"),
        """
**`build_product_page_content` - phần quyết định vector sản phẩm**

Đây là đoạn text đem đi embedding. Format có nhãn giúp model phân biệt “Giá”, “Màu sắc”, “Chất liệu”, “Dịp sử dụng” thay vì đọc một chuỗi hỗn độn.
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="def build_product_page_content"),
        """
def build_product_page_content(item: dict) -> str:
    \"\"\"
    Tạo text có nhãn rõ ràng để embed bằng ViFashionCLIP.

    Vì sao dùng nhãn tiếng Việt?
        Query của user là tiếng Việt. Khi page_content cũng có nhãn tiếng Việt,
        các tín hiệu như màu, chất liệu, dịp sử dụng, size dễ được model bắt hơn.

    Ví dụ output:
        Tên sản phẩm: Áo sơ mi trắng
        Màu sắc: Trắng
        Dịp sử dụng: Đi làm

    Đây là nơi nên chỉnh nếu retrieval thường bỏ sót một loại nhu cầu nào đó.
    \"\"\"
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
    return "\\n".join(f"{label}: {normalize_to_text(value)}" for label, value in fields)


print("[OK] Hàm build Document đã sẵn sàng")
""",
    )

    # Insert preview section after build_product_page_content code cell.
    if not any("PHẦN 3A.1" in "".join(cell.get("source", [])) for cell in cells):
        insert_at = find_cell(cells, startswith="## PHAN 3B")
        cells[insert_at:insert_at] = [
            markdown_cell(
                """
## PHẦN 3A.1: Preview Một Sản Phẩm Trước Khi Index

Đừng index ngay. Trước tiên hãy nhìn một sản phẩm đi qua pipeline:

```text
raw item -> metadata -> page_content -> Document
```

Nếu preview đã sai, Qdrant sẽ lưu sai hàng loạt.
"""
            ),
            code_cell(
                """
def iter_jsonl_items(file_path: str | Path, limit: int | None = None):
    \"\"\"Yield từng product item từ JSONL, bỏ qua dòng rỗng và dòng lỗi JSON.\"\"\"
    file_path = Path(file_path)
    count = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    return
            except json.JSONDecodeError:
                continue


def select_jsonl_item(
    file_path: str | Path,
    index: int | None = None,
    seed: int | None = None,
) -> tuple[int, dict] | None:
    \"\"\"
    Chọn một item hợp lệ từ JSONL.

    - index=None: chọn ngẫu nhiên bằng reservoir sampling, không cần load cả file vào RAM.
    - index=int : lấy đúng item hợp lệ thứ index để tái hiện một case cụ thể.
    - seed      : cố định random nếu bạn muốn kết quả lặp lại được.
    \"\"\"
    rng = random.Random(seed) if seed is not None else random

    if index is not None and index < 0:
        raise ValueError("index phải >= 0 hoặc None")

    selected: tuple[int, dict] | None = None
    seen = 0

    for valid_index, item in enumerate(iter_jsonl_items(file_path)):
        if index is not None:
            if valid_index == index:
                return valid_index, item
            continue

        seen += 1
        if rng.randrange(seen) == 0:
            selected = (valid_index, item)

    return selected


def preview_product_document(
    file_path: str | Path = METADATA_FILE,
    index: int | None = None,
    seed: int | None = None,
) -> Document | None:
    \"\"\"
    In thử một sản phẩm sau khi chuyển thành Document.

    Dùng cell này để kiểm tra:
        - metadata có đủ id/category/brand/price/image_url không.
        - page_content có đủ màu/chất liệu/size/dịp/mô tả không.

    Mặc định index=None nghĩa là chọn ngẫu nhiên một sản phẩm thật.
    Truyền index=123 nếu bạn muốn xem lại đúng một sản phẩm cụ thể.
    \"\"\"
    selected = select_jsonl_item(file_path=file_path, index=index, seed=seed)
    if selected is None:
        print(f"[LỖI] Không chọn được item từ {file_path} với index={index}")
        return None

    selected_index, item = selected
    doc = Document(
        page_content=build_product_page_content(item),
        metadata=build_product_metadata(item),
    )

    print(f"SELECTED INDEX: {selected_index} ({'random' if index is None else 'fixed'})")
    print("RAW KEYS:")
    print(sorted(item.keys()))
    print("\\nMETADATA:")
    print(json.dumps(doc.metadata, ensure_ascii=False, indent=2)[:2000])
    print("\\nPAGE_CONTENT:")
    print(doc.page_content[:2500])
    return doc


# Bỏ comment để xem thử một sản phẩm ngẫu nhiên.
# preview_doc = preview_product_document()

# Nếu muốn random nhưng lặp lại được:
# preview_doc = preview_product_document(seed=42)

# Nếu muốn xem đúng một item cụ thể:
# preview_doc = preview_product_document(index=0)
"""
            ),
        ]

    if not any("PHẦN 3A.2" in "".join(cell.get("source", [])) for cell in cells):
        insert_at = find_cell(cells, startswith="## PHAN 3B")
        cells[insert_at:insert_at] = [
            markdown_cell(
                """
## PHẦN 3A.2: Sanity Check Các Hàm Build `Document`

Trước khi đọc 65k sản phẩm, hãy kiểm tra hàm bằng một sản phẩm giả nhỏ.

Mục tiêu của cell này:

- Nếu hàm sai, lỗi xuất hiện ngay bằng `assert`.
- Nếu hàm đúng, bạn thấy rõ input giả tạo ra `metadata` và `page_content` như thế nào.
- Không gọi embedding, không gọi Qdrant, không tốn thời gian.

Đây là kiểu “đèn báo trên bảng điều khiển”: nhỏ nhưng giúp bạn không debug mù.
"""
            ),
            code_cell(
                """
def run_product_document_sanity_checks() -> Document:
    \"\"\"
    Kiểm tra nhanh 4 hàm:
        - extract_image_urls
        - normalize_to_text
        - build_product_metadata
        - build_product_page_content

    Trả về Document mẫu để bạn nhìn output trực tiếp.
    \"\"\"
    sample_item = {
        "product_id": "TEST001",
        "title": "Áo sơ mi trắng công sở",
        "category": "Áo",
        "department": "Nữ",
        "brand": "Demo Brand",
        "price": 299000,
        "images": [
            {"large": "images/TEST001_MAIN.jpg"},
            {"small": "images/TEST001_SMALL.jpg"},
            {"large": "images/TEST001_BACK.jpg"},
        ],
        "details": {
            "main_color": "Trắng",
            "material": "Cotton",
            "size": ["S", "M", "L"],
            "pattern": "",
        },
        "season": "Xuân hè",
        "occasion": "Đi làm",
        "description": "Áo sơ mi trắng dáng basic, phù hợp môi trường công sở.",
    }

    # 1) Ảnh: chỉ lấy field `large`, bỏ ảnh không có large.
    image_urls = extract_image_urls(sample_item)
    assert image_urls == ["images/TEST001_MAIN.jpg", "images/TEST001_BACK.jpg"]

    # 2) Chuẩn hóa text: None/rỗng/list đều ra text ổn định.
    assert normalize_to_text(None) == "Không rõ"
    assert normalize_to_text("") == "Không rõ"
    assert normalize_to_text(["S", "M", ""]) == "S, M"
    assert normalize_to_text("  Cotton  ") == "Cotton"

    # 3) Metadata: đủ khóa quan trọng cho filter/render/grounding.
    metadata = build_product_metadata(sample_item)
    assert metadata["product_id"] == "TEST001"
    assert metadata["category"] == "Áo"
    assert metadata["image_url"] == "images/TEST001_MAIN.jpg"
    assert metadata["images"] == image_urls

    # 4) Page content: có nhãn rõ và giữ được tín hiệu retrieval chính.
    page_content = build_product_page_content(sample_item)
    required_snippets = [
        "Tên sản phẩm: Áo sơ mi trắng công sở",
        "Mã sản phẩm: TEST001",
        "Danh mục: Áo",
        "Màu sắc: Trắng",
        "Chất liệu: Cotton",
        "Kích cỡ: S, M, L",
        "Dịp sử dụng: Đi làm",
    ]
    for snippet in required_snippets:
        assert snippet in page_content, f"Thiếu trong page_content: {snippet}"

    doc = Document(page_content=page_content, metadata=metadata)
    print("[OK] Sanity checks passed cho các hàm build Document.")
    print("\\nMETADATA MẪU:")
    print(json.dumps(doc.metadata, ensure_ascii=False, indent=2))
    print("\\nPAGE_CONTENT MẪU:")
    print(doc.page_content)
    return doc


sample_doc = run_product_document_sanity_checks()
"""
            ),
            markdown_cell(
                """
## PHẦN 3A.3: Smoke Test Với Một Sản Phẩm Thật

Sau khi test bằng dữ liệu giả, hãy thử một dòng thật trong file JSONL.

Cell này vẫn nhẹ: chỉ đọc một item thật, build `Document`, rồi kiểm tra các điều kiện tối thiểu. Nếu fail, bạn biết dữ liệu thật có vấn đề trước khi chạy index.
"""
            ),
            code_cell(
                """
def run_real_product_smoke_test(
    file_path: str | Path = METADATA_FILE,
    index: int | None = None,
    seed: int | None = None,
) -> Document | None:
    \"\"\"Kiểm tra tối thiểu một sản phẩm thật từ metadata JSONL.\"\"\"
    doc = preview_product_document(file_path=file_path, index=index, seed=seed)
    if doc is None:
        return None

    assert isinstance(doc.page_content, str) and doc.page_content.strip(), "page_content rỗng"
    assert isinstance(doc.metadata, dict), "metadata không phải dict"

    required_metadata_keys = ["product_id", "title", "category", "brand", "price", "images", "image_url"]
    missing_keys = [key for key in required_metadata_keys if key not in doc.metadata]
    assert not missing_keys, f"metadata thiếu key: {missing_keys}"

    required_content_labels = ["Tên sản phẩm:", "Mã sản phẩm:", "Danh mục:", "Giá:", "Mô tả:"]
    missing_labels = [label for label in required_content_labels if label not in doc.page_content]
    assert not missing_labels, f"page_content thiếu nhãn: {missing_labels}"

    print("\\n[OK] Smoke test sản phẩm thật passed.")
    return doc


# Bỏ comment để kiểm tra một sản phẩm thật ngẫu nhiên.
# real_doc = run_real_product_smoke_test()

# Cố định random nếu muốn lặp lại cùng một item:
# real_doc = run_real_product_smoke_test(seed=42)
"""
            ),
        ]

    set_cell(
        cells,
        find_cell(cells, startswith="## PHAN 3B"),
        """
## PHẦN 3B: Đọc JSONL Và Index Vào Qdrant

Sau khi đã tin rằng `Document` được build đúng, ta mới index vào Qdrant.

Luồng đầy đủ:

```text
JSONL -> list[Document] -> ViFashionCLIP embeddings -> Qdrant collection
```

`process_fashion_metadata()` chỉ đọc và build document. Hàm này an toàn để chạy khi debug.

`run_data_pipeline()` sẽ embed và upsert vào Qdrant. Hàm này nặng, chỉ chạy khi bạn thật sự muốn index hoặc rebuild collection.

**Cơ chế resume:** nếu collection đã có `N` points, pipeline bỏ qua `N` document đầu và index tiếp phần còn lại.
"""
    )

    set_cell(
        cells,
        find_cell(cells, startswith="### Doc file JSONL")
        if any(fold(first_line(cell)).startswith(fold("### Doc file JSONL")) for cell in cells)
        else find_cell(cells, startswith="### Hai Muc Do Chay"),
        """
### Hai Mức Độ Chạy

1. Debug nhẹ: chạy `process_fashion_metadata()` hoặc `preview_product_document()` để xem dữ liệu.
2. Index thật: chạy `run_data_pipeline()` khi Qdrant và embedding model/service đã sẵn sàng.

Không nên chạy index thật trước khi preview vài sản phẩm và kiểm tra thống kê missing field.
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="**`process_fashion_metadata`"),
        """
**`process_fashion_metadata` - đọc toàn bộ JSONL thành list `Document`**

Hàm này là “cổng kiểm định dữ liệu”. Nó không embed và không ghi Qdrant, chỉ build document và in thống kê chất lượng.
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="def process_fashion_metadata"),
        """
def process_fashion_metadata(file_path: str | Path) -> list[Document]:
    \"\"\"
    Đọc file JSONL và chuyển từng dòng hợp lệ thành LangChain Document.

    Output:
        list[Document], mỗi Document tương ứng một sản phẩm.

    Thống kê được in ra:
        total_lines        : số dòng JSONL hợp lệ về mặt non-empty.
        json_errors        : số dòng không parse được JSON.
        missing_product_id : sản phẩm thiếu mã, nguy hiểm cho grounding.
        missing_category   : sản phẩm khó filter theo kệ hàng.
        missing_image      : sản phẩm không render được ảnh đại diện.

    Đây là hàm nên chạy trước khi index để biết dữ liệu đầu vào có sạch không.
    \"\"\"
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
        "[THỐNG KÊ DATA] "
        f"dòng={stats['total_lines']} | docs={len(documents)} | "
        f"lỗi_json={stats['json_errors']} | "
        f"thiếu_id={stats['missing_product_id']} | "
        f"thiếu_category={stats['missing_category']} | "
        f"thiếu_ảnh={stats['missing_image']}"
    )
    return documents
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="**`run_data_pipeline`"),
        """
**`run_data_pipeline` - index thật vào Qdrant**

Đây là hàm nặng nhất notebook 02. Nó gọi embedding model/service và ghi vector vào Qdrant.

Chỉ chạy khi:

- `METADATA_FILE` đúng.
- `preview_product_document()` nhìn ổn.
- Qdrant đang bật.
- ViFashionCLIP embedding đã sẵn sàng.
""",
    )

    set_cell(
        cells,
        find_cell(cells, startswith="def run_data_pipeline"),
        """
def run_data_pipeline(
    metadata_file: str | Path = METADATA_FILE,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "fashion_products_vifashionclip_vi_65k_structured_vi",
    batch_size: int = 128,
):
    \"\"\"
    End-to-end indexing Layer A text:

        JSONL -> Document -> ViFashionCLIP vector -> Qdrant

    Resume logic:
        Nếu collection đã có `current_count` points, pipeline bỏ qua đúng số Document đầu.
        Cách này giả định thứ tự file JSONL ổn định giữa các lần chạy.

    Cảnh báo:
        Hàm này có thể chạy lâu vì phải embed nhiều sản phẩm.
        Đừng gọi khi bạn chỉ muốn đọc hiểu/preview dữ liệu.
    \"\"\"
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        print(f"[LỖI] Không tìm thấy metadata file: {metadata_file}")
        return

    if "ViFashionCLIPTextEmbeddings" not in globals():
        raise RuntimeError("Chưa có ViFashionCLIPTextEmbeddings. Hãy chạy notebook 01 trước.")

    print(f"[THÔNG BÁO] Đọc dữ liệu từ: {metadata_file}")
    all_docs = process_fashion_metadata(metadata_file)

    emb = ViFashionCLIPTextEmbeddings()
    client = QdrantClient(url=qdrant_url)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        current_count = 0
        print(f"[OK] Đã tạo collection mới: {collection_name}")
    else:
        current_count = client.get_collection(collection_name).points_count
        print(f"[INFO] Collection đã có {current_count} sản phẩm.")

    remaining = all_docs[current_count:]
    if not remaining:
        print("[OK] Collection đã index đầy đủ, không cần làm thêm.")
        return

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=emb,
    )

    with tqdm(total=len(all_docs), initial=current_count, desc="Index Layer A", unit="SP") as progress:
        for start in range(0, len(remaining), batch_size):
            batch_docs = remaining[start:start + batch_size]
            vector_store.add_documents(documents=batch_docs)
            progress.update(len(batch_docs))

    print("[OK] Index Layer A text hoàn tất.")


# Chỉ bỏ comment khi thật sự muốn index/re-index dữ liệu.
# run_data_pipeline()

print("[OK] Data pipeline đã sẵn sàng. Hãy preview dữ liệu trước khi index.")
""",
    )

    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Refined notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
