# Bắt đầu đọc hệ thống từ đây

Tài liệu này là bản đồ bàn giao chính thức cho `research_demo_v3`. Không cần đọc toàn bộ code hoặc notebook theo thứ tự tên file.

## Cụm tài liệu

| Cụm | Nội dung | File |
|---|---|---|
| [`architecture/`](architecture/) | Kiến trúc, router, thiết kế | 01, 05, 06, INTENT_MODULE, ROUTER_AND_LAYER_B_EMBEDDING_RATIONALE, LANGGRAPH_SYSTEM_DESIGN |
| [`setup/`](setup/) | Cài đặt, deploy, hạ tầng | 02, SETUP_GUIDE, VASTAI_SSH_GUIDE, REMOTE_VIFASHIONCLIP_SERVICE, QDRANT_DOCKER_MIGRATION |
| [`runtime/`](runtime/) | Luồng request, API, dữ liệu | 03, 04, 07 |
| [`quality/`](quality/) | Debug, đánh giá, viết báo cáo | 08, RAG_DEBUG_PLAYBOOK, 09, 10 |
| [`archive/`](archive/) | Tài liệu cũ, đã bị thay thế | SYSTEM_ARCHITECTURE, GIAI_THICH_HE_THONG_RESEARCH_DEMO_V3 |

## Hướng phát triển

Kế hoạch nâng cấp (đánh giá định lượng, phát triển Agent, vá hạn chế): [`HUONG_PHAT_TRIEN.md`](HUONG_PHAT_TRIEN.md).

## Ba đường đọc

| Người đọc | Mục tiêu | Thứ tự |
|---|---|---|
| Hội đồng | Hiểu bài toán, kiến trúc và đóng góp | [`architecture/01_SYSTEM_OVERVIEW.md`](architecture/01_SYSTEM_OVERVIEW.md) |
| Nhóm viết báo cáo | Viết đúng kiến trúc và luồng xử lý | [`01`](architecture/01_SYSTEM_OVERVIEW.md) → [`03`](runtime/03_DATA_AND_INDEXING.md) → [`04`](runtime/04_RUNTIME_REQUEST_FLOW.md) → [`05`](architecture/05_INTENT_ROUTER_DECISION.md) → [`09`](quality/09_EVALUATION.md) → [`10`](quality/10_REPORT_WRITING_GUIDE.md) |
| Người tiếp quản kỹ thuật | Cài, chạy, debug và sửa code | Đọc tuần tự `00` → [`setup/02`](setup/02_SETUP_AND_MODELS.md) → [`runtime/03`](runtime/03_DATA_AND_INDEXING.md) → ... → [`quality/09`](quality/09_EVALUATION.md) |

## Mô hình tinh thần

Đây không phải một LLM nhận câu hỏi rồi tự trả lời. Hệ thống là một bộ điều phối:

```mermaid
flowchart LR
    U[Text / ảnh] --> V[Validation]
    V --> R[Intent router]
    R --> P[Pipeline cố định]
    P --> Q[Qdrant / VLM / Layer B]
    Q --> C[Product cards]
    Q --> L[LLM viết lời tư vấn]
    L --> G[Grounding filter và log]
    C --> UI[Web UI]
    G --> UI
```

LLM có thể hỗ trợ hiểu câu mơ hồ và viết lời tư vấn. Python mới là thành phần quyết định route hợp lệ. Qdrant và product card là nguồn sự thật về sản phẩm.

## Nguồn sự thật của từng phần

| Nội dung | File nguồn |
|---|---|
| Endpoint, state, orchestration, SSE | `apps/api/api.py` |
| Model, collection, threshold | `src/fashion_rag/config.py` |
| Intent, action, slot, route | `src/fashion_rag/core/intent.py` |
| Text/image retrieval | `src/fashion_rag/infrastructure/vectorstores/vector_store.py`, `src/fashion_rag/modules/retrieval/image_search.py` |
| Outfit Layer B → Layer A | `src/fashion_rag/application/recommendation/outfit.py` |
| LLM prompt và chain | `src/fashion_rag/infrastructure/llms/llm.py`, `src/fashion_rag/application/chat/chains.py` |
| VLM | `src/fashion_rag/infrastructure/llms/vision.py` |
| Validation, grounding, log | `src/fashion_rag/core/security.py` |
| Giao diện | `apps/api/static/index.html` |

Nếu tài liệu và code mâu thuẫn, code là nguồn sự thật; sau đó phải cập nhật lại tài liệu trong cùng pull request.

## Notebook nên đọc

`notebooks/research_demo_v3_split/00_INDEX.ipynb` là bản đồ notebook. Notebook giải thích và thử nghiệm; web app dùng code trong `apps/` và `src/fashion_rag/` để chạy thật.

## Quy tắc khi thay đổi hệ thống

1. Thay đổi policy router phải thêm case vào `tests/router_eval_cases.jsonl`.
2. Thay đổi input/output API phải cập nhật [`runtime/07_API_FRONTEND_CONTRACT.md`](runtime/07_API_FRONTEND_CONTRACT.md).
3. Thay đổi model, vector dimension hoặc collection phải cập nhật [`setup/02_SETUP_AND_MODELS.md`](setup/02_SETUP_AND_MODELS.md) và [`runtime/03_DATA_AND_INDEXING.md`](runtime/03_DATA_AND_INDEXING.md).
4. Không dùng `confidence` do LLM tự khai để quyết định route; dùng `certainty` và policy Python.
5. Mã, giá, thương hiệu và ảnh phải đến từ product card/retrieval, không đến từ lời LLM.

