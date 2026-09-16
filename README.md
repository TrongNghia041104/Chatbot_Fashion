# Fashion RAG Chatbot — Tư Vấn Thời Trang

Chatbot tư vấn thời trang sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** + **LLM** với khả năng xử lý đa phương thức (text + hình ảnh).

---

## Tính năng

- 🔍 **Tìm kiếm sản phẩm** thông minh với RAG (BGE-M3 + Qdrant)
- 👗 **Tư vấn phối đồ** theo vóc dáng, tone da, phong cách (Layer B)
- 📷 **Phân tích hình ảnh**: nhận diện vóc dáng người dùng, caption sản phẩm
- 💬 **Lịch sử hội thoại gần** qua Redis; có thể bật tóm tắt khi cần
- ⚡ **Streaming response** qua SSE

---

## Cấu trúc Project

```
Chatbot_Fashion/
├── main.py                       ← Entry point (chạy server)
├── docker-compose.yml            ← Qdrant + Redis
├── requirements.txt
│
├── apps/api/                     ← FastAPI backend
│   ├── api.py
│   └── static/
│       └── index.html            ← Web demo UI
│
├── src/fashion_rag/               ← Package chính
│   ├── config.py                 ← Tất cả cấu hình tập trung
│   ├── core/                     ← Hạ tầng dùng chung + router
│   │   ├── intent.py             ← Intent detection / routing
│   │   ├── security.py           ← Validation, grounding, log
│   │   └── telemetry.py
│   ├── domain/                   ← Entity, value object, port (không phụ thuộc hạ tầng)
│   │   ├── entities/decision.py  ← IntentDecision
│   │   ├── value_objects/enums.py
│   │   └── ports/                ← EmbedderPort/RetrieverPort/LLMPort
│   ├── application/               ← Use case orchestration
│   │   ├── chat/                 ← chains.py, profile.py
│   │   └── recommendation/       ← outfit.py (Layer B)
│   ├── modules/                   ← Pipeline nghiệp vụ
│   │   ├── retrieval/            ← image_search.py
│   │   └── ingestion/            ← product_data.py
│   └── infrastructure/            ← Adapter công nghệ cụ thể
│       ├── vectorstores/         ← Qdrant + Layer B indexing
│       ├── embeddings/           ← BGE-M3 / ViFashionCLIP wrapper
│       ├── llms/                 ← LLM + Prompts, Vision (Qwen2.5-VL)
│       └── cache/                ← Redis chat history
│
├── data/
│   ├── metadata/            ← Fashion product JSONL files (20 categories)
│   └── stylists/            ← Layer B knowledge (Female + Male)
│
├── notebooks/               ← Jupyter notebooks thực nghiệm
├── docs/                    ← Tài liệu
├── tests/
│   └── sample_images/       ← Ảnh mẫu để test
└── storage/                 ← Docker volumes (gitignored)
    ├── qdrant/
    └── redis/
```

---

## Cài đặt & Chạy

### 1. Yêu cầu

- Python 3.10+
- Ollama local hoặc qua SSH tunnel với các model: `bge-m3`, `qwen3:4b-instruct`, `qwen2.5vl:3b`
- Docker & Docker Compose

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

### 3. Khởi động Qdrant & Redis

```bash
docker-compose up -d
```

### 4. Chạy server

```bash
python main.py
# hoặc
PYTHONPATH=src uvicorn apps.api.api:app --reload --port 8000
```

Mở trình duyệt: http://localhost:8000

---

## Cấu hình

Tất cả cấu hình (URLs, model names, thresholds, keywords) nằm trong [`src/fashion_rag/config.py`](src/fashion_rag/config.py).

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (SSH tunnel → Vast.ai) |
| `LLM_MODEL` | `qwen3:4b-instruct` | Model LLM chính |
| `VISION_MODEL` | `qwen2.5vl:3b` | Model xử lý ảnh |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB |
| `REDIS_URL` | `redis://localhost:6379` | Redis chat history |

---

## Tài liệu thêm

Đọc từ [`documentation/00_README_FIRST.md`](documentation/00_README_FIRST.md). Bộ tài liệu được chia thành ba đường đọc cho Hội đồng, nhóm viết báo cáo và người tiếp quản kỹ thuật, và tổ chức theo 4 cụm chủ đề: `architecture/`, `setup/`, `runtime/`, `quality/`.

- [`architecture/01_SYSTEM_OVERVIEW.md`](documentation/architecture/01_SYSTEM_OVERVIEW.md) — Tổng quan ngắn cho Hội đồng.
- [`setup/02_SETUP_AND_MODELS.md`](documentation/setup/02_SETUP_AND_MODELS.md) — Cài đặt, model và endpoint.
- [`runtime/04_RUNTIME_REQUEST_FLOW.md`](documentation/runtime/04_RUNTIME_REQUEST_FLOW.md) — Luồng request end-to-end.
- [`architecture/05_INTENT_ROUTER_DECISION.md`](documentation/architecture/05_INTENT_ROUTER_DECISION.md) — Intent, decision, route và slot policy.
- [`quality/10_REPORT_WRITING_GUIDE.md`](documentation/quality/10_REPORT_WRITING_GUIDE.md) — Khung viết chương kiến trúc/luồng xử lý.
