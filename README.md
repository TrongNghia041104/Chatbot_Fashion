# Fashion RAG Chatbot — Tư Vấn Thời Trang

Chatbot tư vấn thời trang sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** + **LLM** với khả năng xử lý đa phương thức (text + hình ảnh).

---

## Tính năng

- 🔍 **Tìm kiếm sản phẩm** thông minh với RAG (BGE-M3 + Qdrant)
- 👗 **Tư vấn phối đồ** theo vóc dáng, tone da, phong cách (Layer B)
- 📷 **Phân tích hình ảnh**: nhận diện vóc dáng người dùng, caption sản phẩm
- 💬 **Lịch sử hội thoại** tự động tóm tắt (Redis)
- ⚡ **Streaming response** qua SSE

---

## Cấu trúc Project

```
Chatbot_Fashion/
├── main.py                  ← Entry point (chạy server)
├── docker-compose.yml       ← Qdrant + Redis
├── requirements.txt
│
├── app/                     ← Package chính
│   ├── config.py            ← Tất cả cấu hình tập trung
│   ├── api.py               ← FastAPI backend
│   ├── static/
│   │   └── index.html       ← Web demo UI
│   └── core/                ← Logic core
│       ├── embeddings.py    ← BGE-M3 wrapper
│       ├── vector_store.py  ← Qdrant + Layer B indexing
│       ├── llm.py           ← LLM + Prompts
│       ├── vision.py        ← Vision functions (Qwen2.5-VL)
│       ├── intent.py        ← Intent detection
│       ├── outfit.py        ← Outfit matching (Layer B)
│       ├── history.py       ← Redis chat history
│       └── chains.py        ← RAG pipeline assembly
│
├── data/
│   ├── metadata/            ← Fashion product JSONL files (20 categories)
│   └── stylists/            ← Layer B knowledge (Female + Male)
│
├── images/                  ← Ảnh sản phẩm
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
- [Ollama](https://ollama.ai/) với các models: `bge-m3`, `qwen3:4b-instruct`, `qwen2.5vl:3b`
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
uvicorn app.api:app --reload --port 8000
```

Mở trình duyệt: http://localhost:8000

---

## Cấu hình

Tất cả cấu hình (URLs, model names, thresholds, keywords) nằm trong [`app/config.py`](app/config.py).

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (SSH tunnel → Vast.ai) |
| `LLM_MODEL` | `qwen3:4b-instruct` | Model LLM chính |
| `VISION_MODEL` | `qwen2.5vl:3b` | Model xử lý ảnh |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB |
| `REDIS_URL` | `redis://localhost:6379` | Redis chat history |

---

## Tài liệu thêm

Xem thêm trong thư mục [`docs/`](docs/):
- [`SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) — Hướng dẫn cài đặt chi tiết
- [`PROMPT_WebDemo_FashionChatbot.md`](docs/PROMPT_WebDemo_FashionChatbot.md) — Prompt engineering notes
