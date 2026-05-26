# 🚀 Hướng dẫn Setup — Fashion RAG Chatbot

## Yêu cầu hệ thống

| Thành phần | Phiên bản tối thiểu |
|---|---|
| Python | 3.10+ |
| Docker Desktop | 4.x trở lên |
| Git | Bất kỳ |
| RAM máy local | ≥ 8 GB |
| GPU Vast.ai | ≥ 8 GB VRAM (khuyến nghị 16 GB) |

---

## BƯỚC 1 — Cài đặt môi trường Python

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Cài đặt toàn bộ thư viện
pip install -r requirements.txt
```

---

## BƯỚC 2 — Khởi động Docker (Qdrant + Redis)

> Đảm bảo **Docker Desktop đang chạy** trước khi thực hiện lệnh này.

```bash
# Khởi động Qdrant và Redis Stack
docker-compose up -d
```

Kiểm tra các container đã chạy chưa:
```bash
docker ps
```

Bạn sẽ thấy 2 container đang chạy:

| Container | Port | Mô tả |
|---|---|---|
| `fashion_qdrant` | `6333` (API), `6334` (gRPC) | Vector Database |
| `fashion_redis_stack` | `6379` (Redis), `8001` (RedisInsight UI) | Chat History |

Kiểm tra Qdrant đang hoạt động:
```bash
curl http://localhost:6333/collections
# Kết quả mong đợi: {"result":{"collections":[]},...}
```

Xem giao diện RedisInsight (tuỳ chọn):
```
Mở trình duyệt: http://localhost:8001
```

---

## BƯỚC 3 — Thuê GPU trên Vast.ai và cài Ollama

### 3.1 Thuê instance Vast.ai

1. Truy cập [vast.ai](https://vast.ai) → đăng nhập
2. Vào **Search** → lọc:
   - **GPU**: RTX 3090 / 4090 / A100 (≥ 16 GB VRAM)
   - **Template**: `pytorch` hoặc `ubuntu`
   - **Disk**: ≥ 40 GB (để chứa model Ollama)
3. Bật tùy chọn **SSH** khi tạo instance

### 3.2 Lấy thông tin SSH

Vào **Instances** → chọn instance đang chạy → copy lệnh SSH, ví dụ:
```
ssh -p 32523 root@120.238.149.205
```

### 3.3 SSH vào Vast.ai và cài Ollama

```bash
# SSH vào máy Vast.ai
ssh -p <PORT> root@<IP_VASTAI>

# Cài Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Khởi động Ollama server (chạy nền)
ollama serve &

# Chờ 5 giây rồi kiểm tra
sleep 5 && ollama list
```

### 3.4 Tải các model cần thiết

> ⚠️ Đây là bước tốn thời gian nhất. Chạy lần lượt và chờ từng model tải xong.

```bash
# Model Embedding — BGE-M3 (~570 MB)
ollama pull bge-m3

# Model LLM chính — Qwen3 4B (~2.6 GB)
ollama pull qwen3:4b-instruct

# Model Vision — Qwen2.5-VL 3B (~2.3 GB)
ollama pull qwen2.5vl:3b

# Kiểm tra các model đã có
ollama list
```

Kết quả mong đợi của `ollama list`:
```
NAME                    ID              SIZE    MODIFIED
bge-m3:latest           ...             567 MB  ...
qwen3:4b-instruct:latest ...            2.6 GB  ...
qwen2.5vl:3b:latest     ...            2.3 GB  ...
```

---

## BƯỚC 4 — Tạo SSH Tunnel (Local ↔ Vast.ai Ollama)

> Mở **terminal mới** trên máy local (giữ nguyên terminal SSH đang có). Chạy lệnh dưới đây và **để terminal này mở suốt** khi dùng chatbot.

```bash
ssh -p <PORT> root@<IP_VASTAI> -L 11434:localhost:11434 -N
```

**Ví dụ thực tế:**
```bash
ssh -p 32523 root@120.238.149.205 -L 11434:localhost:11434 -N
```

| Tham số | Ý nghĩa |
|---|---|
| `-p 32523` | Port SSH của Vast.ai |
| `root@120.238.149.205` | User và IP Vast.ai |
| `-L 11434:localhost:11434` | Map port 11434 local → port 11434 trên Vast.ai |
| `-N` | Chỉ tạo tunnel, không mở shell |

Kiểm tra tunnel đã hoạt động (trên máy local):
```bash
curl http://localhost:11434/api/tags
# Kết quả mong đợi: danh sách model {"models":[...]}
```

> **Lưu ý**: Nếu Vast.ai instance hiển thị lỗi `channel open failed: connect failed: Connection refused`, hãy vào terminal SSH khác và chắc chắn `ollama serve` đang chạy trên Vast.ai.

---

## BƯỚC 5 — Indexing dữ liệu sản phẩm vào Qdrant

> ⚠️ **Chỉ cần chạy 1 lần duy nhất** khi lần đầu setup. Nếu Qdrant đã có dữ liệu thì bỏ qua bước này.

Đặt các file dữ liệu `.jsonl` vào thư mục `data/metadata/`:
```
Chatbot_Fashion/
└── data/
    └── metadata/
        ├── Fashion_Metadata_Ao.jsonl
        ├── Fashion_Metadata_Quan.jsonl
        ├── Fashion_Metadata_Giay.jsonl
        └── ...
```

Chạy notebook hoặc chạy trực tiếp phần Data Pipeline:

**Cách 1 — Dùng Jupyter Notebook:**
```bash
jupyter notebook notebooks/Chatbot_RAG_MultiModal.ipynb
# Chạy cell "PHẦN 3: Data Pipeline" (có thể mất 15-30 phút)
```

**Cách 2 — Import trực tiếp từ app.core:**
```python
from app.core.vector_store import vector_db
# Sau đó chạy các hàm indexing theo notebook hướng dẫn
```

Sau khi index xong, kiểm tra:
```bash
curl http://localhost:6333/collections/fashion_products_bge_m3
# "points_count" phải > 0
```

---

## BƯỚC 6 — Khởi động Backend API

```bash
# Kích hoạt venv (nếu chưa)
venv\Scripts\activate

# Cách 1: Chạy qua main.py (khuyến nghị)
python main.py

# Cách 2: Chạy trực tiếp uvicorn
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Kết quả mong đợi:
```
[INFO] Đang load app.core...
[INFO] Đang load Embedding model BGE-M3...
[INFO] Đang kết nối Qdrant Docker (localhost:6333)...
[OK] Qdrant + Retriever sẵn sàng!
[OK] Layer B: 880 rules Nữ | 416 rules Nam
[OK] Vision functions sẵn sàng!
[OK] LLM sẵn sàng!
[OK] Redis history sẵn sàng!
[OK] RAG Pipeline sẵn sàng!
[OK] app.core loaded!
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## BƯỚC 7 — Mở giao diện Web

```bash
# Truy cập qua backend (khuyến nghị)
# http://localhost:8000

# Hoặc mở file trực tiếp trong trình duyệt
start app\static\index.html
```

---

## Checklist trước khi dùng

```
[ ] Docker Desktop đang chạy
[ ] docker-compose up -d  → fashion_qdrant + fashion_redis_stack running
[ ] SSH Vast.ai: ollama serve đang chạy
[ ] SSH Tunnel: terminal với lệnh -L 11434:localhost:11434 -N đang mở
[ ] curl http://localhost:11434/api/tags → trả về danh sách model
[ ] Qdrant đã có dữ liệu (points_count > 0)
[ ] python main.py → uvicorn running on http://0.0.0.0:8000
[ ] Mở http://localhost:8000 → status bar hiển thị "Đã kết nối"
```

---

## Xử lý lỗi thường gặp

### ❌ `Connection refused` khi tạo SSH tunnel
```bash
# SSH vào Vast.ai và khởi động lại Ollama
ssh -p <PORT> root@<IP>
ollama serve &
```

### ❌ `Could not connect to Qdrant`
```bash
# Kiểm tra Docker container
docker ps
docker start fashion_qdrant
```

### ❌ `Redis connection error`
```bash
docker start fashion_redis_stack
```

### ❌ `Model not found` khi gọi Ollama
```bash
# SSH vào Vast.ai, kiểm tra model
ssh -p <PORT> root@<IP>
ollama list
# Nếu thiếu model, tải lại
ollama pull bge-m3
ollama pull qwen3:4b-instruct
ollama pull qwen2.5vl:3b
```

### ❌ `GGML_ASSERT` lỗi khi phân tích ảnh
Ảnh quá lớn (> 1024px). Hệ thống đã tự xử lý resize về 512px — nếu vẫn lỗi, giảm `VL_MAX_SIZE = 384` trong `app/core/vision.py`.

### ❌ Chatbot trả lời chậm (> 60s)
- Kiểm tra GPU Vast.ai còn đang chạy không (instance có thể tự tắt)
- Thuê lại instance và tạo lại SSH tunnel

---

## Cấu trúc thư mục dự án

```
Chatbot_Fashion/
│
├── main.py                        # Entry point — chạy server
├── docker-compose.yml             # Qdrant + Redis Stack
├── requirements.txt               # Python dependencies
│
├── app/
│   ├── config.py                  # Cấu hình tập trung (URLs, models, constants)
│   ├── api.py                     # FastAPI backend + SSE streaming
│   ├── static/
│   │   └── index.html             # Giao diện web chat
│   └── core/
│       ├── embeddings.py          # BGE-M3 embedding wrapper
│       ├── vector_store.py        # Qdrant + Layer B indexing
│       ├── llm.py                 # LLM (Qwen) + tất cả prompts
│       ├── vision.py              # Xử lý ảnh (Qwen2.5-VL)
│       ├── intent.py              # Phân loại intent người dùng
│       ├── outfit.py              # Logic phối đồ (Layer B)
│       ├── history.py             # Redis chat history + summarization
│       └── chains.py              # Lắp ráp RAG pipeline
│
├── data/
│   ├── metadata/                  # Dữ liệu sản phẩm .jsonl (không commit)
│   └── stylists/
│       ├── Layer_B_Female_Knowledge.json  # 880 rules phối đồ Nữ
│       └── Layer_B_Male_Knowledge.json    # 416 rules phối đồ Nam
│
├── notebooks/                     # Jupyter notebooks thực nghiệm
├── docs/                          # Tài liệu (file này nằm đây)
├── tests/
│   └── sample_images/             # Ảnh mẫu để test
└── storage/                       # Docker volumes (không commit)
    ├── qdrant/
    └── redis/
```

---

## Các port quan trọng

| Port | Dịch vụ | Ghi chú |
|---|---|---|
| `8000` | FastAPI API | Backend chatbot |
| `6333` | Qdrant REST API | Vector database |
| `6334` | Qdrant gRPC | Vector database |
| `6379` | Redis | Chat history storage |
| `8001` | RedisInsight | Giao diện quản lý Redis |
| `11434` | Ollama (via tunnel) | LLM & Embedding trên Vast.ai |
