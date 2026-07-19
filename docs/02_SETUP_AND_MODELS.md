# Cài đặt, dịch vụ và model

## Thành phần cần chạy

| Thành phần | Endpoint mặc định | Vai trò |
|---|---|---|
| FastAPI | `http://localhost:8000` | Backend và web UI |
| Qdrant | `http://localhost:6333` | Vector database |
| Redis | `redis://localhost:6379` | Lịch sử LangChain |
| Ollama | `http://localhost:11434` | Qwen LLM, Qwen-VL, BGE-M3 |
| ViFashionCLIP service | `http://localhost:18080` | Text embedding Layer A khi backend=`remote` |

Các endpoint localhost có thể là dịch vụ local hoặc đầu SSH tunnel đến VastAI. Không suy ra vị trí GPU chỉ từ URL; kiểm tra cách tunnel được khởi động.

## Model và embedding space

| Tác vụ | Model mặc định | Nơi cấu hình | Vector |
|---|---|---|---:|
| Sinh câu trả lời/router fallback | `qwen3:4b-instruct` | `LLM_MODEL` | — |
| Hiểu ảnh/profile | `qwen2.5vl:3b` | `VISION_MODEL` | — |
| Rule Layer B | `bge-m3` | `EMBEDDING_MODEL` | 1024 |
| Product text Layer A | ViFashionCLIP checkpoint/service | `PRODUCT_EMBEDDING_BACKEND` | 512 |
| Product image Layer A | `patrickjohncyh/fashion-clip` | `TEACHER_MODEL_NAME` | 512 |

Text product và image product phải cùng đúng embedding family/space với collection đã index. Không đổi model query nếu chưa index lại collection tương ứng.

## Khởi động

```powershell
docker compose up -d
python main.py
```

Mở `http://127.0.0.1:8000`. Nếu dùng VastAI, tạo SSH tunnel cho Ollama và/hoặc embedding service trước khi chạy app.

## Biến môi trường quan trọng

```text
OLLAMA_BASE_URL
LLM_MODEL
VISION_MODEL
EMBEDDING_MODEL
PRODUCT_EMBEDDING_BACKEND=local|remote
VIFASHIONCLIP_SERVICE_URL
REMOTE_EMBEDDING_FALLBACK_LOCAL
ENABLE_PRODUCT_RERANKER
HISTORY_ENABLE_SUMMARIZATION
PRELOAD_IMAGE_ENCODER
DEBUG_ROUTER_TRACE
```

Mặc định hiện tại ưu tiên tốc độ demo: history summarization tắt, reranker tắt, FashionCLIP image encoder preload khi startup.

## Đường dẫn dữ liệu cần kiểm tra trên máy mới

`IMAGES_DIR` và `PRODUCT_IMAGE_ROOT` trong `app/config.py` đang là đường dẫn tuyệt đối tới bộ ảnh 65k. Người tiếp quản phải đổi hai biến này hoặc đưa chúng ra biến môi trường trước khi chuyển máy.

Kiểm tra thêm:

- File metadata Layer A có tồn tại.
- Hai JSON Layer B nam/nữ có tồn tại.
- Checkpoint ViFashionCLIP có tồn tại nếu dùng backend local.
- Bốn collection Qdrant đúng tên và đúng dimension.

## Smoke test

```powershell
python -m compileall -q app
python -m unittest discover -s tests -p "test_*.py" -v
```

Sau đó thử lần lượt: lời chào, text search, text outfit, image search, image outfit và profile analysis. Bật Developer Mode để xác nhận `route`, `certainty`, `model_calls` và `timings`.

## Lỗi setup thường gặp

| Triệu chứng | Kiểm tra trước |
|---|---|
| Text query treo lâu | tunnel Ollama/ViFashionCLIP, timeout, model đang load |
| Card không có ảnh | `IMAGES_DIR`, metadata `images/image_url`, mount `/images` |
| Qdrant dimension error | model query không khớp collection đã index |
| Lượt đầu ảnh rất chậm | preload FashionCLIP hoặc thiếu RAM/VRAM |
| History lỗi | Redis chưa chạy; kiểm tra cổng 6379 |

