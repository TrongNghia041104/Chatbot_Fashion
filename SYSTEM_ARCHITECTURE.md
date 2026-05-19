# 🧠 Fashion RAG Chatbot — Kiến trúc & Luồng xử lý hệ thống

## Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MÁY LOCAL (Windows)                            │
│                                                                         │
│   Browser (index.html)  ◄──SSE Stream──►  FastAPI (api.py :8000)       │
│                                                  │                      │
│                              ┌───────────────────┤                      │
│                              ▼                   ▼                      │
│                    Qdrant (:6333)          Redis (:6379)                │
│                    [Docker]                [Docker]                     │
│                              │                                          │
│         SSH Tunnel :11434 ◄──┘                                          │
└─────────────────────────────────────────────────────────────────────────┘
                    │ SSH Tunnel
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MÁY VAST.AI (GPU)                              │
│                                                                         │
│   Ollama (:11434)                                                       │
│   ├── bge-m3          (Embedding 1024 chiều)                            │
│   ├── qwen3:4b-instruct  (LLM chính — Chat & Intent)                   │
│   └── qwen2.5vl:3b   (Vision LLM — Phân tích ảnh)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Các thành phần hệ thống

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| **Frontend** | HTML + Vanilla JS | Giao diện chat, upload ảnh, hiển thị sản phẩm |
| **Backend API** | FastAPI + SSE | Điều phối luồng xử lý, streaming response |
| **Core Logic** | `chatbot_core.py` | Toàn bộ logic RAG, intent, outfit |
| **Vector DB** | Qdrant (Docker) | Lưu trữ embedding sản phẩm & rules Layer B |
| **Chat History** | Redis (Docker) | Lưu lịch sử hội thoại theo session |
| **Embedding** | BGE-M3 via Ollama | Chuyển văn bản → vector 1024 chiều |
| **LLM chính** | Qwen3-4B-Instruct | Sinh câu trả lời, phân loại intent |
| **Vision LLM** | Qwen2.5-VL-3B | Phân tích ảnh người / ảnh sản phẩm |

---

## Luồng xử lý chi tiết

### 1. Khởi tạo phiên làm việc

```
Browser → POST /api/session → Tạo session_id mới (UUID)
                            → Lưu vào localStorage
```

---

### 2. Người dùng gửi tin nhắn (± ảnh)

```
Browser gửi FormData:
  ├── message: "tôi muốn mặc đồ đi tiệc"
  ├── session_id: "abc-123-..."
  └── image: <file> (nếu có)
         │
         ▼
POST /api/chat  →  SSE Stream bắt đầu
```

---

### 3. Xử lý ảnh (nếu có)

```
Image Upload
    │
    ▼
detect_image_type(image, query)  →  Qwen2.5-VL
    │
    ├── "PERSON" ──► analyze_person_image()  →  Qwen2.5-VL
    │                   ├── Dáng người (8 loại)
    │                   ├── Tone da (4 loại)
    │                   └── Nhận xét phối đồ
    │                   → Lưu vào user_profile (session)
    │                   → SSE: person_analyzed
    │                   → Trả lời → KẾT THÚC (không chạy RAG)
    │
    └── "PRODUCT" ─► caption_product_image()  →  Qwen2.5-VL
                        └── Mô tả sản phẩm bằng tiếng Việt
                        → Ghép vào final_query: "{caption}. Yêu cầu: {message}"
```

---

### 4. Phát hiện Intent — Hybrid 2 tầng

```
final_query
    │
    ▼
Tầng 1: Keyword matching (không gọi LLM — nhanh)
    ├── DEFINITE_OUTFIT   → ["phối đồ", "mix match", "mặc với gì", ...]
    ├── DEFINITE_SEARCH   → ["còn hàng không", "giá bao nhiêu", ...]
    ├── DEFINITE_GREETING → ["xin chào", "hello", "hi bạn", ...]
    └── DEFINITE_CHITCHAT → ["cảm ơn", "tạm biệt", "bye", ...]
    
    Nếu không khớp keyword nào:
    │
    ▼
Tầng 2: LLM classify  →  Qwen3-4B-Instruct
    Prompt gồm: câu hỏi + context bot message trước
    Output: OUTFIT / SEARCH / CHITCHAT / GREETING
    │
    ▼
SSE: intent_detected {intent, gender}
```

---

### 5. Phát hiện giới tính & Quản lý Profile

```
detect_gender(final_query)
    ├── Tìm từ khóa: ["nam", "con trai", "anh", "bạn trai", ...]
    ├── Nếu detect "male" → Lưu vào user_profile["gender"]
    └── Ưu tiên lấy gender đã lưu trước đó (không bị quên)

user_profile = {
    "gender":     "female" | "male",
    "dang_nguoi": "Dáng quả lê" | ...,   ← từ phân tích ảnh người
    "tone_da":    "Da sáng" | ...,        ← từ phân tích ảnh người
}
```

---

### 6A. Luồng GREETING / CHITCHAT (không dùng RAG)

```
intent == "greeting"  → get_greeting_response()  → SSE: token → done
intent == "chitchat"  → get_chitchat_response()  → SSE: token → done
```

---

### 6B. Luồng OUTFIT — Layer B → Layer A

```
build_outfit_context(query, gender, profile)
    │
    ▼
[LAYER B — Tìm công thức phối đồ]
    │
    ├── BƯỚC 1: find_matching_rule()
    │     Embed query bằng BGE-M3
    │     Tìm trong Qdrant collection: layer_b_female | layer_b_male
    │     Filter theo dáng người + tone da (nếu có trong profile)
    │     Fallback 1: bỏ tone da, chỉ giữ dáng
    │     Fallback 2: không filter, lấy gần nhất
    │     → base_rule: {phong_cach, boi_canh, goi_y_phoi_cung, ly_do_tu_van}
    │
    └── BƯỚC 2: find_outfit_details()
          Lặp qua từng món trong goi_y_phoi_cung
          Tìm exact match trong knowledge base → semantic search fallback
          → outfit_rules: {Áo → rule, Quần → rule, Giày → rule, ...}
    │
    ▼
[CATEGORY MAPPING — Dịch Layer B sang Layer A]
    │
    CATEGORY_MAPPING: "Áo mặc trong" → ["Áo"]
                      "Quần/Chân váy" → ["Quần", "Chân váy"]
                      "Phụ kiện" → PHU_KIEN_KEYWORD_ROUTER (soi từ khóa EN)
    │
    ▼
[LAYER A — Tìm sản phẩm thực tế]
    │
    get_products_for_outfit() → Qdrant collection: fashion_products_bge_m3
    Filter theo category + similarity search
    → products: list[Document] (page_content + metadata)
    │
    ▼
[TRÍCH XUẤT ẢNH]
    metadata.images → images_data list
    → SSE: product_images  (gửi TRƯỚC khi stream text)
    │
    ▼
Ghép context string → outfit_chain_with_history.stream()
    → SSE: token (stream từng chữ)
    → SSE: done
```

---

### 6C. Luồng SEARCH — RAG Pipeline

```
final_query
    │
    ▼
[QUERY REWRITING]
create_history_aware_retriever
    LLM (Qwen3) đọc lịch sử hội thoại → viết lại câu hỏi thành độc lập
    Ví dụ: "Còn màu khác không?" → "Áo thun trắng nữ trên có màu khác không?"
    │
    ▼
[RETRIEVAL]
Qdrant similarity_score_threshold
    collection: fashion_products_bge_m3
    BGE-M3 embed query → top-5 documents (score ≥ 0.7)
    → SSE: product_images (ảnh từ retrieved docs)
    │
    ▼
[GENERATION]
create_stuff_documents_chain
    Nhồi documents vào QA_PROMPT
    LLM (Qwen3) sinh câu trả lời (anti-hallucination prompt)
    → SSE: token (streaming)
    → SSE: done
```

---

### 7. Quản lý lịch sử hội thoại

```
RedisChatMessageHistory
    ├── Lưu theo session_id
    ├── Giữ tối đa 6 messages gần nhất (3 lượt chat)
    └── Tự cắt bớt nếu vượt quá → tránh context quá dài
```

---

### 8. Streaming SSE — Các event type

| Event | Dữ liệu | Ý nghĩa |
|---|---|---|
| `person_analyzed` | `{dang_nguoi, tone_da, nhan_xet}` | Kết quả phân tích ảnh người |
| `product_captioned` | `{caption}` | Mô tả ảnh sản phẩm |
| `intent_detected` | `{intent, gender}` | Intent đã phân loại |
| `product_images` | `{images: [{product_id, category, images:[url]}]}` | Ảnh sản phẩm gợi ý |
| `token` | `{content}` | Từng token LLM stream |
| `done` | `{ttft, total}` | Hoàn tất, kèm thời gian |
| `error` | `{message}` | Lỗi xảy ra |

---

## Cấu trúc dữ liệu Qdrant

### Collection `fashion_products_bge_m3`
```
Vector size: 1024 (BGE-M3 Cosine)
Payload (metadata):
  ├── product_id  : string
  ├── category    : string  ("Áo", "Quần", "Giày", ...)
  ├── department  : string  ("Nam", "Nữ", "Unisex")
  ├── brand       : string
  ├── price       : number  (VNĐ)
  └── images      : list[string]  (URL ảnh sản phẩm)
```

### Collection `layer_b_female` / `layer_b_male`
```
Vector size: 1024 (BGE-M3 Cosine)
Payload (rule):
  ├── rule_key      : "Áo mặc trong | Áo thun basic cổ tròn"
  ├── phong_cach    : "Thanh lịch"
  ├── boi_canh      : "Đi làm"
  ├── dang_nguoi    : "Dáng quả lê"
  ├── tone_da       : "Da trung bình"
  ├── goi_y_phoi_cung: ["Áo mặc trong", "Quần/Chân váy", "Giày dép"]
  └── ly_do_tu_van  : "Áo thun basic giúp cân bằng phần hông..."
```

---

## Sơ đồ tổng thể (text)

```
User Input (text + image?)
        │
        ├─[có ảnh]─► Qwen2.5-VL
        │               ├── PERSON → profile update → trả lời ngay
        │               └── PRODUCT → caption → ghép vào query
        │
        ▼
    Intent Detection (2-tier)
        ├── GREETING / CHITCHAT → trả lời trực tiếp
        │
        ├── OUTFIT ─────────────────────────────────────────────┐
        │   Layer B (Qdrant)                                    │
        │   → công thức phối đồ                                 │
        │   → chi tiết từng món                                 │
        │   Layer A (Qdrant)                                    │
        │   → sản phẩm thực tế                                  │
        │   → extract ảnh → SSE product_images                 │
        │   → Qwen3 sinh lời tư vấn (streaming)                 │
        │                                                       │
        └── SEARCH ─────────────────────────────────────────────┘
            Query Rewriting (Qwen3)
            → Qdrant similarity search
            → extract ảnh → SSE product_images
            → Qwen3 sinh câu trả lời (streaming)
                    │
                    ▼
            Redis lưu lịch sử session
                    │
                    ▼
            SSE stream về Frontend
            → Hiển thị ảnh sản phẩm + text tư vấn
```
