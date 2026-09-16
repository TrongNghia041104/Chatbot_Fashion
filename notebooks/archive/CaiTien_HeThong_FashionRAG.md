# Danh sách cải tiến hệ thống Fashion RAG Chatbot

## Phân loại theo mức độ ưu tiên

---

## 🔴 Nhóm 1 — Phải có (ảnh hưởng trực tiếp độ tin cậy)

### 1.1 Intent Detection — thu hẹp keyword, mở rộng LLM

**Vấn đề hiện tại:** Keyword list quá rộng gây false positive.
Ví dụ: "thời tiết se lạnh mặc gì" bị chặn bởi keyword "thời tiết" → OUT_OF_SCOPE sai.

**Giải pháp:** Chỉ giữ keyword 100% chắc chắn ở Tier 1. Bỏ `DEFINITE_OUT_OF_SCOPE`.
Để LLM tự phân biệt ngữ cảnh ở Tier 2.

**Thứ tự check đúng:**
```
greeting → chitchat → profile_inquiry → outfit → search → LLM classify
```
Greeting/chitchat phải check trước để tránh bị override bởi outfit/search.

---

### 1.2 Thêm intent `profile_inquiry`

**Vấn đề:** User hỏi lại thông tin đã chia sẻ ("dáng người tôi là gì nhỉ?")
→ rơi vào `search` → RAG chạy → không tìm thấy gì → LLM bịa hoặc nói không biết.
Trong khi `user_profile` đang lưu sẵn thông tin đó.

**Giải pháp:** Route về hàm đọc thẳng `user_profile` dict, không cần gọi LLM lớn.

---

### 1.3 Thêm intent `out_of_scope`

**Vấn đề:** Câu hỏi ngoài thời trang ("thời tiết hôm nay", "1+1=?")
→ rơi vào `search` → RAG chạy → LLM hallucinate hoặc trả lời vô nghĩa.

**Giải pháp:** LLM classify phát hiện OUT_OF_SCOPE → trả về template từ chối lịch sự,
không để RAG chạy.

---

### 1.4 Thêm intent `unclear` và cơ chế hỏi lại

**Vấn đề:** Câu mơ hồ ("tôi muốn thứ gì đó đẹp", "cái kia thế nào?")
→ bị classify sai → pipeline chạy không đúng hướng.

**Giải pháp:**
- LLM trả về `UNCLEAR` khi không đủ thông tin phân loại
- Gọi `get_clarification_question()` — hỏi lại có định hướng (không hỏi chung chung)
- Đếm `unclear_count`: sau 2 lần UNCLEAR liên tiếp → fallback về search,
  tránh loop vô hạn

---

### 1.5 Thêm `last_bot_msg` vào main loop

**Vấn đề:** `detect_intent_llm()` nhận `last_bot_msg` để hiểu context
nhưng main loop không truyền vào → LLM classify không có ngữ cảnh.

**Giải pháp:** Thêm biến `last_bot_msg = ""`, cập nhật sau mỗi lượt bot trả lời,
truyền vào `detect_intent(final_query, last_bot_msg)`.

---

### 1.6 Thông báo rõ khi fallback outfit → search

**Vấn đề:** Khi `build_outfit_context()` trả về rỗng, hệ thống âm thầm
chuyển sang search mà user không biết.

**Giải pháp:** In thông báo trước khi fallback:
> "Mình chưa có công thức phối đồ cụ thể cho yêu cầu này,
> nhưng để mình tìm sản phẩm phù hợp cho bạn nhé!"

---

### 1.7 Validate input ảnh không có text

**Vấn đề:** User gửi ảnh mà không gõ text → `user_input = ""`
→ `detect_image_type(raw_img, "")` không có context → kết quả không ổn định.

**Giải pháp:** Check `final_query.strip()` sau khi ghép caption.
Nếu rỗng → hỏi lại user thay vì tiếp tục pipeline.

---

### 1.8 Nâng score threshold outfit 0.20 → 0.30

**Vấn đề:** Threshold 0.20 quá thấp cho ViFashionCLIP 512d
→ sản phẩm không liên quan lọt vào outfit context → LLM gợi ý sai.

**Giải pháp:** Đổi `if score >= 0.20` thành `if score >= 0.30` trong
`get_products_for_outfit()`. Test thực tế để điều chỉnh nếu cần.

---

## 🟡 Nhóm 2 — Nên có (cải thiện chất lượng rõ rệt)

### 2.1 Cross-encoder Reranker

**Model:** `BAAI/bge-reranker-v2-m3`

**Luồng:** Qdrant trả top-10 (bi-encoder, nhanh)
→ Reranker chọn lại top-3 (cross-encoder, chính xác hơn)
→ Đưa vào LLM

**Lý do:** Bi-encoder embed query và document riêng lẻ.
Cross-encoder nhìn query + document cùng lúc → chính xác hơn nhiều.

---

### 2.2 History Summarization (đã có, cần kiểm tra hoạt động đúng)

**Cơ chế hiện tại:** Khi history > 8 messages → LLM tóm tắt phần cũ,
giữ 4 messages gần nhất + 1 SystemMessage tóm tắt.

**Cần kiểm tra:** `summarize_history()` và `get_message_history()` đang
được gọi đúng không, hay chỉ có trong code mà chưa được invoke.

---

### 2.3 Fine-tune BGE-M3 trên domain thời trang tiếng Việt

**Bài toán giải quyết:** Cross-lingual gap giữa query tiếng Việt và
mô tả sản phẩm tiếng Anh trong Layer B (quy tắc phối đồ).

**Pipeline:**
1. Dùng Qwen3 sinh synthetic queries tiếng Việt từ 49k sản phẩm (~5k cặp)
2. Fine-tune với `MultipleNegativesRankingLoss`, 3 epochs, fp16
3. Re-index Qdrant collection `layer_b_female` và `layer_b_male`

**Thời gian ước tính trên RTX 3050:**
- Sinh data: ~3-4 tiếng
- Fine-tune: ~1-2 tiếng

---

### 2.4 Output Hallucination Check

**Cơ chế:** Sau khi LLM trả về response, dùng regex trích xuất mã SP
được đề cập → so sánh với mã SP trong retrieved_docs.

**Xử lý:** Nếu có mã SP lạ (không trong retrieved_docs) → log warning.
Không crash hệ thống, chỉ ghi log để debug.

---

## 🟢 Nhóm 3 — Nâng cao (production-grade)

### 3.1 Conversation State Tracking

**4 trạng thái hành trình mua sắm:**
```
exploring   → User đang khám phá, chưa có SP cụ thể
comparing   → User đang so sánh 2-3 sản phẩm
deciding    → User gần ra quyết định
post_select → User đã chọn, có thể upsell phụ kiện
```

**Ứng dụng:** Điều chỉnh tone và strategy tư vấn theo từng state.
Ví dụ: state `post_select` → chủ động gợi ý phụ kiện phù hợp.

---

### 3.2 Contextual Memory ngoài chat history

**Lưu vào Redis (ngoài message history):**
```python
user_context = {
    "liked_products":    ["B0B9R3P94M"],   # SP đã quan tâm
    "disliked_aspects":  ["quá đắt", "không thích màu đỏ"],
    "stated_occasion":   "đi làm văn phòng",
    "stated_budget":     "dưới 500k",
    "size_mentioned":    "M",
}
```

**Lợi ích:** LLM biết những gì user đã nói → không hỏi lại,
tư vấn cá nhân hóa hơn qua nhiều lượt chat.

---

### 3.3 Health Check khi khởi động

**Cần check:**
- Qdrant có kết nối được không?
- Redis có kết nối được không?
- Ollama có đang chạy không? Model đã load chưa?

**Xử lý:** Nếu service nào đó chết → thông báo rõ ràng thay vì crash
im lặng giữa chừng khi đang chat.

---

### 3.4 Retry Logic cho Ollama Timeout

**Vấn đề hiện tại:** `timeout=120` trong `ChatOllama` nhưng không có retry.
Nếu Qwen3 đang load model lần đầu (> 120s) → crash.

**Giải pháp:** Thêm retry 2-3 lần với exponential backoff.
Hoặc tăng timeout lên 300s cho lần khởi động đầu.

---

### 3.5 Auto Evaluation Framework

**Chạy ngầm sau mỗi lượt chat:**
- Relevance: response có trả lời đúng câu hỏi không?
- Groundedness: có bịa thông tin ngoài retrieved context không?
- Helpfulness: có thực sự giúp ích cho user không?

**Thang điểm:** 1-5 cho mỗi tiêu chí, log vào file để phân tích sau.

---

## Kiến trúc Embedding — vai trò từng model

| Model | Input | Bài toán | Collection Qdrant |
|---|---|---|---|
| BGE-M3 | Text | Text-to-text (Layer B) | `layer_b_female`, `layer_b_male` |
| ViFashionCLIP text encoder | Text tiếng Việt | Text-to-text (Layer A) | `fashion_products_vifashionclip_vi_65k` |
| FashionCLIP vision encoder | Ảnh sản phẩm | Image-to-image | Cần tạo: `fashion_clip_vi_images` |

**Lưu ý quan trọng:** ViFashionCLIP text encoder và FashionCLIP vision encoder
cùng không gian vector 512d — cho phép dùng 1 collection Qdrant để search
từ cả text query lẫn image query.

**Việc còn thiếu:** Thêm `ViFashionCLIPVisionEmbeddings` class vào notebook
để load FashionCLIP vision encoder, pre-embed ảnh catalog offline,
và dùng khi user gửi ảnh sản phẩm.

---

## Intent taxonomy đầy đủ — 7 intent

| Intent | Pipeline | Gọi LLM lớn? | Latency |
|---|---|---|---|
| `greeting` | Template response | ❌ | ~0ms |
| `chitchat` | Template response | ❌ | ~0ms |
| `out_of_scope` | Template từ chối | ❌ | ~0ms |
| `profile_inquiry` | Đọc `user_profile` dict | ❌ | ~0ms |
| `unclear` | Hỏi lại (clarification) | ✅ (nhỏ, ~0.3s) | ~300ms |
| `product_search` | BGE-M3 → Qdrant → Qwen3 4B | ✅ | ~3-8s |
| `outfit_advice` | Layer B × 2 → Qdrant → Qwen3 4B | ✅ | ~5-10s |

---

## Thứ tự ưu tiên triển khai

```
Tuần này:
  ✅ 1.1  Fix thứ tự keyword trong detect_intent()
  ✅ 1.2  Thêm profile_inquiry
  ✅ 1.3  Thêm out_of_scope
  ✅ 1.4  Thêm unclear + clarification
  ✅ 1.5  Thêm last_bot_msg vào main loop
  ✅ 1.6  Thông báo fallback outfit → search
  ✅ 1.7  Validate input ảnh rỗng
  ✅ 1.8  Nâng threshold 0.20 → 0.30

Sau đó:
  🔲 2.1  Cross-encoder reranker
  🔲 2.3  Fine-tune BGE-M3 Layer B
  🔲 2.4  Hallucination check
  🔲      ViFashionCLIP vision encoder cho image retrieval

Dài hạn:
  🔲 3.1  Conversation state tracking
  🔲 3.2  Contextual memory
  🔲 3.3  Health check
  🔲 3.4  Retry logic
  🔲 3.5  Auto evaluation framework
```

---

*Ghi chú: Nhóm 1 đã được áp dụng vào file `Chatbot_RAG_MultiModal_ViFashionCLIP_v2.ipynb`*
