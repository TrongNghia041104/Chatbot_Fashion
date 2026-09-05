# Ghi chú hướng cải tiến — Fashion RAG Chatbot (ViFashionCLIP-vi + BGE-M3)

> Góc nhìn đánh giá hệ thống hiện tại (đọc trực tiếp từ notebook), sắp xếp theo độ ưu tiên triển khai cho đồ án tốt nghiệp. Ưu tiên dựa trên 2 tiêu chí: **(1) tác động tới chất lượng/độ tin cậy câu trả lời**, **(2) chi phí implement trong quỹ thời gian còn lại + phần cứng RTX 3050 4GB VRAM**.

---

## TIER 0 — Đã chốt (không cần bàn thêm)
- **Image-to-image retrieval**: dùng thẳng **FashionCLIP gốc (teacher)** cho cả encode ảnh index lẫn ảnh query. Không dùng ViFashionCLIP-vi (student) cho nhánh ảnh — hợp lý vì student chỉ được distill để map **text tiếng Việt** vào không gian ảnh của teacher, bản thân nó không có vision encoder riêng biệt tốt hơn teacher.

---

## TIER 1 — Ưu tiên cao (ảnh hưởng trực tiếp đến chất lượng câu trả lời, chi phí implement thấp–vừa)

### 1. Reranking sau retrieval — **CHƯA có, đây là gap lớn nhất còn lại**
- Hiện tại: Qdrant trả top-30 → chỉ qua `diversity_filter_documents` (lọc trùng/brand), **không có bước chấm lại độ liên quan** giữa query và từng candidate.
- Vấn đề: cosine similarity của bi-encoder (ViFashionCLIP-vi) vốn kém chính xác hơn cross-encoder ở việc phân biệt các sản phẩm gần giống nhau — đặc biệt quan trọng khi bạn đã ghi nhận **recall degradation** so với baseline tiếng Anh.
- Đề xuất: thêm cross-encoder reranker sau khi lấy top-30 từ Qdrant, chỉ giữ lại top-5 đã rerank để đưa vào LLM.
  - Model gợi ý: `BAAI/bge-reranker-v2-m3` (multilingual, có tiếng Việt, nhẹ — chạy được trên 4GB VRAM ở batch nhỏ) hoặc `namdp-ptit/ViRanker` nếu cần thuần Việt hơn.
  - Vị trí chèn: ngay trong `DiversityFilteredRetriever._get_relevant_documents`, trước bước diversity filter (rerank trước, diversify sau, hoặc ngược lại tuỳ mục tiêu — nên thử nghiệm cả 2 thứ tự).
- Đây cũng là một **research contribution hợp lý** để đưa vào báo cáo: "reranking bù đắp cho recall degradation của student model".

### 2. Bộ eval tự động (RAGAS / LLM-as-judge) — **CHƯA có trong notebook**
- Hiện tại notebook không có cell nào đo faithfulness / answer relevancy / context precision, dù report có nhắc RAGAS.
- Không có eval tự động thì mọi thay đổi (thêm reranker, đổi threshold, đổi prompt) đều phải đánh giá bằng mắt — rất chậm và không có số liệu để đưa vào luận văn.
- Đề xuất tối thiểu: viết một script eval nhỏ trong notebook, dùng ~30–50 câu query test (đã có sẵn tinh thần này ở `TEST_CASES` cho intent) + `qwen3:4b` làm LLM-judge chấm 3 tiêu chí: có bịa sản phẩm không (faithfulness), có đúng nhu cầu không (relevancy), context có đủ không (precision).
- Nếu có RAGAS lib cài được: dùng RAGAS chuẩn để số liệu có tính học thuật hơn (recall@k, MRR cho retrieval; faithfulness/answer_relevancy cho generation).

### 3. Hybrid search (dense + sparse/BM25) cho Layer A text
- Hiện tại Layer A chỉ dùng dense retrieval thuần (ViFashionCLIP-vi cosine similarity). Với truy vấn có từ khoá chính xác (mã sản phẩm, tên brand, số cụ thể như "size L", "dưới 200k") thì dense embedding thường yếu hơn sparse.
- Đề xuất: thêm BM25/sparse search song song (Qdrant hỗ trợ sparse vectors native), rồi fusion điểm (RRF — Reciprocal Rank Fusion) với dense score. Đỡ tốn công hơn train lại embedding model, và giải quyết đúng điểm yếu recall bạn đang gặp.

---

## TIER 2 — Ưu tiên trung bình (cải thiện độ ổn định/trải nghiệm, effort vừa)

### 4. Guardrail chống hallucination mạnh hơn ở tầng generation
- Prompt hiện tại đã có "QUY TẮC TỐI CAO: chỉ dùng data trong context" — tốt, nhưng đây là **soft constraint** (LLM 4B vẫn có thể bịa nếu context yếu).
- Đề xuất: thêm bước **post-check** đơn giản bằng code — sau khi LLM trả lời, regex/parse ra các `[MÃ_SP]` LLM nhắc tới, đối chiếu với product_id thật sự có trong context đã gửi. Nếu LLM bịa mã không tồn tại → tự động fallback (báo lỗi hoặc gọi lại với prompt nhắc nhở). Đây là guardrail rẻ, không cần model thêm.

### 5. Streaming answer_relevancy check / early-exit khi Layer B không khớp
- `build_outfit_context` trả `""` khi không tìm được rule → hiện tại fallback về `search`. Cơ chế fallback đã có, ổn — nhưng nên log lại tần suất fallback này để biết Layer B miss bao nhiêu % câu hỏi outfit thực tế (phục vụ đánh giá coverage của 1,296 rules).

### 6. Cache embedding cho query lặp lại
- Intent detection + rule matching gọi embed_query khá thường xuyên trong 1 phiên chat. Có thể thêm `functools.lru_cache` nhỏ cho các câu hỏi lặp/gần giống để giảm latency, nhất là khi chạy trên GPU 4GB (không có nhiều đầu để chạy song song).

### 7. Structured output thay vì parse text tự do
- `analyze_person_image`, `detect_intent_llm` hiện đang parse output LLM bằng string matching (`"DÁNG" in upper`, tìm `":"` ...). Khá dễ vỡ nếu LLM đổi format output.
- Đề xuất: dùng `format="json"` của Ollama (Qwen3 hỗ trợ structured output/JSON mode) để LLM trả JSON có schema cố định, parse an toàn hơn bằng `json.loads` thay vì regex/string split.

---

## TIER 3 — Ưu tiên thấp / nice-to-have (nếu còn thời gian, hoặc để mở hướng phát triển trong phần "Hướng phát triển" của báo cáo)

### 8. Logging có cấu trúc thay vì print
- Hiện tại TTFT/latency chỉ `print` ra màn hình, mất khi tắt notebook. Ghi log JSON theo từng lượt chat (intent, latency, có fallback không, rerank score...) sẽ giúp làm biểu đồ phân tích cho phần Chapter 4 đánh giá hệ thống.

### 9. Quantization / tối ưu inference cho VRAM 4GB
- Đã dùng Ollama (đã quantize sẵn) nên phần LLM/VLM ổn. Điểm có thể tối ưu thêm: ViFashionCLIP-vi + FashionCLIP đang chạy full-precision (fp32) trên GPU theo code — có thể chuyển sang fp16 (`.half()`) để giảm VRAM và tăng tốc encode, quan trọng khi cần chạy đồng thời cả 2 model (text + image) trong 4GB.

### 10. A/B so sánh có/không reranker, có/không hybrid search
- Sau khi làm xong Tier 1, nên có 1 bảng so sánh định lượng (recall@5, faithfulness score) giữa các phiên bản pipeline — vừa chứng minh cải tiến có tác dụng thật, vừa là nội dung tốt cho Chapter 4 của luận văn.

### 11. Đánh giá lại category mapping Layer B → Layer A bằng dữ liệu thật
- `CATEGORY_MAPPING` và `PHU_KIEN_KEYWORD_ROUTER` hiện là từ điển tay (rule-based, khớp theo keyword tiếng Anh). Nếu có thời gian, review lại coverage của các keyword này trên tập 65k sản phẩm thật để tránh sản phẩm bị rơi vào "Phụ kiện hỗ trợ" (default fallback) quá nhiều.

---

## Tóm tắt thứ tự làm nếu chỉ chọn 3 việc quan trọng nhất
1. **Reranker** (bù recall degradation, dễ làm, tác động rõ) 
2. **Eval script (RAGAS-lite hoặc LLM-judge)** (bắt buộc phải có để đo được cải tiến #1 có hiệu quả không, và để viết Chapter 4)
3. **Hybrid search (BM25 + dense)** (giải quyết lớp lỗi khác với reranker — truy vấn có từ khoá chính xác)
