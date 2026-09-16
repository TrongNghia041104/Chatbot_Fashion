# RAG Debug Playbook

> **Tài liệu bổ trợ:** Dùng cùng [`08_DEBUGGING_AND_LOGGING.md`](08_DEBUGGING_AND_LOGGING.md). Tên field/runtime contract mới nhất nằm trong bộ tài liệu đánh số từ [`00_README_FIRST.md`](../00_README_FIRST.md).

Mục tiêu của debug RAG không phải là nhìn câu trả lời cuối rồi đoán sai ở đâu. Mục tiêu là tách hệ thống thành từng tầng, in output thật của từng tầng, rồi hỏi: tầng này đã đưa đúng tín hiệu cho tầng sau chưa?

## 1. Stack là gì?

Trong dự án này, "stack" là bộ công nghệ đang ghép lại thành chatbot:

| Thành phần | Vai trò trong hệ thống |
|---|---|
| LangChain | Lớp điều phối: prompt, retriever, chain, history, document format. Nó không tự là database hay model. |
| Qdrant | Vector database: lưu 65k vector sản phẩm, vector ảnh, vector rule stylist; trả về các điểm gần query nhất. |
| Embedding model | Biến text/ảnh thành vector. Dự án dùng ViFashionCLIP cho sản phẩm, FashionCLIP cho ảnh, BGE-M3 cho rule Layer B. |
| Reranker | Chấm lại các ứng viên top-k sau khi Qdrant lấy ra. Qdrant nhanh nhưng thô; reranker chậm hơn nhưng đọc kỹ hơn. |
| LLM | Viết câu trả lời cuối dựa trên context. Nếu retrieval sai hoặc context thiếu, LLM rất dễ trả lời lệch/hallucinate. |
| FastAPI | API phục vụ web/chat. |

Điểm cần nhớ: LangChain là "ống dẫn", Qdrant là "kho tìm kiếm vector", embedding là "cách biến dữ liệu thành tọa độ", reranker là "ban giám khảo vòng 2", còn LLM là "người viết câu trả lời".

## 2. Bản đồ pipeline hiện tại

Luồng text search chính:

```text
message
  -> validate_user_query
  -> route_user_request
  -> active_query / rewrite_query
  -> Qdrant Layer A top-30
  -> optional rerank top-15
  -> diversity_filter_documents, tối đa 5 docs
  -> QA_PROMPT + LLM
  -> grounding check product_id
  -> chat_turns_research_demo_v3.jsonl
```

Luồng outfit:

```text
message
  -> route_user_request = outfit_advice
  -> find_matching_rule ở Layer B
  -> find_outfit_details
  -> map category Layer B sang category Layer A
  -> search sản phẩm thật trong Layer A
  -> outfit_prompt + LLM
  -> grounding check product_id
```

## 3. Cách chạy debug từng tầng

Chạy một query sản phẩm:

```powershell
python scripts/debug_rag_pipeline.py --query "tìm áo sơ mi trắng đi làm cho nữ"
```

Chạy query outfit kèm profile:

```powershell
python scripts/debug_rag_pipeline.py --query "phối đồ đi tiệc cho dáng quả lê" --profile-json "{\"dang_nguoi\":\"Dáng quả lê\"}" --force-outfit
```

Chạy bộ case mẫu và lưu output:

```powershell
python scripts/debug_rag_pipeline.py --cases tests/retrieval_debug_cases.example.jsonl --json-out eval_outputs/debug_run.jsonl
```

Nếu Ollama router chậm hoặc chưa bật, debug route bằng keyword trước:

```powershell
python scripts/debug_rag_pipeline.py --query "cho tôi xem váy đen đi tiệc" --router-mode keyword
```

Trước khi debug retrieval, kiểm tra service nền:

```powershell
python scripts/debug_rag_pipeline.py --check-services
```

Kết quả cần có:

```text
Qdrant  : OK
Ollama  : OK
Embedder: OK
```

Nếu Qdrant fail, bật Qdrant bằng Docker Compose trong thư mục project:

```powershell
docker compose up -d qdrant
```

Nếu Embedder fail, kiểm tra service ViFashionCLIP ở `localhost:18080` hoặc tunnel Vast.ai. Nếu chỉ muốn xem router trước, dùng `--route-only` để không gọi retrieval.

## 4. Đọc output như thế nào?

Nhìn theo thứ tự này:

1. Router có chọn đúng route không?
   Nếu câu "phối đồ" bị route thành product_search, lỗi ở intent/router.

2. `active_query` có đúng ý không?
   Nếu rewrite query mất màu, mất dịp, mất giới tính, retrieval sẽ lệch dù Qdrant hoạt động đúng.

3. Qdrant raw candidates có đúng nhóm hàng không?
   Nếu top-30 đã sai nhiều, nghi embedding, dữ liệu page_content, metadata category, hoặc query quá mơ hồ.

4. Reranker có đẩy sản phẩm đúng lên không?
   Nếu Qdrant có sản phẩm đúng nhưng reranker đẩy xuống, nghi reranker hoặc text đưa vào reranker chưa đủ tốt.

5. Diversity filter có làm mất sản phẩm tốt không?
   Nếu nhiều món cùng brand/sản phẩm lặp lại, filter hữu ích. Nếu filter bỏ hết nhóm tốt, cần chỉnh `PRODUCT_SEARCH_BRAND_LIMIT` hoặc dedupe key.

6. Final docs sent to LLM có đủ bằng chứng không?
   Nếu final docs sai, đừng sửa prompt trước. LLM chỉ đang viết dựa trên context sai.

7. Log sau mỗi lượt chat:
   `logs/chat_turns_research_demo_v3.jsonl` giờ có thêm `raw_query`, `route_confidence`, `route_reason`, `retrieved_count`, `retrieved_product_ids`.

## 5. Checklist lỗi hay gặp

| Hiện tượng | Nghi tầng nào trước |
|---|---|
| Trả lời lặp nhiều món giống nhau | diversity filter, product_id trùng, brand limit, hoặc dữ liệu có nhiều variant giống nhau |
| Trả lời đúng format nhưng sai sản phẩm | retrieval/rerank trước, prompt sau |
| Bịa mã sản phẩm | grounding check, context rỗng/yếu, prompt schema |
| Câu phối đồ chỉ tìm sản phẩm đơn lẻ | router chọn sai route |
| Outfit có rule đúng nhưng sản phẩm sai | category mapping Layer B -> Layer A hoặc `score >= 0.30` |
| Không lấy được sản phẩm dù data có | query rewrite, metadata filter, embedding, threshold |

## 6. Cách tự tạo test case

Mỗi case nên có một mục tiêu rõ:

```json
{"query":"tìm áo khoác denim nam","expected_route":"product_search","expected_categories":["Áo khoác"],"expected_product_ids":[]}
```

Khi chưa biết `expected_product_ids`, cứ để trống và dùng output debug để chọn 3-5 case chuẩn. Sau đó mới dùng chúng làm bộ regression nhỏ để so sánh trước/sau khi tối ưu.
