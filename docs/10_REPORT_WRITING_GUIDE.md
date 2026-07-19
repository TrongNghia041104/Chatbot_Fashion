# Hướng dẫn viết phần kiến trúc và luồng xử lý trong báo cáo

Tài liệu này dành cho nhóm viết khóa luận. Nội dung chi tiết kỹ thuật vẫn lấy từ các file `01`–`09`; không copy toàn bộ code vào báo cáo.

## Bố cục đề xuất

### 1. Bài toán và yêu cầu

Trình bày ba nhu cầu: tìm sản phẩm đa phương thức, phối đồ dựa trên tri thức stylist, và trả lời hội thoại có căn cứ. Nêu các yêu cầu phi chức năng: độ trễ, khả năng quan sát, hạn chế hallucination và giao diện demo trực quan.

### 2. Kiến trúc tổng thể

Dùng sơ đồ trong `01_SYSTEM_OVERVIEW.md`. Giải thích theo trách nhiệm, không kể tên file trước:

```text
giao diện → API/orchestrator → router → pipeline chuyên biệt
→ retrieval/knowledge → cards + LLM answer → grounding/logging
```

### 3. Chuẩn bị dữ liệu và vector database

Dùng bảng ba embedding space trong `03_DATA_AND_INDEXING.md`. Nhấn mạnh lý do không dùng một embedding cho mọi dữ liệu:

- sản phẩm text cần tương thích không gian thời trang;
- ảnh cần encoder thị giác cùng family;
- rule stylist là semantic text dài nên dùng BGE-M3.

### 4. Router và điều phối

Giải thích `intent → modality → action → route`. Nêu hybrid router: rule/state xử lý trường hợp rõ; LLM chỉ hỗ trợ câu mơ hồ; Python ép enum và chọn route.

Không mô tả numeric `confidence` là độ tin cậy. Dùng `certainty` như provenance vận hành và dùng evaluation accuracy để báo cáo chất lượng.

### 5. Tìm kiếm sản phẩm

Mô tả query embedding, Qdrant candidate retrieval, optional reranking, dedupe/diversity và product cards. Card là dữ liệu xác định; LLM chỉ viết diễn giải.

### 6. Phối đồ hai lớp

Mô tả Layer B chọn công thức/slot, sau đó Layer A gắn sản phẩm thật vào từng slot. Đây là phần nên có sequence diagram riêng vì thể hiện đóng góp thiết kế rõ nhất.

### 7. Xử lý ảnh

Tách hai nhiệm vụ:

- VLM hiểu nội dung/loại ảnh và profile;
- FashionCLIP tìm sản phẩm tương tự.

Không viết rằng VLM trực tiếp tìm nearest products trong Qdrant.

### 8. An toàn, grounding và quan sát

Nêu validation đầu vào, product-card source of truth, commerce-fact streaming filter, ID grounding check, JSONL log và Developer Mode telemetry. Trình bày giới hạn: grounding không chứng minh toàn bộ lời diễn giải là đúng.

### 9. Thực nghiệm

Tách router, retrieval, outfit, grounding, latency và human evaluation. Ghi rõ kích thước tập test và metric; không dùng model self-confidence.

## Đoạn mô tả kiến trúc tham khảo

> Hệ thống được thiết kế theo kiến trúc điều phối đa pipeline thay vì giao toàn bộ tác vụ cho một mô hình ngôn ngữ. Sau khi đầu vào được kiểm tra, router lai kết hợp luật xác định, trạng thái hội thoại và mô hình ngôn ngữ để suy ra intent, modality và action. Route thực thi được ánh xạ bằng chính sách Python hữu hạn. Tùy route, hệ thống truy vấn kho sản phẩm Layer A bằng ViFashionCLIP/FashionCLIP, truy vấn tri thức phối đồ Layer B bằng BGE-M3 hoặc dùng VLM để quan sát ảnh. Kết quả retrieval được hiển thị trực tiếp dưới dạng product card, đồng thời được đưa vào LLM để tạo lời tư vấn. Các trường thương mại trên card được xem là nguồn sự thật và được tách khỏi nội dung sinh nhằm giảm hallucination.

Hãy chỉnh văn phong theo chuẩn của trường và bổ sung citation học thuật cho RAG, multimodal retrieval, vector database, FashionCLIP/BGE/LLM; không coi tài liệu nội bộ này là nguồn học thuật.

## Danh sách hình và bảng nên có

1. Sơ đồ kiến trúc tổng thể.
2. Sequence diagram một request text search.
3. Sequence diagram image outfit.
4. Bảng model/embedding/collection/dimension.
5. Bảng intent/action/route.
6. Bảng input/output API.
7. Bảng metric và kết quả thực nghiệm.
8. Ảnh giao diện card + Developer Mode.

## Phân công nhóm gợi ý

| Thành viên | Phần | Tài liệu nguồn |
|---|---|---|
| A | Tổng quan, yêu cầu, kiến trúc | `01`, `04` |
| B | Data, embedding, indexing | `02`, `03` |
| C | Router, API, history | `05`, `07` |
| D | Retrieval, outfit, vision | `06`, notebooks 03–06 |
| E | Grounding, logging, evaluation | `08`, `09` |

Một người cuối cùng phải rà soát thuật ngữ để không gọi intent là route, không gọi similarity là confidence và không nói LLM tự tìm sản phẩm.

## Checklist trước khi chốt chương

- Mỗi sơ đồ có input/output và trách nhiệm từng khối.
- Tên model/collection/dimension khớp `app/config.py`.
- Luồng text và image được tách rõ.
- Phân biệt Layer A và Layer B.
- Có giới hạn hệ thống và dữ liệu tồn kho.
- Kết quả evaluation là số đo trên tập nhãn, không phải model tự nhận.
- Nội dung báo cáo khớp code/test tại commit dùng để demo.

