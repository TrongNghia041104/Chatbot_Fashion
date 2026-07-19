# Hợp đồng API, SSE và giao diện

## REST endpoints

| Method | Endpoint | Mục đích |
|---|---|---|
| `POST` | `/api/session` | Tạo session |
| `GET` | `/api/profile/{session_id}` | Đọc profile phiên |
| `DELETE` | `/api/session/{session_id}` | Xóa state phiên |
| `POST` | `/api/chat` | Chat multipart + SSE |

## Input `/api/chat`

| Field | Kiểu | Bắt buộc | Ý nghĩa |
|---|---|---|---|
| `session_id` | string | Có | khóa state/history |
| `message` | string | Không nếu có ảnh | text người dùng |
| `image` | file | Không | ảnh sản phẩm/người |
| `developer_mode` | boolean | Không | hiện diagnostic events |

## Session state trong FastAPI

```json
{
  "profile": {},
  "last_bot_msg": "",
  "last_route_decision": null,
  "last_query": "",
  "pending_profile_candidate": null,
  "pending_image_context": null,
  "pending_image_docs": [],
  "unclear_count": 0
}
```

State này nằm trong memory của process; restart app sẽ mất. Lịch sử message của LangChain nằm ở Redis. Hai loại state không đồng nhất và cần được phân biệt khi triển khai nhiều worker.

## SSE events chính

| Type | Khi nào | Payload đáng chú ý |
|---|---|---|
| `progress` | chuyển stage | `phase`, `message`, `filters` |
| `route_detected` | có decision | intent/action/route/source/certainty |
| `image_understood` | VLM xong | subject/caption/fashion_item |
| `product_images` | retrieval xong | cards, group, replace |
| `image_search_ready` | image candidates sẵn sàng | count, route |
| `image_item_context` | suy món gốc | diagnostics |
| `clarification` | thiếu slot | question/options/missing_slots |
| `suggested_follow_up` | CTA tiếp tục | question/options |
| `token` | lời tư vấn | content |
| `stage_metrics` | Developer Mode | timings/model calls/vectors |
| `grounding_filter` | đã loại fact LLM | removed_lines |
| `error` | lỗi stage | message |
| `done` | kết thúc | ttft/total và telemetry tùy mode |

## Product card contract

```json
{
  "product_id": "...",
  "title": "...",
  "category": "...",
  "brand": "...",
  "price": 299000,
  "images": ["/images/..."],
  "slot": "Áo mặc trong",
  "alternatives": []
}
```

`slot` và `alternatives` chủ yếu dùng cho outfit. Modal “Xem món này” dùng đúng `product_id` và toàn bộ `images` của card được chọn.

## Các group UI

- `search_results`: kết quả tìm sản phẩm.
- `source_item`: món catalog gần ảnh làm điểm bắt đầu.
- `outfit`: các slot của set được chọn.

Frontend không được trộn các group thành một danh sách rồi suy category từ text LLM.

## Grounding contract

Card lấy trực tiếp từ retrieved metadata và là nguồn sự thật. Lời LLM chỉ giải thích đặc điểm/lý do phối; filter backend loại dòng do LLM tự sinh có nhãn mã, giá, thương hiệu hoặc ảnh trước khi stream đến browser.

## Responsive và copywriting

UI desktop-first, responsive cơ bản. Developer Mode là khu vực kỹ thuật duy nhất; màn hình khách hàng không hiển thị các thuật ngữ như vector payload, intent score hay RAG trace.

