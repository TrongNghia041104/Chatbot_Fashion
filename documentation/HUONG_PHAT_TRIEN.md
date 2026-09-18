# Hướng phát triển & lộ trình nâng cấp

> File tổng hợp các hướng cải tiến để đạt kết quả cao khi bảo vệ khóa luận.
> Đọc kèm: [`architecture/LANGGRAPH_SYSTEM_DESIGN.md`](architecture/LANGGRAPH_SYSTEM_DESIGN.md) (xương sống điều phối),
> [`quality/09_EVALUATION.md`](quality/09_EVALUATION.md) (kế hoạch đánh giá chi tiết),
> [`architecture/05_INTENT_ROUTER_DECISION.md`](architecture/05_INTENT_ROUTER_DECISION.md) (router hiện tại),
> [`runtime/04_RUNTIME_REQUEST_FLOW.md`](runtime/04_RUNTIME_REQUEST_FLOW.md) (luồng hiện tại).

---

## 0. Hiện trạng một câu

Hệ thống hiện là một **RAG chatbot đa phương thức có bộ điều phối (router) lai deterministic** — kiến trúc sạch, chống hallucination tốt. Điểm mạnh nhất khi bảo vệ: **router 4 lớp không cho LLM tự chọn pipeline** và **grounding filter** chặn LLM bịa mã SP/giá/brand. Điểm còn thiếu: (1) số liệu đánh giá định lượng đầy đủ, (2) phân loại ý định còn dựa nhiều vào keyword nên dễ nhầm khi câu mơ hồ, (3) chưa phải "agent" đúng nghĩa, (4) vài lỗ hổng vận hành.

Các hướng phát triển, xếp theo tỉ lệ **điểm/công sức**:

| Ưu tiên | Hướng | Vì sao |
|---|---|---|
| ⭐⭐⭐ | [1. Đánh giá định lượng](#1-đánh-giá-định-lượng-ưu-tiên-cao-nhất) | Hội đồng luôn hỏi "chứng minh nó tốt bằng số liệu nào?". Biến demo thành khóa luận. |
| ⭐⭐ | [2. Phân loại ý định mạnh hơn](#2-phân-loại-ý-định-mạnh-hơn-semantic-routing) | Chữa đúng điểm yếu bạn đang lấn cấn; tận dụng bge-m3 đã có, ít công sức. |
| ⭐⭐ | [3. Nâng lên Agent](#3-hướng-phát-triển-agent-trọng-tâm) | Câu hỏi trực tiếp của đề tài. Tạo đóng góp kỹ thuật mới + thêm một bảng so sánh. |
| ⭐ | [4. Vá hạn chế vận hành](#4-vá-hạn-chế-vận-hành) | Trả lời sạch các câu hỏi hội đồng hay soi. |

> **Lưu ý chiến lược:** hướng 1 và 2 đã đủ cho một khóa luận vững. Hướng 3 (agent) là **điểm cộng nâng cao**, nên làm như một module tách rời — nếu quỹ thời gian hẹp, hoàn toàn có thể trình bày agent ở dạng "thiết kế + prototype một use case" thay vì thay thế cả hệ thống. Đừng để agent làm rối phần đang chạy tốt.

---

## 1. Đánh giá định lượng (ưu tiên cao nhất)

Kế hoạch chi tiết đã có trong [`quality/09_EVALUATION.md`](quality/09_EVALUATION.md). Phần còn **thiếu và cần bổ sung** để đủ sức bảo vệ:

- **Chạy thật và ra bảng số**, không chỉ liệt kê metric. Hệ thống đã log sẵn nhiều thứ (`append_chat_turn_log` trong `core/security.py`): `intent`, `route`, `source`, `grounding_ok`, `unknown_ids`, `ttft_sec`, `total_sec`, model call counts. → Viết một script tổng hợp log `.jsonl` thành bảng %/biểu đồ.
- **Ablation study** (rất được đánh giá cao vì chứng minh từng thành phần có ích):
  - Có / không **reranker** (`is_reranker_enabled`).
  - Có / không **LLM rewrite**: đã có sẵn 2 chain `get_fast_search_chain` (bỏ rewrite) vs `get_full_chat_chain` (có rewrite) → gần như chỉ cần bật/tắt và đo Recall@k, latency.
  - Router **keyword-only** vs **keyword + semantic + LLM fallback** (xem mục 2).
- **Router accuracy + confusion matrix** intent/action/route, dựng từ `tests/router_eval_cases.jsonl` (đã có >40 case). Tách **development set** và **held-out set** khi báo cáo.
- **Hallucination rate (%)**: tổng hợp `grounding_ok` / `unknown_ids` đã log → con số thuyết phục cho phần guardrail.
- **Retrieval**: Recall@k, MRR, nDCG trên ~50–100 query có nhãn (tự gán cũng được, ghi rõ cách gán).
- **Latency**: p50/p95 cho TTFT và total, theo từng route.

> Mẹo bảo vệ: một bảng ablation "reranker +X% Recall@5, LLM-rewrite +Y% nhưng +Z giây" cho thấy bạn hiểu trade-off, không chỉ ghép thư viện.

---

## 2. Phân loại ý định mạnh hơn (semantic routing)

### 2.1. Vấn đề: keyword dễ nhầm khi người dùng viết mơ hồ

Router hiện tại nhận diện ý định chủ yếu bằng **keyword matching** (Layer 3), chỉ khi trượt mới rơi xuống **LLM fallback** (Layer 4). Keyword chính xác cao nhưng **giòn**: người dùng viết "nay đi cưới bạn mặc gì giờ", "kiếm cái mặc lên đồ được không" — không trúng từ khóa nào → phải nhờ LLM hoặc hỏi lại. Đây chính là chỗ bạn lấn cấn.

**Điểm quan trọng cần hiểu:** hệ thống của bạn **đã đúng hướng** — nó là *hybrid cascade* (keyword rẻ trước, LLM cho câu khó sau), đúng như cách các hệ thống thực tế làm. Vấn đề chỉ là **giữa keyword và LLM đang thiếu một tầng**.

### 2.2. Phổ các cách phân loại ý định (để đối chiếu và viết báo cáo)

| Cách | Cơ chế | Ưu | Nhược | Cần gì |
|---|---|---|---|---|
| **Rule / keyword** | So khớp từ khóa, regex | Nhanh, rẻ, giải thích được | Giòn, chết với cách nói lạ | Không cần data |
| **ML classifier** | Huấn luyện phân loại (fastText, SVM, BERT fine-tune) | Ổn định với biến thể câu chữ | Cần **data có nhãn**, phải train lại khi thêm intent | Tập nhãn |
| **Semantic routing (embedding)** | Nhúng câu bằng embedding, so cosine với các câu mẫu/centroid mỗi intent | Chịu được diễn đạt lạ, **không cần train**, không gọi LLM | Cần chọn ngưỡng, cần vài câu mẫu/intent | **Đã có bge-m3** |
| **LLM zero/few-shot** | Prompt LLM phân loại (trả JSON) | Linh hoạt nhất, hiểu câu mơ hồ | Chậm/tốn hơn, đôi khi không nhất quán | Không cần data |
| **LLM function-calling** | Cho LLM chọn tool = chọn ý định | Gộp luôn vào bước gọi tool | Chính là bước tiến sang agent (mục 3) | Model biết tool-calling |
| **Hybrid cascade** | Rẻ trước → khó sau (rule → semantic → LLM) | Cân bằng tốc độ/độ chính xác/chi phí | Nhiều tầng, cần điều phối | Kết hợp trên |

> Cách các sản phẩm thực tế làm (Dialogflow, Rasa, trợ lý ảo lớn) gần như luôn là **hybrid cascade** — không ai chỉ dùng thuần Python keyword, cũng không ai gọi LLM cho *mọi* câu.

### 2.3. Đề xuất cụ thể: chèn tầng semantic dùng bge-m3 sẵn có

Bạn đang chạy **bge-m3** cho retrieval rồi → tái dùng nó để phân loại ý định gần như miễn phí:

```text
Layer 1  Modality gate (có ảnh?)          ← giữ nguyên
Layer 2  Session state (đang chờ xác nhận?) ← giữ nguyên
Layer 3  Keyword (độ chính xác cao)         ← giữ nguyên
Layer 3b SEMANTIC (MỚI): nhúng câu → so cosine với câu mẫu mỗi intent
            - điểm cao & cách biệt rõ  → chọn intent, KHÔNG gọi LLM
            - điểm thấp/nhập nhằng     → mới rơi xuống Layer 4
Layer 4  LLM fallback                       ← giữ nguyên, nhưng ít bị gọi hơn
```

Cách làm: soạn ~5–10 **câu mẫu cho mỗi intent** (`product_discovery`, `outfit_advice`, `profile_*`, `social`, `out_of_scope`), nhúng sẵn 1 lần. Khi có query: nhúng query, so cosine, lấy intent gần nhất nếu vượt ngưỡng. Tái dùng `EmbedderPort` đã có; không thêm phụ thuộc nặng.

**Lợi ích cho khóa luận:** bắt được nhiều câu nói tự nhiên mà keyword trượt, **giảm số lần gọi LLM** (nhanh + rẻ hơn), và tạo thêm một dòng ablation "keyword-only vs keyword+semantic vs +LLM".

**✅ Đã triển khai (Layer 3b):**

- Logic ở `src/fashion_rag/core/semantic_router.py`; cắm vào `route_user_request()` (`core/intent.py`) *giữa* keyword và LLM. Keyword trúng vẫn ưu tiên như cũ → không phá luồng deterministic.
- Dùng lại BGE-M3 qua `get_rule_embeddings()` (không thêm model). Câu mẫu + ngưỡng nằm ở `config.py`: `SEMANTIC_INTENT_EXAMPLES`, `SEMANTIC_ROUTER_THRESHOLD` (0.55), `SEMANTIC_ROUTER_MARGIN` (0.05).
- Quy tắc chốt: điểm cosine cao nhất `>= threshold` **và** cách biệt intent nhì `>= margin`; nếu không → trả None, nhường LLM. Có `source="semantic"` + `certainty="semantic"` riêng để đo tách bạch (telemetry không đếm nhầm là gọi LLM).
- Bật/tắt bằng env `SEMANTIC_ROUTER_ENABLED` (mặc định bật) → chính là công tắc để chạy ablation.
- Self-check thuần logic (không cần Ollama): `python -m fashion_rag.core.semantic_router`.
- **Việc cần làm tiếp:** calibrate `THRESHOLD`/`MARGIN` trên `tests/router_eval_cases.jsonl` và bổ sung câu mẫu cho intent nào hay bị nhầm.

### 2.4. Câu mơ hồ không phải "lỗi" — clarification là thiết kế đúng

Ngay cả nhân viên bán hàng giỏi cũng hỏi lại "bạn muốn tìm món cụ thể hay để mình tư vấn phối đồ?". Hệ thống của bạn đã có nhánh **clarification** (Layer 4 fallback hỏi lại) — đó là **hành vi đúng**, không phải thất bại. Việc cần làm là *đo* tỉ lệ clarification và giữ nó ở mức hợp lý, không phải cố xóa nó về 0.

---

## 3. Hướng phát triển Agent (trọng tâm)

### 3.1. Vì sao hiện tại **chưa phải** agent

| Cấp | Bản chất | Ai quyết định các bước? | Dự án hiện tại |
|---|---|---|---|
| **RAG / Chatbot** | Làm theo quy trình cố định | Con người viết sẵn luồng | ✅ Đang ở đây |
| **Workflow (có nhánh)** | State graph phức tạp | Con người viết sẵn các nhánh | Có một phần (router 4 lớp) |
| **Agent** | Được giao **mục tiêu**, tự xoay xở | **LLM tự quyết** bước nào, thứ tự nào, dùng tool nào, lặp đến khi xong | ❌ Chưa |

Trong `apps/api/api.py`, **toàn bộ luồng do bạn viết sẵn**: có ảnh → gọi VLM → route → chọn chain → sinh câu trả lời. LLM chỉ (a) phân loại câu mơ hồ và (b) viết lời tư vấn. **LLM không tự lập kế hoạch, không tự gọi công cụ, không có vòng lặp quyết định.** Đó là lý do nó là RAG orchestration chứ chưa phải agent.

### 3.2. Agent là gì (định nghĩa dùng được trong báo cáo)

Agent = hệ thống mà **LLM giữ vai trò điều khiển (controller)**, đặc trưng bởi 3 tính chất:

1. **Planning** — tự chia mục tiêu lớn thành nhiều bước, không theo script cứng.
2. **Tool use (function calling)** — tự chọn và gọi công cụ, rồi **đọc kết quả để quyết bước tiếp theo**.
3. **Loop (ReAct: Reason → Act → Observe)** — lặp quan sát–suy nghĩ–hành động, tự dừng khi đạt mục tiêu.

### 3.3. Bạn đã có gần đủ nguyên liệu (không phải viết lại)

Các hàm nghiệp vụ hiện có chính là các **tool** tương lai:

| Tool (đề xuất) | Hàm sẵn có tái sử dụng | Trả về |
|---|---|---|
| `search_products(query, max_price, size, color)` | `chains.get_fast_search_chain` / `vector_store.get_product_retriever` | list sản phẩm **thật** (id, title, price) |
| `build_outfit(query, gender, profile)` | `recommendation.outfit.build_outfit_context` | công thức Layer B + sản phẩm Layer A |
| `search_by_image(image_path)` | `retrieval.image_search.search_products_by_image` | sản phẩm giống ảnh |
| `analyze_person(image_path)` | `llms.vision.analyze_person_image` | dáng người, tone da |
| `get_profile()` / `update_profile()` | `application.chat.profile` | đọc/ghi hồ sơ |
| `finish(summary)` | — | tín hiệu dừng vòng lặp |

### 3.4. Có nên đổi sang model nhỏ hơn 4B không? — **Không**

Đây là chỗ dễ hiểu nhầm. Việc mà agent cần — *lập kế hoạch, chọn đúng tool, đọc kết quả rồi quyết bước tiếp* — **chính là chỗ model nhỏ sụp đổ đầu tiên**: sinh tool-call sai cú pháp, chọn nhầm tool, lặp vô nghĩa. Nghịch lý là model nhỏ **lặp nhiều hơn → tổng thời gian còn chậm hơn** model to gọi một phát trúng. 4B là **sàn tối thiểu** cho tool-calling ổn định; đi xuống 1.5B/3B cho phần "não" sẽ làm agent mất tin cậy.

"Nặng" ở đây gần như là **latency do nhiều vòng gọi LLM** (bản chất của agent), không phải do model to. Cách chữa đúng:

| Vấn đề | Cách sai | Cách đúng |
|---|---|---|
| Agent chậm | Đổi model nhỏ hơn | **Gate**: chỉ query phức tạp mới vào agent; còn lại đi router cũ (nhanh, rẻ) |
| Tool-call sai cú pháp | Đổi model | **Constrained decoding** — ép output theo JSON schema (đã dùng `format="json"` trong `classify_intent_llm`, mở rộng thêm) |
| GPU / chi phí | Model nhỏ | Quantize (Q4_K_M), hoặc dùng **vLLM** thay Ollama cho concurrency tốt hơn |

**Khuyến nghị:**
- **Não (controller/planner):** giữ Qwen3-4B, hoặc thử **Qwen2.5-7B-Instruct** (tool-calling tốt hơn rõ) rồi đo — đừng xuống dưới 4B.
- **Tay viết câu trả lời** (sinh lời tư vấn từ context đã lấy): việc dễ, 4B/3B thừa sức. Nếu cần tiết kiệm, **đây** mới là chỗ dùng model nhỏ.
- Nguyên tắc: **tách vai trò**, không "đổi đồng loạt sang nhỏ".

### 3.5. Kiến trúc đề xuất: Router-gated single agent

Không agent-hóa mọi thứ (đó là lỗi làm hệ thống vừa chậm vừa rối). Chỉ đẩy vào agent các câu **nhiều ràng buộc / nhiều bước**:

```text
Tầng 0  Fast path: router deterministic hiện tại xử lý 80–90% câu (đây cũng là BASELINE để so sánh)
Tầng 1  Cổng "phức tạp": phát hiện query nhiều ràng buộc (keyword rẻ, không cần LLM)
            vd: "phối cả bộ ngân sách 2tr", "đi Đà Lạt 3 ngày mặc gì", "so sánh 2 áo rồi chọn hợp dáng tôi"
Tầng 2  Vòng lặp agent (LangGraph):
            System prompt: vai trò + tool + ràng buộc (BẮT BUỘC bám catalog, tôn trọng ngân sách)
            ↺ LLM chọn tool (JSON schema) → chạy hàm Python sẵn có → nối observation → suy luận tiếp
            → gọi finish() hoặc chạm max_steps (5–6) → tổng hợp
            → grounding filter (GIỮ NGUYÊN) → stream ra UI
```

LangGraph chính là xương sống này (xem [`LANGGRAPH_SYSTEM_DESIGN.md`](architecture/LANGGRAPH_SYSTEM_DESIGN.md)): cho vòng lặp + state + checkpoint + human-in-the-loop, thay cho việc tự quản `queue.Queue`/thread.

**Ví dụ trace — "phối đồ đám cưới, ngân sách 2tr":**

```text
Step 1  LLM → search_products("áo sơ mi nam dự tiệc", max_price=800k)
        obs: [SP_A 650k, SP_B 720k, ...]
Step 2  LLM → search_products("quần âu nam", max_price=700k)
        obs: [SP_C 550k, ...]
Step 3  LLM → search_products("giày tây nam", max_price=900k)
        obs: [SP_D 1.2tr, SP_E 850k, ...]
Step 4  LLM suy luận: 650k+550k+1.2tr = 2.4tr > 2tr → chọn SP_E thay SP_D
        tổng = 2.05tr vẫn nhỉnh → tìm áo rẻ hơn
Step 5  LLM → search_products("áo sơ mi nam", max_price=500k) → SP_F 450k
        tổng = 1.85tr ✓ đủ slot ✓
Step 6  LLM → finish("bộ 3 món trong 1.85tr ...")
```

Agent chỉ **chọn và sắp xếp** trên dữ liệu thật do tool trả về; id/giá/ảnh vẫn render từ tool output, không từ chữ LLM.

### 3.6. Giữ nguyên guardrails khi lên agent (điểm bảo vệ quan trọng)

Agent tự do hơn → rủi ro hallucination cao hơn. **Không được bỏ** các bảo vệ đang là điểm mạnh:

- **Grounding**: mã SP/giá/brand/ảnh vẫn đến từ product card/retrieval, không từ lời LLM. Giữ `CommerceFactStreamFilter` + `check_answer_grounding` ở đầu ra agent.
- **Deterministic policy làm rào chắn**: agent tự chọn *thứ tự* và *tham số* tool, nhưng ràng buộc an toàn vẫn do Python kiểm. Đây là "agent có kiểm soát" — dễ bảo vệ hơn agent tự do hoàn toàn.
- **Giới hạn vòng lặp** (`max_steps`) tránh gọi tool vô hạn và latency phình.

### 3.7. Năm quyết định thiết kế nên nhấn mạnh khi bảo vệ

1. **Hybrid gating** — không agent-hóa mọi thứ → kiểm soát latency/chi phí, có baseline so sánh.
2. **Constrained tool-call (JSON schema)** — để model 4B gọi tool đáng tin.
3. **max_steps + budget guard** — chặn lặp vô hạn.
4. **Grounding giữ nguyên** — "tay" (tool Python) luôn đáng tin, chỉ "não" là LLM.
5. **Đo lường**: agent vs baseline trên cùng bộ query — task completion rate, số bước TB, tool-call success rate, latency p50/p95, hallucination rate.

### 3.8. Lộ trình từng bước (module tách rời, không làm rối luồng cũ)

1. Định nghĩa tool schema cho 2 hàm quan trọng nhất trước: `search_products`, `build_outfit`.
2. Viết vòng lặp ReAct tối giản (hoặc LangGraph) cho **một** use case: *phối đồ theo ngân sách*.
3. Thêm endpoint **riêng** `/api/chat_agent`, **giữ nguyên** `/api/chat` cũ. Agent là module độc lập — hiểu và sửa nó không đụng vào luồng hiện tại.
4. So sánh agent vs baseline trên cùng bộ query.
5. Chỉ mở rộng multi-agent (`ProductSearchAgent`, `OutfitStylistAgent`, `VisionAgent`, `SafetyAgent`) khi single-agent đã ổn — và chỉ khi còn thời gian.

### 3.9. Rủi ro cần nêu trung thực trong báo cáo

- **Model 4B gọi tool có thể sai/không ổn định** → đó là lý do giữ deterministic fallback.
- **Latency tăng** do nhiều vòng gọi tool → đo và báo cáo thật.
- **Khó reproduce** hơn workflow cố định → cần `temperature=0` và log đầy đủ trace.

---

## 4. Vá hạn chế vận hành

Sửa được thì sửa; không thì đưa vào mục "Hạn chế & hướng phát triển" một cách chủ động (nêu trước vẫn ghi điểm).

| Hạn chế | Hiện trạng | Cách xử lý |
|---|---|---|
| Session in-memory | `sessions: dict` trong `api.py` → mất khi restart, không scale nhiều worker | Đẩy session/profile sang **Redis** (đã có sẵn) |
| CORS mở toàn bộ | `allow_origins=["*"]`, chưa có auth | Giới hạn origin + auth tối thiểu, hoặc ghi rõ là phạm vi nghiên cứu |
| Không có tồn kho realtime | Code đã tự thú nhận ở `stock_check` | Đưa vào future work (tích hợp API kho) |
| Thiếu test cho pipeline sinh | Chủ yếu test router | Thêm test cho grounding filter + outfit context |

---

## 5. Trình bày trong khóa luận

- Làm nổi bật 2 đóng góp: **router lai chống hallucination** và **grounding guardrail** — kèm số liệu ở mục 1.
- Mạch chuyện mạnh: trình bày **tiến hoá kiến trúc** RAG → phân loại ý định lai (semantic) → workflow (LangGraph) → agent, mỗi mức kèm bảng so sánh.
- Luôn kèm phần **hạn chế** trung thực; hội đồng đánh giá cao việc tự nhận biết giới hạn hơn là giấu.
