# Giải Thích Hệ Thống `Chatbot_RAG_MultiModal_ViFashionCLIP_research_demo_v3.ipynb`

> **Tài liệu tham khảo cũ:** Một số ví dụ `confidence`, prompt và log trong file này phản ánh giai đoạn trước khi router được chuẩn hóa. Khi bàn giao hoặc viết báo cáo, bắt đầu từ [`00_README_FIRST.md`](00_README_FIRST.md); code trong `apps/` và `src/fashion_rag/` là nguồn sự thật.

Tài liệu này giải thích notebook `Chatbot_RAG_MultiModal_ViFashionCLIP_research_demo_v3.ipynb` theo hướng đọc từ con số 0 vẫn hiểu được. Mục tiêu không chỉ là biết “hàm nào làm gì”, mà là đổi cách nhìn: đây không phải một chatbot gọi LLM đơn thuần, mà là một hệ thống điều phối nhiều loại trí tuệ nhỏ, mỗi loại chịu trách nhiệm cho một phần rất cụ thể của bài toán tư vấn thời trang.

---

## 1. Nhìn hệ thống bằng một mô hình tinh thần mới

Nếu chỉ nhìn bề mặt, hệ thống giống một chatbot:

```text
Người dùng hỏi -> Bot trả lời
```

Nhưng thực tế bên trong là:

```text
Người dùng gửi text/ảnh
        |
        v
Router quyết định người dùng muốn gì
        |
        +-- Tìm sản phẩm bằng text
        +-- Tìm sản phẩm bằng ảnh
        +-- Tư vấn phối đồ theo công thức stylist
        +-- Phân tích dáng người/tone da
        +-- Hỏi lại nếu chưa đủ rõ
        +-- Từ chối mềm nếu ngoài phạm vi
        |
        v
Retriever lấy bằng chứng sản phẩm/quy tắc
        |
        v
LLM chỉ được phép viết câu trả lời dựa trên bằng chứng đó
        |
        v
Guardrail kiểm tra câu trả lời có bịa mã sản phẩm không
```

Nói cách khác, LLM không phải “bộ não duy nhất”. LLM trong hệ thống này giống người viết câu trả lời cuối cùng. Còn việc hiểu ảnh, tìm sản phẩm, nhớ lịch sử, chọn route, lọc trùng, kiểm tra bịa đặt đều được giao cho các thành phần riêng.

Đây là tư duy cốt lõi của một hệ RAG tốt: **không để LLM tự nhớ, tự tìm, tự đoán và tự bịa**. Hệ thống phải đưa dữ liệu đúng cho LLM, rồi ép LLM trả lời trong phạm vi dữ liệu đó.

---

## 2. Hệ thống giải quyết bài toán gì?

Notebook xây dựng một chatbot tư vấn thời trang đa phương thức, nghĩa là nhận cả text và ảnh.

Các năng lực chính:

1. Tìm sản phẩm bằng tiếng Việt.
2. Tìm sản phẩm tương tự bằng ảnh upload.
3. Phân tích ảnh người dùng để lưu dáng người và tone da.
4. Tư vấn phối đồ dựa trên quy tắc stylist Layer B.
5. Dùng lịch sử hội thoại để hiểu các câu hỏi tiếp nối như “còn màu khác không?”.
6. Chặn input nguy hiểm kiểu prompt injection.
7. Kiểm tra câu trả lời có nhắc mã sản phẩm ngoài context hay không.
8. Ghi log và chạy eval-lite phục vụ nghiên cứu/luận văn.

---

## 3. Kiến trúc 2 lớp: Layer A và Layer B

Notebook dùng kiến trúc RAG hai lớp:

| Lớp | Vai trò | Dữ liệu | Embedding | Vector dim | Collection |
|---|---|---|---|---:|---|
| Layer A text | Tìm sản phẩm thật trong kho bằng text | Metadata 65k sản phẩm | ViFashionCLIPTextEmbeddings | 512 | `fashion_products_vifashionclip_vi_65k_structured_vi` |
| Layer A image | Tìm sản phẩm thật bằng ảnh | Ảnh MAIN sản phẩm | FashionCLIPImageEmbeddings | 512 | `fashion_products_fashionclip_image_main_65k` |
| Layer B | Tìm công thức phối đồ | Rule stylist nam/nữ | BGE-M3 | 1024 | `layer_b_female`, `layer_b_male` |

Điểm quan trọng:

- **Layer A** trả lời câu hỏi: “Trong kho có sản phẩm nào giống nhu cầu này?”
- **Layer B** trả lời câu hỏi: “Theo stylist, nên phối loại đồ nào với loại đồ nào?”

Khi người dùng hỏi “phối đồ đi tiệc cho dáng quả lê”, hệ thống không chỉ tìm sản phẩm. Nó tìm một công thức phối đồ phù hợp ở Layer B, rồi dùng công thức đó để quay lại Layer A lấy sản phẩm thật.

---

## 4. Vì sao không dùng một embedding cho tất cả?

Notebook cố tình dùng hai loại embedding khác nhau:

### ViFashionCLIP cho Layer A

Layer A là sản phẩm thời trang: tên, màu, chất liệu, ảnh, phong cách, mô tả. ViFashionCLIP được fine-tune để đưa text tiếng Việt về cùng không gian ngữ nghĩa với FashionCLIP. Nó phù hợp cho retrieval sản phẩm vì hiểu ngôn ngữ thời trang và liên hệ text với hình ảnh.

### FashionCLIP image encoder cho ảnh sản phẩm

Khi người dùng gửi ảnh áo/váy/giày, hệ thống không caption ảnh rồi mới search ngay. Nó ưu tiên encode ảnh bằng FashionCLIP image encoder và tìm vector tương tự trong Qdrant. Đây là cách đúng hơn vì ảnh chứa nhiều tín hiệu thị giác mà caption có thể bỏ sót.

### BGE-M3 cho Layer B

Layer B là rule stylist dạng text: phong cách, bối cảnh, dáng người, tone da, lý do tư vấn. Đây là tri thức ngôn ngữ thuần, không phải sản phẩm/ảnh. BGE-M3 phù hợp cho rule semantic search và vector 1024 chiều.

Góc nhìn mới ở đây: **embedding không phải công cụ chung chung**. Mỗi embedding là một “ngôn ngữ hình học” khác nhau. Dùng sai embedding giống như dùng bản đồ thành phố để đi trong siêu thị: vẫn có vector, nhưng vector không còn đúng ngữ cảnh.

---

## 5. Các pha lớn trong notebook

Notebook có thể chia thành 12 khối tư duy:

| Pha | Nội dung | Ý nghĩa |
|---|---|---|
| 0 | Kiểm tra môi trường | Biết Python đang chạy ở đâu |
| 1 | Import thư viện | Chuẩn bị LangChain, Qdrant, Ollama, Torch, Transformers |
| 2 | Model config và embedding model | Định nghĩa LLM, VL model, ViFashionCLIP, FashionCLIP, BGE-M3 |
| 3 | Data pipeline | Chuyển JSONL sản phẩm thành Document để index |
| 4 | Product retriever | Kết nối Qdrant, rerank, diversity filter |
| 4.2-4.4 | Image retrieval | Index và search ảnh MAIN sản phẩm |
| 5 | Layer B stylist rules | Load và index quy tắc phối đồ |
| 6 | Category mapping | Dịch category Layer B sang category sản phẩm Layer A |
| 7 | Vision module | Phân loại ảnh, phân tích người, caption sản phẩm |
| 8-10 | LLM và chain | Prompt, Redis history, RAG chain |
| 11 | Router | Quyết định request thuộc luồng nào |
| 12-13 | Security, logging, eval, chat loop | Guardrail, log, eval, vòng lặp chat chính |

---

## 6. Khối cấu hình hệ thống

Các hằng số quan trọng:

| Tên | Vai trò |
|---|---|
| `LLM_MODEL` | Model sinh câu trả lời chính, ví dụ `qwen3:4b-instruct` |
| `QWEN_VL_MODEL` | Model vision-language để hiểu ảnh, ví dụ `qwen2.5vl:3b` |
| `TEACHER_MODEL_NAME` | FashionCLIP gốc, dùng cho image encoder |
| `STUDENT_MODEL_NAME` | Vietnamese embedding model làm student encoder |
| `VIFASHIONCLIP_CHECKPOINT` | Checkpoint fine-tuned của ViFashionCLIP |
| `PRODUCT_COLLECTION` | Collection sản phẩm text 512 chiều |
| `PRODUCT_IMAGE_COLLECTION` | Collection ảnh sản phẩm 512 chiều |
| `REDIS_URL` | Redis lưu lịch sử chat |

Hệ thống local và Vast.ai kết nối qua ý tưởng:

```text
Local FastAPI -> localhost:11434 -> SSH tunnel -> Ollama trên Vast.ai GPU
```

Với app local, Ollama luôn có vẻ như đang chạy ở `localhost:11434`, dù thật ra model nằm trên Vast.ai.

---

## 7. Nhóm hàm ViFashionCLIP text embedding

### `ResidualMLPBlock`

Đây là block MLP có skip connection:

```text
output = input + MLP(LayerNorm(input))
```

Công dụng:

- Giúp projection head học biến đổi sâu hơn mà không làm mất gradient.
- Dùng trong mô hình student để map embedding tiếng Việt sang không gian FashionCLIP.

### `ProjectionHead`

Nhận embedding từ student encoder và chiếu sang vector 512 chiều.

Luồng:

```text
Vietnamese text
    -> Student encoder hidden state
    -> mean pooling
    -> ProjectionHead
    -> 512-dim FashionCLIP-like vector
```

### `mean_pool`

Transformer trả về embedding cho từng token. Nhưng retrieval cần một vector cho cả câu. `mean_pool` lấy trung bình token embedding, bỏ qua padding token.

Công dụng:

- Biến output dạng `[batch, tokens, hidden]` thành `[batch, hidden]`.
- Tránh padding làm sai vector câu.

### `Stage2StudentProjection`

Gói `encoder + projection_head` thành một model inference hoàn chỉnh.

Nó làm đúng thứ checkpoint đã train:

```text
input_ids + attention_mask
    -> encoder
    -> mean_pool
    -> projection_head
    -> vector 512 chiều
```

### `ViFashionCLIPTextEmbeddings`

Đây là wrapper LangChain cho model text retrieval Layer A.

Các việc chính:

1. Load checkpoint `best_stage2_model.pt`.
2. Lấy `student_model_name` và `teacher_dim` từ checkpoint.
3. Load tokenizer và encoder.
4. Load weight cho encoder và projection head.
5. Freeze toàn bộ model.
6. Encode text theo batch.
7. L2 normalize vector trước khi đưa vào Qdrant.

Hàm chính:

| Hàm | Công dụng |
|---|---|
| `_encode(texts)` | Encode list text thành vector 512 chiều |
| `embed_documents(texts)` | LangChain gọi khi index documents |
| `embed_query(text)` | LangChain gọi khi search query |

Điểm nên nhớ: vector được normalize, nên cosine similarity hoạt động ổn định hơn.

---

## 8. Nhóm hàm BGE-M3 cho Layer B

### `BGEM3Embeddings`

Wrapper dùng `OllamaEmbeddings` để gọi `bge-m3` qua Ollama.

Công dụng:

- Embed các rule stylist Layer B.
- Embed query phối đồ để tìm rule phù hợp.

Nó không dùng cho sản phẩm Layer A trong bản v3. Đây là thay đổi lớn so với bản cũ.

---

## 9. Nhóm hàm FashionCLIP image embedding

### `FashionCLIPImageEmbeddings`

Encoder ảnh dùng FashionCLIP gốc.

Các hàm:

| Hàm | Công dụng |
|---|---|
| `_open_image(image_path)` | Mở ảnh và convert RGB |
| `encode_image_paths(image_paths)` | Encode nhiều ảnh thành vector 512 chiều |
| `embed_image(image_path)` | Encode một ảnh query |

Luồng image embedding:

```text
Image file
    -> PIL RGB
    -> CLIPProcessor
    -> CLIPModel.get_image_features()
    -> L2 normalize
    -> vector 512 chiều
```

Notebook lazy-load image encoder bằng `get_image_embeddings()` vì FashionCLIP khá nặng. Đây là tư duy vận hành đúng: đừng load model lớn nếu request hiện tại không cần ảnh.

---

## 10. Nhóm hàm xử lý metadata sản phẩm

### `extract_image_urls(item)`

Lấy danh sách `large` image URL/path từ field `images`.

Ví dụ input:

```json
{
  "images": [
    {"large": "images/ABC_MAIN.jpg", "variant": "MAIN"}
  ]
}
```

Output:

```text
["images/ABC_MAIN.jpg"]
```

### `normalize_to_text(value, default="Không rõ")`

Chuẩn hóa dữ liệu metadata về text:

- `None` hoặc chuỗi rỗng -> `Không rõ`
- list -> nối bằng dấu phẩy
- giá trị khác -> `str(value).strip()`

### `build_product_metadata(item)`

Tạo metadata ngắn để filter/render:

```text
product_id, title, category, department, brand, price, images, image_url
```

Metadata dùng cho:

- Lọc category trong Qdrant.
- Hiển thị ảnh.
- Kiểm tra mã sản phẩm trong guardrail.

### `build_product_page_content(item)`

Tạo đoạn text có nhãn rõ ràng để embed bằng ViFashionCLIP.

Ví dụ:

```text
Tên sản phẩm: Áo tank top nữ
Mã sản phẩm: B0B8MHXHBW
Danh mục: Áo
Đối tượng: Nữ
Thương hiệu: Colorfulkoala
Giá: 479759 VND
Màu sắc: Không rõ
Chất liệu: 85% Polyester tái chế, 15% Spandex
...
```

Tại sao format có nhãn quan trọng?

Vì embedding model không chỉ đọc từ khóa, nó đọc cấu trúc. Khi có nhãn như “Màu sắc”, “Chất liệu”, “Dịp sử dụng”, vector sẽ phản ánh đúng loại thông tin hơn.

### `process_fashion_metadata(file_path)`

Đọc JSONL sản phẩm thành list `Document`.

Mỗi `Document` gồm:

```text
page_content = text có nhãn để embed
metadata = thông tin ngắn để render/filter
```

Hàm cũng thống kê:

- Tổng số dòng.
- Lỗi JSON.
- Thiếu product_id.
- Thiếu category.
- Thiếu ảnh.

---

## 11. Index sản phẩm text vào Qdrant

### `run_data_pipeline(...)`

Pipeline index Layer A text:

```text
JSONL metadata
    -> process_fashion_metadata()
    -> ViFashionCLIPTextEmbeddings
    -> Qdrant collection 512 chiều
```

Nó có resume:

```text
current_count = số point hiện có trong collection
remaining = all_docs[current_count:]
```

Nếu đang index bị dừng giữa chừng, chạy lại sẽ tiếp tục từ điểm đã có. Đây là chi tiết rất thực tế khi index 65k sản phẩm.

---

## 12. Product retriever và chống kết quả nghèo nàn

Retrieval đơn giản thường bị lỗi:

- Trả nhiều sản phẩm trùng nhau.
- Trả nhiều sản phẩm cùng brand.
- Vector search đúng đại khái nhưng thứ tự chưa tốt.

Notebook xử lý bằng ba lớp:

```text
Qdrant top-30
    -> optional reranker top-15
    -> diversity filter còn 5 sản phẩm
```

### `normalize_product_metadata(doc)`

Đảm bảo mỗi document có `image_url` ổn định.

### `diversity_filter_documents(...)`

Lọc kết quả theo:

- Không trùng `product_id`.
- Tối đa `PRODUCT_SEARCH_BRAND_LIMIT` sản phẩm cùng brand.
- Nếu chưa đủ slot, fill thêm sản phẩm chưa trùng.

Đây là thứ làm câu trả lời có vẻ “có gu” hơn. Không phải vì LLM thông minh hơn, mà vì context đưa vào LLM đã đa dạng hơn.

### `ProductCrossEncoderReranker`

Dense retrieval như Qdrant chỉ so vector query và vector document riêng lẻ. Cross-encoder đọc cặp `(query, document)` cùng lúc nên đánh giá phù hợp tốt hơn.

Hàm:

| Hàm | Công dụng |
|---|---|
| `score_pairs(query, docs)` | Tính điểm rerank cho từng cặp query-document |

### `get_product_reranker()`

Lazy-load reranker. Nếu lỗi, tự tắt reranker và fallback về dense retrieval.

Đây là thiết kế tốt vì reranker là phần “cải thiện”, không nên làm sập toàn chatbot.

### `rerank_documents(query, docs)`

Sắp xếp lại docs theo điểm cross-encoder.

### `safe_base_retrieve(base_retriever, query)`

Retry retrieval nếu Qdrant lỗi tạm thời. Nếu vẫn lỗi thì trả `[]` thay vì crash.

### `DiversityFilteredRetriever`

Đây là retriever chính thức đưa vào RAG chain.

Luồng:

```text
query
    -> base_retriever.invoke(query)
    -> rerank_documents()
    -> diversity_filter_documents()
    -> docs cuối cùng cho LLM
```

---

## 13. Image retrieval: tìm sản phẩm bằng ảnh

Notebook không chỉ caption ảnh. Nó xây một collection ảnh riêng.

### `get_main_image_relative_path(item)`

Lấy ảnh MAIN của sản phẩm theo fallback:

1. Ảnh có `variant == "MAIN"`.
2. Ảnh có `_MAIN` trong tên file.
3. Ảnh đầu tiên trong list.
4. Nếu vẫn không có, đoán theo `product_id_MAIN.jpg`.

### `resolve_main_image_path(relative_path, image_root)`

Chuyển path metadata như:

```text
images/B0B8MHXHBW_MAIN.jpg
```

thành path thật trên disk.

### `iter_products_with_main_image(...)`

Generator duyệt metadata và chỉ yield sản phẩm có ảnh MAIN tồn tại.

Output mỗi item:

```text
(item, rel_path, img_path)
```

### `build_image_payload(item, rel_path, img_path)`

Tạo payload lưu trong Qdrant image collection.

Payload gồm:

- product_id
- title
- category
- department
- brand
- price
- image_url
- main_image_path
- page_content
- metadata

Tại sao payload ảnh cũng có `page_content`?

Vì sau khi tìm sản phẩm bằng ảnh, LLM vẫn cần text chi tiết để viết câu trả lời. Vector ảnh chỉ dùng để tìm. Câu trả lời vẫn phải dựa trên metadata sản phẩm.

### `run_main_image_index_pipeline(...)`

Index ảnh MAIN:

```text
metadata JSONL
    -> tìm ảnh MAIN tồn tại
    -> FashionCLIP image encoder
    -> Qdrant image collection
```

Nó cũng resume theo `current_count`.

### `image_point_to_document(point)`

Chuyển Qdrant point ảnh thành LangChain `Document` để dùng chung prompt product.

### `search_products_by_image(...)`

Luồng tìm sản phẩm bằng ảnh upload:

```text
uploaded image
    -> FashionCLIP image vector
    -> Qdrant image collection
    -> top_k points
    -> bỏ trùng product_id
    -> tối đa max_products documents
```

Nếu image collection chưa có, hàm trả `[]` để hệ thống fallback sang caption bằng Qwen-VL.

---

## 14. Layer B: tri thức phối đồ

Layer B là nơi hệ thống có “gu stylist” có cấu trúc.

Mỗi rule thường có:

- `rule_key`
- `phong_cach`
- `boi_canh`
- `dang_nguoi`
- `tone_da`
- `ly_do_tu_van`
- `goi_y_phoi_cung`

### `load_layer_b(file_path)`

Load JSON rule nam/nữ.

### `index_layer_b(data, collection_name)`

Index rule vào Qdrant bằng BGE-M3.

Text embed thường ghép:

```text
rule_key + phong_cach + boi_canh + ly_do_tu_van
```

### `find_matching_rule(user_query, gender, profile)`

Tìm rule chính phù hợp nhất.

Luồng fallback:

```text
1. Lọc theo dáng người + tone da
2. Nếu không có, chỉ lọc dáng người
3. Nếu vẫn không có, bỏ filter và semantic search
```

Ý nghĩa:

- Nếu có ảnh người dùng, tư vấn cá nhân hóa hơn.
- Nếu profile chưa đủ, hệ thống vẫn không bị tắc.

### `find_outfit_details(base_rule, gender)`

Sau khi có rule chính, tìm rule chi tiết cho từng món cần phối.

Ví dụ:

```text
base_rule gợi ý phối cùng:
- Áo mặc trong
- Quần/Chân váy
- Giày dép
```

Hàm sẽ tìm rule chi tiết cho từng category đó.

Nó có hai cách:

1. Khớp chính xác bằng text.
2. Fallback semantic search trong Qdrant nếu không khớp.

---

## 15. Mapping Layer B sang Layer A

Layer B nói theo ngôn ngữ stylist:

```text
Áo mặc trong (áo thun/sơ mi)
Quần/Chân váy
Giày dép
```

Layer A nói theo ngôn ngữ kho sản phẩm:

```text
Áo
Quần
Chân váy
Giày
```

Vì vậy cần mapping.

### `CATEGORY_MAPPING`

Dịch category Layer B sang category Layer A.

Ví dụ:

```text
"Áo mặc trong (áo thun/sơ mi)" -> ["Áo"]
"Đầm/Jumpsuit" -> ["Đầm", "Jumpsuit"]
"Giày dép" -> ["Giày"]
```

### `PHU_KIEN_KEYWORD_ROUTER`

Riêng “Phụ kiện” quá rộng. Notebook dùng keyword router để đoán loại phụ kiện:

```text
watch -> Đồng hồ
necklace -> Dây chuyền
gloves -> Găng tay
```

### `get_layer_a_categories(layer_b_category, product_type)`

Trả về category Layer A tương ứng.

### `get_products_for_outfit(product_type, layer_b_category, phong_cach, vdb)`

Tìm sản phẩm thật cho một món trong outfit.

Luồng:

```text
product_type + phong_cach
    -> map category Layer B sang Layer A
    -> Qdrant similarity_search_with_score
    -> giữ score >= 0.30
    -> diversity_filter_documents(max_docs=3)
```

---

## 16. Vision module: hiểu ảnh người và ảnh sản phẩm

### `VL_MAX_SIZE`

Resize ảnh về tối đa 512px để tránh lỗi Qwen-VL khi ảnh quá lớn.

### `_preprocess_image(image_path)`

Làm 3 việc:

1. Mở ảnh.
2. Convert RGB.
3. Resize nếu cạnh lớn nhất > 512.
4. Encode JPEG base64.

### `_call_vl(image_path, prompt)`

Gọi Qwen-VL qua Ollama.

Nếu lỗi, trả chuỗi rỗng thay vì làm crash chatbot.

### `detect_image_type(image_path, user_query)`

Phân loại ảnh thành:

- `product`
- `person`

Điểm thông minh: nó không chỉ nhìn ảnh, mà nhìn cả câu hỏi đi kèm.

Ví dụ:

- Ảnh người mẫu mặc váy + user hỏi “tìm váy giống ảnh” -> `product`
- Ảnh selfie + user hỏi “tôi hợp đồ gì” -> `person`

### `analyze_person_image(image_path)`

Trích xuất:

- `dang_nguoi`
- `tone_da`
- `nhan_xet`

Danh sách dáng và tone được ép khớp với Layer B:

```text
Dáng quả lê, Dáng quả táo, Dáng đồng hồ cát, ...
Da sáng, Da trung bình, Da ngăm, Da ấm
```

Nếu VL model trả nhãn lệch khỏi Layer B, filter Qdrant sẽ kém. Vì vậy prompt bắt model chọn đúng trong danh sách.

### `caption_product_image(image_path, user_query)`

Fallback khi image vector search không khả dụng.

Nó tạo mô tả món đồ:

- loại sản phẩm
- màu sắc
- kiểu dáng
- chất liệu nếu nhận ra
- phong cách

Sau đó caption được ghép vào query text để chạy product search.

---

## 17. Prompt và LLM chain

### Search prompt

Prompt search yêu cầu LLM:

- Chỉ dùng dữ liệu trong context.
- Không bịa mã, giá, ảnh, đặc điểm.
- Tối đa 5 sản phẩm.
- Trả lời theo schema cố định.
- Kết thúc bằng một câu hỏi gợi mở.

Schema:

```text
1. **Tên sản phẩm**
- Mã SP: [MÃ_SP]
- Giá: [GIÁ] VND
- Đặc điểm: ...
- Lý do phù hợp: ...
- Ảnh: ![Sản phẩm]([IMAGE_URL])
```

Schema này không chỉ để đẹp. Nó giúp:

- Frontend dễ render.
- Guardrail dễ extract mã sản phẩm.
- Luận văn dễ đánh giá output.

### `contextualize_q_prompt`

Viết lại câu hỏi dựa trên lịch sử chat.

Ví dụ:

```text
User: Có màu khác không?
```

Nếu trước đó bot vừa nói về áo thun đỏ, nó viết lại thành:

```text
Áo thun đỏ ở trên có màu khác không?
```

Điểm quan trọng: prompt cấm LLM trả lời câu hỏi. Nó chỉ được viết lại query.

### `doc_prompt`

Format từng Document thành context:

```text
[MÃ_SP: ...]
IMAGE_URL: ...
THÔNG TIN CHI TIẾT: ...
```

### `format_documents_for_llm(docs)`

Ghép nhiều document thành một context string cho prompt.

### Outfit prompt

Prompt outfit yêu cầu:

- Chỉ giới thiệu sản phẩm trong “SẢN PHẨM GỢI Ý”.
- Tối đa 3 sản phẩm.
- Gắn lý do với công thức phối đồ.
- Có ảnh nếu có `IMAGE_URL`.

---

## 18. Redis chat history và tóm tắt lịch sử

### `summarize_history(messages)`

Khi lịch sử dài, gọi LLM để tóm tắt các message cũ.

Giữ lại:

- Sản phẩm đã hỏi.
- Phong cách khách thích.
- Dáng người/tone da nếu có.

Bỏ qua:

- Lời chào.
- Câu xã giao.

### `get_message_history(session_id)`

Lấy lịch sử từ Redis.

Chiến lược:

```text
Nếu <= 8 messages:
    giữ nguyên
Nếu > 8 messages:
    tóm tắt phần cũ
    giữ 4 messages gần nhất
```

Lợi ích:

- Không vượt context window.
- Vẫn giữ memory quan trọng.
- Tăng ổn định khi chat dài.

---

## 19. RAG pipeline trong LangChain

Notebook lắp 3 chain:

### `full_chat_chain`

Dùng cho text search.

Luồng:

```text
input
    -> history-aware retriever viết lại query
    -> product retriever lấy docs
    -> document_chain nhét docs vào QA_PROMPT
    -> LLM sinh answer
    -> Redis lưu history
```

### `product_answer_chain_with_history`

Dùng cho image search.

Vì image search đã lấy docs trước rồi, chain này không gọi retriever nữa.

Luồng:

```text
uploaded image
    -> search_products_by_image()
    -> docs
    -> format_documents_for_llm()
    -> QA_PROMPT | LLM
```

### `outfit_chain_with_history`

Dùng cho tư vấn phối đồ.

Vì `build_outfit_context()` đã tự tìm rule và sản phẩm, chain này cũng không gọi retriever chung.

Luồng:

```text
query
    -> build_outfit_context()
    -> outfit_prompt | LLM
```

---

## 20. Router: trái tim điều phối

Nếu không có router, mọi câu hỏi đều bị đẩy vào RAG. Điều này rất tệ.

Ví dụ:

- “xin chào” không cần search Qdrant.
- “bạn nhớ gì về tôi?” không cần LLM sản phẩm.
- “2+2=?” không phải thời trang.
- “xem thêm” cần biết lượt trước là gì.

Router giải quyết điều đó.

### Route chính

| Route | Intent cũ | Khi nào dùng |
|---|---|---|
| `product_search` | `search` | Tìm sản phẩm, hỏi giá, size, stock |
| `image_product_search` | `image_search` | Ảnh sản phẩm đã tìm được docs |
| `outfit_advice` | `outfit` | Phối đồ, mặc gì, mix-match |
| `profile_inquiry` | `profile_inquiry` | Hỏi bot đang nhớ gì về user |
| `out_of_scope` | `out_of_scope` | Ngoài thời trang/mua sắm |
| `greeting` | `greeting` | Chào hỏi |
| `chitchat` | `chitchat` | Cảm ơn, tạm biệt |
| `clarify` | `clarify` | Câu quá mơ hồ |

### `RouteDecision`

Dataclass lưu kết quả route:

```text
route
action
confidence
rewrite_query
entities
missing_slots
reason
source
```

Đây là một thay đổi lớn về tư duy: thay vì trả một string `intent`, router trả một quyết định có cấu trúc. Nhờ vậy hệ thống biết không chỉ “đi đâu”, mà còn “vì sao đi đường đó” và “cần hành động gì”.

### Text normalization helpers

| Hàm | Công dụng |
|---|---|
| `strip_vietnamese_accents` | Bỏ dấu tiếng Việt |
| `normalize_text` | Lowercase và chuẩn hóa khoảng trắng |
| `plain_text` | Bỏ dấu + normalize |
| `keyword_hit` | Match keyword cả có dấu và không dấu |

### `is_more_request(query)`

Nhận diện các câu như:

```text
xem thêm
cho xem thêm
mẫu khác
gợi ý thêm
```

Nếu có previous route, nó reuse route/query trước.

### `infer_product_action(query)`

Phân loại hành động nhỏ trong product search:

- `size_check`
- `price_check`
- `stock_check`
- `compare`
- `more`
- `search`

### `extract_basic_entities(query)`

Trích entity đơn giản:

- màu
- category
- dịp sử dụng
- size
- budget

### `route_from_keywords(query, state)`

Router tầng nhanh:

1. Check follow-up “xem thêm”.
2. Check greeting.
3. Check chitchat.
4. Check profile inquiry.
5. Check out-of-scope.
6. Check outfit keyword.
7. Check product search keyword.

Nếu không match thì trả `None`.

### `classify_route_llm(query, last_bot_msg)`

Router tầng chậm dùng LLM.

Nó yêu cầu LLM trả JSON:

```json
{
  "route": "product_search",
  "action": "search",
  "confidence": 0.8,
  "rewrite_query": "áo thun trắng nữ",
  "entities": {},
  "missing_slots": [],
  "reason": "..."
}
```

### `route_user_request(...)`

Hàm router chính:

```text
Nếu force_image_search:
    image_product_search
Nếu keyword route bắt được:
    dùng keyword route
Ngược lại:
    dùng LLM router
```

---

## 21. Simple response handlers

Các intent đơn giản không cần gọi RAG:

| Hàm | Công dụng |
|---|---|
| `get_greeting_response()` | Trả lời chào hỏi |
| `get_chitchat_response(query)` | Trả lời cảm ơn/tạm biệt |
| `get_profile_inquiry_response(profile)` | Trả lời thông tin bot đang nhớ |
| `get_out_of_scope_response(query)` | Từ chối mềm và kéo về thời trang |
| `get_clarify_response(decision)` | Hỏi lại khi thiếu thông tin |

Đây là một nguyên tắc vận hành tốt: **không phải request nào cũng đáng gọi LLM lớn**.

---

## 22. `build_outfit_context`: nơi Layer B gặp Layer A

Đây là hàm quan trọng nhất cho phối đồ.

Luồng:

```text
user_query + gender + profile
    -> find_matching_rule()
    -> find_outfit_details()
    -> với từng món trong outfit:
           get_products_for_outfit()
    -> build context:
           CÔNG THỨC PHỐI ĐỒ
           SẢN PHẨM GỢI Ý
```

Output là text context:

```text
CÔNG THỨC PHỐI ĐỒ:
  Phong cách: ...
  Bối cảnh : ...
  Lý do    : ...
  Dáng người: ...
  Tone da   : ...

SẢN PHẨM GỢI Ý:

[Áo mặc trong - Áo thun cổ tròn]
  Lý do: ...
  - (Mã SP: ... | Giá: ... VND | IMAGE_URL: ...)
    Tên sản phẩm: ...
    Màu sắc: ...
```

LLM sau đó không tự nghĩ sản phẩm. Nó chỉ viết lại từ context này.

---

## 23. Security và guardrail

### `validate_user_query(query)`

Chặn:

- Tin nhắn quá dài.
- Prompt injection kiểu “ignore previous instructions”.
- Yêu cầu tiết lộ system prompt.

Đây là lớp kiểm tra trước khi vào RAG.

### `extract_product_ids_from_docs(docs)`

Lấy danh sách mã sản phẩm hợp lệ từ context.

### `extract_product_ids_from_text(text)`

Regex tìm mã sản phẩm trong câu trả lời LLM.

### `check_answer_grounding(answer, allowed_product_ids, query, route)`

So sánh:

```text
mã sản phẩm LLM nhắc tới
        vs
mã sản phẩm có trong context
```

Nếu LLM nhắc mã lạ, log vào file hallucination warning.

### `append_chat_turn_log(record)`

Ghi log mỗi lượt:

- session_id
- query
- route
- action
- intent
- TTFT
- total latency
- reranker enabled
- grounding ok
- unknown ids

Đây là phần biến chatbot thành hệ thống nghiên cứu đo được, không chỉ demo cảm tính.

---

## 24. Eval-lite cho luận văn

Notebook có nhóm hàm eval:

| Hàm | Công dụng |
|---|---|
| `doc_text_for_eval` | Chuẩn hóa doc thành text để chấm |
| `score_retrieval_case` | Chấm một case retrieval |
| `run_retrieval_eval` | Chạy nhiều case |
| `summarize_eval_rows` | Tổng hợp kết quả |
| `save_eval_rows` | Lưu kết quả ra file |
| `run_retrieval_ab_eval` | So sánh baseline vs improved retrieval |

Mục đích:

- Chứng minh retrieval có cải thiện.
- So sánh dense retrieval với dense + reranker.
- Có số liệu cho luận văn thay vì chỉ screenshot chatbot.

---

## 25. Chat loop chính: toàn bộ hệ thống chạy như thế nào?

Đây là luồng notebook khi người dùng nhập một lượt chat.

### Bước 1: Nhận input

```text
user_input = input("Bạn: ")
raw_img = input("Ảnh: ")
```

Nếu nhập `0`, thoát.

### Bước 2: Validate text

```text
validate_user_query(user_input)
```

Nếu vi phạm, trả message nhẹ nhàng và dừng lượt.

### Bước 3: Xử lý ảnh nếu có

Nếu có ảnh:

```text
detect_image_type(raw_img, user_input)
```

#### Nếu là ảnh người

```text
analyze_person_image()
    -> lưu dang_nguoi, tone_da vào user_profile
    -> trả lời đã lưu profile
    -> kết thúc lượt
```

#### Nếu là ảnh sản phẩm

Ưu tiên:

```text
search_products_by_image()
```

Nếu có docs:

```text
force_image_search = True
intent = image_search
```

Nếu không có docs:

```text
caption_product_image()
final_query = caption + user_input
```

### Bước 4: Validate lại `final_query`

Vì caption ảnh có thể làm query dài hơn hoặc chứa nội dung lạ, hệ thống validate lần nữa.

### Bước 5: Router quyết định đường đi

```text
decision = route_user_request(
    final_query,
    last_bot_msg,
    chat_state,
    force_image_search
)
```

Output:

```text
route, action, intent, source, rewrite_query
```

### Bước 6: Detect gender

```text
detect_gender(active_query)
```

Chỉ trả `"male"` nếu query có keyword nam rõ ràng. Nếu không, giữ profile cũ hoặc default female.

### Bước 7: Xử lý route đơn giản

Các route không cần RAG:

- greeting
- chitchat
- profile_inquiry
- out_of_scope
- clarify

### Bước 8: Route `image_search`

```text
image_context = format_documents_for_llm(image_search_docs)
product_answer_chain_with_history.stream({
    "input": active_query,
    "context": image_context
})
```

LLM trả lời dựa trên sản phẩm tìm được bằng ảnh.

### Bước 9: Route `outfit`

```text
outfit_context = build_outfit_context(active_query, gender, user_profile)
```

Nếu không có context:

```text
fallback sang product_search
```

Nếu có:

```text
outfit_chain_with_history.stream({
    "input": active_query,
    "outfit_context": outfit_context
})
```

### Bước 10: Route `search`

```text
full_chat_chain.stream({"input": active_query})
```

Chain tự:

1. Viết lại query theo history.
2. Retrieve sản phẩm.
3. Nhét docs vào prompt.
4. Stream answer.

### Bước 11: Metrics

Notebook tính:

- TTFT: time to first token.
- Tổng thời gian.

### Bước 12: Grounding check

```text
check_answer_grounding(answer_text, allowed_product_ids, active_query, route)
```

### Bước 13: Log và cập nhật state

Lưu:

- last_route_decision
- last_query
- last_bot_msg

Nhờ vậy lượt sau “xem thêm” hiểu được đang xem thêm cái gì.

---

## 26. Ba luồng quan trọng nhất dưới dạng sơ đồ

### 26.1 Text search

```text
User: "Tìm áo thun trắng nữ"
        |
        v
validate_user_query
        |
        v
route_user_request -> product_search
        |
        v
full_chat_chain
        |
        +-> contextualize query
        +-> Qdrant ViFashionCLIP text retrieval
        +-> reranker
        +-> diversity filter
        +-> QA_PROMPT
        |
        v
LLM answer with product IDs and images
        |
        v
grounding check + log
```

### 26.2 Image product search

```text
User uploads product image
        |
        v
detect_image_type -> product
        |
        v
FashionCLIP encode uploaded image
        |
        v
Qdrant image collection search
        |
        v
Documents for similar products
        |
        v
QA_PROMPT | LLM
        |
        v
grounding check + log
```

Nếu image search không có kết quả:

```text
Qwen-VL caption image -> text search fallback
```

### 26.3 Outfit advice

```text
User: "Phối đồ đi tiệc cho dáng quả lê"
        |
        v
route_user_request -> outfit_advice
        |
        v
find_matching_rule in Layer B
        |
        v
find_outfit_details
        |
        v
For each item type:
    get_products_for_outfit from Layer A
        |
        v
build_outfit_context
        |
        v
OUTFIT prompt | LLM
        |
        v
grounding check + log
```

---

## 27. Điều làm notebook này “research-demo v3”

Bản v3 không chỉ là chatbot chạy được. Nó có các yếu tố nghiên cứu:

1. **ViFashionCLIP tiếng Việt** cho retrieval sản phẩm.
2. **Image retrieval thật** bằng FashionCLIP, không chỉ caption ảnh.
3. **Reranker** để cải thiện thứ tự kết quả.
4. **Diversity filter** để output không trùng lặp.
5. **Router có cấu trúc** thay vì intent string đơn giản.
6. **Grounding guardrail** kiểm tra hallucination mã sản phẩm.
7. **Eval-lite** để đo retrieval.
8. **Logging latency và grounding** để phân tích vận hành.
9. **Profile-aware outfit advice** dùng dáng người/tone da.
10. **Fallback mềm** ở nhiều tầng, tránh crash hoặc trả lời bừa.

---

## 28. Những điểm dễ hiểu nhầm

### Hiểu nhầm 1: “LLM tìm sản phẩm”

Không đúng. Qdrant/retriever tìm sản phẩm. LLM chỉ viết câu trả lời dựa trên sản phẩm đã tìm.

### Hiểu nhầm 2: “Ảnh sản phẩm được xử lý bằng Qwen-VL là chính”

Không hẳn. Qwen-VL dùng để phân loại ảnh và fallback caption. Tìm sản phẩm bằng ảnh chính thức dùng FashionCLIP image embedding.

### Hiểu nhầm 3: “BGE-M3 là embedding toàn hệ thống”

Ở v3, BGE-M3 chỉ dùng cho Layer B. Sản phẩm Layer A dùng ViFashionCLIP.

### Hiểu nhầm 4: “Router chỉ là intent classifier”

Router v3 là decision object. Nó có route, action, rewrite_query, entities, reason, source. Nó là bộ điều phối, không chỉ phân loại.

### Hiểu nhầm 5: “Prompt mạnh là đủ chống bịa”

Không đủ. Notebook thêm grounding check sau khi LLM trả lời. Prompt là lời dặn, guardrail là kiểm tra.

---

## 29. Cách đọc notebook nếu bạn mới bắt đầu

Đừng đọc từ cell 1 đến cell cuối như đọc truyện. Hãy đọc theo vòng:

### Vòng 1: Hiểu mục tiêu

Đọc:

- Cell giới thiệu.
- Constants.
- Chat loop cuối.

Mục tiêu là hiểu hệ thống nhận gì và trả gì.

### Vòng 2: Hiểu dữ liệu

Đọc:

- `build_product_page_content`
- `process_fashion_metadata`
- `run_data_pipeline`
- `load_layer_b`
- `index_layer_b`

Mục tiêu là hiểu cái gì được index vào Qdrant.

### Vòng 3: Hiểu retrieval

Đọc:

- `ViFashionCLIPTextEmbeddings`
- `FashionCLIPImageEmbeddings`
- `DiversityFilteredRetriever`
- `search_products_by_image`

Mục tiêu là hiểu “bằng chứng” được lấy ra thế nào.

### Vòng 4: Hiểu reasoning có kiểm soát

Đọc:

- Prompt search.
- Prompt outfit.
- `build_outfit_context`.
- `check_answer_grounding`.

Mục tiêu là hiểu LLM bị giới hạn thế nào.

### Vòng 5: Hiểu vận hành

Đọc:

- Redis history.
- Router.
- Validation.
- Logging.
- Eval.

Mục tiêu là hiểu vì sao hệ thống chạy bền hơn notebook demo thông thường.

---

## 30. Tóm tắt một câu cho từng khối hàm

| Khối | Một câu tóm tắt |
|---|---|
| ViFashionCLIP | Biến text sản phẩm tiếng Việt thành vector thời trang 512 chiều |
| FashionCLIP image | Biến ảnh sản phẩm thành vector thị giác 512 chiều |
| BGE-M3 | Biến rule stylist thành vector ngữ nghĩa 1024 chiều |
| Product metadata | Biến JSONL thô thành Document có text và metadata |
| Product retriever | Lấy sản phẩm phù hợp, rerank và lọc đa dạng |
| Image retrieval | Tìm sản phẩm tương tự ảnh upload |
| Layer B | Chọn công thức phối đồ theo phong cách, bối cảnh, dáng người, tone da |
| Category mapping | Dịch ngôn ngữ stylist sang ngôn ngữ kho hàng |
| Vision module | Phân loại ảnh, phân tích người, caption sản phẩm fallback |
| Prompt | Ép LLM trả lời theo schema và không bịa |
| History | Nhớ hội thoại dài bằng Redis và tóm tắt |
| Router | Chọn đường xử lý đúng cho từng request |
| Security | Chặn input nguy hiểm và kiểm tra hallucination |
| Eval | Đo retrieval phục vụ nghiên cứu |
| Chat loop | Dàn nhạc điều phối toàn bộ hệ thống |

---

## 31. Bức tranh cuối cùng

Hệ thống này nên được nhìn như một pipeline có kiểm soát:

```text
Input của user
    -> hiểu ý định
    -> hiểu modality text/ảnh
    -> lấy bằng chứng đúng
    -> dùng LLM viết câu trả lời
    -> kiểm tra câu trả lời
    -> lưu trạng thái cho lượt sau
```

Giá trị thật của hệ thống không nằm ở việc “gọi một model mạnh”. Nó nằm ở việc chia bài toán thành các tầng nhỏ:

- Tầng hiểu ảnh.
- Tầng hiểu text.
- Tầng tìm sản phẩm.
- Tầng tìm công thức phối đồ.
- Tầng nhớ lịch sử.
- Tầng điều phối route.
- Tầng viết câu trả lời.
- Tầng kiểm tra bịa đặt.

Khi nhìn như vậy, bạn sẽ thấy RAG không phải là “nhét dữ liệu vào prompt”. RAG là thiết kế một hệ thống biết **khi nào cần tìm**, **tìm ở đâu**, **tìm bằng mô hình nào**, **đưa bằng chứng nào cho LLM**, và **kiểm tra LLM sau khi nó nói**.

Đó là góc nhìn quan trọng nhất của notebook v3.
