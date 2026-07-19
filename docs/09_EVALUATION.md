# Kế hoạch và cách đánh giá hệ thống

## Nguyên tắc

Không dùng `confidence` do LLM tự khai làm độ chính xác. Độ tin cậy phải đến từ tập dữ liệu có nhãn và metric đo được.

## Router evaluation

Tập hiện tại: `tests/router_eval_cases.jsonl`, gồm hơn 40 câu có dấu/không dấu, text/ảnh, social, profile, out-of-scope, clarification và negative boundary cases.

Chạy:

```powershell
python -m unittest tests.test_router_eval_dataset -v
```

Metric cần báo cáo:

- intent accuracy;
- action accuracy;
- route accuracy;
- clarification precision/recall;
- false-positive rate trên negative cases;
- tỷ lệ request cần LLM fallback;
- router latency p50/p95.

Tập hiện tại là regression suite, chưa đủ để tuyên bố khả năng tổng quát. Khi viết báo cáo, tách development cases và held-out evaluation cases.

## Retrieval evaluation

### Layer A text/image

- Recall@K: sản phẩm/category đúng có trong top K hay không.
- Precision@K: tỷ lệ kết quả phù hợp.
- MRR hoặc nDCG: thứ hạng item đúng.
- Diversity: số brand/category khác nhau.
- Image retrieval: đánh giá item/category visually similar bằng nhãn người chấm.

### Layer B outfit

- Rule relevance: rule có khớp dịp/phong cách/profile.
- Slot correctness: category mỗi slot hợp rule.
- Slot coverage: đủ số slot cần thiết.
- Duplicate-category error rate.
- Product availability in catalog: mỗi slot lấy được sản phẩm thật.

## Grounding evaluation

Tạo câu trả lời có chủ ý chứa:

- ID hợp lệ;
- ID lạ;
- dòng giá/thương hiệu/ảnh do model tự viết;
- ID bị chia giữa nhiều stream chunk.

Đo:

- unknown-ID detection recall;
- commerce-fact filtering recall;
- tỷ lệ loại nhầm nội dung tư vấn hợp lệ;
- số câu trả lời cần retry — policy hiện tại đặt mục tiêu bằng 0.

## End-to-end và latency

Cho mỗi route, ghi:

```text
router time
time-to-first-card
time-to-first-answer-token
total time
model call counts
vector counts
error/timeout rate
```

Tối thiểu có các scenario: text search, image search, text outfit, image outfit, profile VLM, follow-up, ambiguous clarification.

## Human evaluation

Người chấm dùng thang 1–5 cho:

- mức phù hợp nhu cầu;
- tính nhất quán giữa card và lời tư vấn;
- tính hợp lý của outfit;
- tính tự nhiên/thân thiện;
- mức dễ hiểu của câu hỏi làm rõ.

Mỗi mẫu nên có ít nhất hai người chấm; báo cáo tiêu chí, số mẫu và cách xử lý bất đồng.

## Những gì không được tuyên bố

- Không gọi `certainty=deterministic` là “độ chính xác 100%”.
- Không gọi VLM confidence là router accuracy.
- Không dùng vài câu demo thành benchmark chính thức.
- Không đánh đồng Qdrant similarity score giữa các collection/model khác nhau.

